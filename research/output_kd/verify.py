"""Real checkpoint + synthetic engineering tests; no accuracy claims."""
import argparse
import copy
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from runtime import components
from teacher import build_teacher, output_loss, teacher_prediction, sha256
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--teacher-checkpoint', type=Path, required=True)
    parser.add_argument('--production-batch', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((HERE / 'configs/SPAN-KD-Output.json').read_text())
    cfg['distillation']['checkpoint'] = str(args.teacher_checkpoint)
    arch, _ = components()
    torch.manual_seed(10)
    student = arch.SPAN(**cfg['model']).to(args.device)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    teacher = build_teacher(cfg, args.device)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    teacher_state = {k: v.clone() for k, v in teacher.state_dict().items()}
    # Compare chunking in full FP32: TF32 convolutions can differ by batch shape.
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for shape in ((1, 3, 13, 17), (2, 3, 16, 24)):
        lr = torch.rand(shape, device=args.device)
        target = teacher_prediction(teacher, lr)
        with torch.no_grad():
            direct = teacher(lr)
        torch.testing.assert_close(target, direct, atol=2e-6, rtol=2e-5)
        prediction = student(lr)
        loss, rec, kd = output_loss(prediction, torch.zeros_like(prediction), target, .1)
        torch.testing.assert_close(loss, rec+.1*kd)
        assert not target.requires_grad and torch.isfinite(loss)
    torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
    torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
    lr = torch.rand(1, 3, 8, 8, device=args.device)
    hr = torch.rand(1, 3, 32, 32, device=args.device)
    target = teacher_prediction(teacher, lr)
    opt = torch.optim.Adam((p for p in student.parameters() if p.requires_grad), lr=2e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [200000,400000,600000,800000], gamma=.5)
    before = {k: p.clone() for k, p in student.named_parameters() if p.requires_grad}
    def step(net, optimizer, scheduler):
        net.train(); optimizer.zero_grad(set_to_none=True)
        total, _, _ = output_loss(net(lr), hr, target, .1)
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters() if p.requires_grad)
        optimizer.step(); scheduler.step()
    step(student, opt, sched)
    assert any(not torch.equal(before[k], p) for k, p in student.named_parameters() if p.requires_grad)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(v, teacher.state_dict()[k]) for k, v in teacher_state.items())
    stream = io.BytesIO()
    torch.save(dict(model=student.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict()), stream)
    stream.seek(0); state = torch.load(stream, weights_only=True)
    clone = arch.SPAN(**cfg['model']).to(args.device)
    clone.load_state_dict(state['model'], strict=True)
    opt2 = torch.optim.Adam((p for p in clone.parameters() if p.requires_grad), lr=2e-4)
    sched2 = torch.optim.lr_scheduler.MultiStepLR(opt2, [200000,400000,600000,800000], gamma=.5)
    opt2.load_state_dict(state['optimizer']); sched2.load_state_dict(state['scheduler'])
    step(student,opt,sched); step(clone,opt2,sched2)
    different = {k: float((v-clone.state_dict()[k]).abs().max()) for k,v in student.state_dict().items()
                 if not torch.equal(v, clone.state_dict()[k])}
    assert not different, different
    assert sched.state_dict() == sched2.state_dict()
    student.eval(); clone.eval()
    with torch.no_grad():
        torch.testing.assert_close(student(lr),clone(lr),rtol=0,atol=0)
    assert sum(p.numel() for p in teacher.parameters()) > 10000000
    assert not any('teacher' in k for k in student.state_dict())
    from deploy import fused_copy
    deployed = fused_copy(student, arch.Conv3XC).to(args.device).eval()
    assert sum(p.numel() for p in deployed.parameters()) == 498208
    with torch.no_grad():
        torch.testing.assert_close(student(lr), deployed(lr), rtol=2e-5, atol=2e-5)
    if args.production_batch:
        assert args.device.startswith('cuda')
        production_lr = torch.rand(16,3,64,64,device=args.device)
        production_hr = torch.rand(16,3,256,256,device=args.device)
        production_target = teacher_prediction(teacher, production_lr)
        student.train(); opt.zero_grad(set_to_none=True)
        total,_,_ = output_loss(student(production_lr),production_hr,production_target,.1)
        total.backward()
        assert torch.isfinite(total)
        assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.requires_grad)
    # KD gradient must exist even with the HR target equal to the student output.
    probe = torch.zeros(1,3,4,4,device=args.device,requires_grad=True)
    total,_,_ = output_loss(probe,probe.detach(),torch.ones_like(probe),.1)
    total.backward(); assert probe.grad.abs().sum() > 0
    report = dict(status='PASS', device=args.device, teacher_sha256=sha256(args.teacher_checkpoint),
                  teacher_parameters=sum(p.numel() for p in teacher.parameters()),
                  checks=['strict official weight loading','teacher frozen/no grad/no state drift',
                          'RNG preservation','odd rectangular batch shapes','teacher chunk equivalence',
                          'loss formula and KD gradient','student updates','exact serialized continuation',
                          'student-only checkpoint and eval'], evidence='engineering only')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
