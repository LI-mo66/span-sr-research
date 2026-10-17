"""Synthetic checks only; no real-data accuracy claims."""
import argparse
import io
import json
from pathlib import Path
import torch
from entry import existing
from sg_model import build_model
from gb_model import build_model as build_gb
from deploy import fused_copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cfg = json.loads((Path(__file__).parent/'configs/LG6-SiLU-SG6.json').read_text())
    arch, _ = existing.fs_runtime.components()
    torch.manual_seed(10)
    gb = build_gb(arch, cfg).to(args.device)
    rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state_all()
    torch.manual_seed(10)
    model = build_model(arch, cfg).to(args.device)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(gpu_rng,torch.cuda.get_rng_state_all()))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in gb.state_dict().items())
    for shape in ((1,3,13,17),(2,3,16,20)):
        x = torch.rand(*shape,device=args.device)
        for training in (True,False):
            gb.train(training); model.train(training)
            with torch.no_grad():
                y = model(x)
                torch.testing.assert_close(y,gb(x),rtol=0,atol=0)
                assert y.shape == (shape[0],3,4*shape[2],4*shape[3])
    # Preserve trained GB state keys and exact zero-correction mapping as well.
    with torch.no_grad():
        gb.image_residual.weight.normal_(0,.01)
        gb.image_gate.weight.normal_(0,.01)
        for i in range(1,7):
            getattr(gb,f'block_{i}').local_gate.weight.normal_(0,.001)
    merged = model.state_dict()
    merged.update(gb.state_dict())
    model.load_state_dict(merged,strict=True)
    with torch.no_grad():
        for i in range(1,7):
            b=getattr(model,f'block_{i}')
            a=getattr(gb,f'block_{i}')
            feature=torch.ones(2,52,7,9,device=args.device)
            for actual,expected in zip(b(feature),a(feature)):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(model(x),gb(x),rtol=0,atol=0)
    # Restart from protocol initialization for staged gradient checks.
    torch.manual_seed(10)
    model = build_model(arch,cfg).to(args.device)
    params=[v for v in model.parameters() if v.requires_grad]
    opt=torch.optim.Adam(params,lr=.0002,betas=(.9,.999),eps=1e-8)
    ids=[id(v) for g in opt.param_groups for v in g['params']]
    assert len(ids)==len(set(ids))==len(params)
    x=torch.rand(2,3,16,20,device=args.device)
    target=torch.rand(2,3,64,80,device=args.device)
    before={i:getattr(model,f'block_{i}').spatial_gate.depth.weight.detach().clone() for i in range(1,7)}
    started={i:False for i in before}
    losses=[]
    for step in range(20):
        model.train(); opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.l1_loss(model(x),target)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
        for i in before:
            s=getattr(model,f'block_{i}').spatial_gate
            assert s.project.weight.grad.abs().sum()>0
            if step==0:
                assert s.depth.weight.grad.abs().sum()==0
            else:
                started[i] |= bool(s.depth.weight.grad.abs().sum()>0)
        opt.step(); losses.append(float(loss.detach()))
    assert all(started.values())
    assert all(not torch.equal(v,getattr(model,f'block_{i}').spatial_gate.depth.weight) for i,v in before.items())
    buffer=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),buffer);buffer.seek(0)
    state=torch.load(buffer,map_location=args.device,weights_only=False)
    clone=build_model(arch,cfg).to(args.device);clone.load_state_dict(state['model'],strict=True)
    clone_opt=torch.optim.Adam((v for v in clone.parameters() if v.requires_grad),lr=.0002);clone_opt.load_state_dict(state['optimizer'])
    for net,optimizer in ((model,opt),(clone,clone_opt)):
        net.train();optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.l1_loss(net(x),target).backward();optimizer.step()
    assert all(torch.equal(v,clone.state_dict()[k]) for k,v in model.state_dict().items())
    model.eval();deployed=fused_copy(model,arch.Conv3XC)
    with torch.no_grad():
        torch.testing.assert_close(model(x),deployed(x),rtol=1e-5,atol=1e-6)
    assert sum(v.numel() for v in deployed.parameters())==535772
    assert model.extra_parameters==37564
    print(json.dumps(dict(status='PASS',device=args.device,first_loss=losses[0],last_loss=losses[-1],
        deployed_parameters=535772,extra_over_GB=19656,
        checks=['GB exact initialization and shared RNG','nonzero GB weights and block return semantics',
                'rectangular batch1/2 train/eval','six projections first step; six depth kernels later updates',
                'finite gradients and optimizer coverage','strict state and optimizer resume equality',
                'deployment agreement'],evidence='Synthetic only; no real PSNR measured'),indent=2))


if __name__=='__main__':main()
