"""Inference identity, phased reconstruction, staged learning and resume audit."""
import argparse
import copy
import io
import json
from pathlib import Path
import torch
from entry import existing
from ms_model import build_model
from lg_model import build_model as build_a
from deploy import fused_copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cfg = json.loads((Path(__file__).parent/'configs/LG6-SiLU-MS.json').read_text())
    arch, _ = existing.fs_runtime.components()
    torch.manual_seed(10)
    a = build_a(arch, cfg).to(args.device)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    torch.manual_seed(10)
    model = build_model(arch, cfg).to(args.device)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(u,v) for u,v in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in a.state_dict().items())
    for shape in ((1,3,13,17),(2,3,16,20)):
        x = torch.rand(*shape, device=args.device)
        for training in (True,False):
            a.train(training); model.train(training)
            with torch.no_grad():
                expected, actual = a(x), model(x)
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                assert actual.shape == (shape[0],3,shape[2]*4,shape[3]*4)
    # Inject known channel/phase corrections and spatial gate, compare to RGB-space oracle.
    model.eval(); a.eval()
    with torch.no_grad():
        model.image_residual.project.bias.copy_(torch.linspace(-.1,.1,48,device=args.device))
        model.image_gate.weight[0,0,1,1] = 1
        expected = a(x) + torch.nn.functional.pixel_shuffle(
            2*torch.sigmoid(model.image_gate(x))*model.image_residual(x),4)
        torch.testing.assert_close(model(x),expected,rtol=1e-5,atol=1e-6)
        model.image_residual.project.bias.zero_();model.image_gate.weight.zero_()
    params = [v for v in model.parameters() if v.requires_grad]
    opt = torch.optim.Adam(params, lr=.0002, betas=(.9,.999),eps=1e-8)
    ids = [id(v) for g in opt.param_groups for v in g['params']]
    assert len(ids)==len(set(ids))==len(params)
    target=torch.rand(2,3,64,80,device=args.device)
    x=torch.rand(2,3,16,20,device=args.device)
    losses=[]; gate_started=False
    hidden_started={name:False for name in ('stem','local','wide')}
    hidden_before={name:getattr(model.image_residual,name).weight.detach().clone() for name in hidden_started}
    for step in range(20):
        model.train();opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.l1_loss(model(x),target)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
        assert model.image_residual.project.weight.grad.abs().sum()>0
        for name in hidden_started:
            gradient=getattr(model.image_residual,name).weight.grad
            if step==0:assert gradient.abs().sum()==0
            else:hidden_started[name] |= bool(gradient.abs().sum()>0)
        if step==0:
            assert model.image_gate.weight.grad.abs().sum()==0
            assert model.image_gate.bias.grad.abs().sum()==0
        else:
            gate_started |= bool(model.image_gate.weight.grad.abs().sum()>0)
        opt.step();losses.append(float(loss.detach()))
    assert all(hidden_started.values())
    assert all(not torch.equal(value,getattr(model.image_residual,name).weight) for name,value in hidden_before.items())
    assert gate_started and model.image_gate.weight.abs().sum()>0 and model.image_residual.project.weight.abs().sum()>0
    stream=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),stream);stream.seek(0)
    state=torch.load(stream,map_location=args.device,weights_only=False)
    clone=build_model(arch,cfg).to(args.device);clone.load_state_dict(state['model'],strict=True)
    clone_opt=torch.optim.Adam((v for v in clone.parameters() if v.requires_grad),lr=.0002)
    clone_opt.load_state_dict(state['optimizer'])
    for net,optimizer in ((model,opt),(clone,clone_opt)):
        net.train();optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.l1_loss(net(x),target).backward();optimizer.step()
    assert all(torch.equal(v,clone.state_dict()[k]) for k,v in model.state_dict().items())
    model.eval();deployed=fused_copy(model,arch.Conv3XC)
    with torch.no_grad():torch.testing.assert_close(model(x),deployed(x),rtol=1e-5,atol=1e-6)
    assert sum(v.numel() for v in deployed.parameters())==517380
    assert model.extra_parameters==19172
    print(json.dumps(dict(status='PASS',device=args.device,first_loss=losses[0],last_loss=losses[-1],
        deployed_parameters=517380,extra_over_A=2636,extra_over_GB=1264,
        checks=['A initial exact identity train/eval','shared weights and RNG','rectangular batch1/2',
                'signed RGB phase oracle','projection first-step; stem/DW3/DW5/gate later-step learning','optimizer coverage',
                'finite gradients and 20 updates','strict reload and resumed step equality','deployment agreement'],
        evidence='Synthetic only; no real PSNR or long-run stability claim'),indent=2))


if __name__=='__main__':main()
