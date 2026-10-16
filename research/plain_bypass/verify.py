"""Inference identity, phased reconstruction, staged learning and resume audit."""
import argparse
import copy
import io
import importlib.util
import json
from pathlib import Path
import torch
from entry import existing
from pb_model import build_model
from lg_model import build_model as build_a
from deploy import fused_copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cfg = json.loads((Path(__file__).parent/'configs/LG6-SiLU-PB.json').read_text())
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
        model.image_residual.bias.copy_(torch.linspace(-.1,.1,48,device=args.device))
        expected = a(x) + torch.nn.functional.pixel_shuffle(
            model.image_residual(x),4)
        torch.testing.assert_close(model(x),expected,rtol=1e-5,atol=1e-6)
        model.image_residual.bias.zero_()
    # Hold all learned tensors fixed: plain bypass must equal gated bypass with G=1.
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('fixed_gb_reference', root/'research/gated_bypass/gb_model.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    gb_cfg = json.loads((root/'research/gated_bypass/configs/LG6-SiLU-GB.json').read_text())
    gated = module.build_model(arch,gb_cfg).to(args.device).eval()
    loaded = gated.load_state_dict(model.state_dict(),strict=False)
    assert set(loaded.missing_keys)=={'image_gate.weight','image_gate.bias'} and not loaded.unexpected_keys
    with torch.no_grad():
        model.image_residual.weight.fill_(.001)
        gated.image_residual.weight.copy_(model.image_residual.weight)
        torch.testing.assert_close(model(x),gated(x),rtol=0,atol=0)
        model.image_residual.weight.zero_()
    params = [v for v in model.parameters() if v.requires_grad]
    opt = torch.optim.Adam(params, lr=.0002, betas=(.9,.999),eps=1e-8)
    ids = [id(v) for g in opt.param_groups for v in g['params']]
    assert len(ids)==len(set(ids))==len(params)
    target=torch.rand(2,3,64,80,device=args.device)
    x=torch.rand(2,3,16,20,device=args.device)
    losses=[]; gate_started=False
    for step in range(20):
        model.train();opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.l1_loss(model(x),target)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
        assert model.image_residual.weight.grad.abs().sum()>0
        opt.step();losses.append(float(loss.detach()))
    assert model.image_residual.weight.abs().sum()>0
    assert not any('image_gate' in k for k in model.state_dict())
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
    assert sum(v.numel() for v in deployed.parameters())==516088
    assert model.extra_parameters==17880
    print(json.dumps(dict(status='PASS',device=args.device,first_loss=losses[0],last_loss=losses[-1],
        deployed_parameters=516088,extra_over_A=1344,
        checks=['A initial exact identity train/eval','shared weights and RNG','rectangular batch1/2',
                'signed RGB phase oracle','residual first-step learning and no image gate parameters','optimizer coverage',
                'exact fixed-weight gated-G1 equivalence','finite gradients and 20 updates','strict reload and resumed step equality','deployment agreement'],
        evidence='Synthetic only; no real PSNR or long-run stability claim'),indent=2))


if __name__=='__main__':main()
