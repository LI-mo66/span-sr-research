"""Synthetic engineering checks only: full loss, all six adapters, no PSNR claim."""
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('a6_runtime', HERE / 'runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
import torch
from torch.nn import functional as F
from model import GridResidual, GridAttention, build_model
from deploy import fused_copy


def reference_tokens(x, centers=False):
    """Explicit coordinate oracle for upstream center-grid sample order."""
    B, C, H, W = x.shape
    h, w = H // 8, W // 8
    if centers:
        x = F.pad(x, (0, w//2, 0, h//2), mode='reflect')[:, :, h//2:h//2+H, w//2:w//2+W]
    p = 7 if centers else 8
    return torch.stack([torch.stack([x[b, :, i*h+y, j*w+z]
                                     for i in range(p) for j in range(p)])
                        for b in range(B) for y in range(h) for z in range(w)])


def check_reference():
    # Independent coordinate assembly versus production reshape/index_select.
    m = GridAttention().double()
    for H, W in ((8, 8), (16, 24), (24, 16)):
        x = torch.randn(1, 26, H, W, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        base = reference_tokens(y)
        tokens = torch.cat((base, reference_tokens(y, centers=True)), dim=1)
        q = m.q(base) * 26**-0.5
        k, v = m.kv(tokens).chunk(2, dim=-1)
        bias = torch.cat((m.relative_bias(m.relative_index).squeeze(-1).unsqueeze(0), m.extra_bias), -1)
        ref = m.proj(((q @ k.transpose(-2, -1)) + bias).softmax(-1) @ v)
        # Scatter in explicit coordinates, retaining autograd for the oracle.
        hh, ww = H//8, W//8
        ref = ref.reshape(1, hh, ww, 8, 8, 26).permute(0, 5, 3, 1, 4, 2).reshape(1, 26, H, W)
        got = m(x)
        torch.testing.assert_close(got, ref, atol=1e-12, rtol=1e-11)
        weights = torch.randn_like(got)
        gx = torch.autograd.grad((got*weights).sum(), x)[0]
        gy = torch.autograd.grad((ref*weights).sum(), y)[0]
        torch.testing.assert_close(gx, gy, atol=1e-12, rtol=1e-10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--output')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    check_reference()
    arch, _ = runtime.components()
    cfg = json.loads((HERE/'configs/A6-lr2e4.json').read_text())
    torch.manual_seed(10)
    baseline = arch.SPAN(**cfg['model']).to(args.device)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    torch.manual_seed(10)
    model = build_model(arch, cfg).to(args.device)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in baseline.state_dict().items())
    assert len({id(a.attention.q.weight) for a in model.grid_adapters}) == 6
    shapes = ((1, 3, 1, 1), (2, 3, 3, 5), (1, 3, 13, 17),
              (1, 3, 64, 64), (1, 3, 65, 71))
    for shape in shapes:
        x = torch.rand(*shape, device=args.device)
        for training in (True, False):
            model.train(training); baseline.train(training)
            with torch.no_grad():
                out = model(x)
                assert out.shape == (shape[0], 3, shape[2]*4, shape[3]*4)
                assert torch.isfinite(out).all() and out.device == x.device and out.dtype == x.dtype
                scales = [a.gamma.clone() for a in model.grid_adapters]
                for a in model.grid_adapters:
                    a.gamma.zero_()
                torch.testing.assert_close(model(x), baseline(x), atol=0, rtol=0)
                for a, value in zip(model.grid_adapters, scales):
                    a.gamma.copy_(value)
    adapter = model.grid_adapters[0]
    features = torch.randn(2, 52, 17, 13, device=args.device).transpose(2, 3)
    old = features.clone()
    with torch.no_grad():
        y = adapter(features)
        assert torch.equal(features, old)
        torch.testing.assert_close(y[:1], adapter(features[:1]), atol=2e-6, rtol=2e-5)
    for h, w in ((1, 1), (3, 5), (13, 17)):
        x = torch.zeros(1, 52, w, h, device=args.device).transpose(2, 3).requires_grad_()
        adapter.zero_grad(set_to_none=True)
        adapter(x).square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=2e-4, betas=(.9, .999), eps=1e-8)
    ids = [id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(params)
    before = {k: p.detach().clone() for k, p in model.grid_adapters.named_parameters()}
    x = torch.rand(2, 3, 13, 17, device=args.device)
    target = torch.rand(2, 3, 52, 68, device=args.device)
    losses, grads = [], {}
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        loss = F.l1_loss(model(x), target)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        if step == 0:
            grads = {k: p.grad.detach().abs().max().item() for k, p in model.grid_adapters.named_parameters()}
            assert all(v > 0 for v in grads.values()), grads
        opt.step(); losses.append(loss.item())
    updated = {k: not torch.equal(before[k], v) for k, v in model.grid_adapters.named_parameters()}
    assert all(updated.values()), updated
    model.eval()
    with torch.no_grad():
        expected = model(x)
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
    clone = build_model(arch, cfg).to(args.device).eval()
    keys = clone.load_state_dict(torch.load(buffer, map_location=args.device, weights_only=True), strict=True)
    assert not keys.missing_keys and not keys.unexpected_keys
    deployed = fused_copy(model, arch.Conv3XC)
    with torch.no_grad():
        torch.testing.assert_close(expected, clone(x), atol=0, rtol=0)
        torch.testing.assert_close(expected, deployed(x), atol=1e-6, rtol=1e-5)
    counts = dict(extra=model.extra_parameters, deployed=sum(p.numel() for p in deployed.parameters()),
                  stored=sum(p.numel() for p in model.parameters()),
                  trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert counts == dict(extra=45438, deployed=543646, stored=2667402, trainable=2202574), counts
    try:
        build_model(arch, dict(cfg, a6=False))
    except ValueError:
        pass
    else:
        raise AssertionError('Baseline training configuration accepted')
    report = dict(status='PASS', device=args.device, torch=torch.__version__, counts=counts,
                  deterministic=True, shapes=[list(s) for s in shapes],
                  first_step_grad_max=grads, adapter_parameters_updated=updated,
                  synthetic_loss=losses, strict_reload=True, prefusion=True,
                  checks=['coordinate oracle values/input gradients', 'baseline initial weights/RNG',
                          'six distinct parameter sets', 'zero-gamma backbone equivalence',
                          'tiny/odd/non-square train/eval', 'no input mutation/batch mixing',
                          'full-loss finite gradients and actual updates in all six branches',
                          'optimizer coverage', 'strict serialization', 'baseline training rejected'],
                  evidence='synthetic engineering only; no real-data PSNR training')
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
