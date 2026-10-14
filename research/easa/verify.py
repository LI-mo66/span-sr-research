"""Synthetic checks of interface, actual learning, state and deployment."""
import argparse
import copy
import importlib.util
import io
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('easa_runtime', HERE / 'runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
import torch
from model import EASAResidual, adaptive_structure_pool, build_model
from deploy import fused_copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    # Match native adaptive max on CPU, including overlapping bins and ties.
    for h, w, size in ((64, 64, (8, 8)), (65, 71, (8, 8)),
                       (13, 17, (1, 2)), (1, 1, (1, 1))):
        for tied in (False, True):
            values = torch.zeros(2, 3, h, w) if tied else torch.randn(2, 3, h, w)
            reference = values.clone().requires_grad_()
            actual = values.to(args.device).requires_grad_()
            ref = torch.nn.functional.adaptive_max_pool2d(reference, size)
            got = adaptive_structure_pool(actual, size)
            torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=0)
            weights = torch.randn_like(ref)
            (ref * weights).sum().backward()
            (got * weights.to(args.device)).sum().backward()
            torch.testing.assert_close(actual.grad.cpu(), reference.grad, rtol=1e-6, atol=1e-6)
    arch, _ = runtime.components()
    cfg = json.loads((HERE / 'configs/E1-mid-lr2e4.json').read_text(encoding='utf-8-sig'))
    base_cfg = dict(cfg, easa=False)
    torch.manual_seed(cfg['seed'])
    baseline = build_model(arch, base_cfg).to(args.device)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    torch.manual_seed(cfg['seed'])
    model = build_model(arch, cfg).to(args.device)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in baseline.state_dict().items())
    shapes = ((1, 3, 1, 1), (2, 3, 3, 5), (1, 3, 7, 9),
              (2, 3, 13, 17), (1, 3, 64, 64), (1, 3, 65, 71))
    for shape in shapes:
        x = torch.rand(*shape, device=args.device)
        for training in (True, False):
            baseline.train(training)
            model.train(training)
            with torch.no_grad():
                result = model(x)
                assert result.shape == (shape[0], 3, shape[2]*4, shape[3]*4)
                assert result.dtype == x.dtype and result.device == x.device
                assert torch.isfinite(result).all()
                # Bypass diagnostic only: verifies unchanged backbone wiring.
                old_gamma = model.easa_adapter.gamma.detach().clone()
                model.easa_adapter.gamma.zero_()
                torch.testing.assert_close(model(x), baseline(x), rtol=0, atol=0)
                model.easa_adapter.gamma.copy_(old_gamma)
    # Feature-level check: no input mutation; no cross-image statistics.
    adapter = model.easa_adapter
    features = torch.randn(2, 52, 13, 17, device=args.device)
    original = features.clone()
    with torch.no_grad():
        result = adapter(features)
        assert torch.equal(features, original)
        torch.testing.assert_close(result[:1], adapter(features[:1]), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result, features + adapter.gamma * adapter.correction(features))
    # Cover noncontiguous inputs and degenerate zero features with backward.
    for h, w in ((1, 1), (3, 5), (13, 17), (64, 64)):
        value = torch.zeros(1, 52, w, h, device=args.device).transpose(2, 3).requires_grad_()
        adapter.zero_grad(set_to_none=True)
        adapter(value).square().mean().backward()
        assert value.grad is not None and torch.isfinite(value.grad).all()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg['lr'])
    optimizer_ids = [id(p) for group in opt.param_groups for p in group['params']]
    assert len(optimizer_ids) == len(set(optimizer_ids)) == len(trainable)
    before = {k: p.detach().clone() for k, p in adapter.named_parameters()}
    x = torch.rand(2, 3, 13, 17, device=args.device)
    target = torch.rand(2, 3, 52, 68, device=args.device)
    losses, gradients = [], {}
    # This must be the full output loss, not an adapter-only surrogate.
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(model(x), target)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        if step == 0:
            gradients = {k: float(p.grad.abs().max()) for k, p in adapter.named_parameters()}
            assert all(v > 0 for v in gradients.values())
        opt.step()
        losses.append(float(loss.detach()))
    updated = {k: not torch.equal(before[k], p) for k, p in adapter.named_parameters()}
    assert all(updated.values())
    assert losses[-1] < losses[0], losses
    model.eval()
    with torch.no_grad():
        expected = model(x)  # refresh derived Conv3XC eval kernels before save
    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    clone = build_model(arch, cfg).to(args.device).eval()
    status = clone.load_state_dict(torch.load(payload, map_location=args.device,
                                             weights_only=True), strict=True)
    assert not status.missing_keys and not status.unexpected_keys
    deployed = fused_copy(model, arch.Conv3XC)
    with torch.no_grad():
        torch.testing.assert_close(expected, clone(x), rtol=0, atol=0)
        torch.testing.assert_close(expected, deployed(x), rtol=1e-5, atol=1e-6)
    assert model.extra_parameters == 8944
    assert sum(p.numel() for p in deployed.parameters()) == 507152
    # Both approved insertion sites must preserve baseline with a zero diagnostic scale.
    tail_cfg = copy.deepcopy(cfg)
    tail_cfg['easa_settings']['position'] = 'tail'
    torch.manual_seed(cfg['seed'])
    tail = build_model(arch, tail_cfg).to(args.device).eval()
    with torch.no_grad():
        tail.easa_adapter.gamma.zero_()
        torch.testing.assert_close(tail(x), baseline.eval()(x), rtol=0, atol=0)
    for bad in (dict(cfg, easa='yes'), dict(cfg, moe_fs=True),
                dict(cfg, easa_settings=dict(position='invalid'))):
        try:
            build_model(arch, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid configuration accepted')
    report = dict(status='PASS', device=args.device, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                  shapes=[list(s) for s in shapes], deterministic=True,
                  extra_parameters=8944, deployed_parameters=507152,
                  first_step_adapter_gradient_max=gradients, adapter_updated=updated,
                  synthetic_l1=losses, checkpoint_missing=[], checkpoint_unexpected=[],
                  checks=['native adaptive-max values/tie gradients',
                          'baseline weights/RNG', 'mid/tail baseline bypass wiring',
                          'tiny/odd/non-square NCHW train/eval', 'no input mutation',
                          'batch independence', 'zero/noncontiguous backward',
                          'full-loss finite gradients and all adapter updates',
                          'optimizer coverage', 'serialized strict reload', 'prefusion'],
                  evidence='engineering only; no real-data PSNR measured')
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
