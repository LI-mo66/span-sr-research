"""Synthetic engineering checks only; never reports real-data PSNR gains."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fs_runtime', HERE / 'runtime.py')
fs_runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs_runtime)
import torch
from model import FusionGate4, build_model
from deploy import fused_copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    arch, _ = fs_runtime.components()
    cfg = json.loads((HERE / 'configs/A1-G4.json').read_text())
    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg['moe_fs'] = False
    torch.manual_seed(cfg['seed'])
    baseline = build_model(arch, baseline_cfg).to(args.device)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    torch.manual_seed(cfg['seed'])
    candidate = build_model(arch, cfg).to(args.device)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert all(torch.equal(v, candidate.state_dict()[k]) for k, v in baseline.state_dict().items())
    for shape in ((1, 3, 13, 17), (2, 3, 16, 20)):
        x = torch.rand(*shape, device=args.device)
        for training in (True, False):
            baseline.train(training)
            candidate.train(training)
            with torch.no_grad():
                a, b = baseline(x), candidate(x)
                assert b.shape == (shape[0], 3, shape[2] * 4, shape[3] * 4)
                assert torch.isfinite(b).all()
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Check expert axis independently of full-network coincidences.
    gate = FusionGate4(52).to(args.device)
    experts = [torch.rand(2, 52, 13, 17, device=args.device) for _ in range(4)]
    with torch.no_grad():
        for a, b in zip(experts, gate(experts)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        gate.logits[0].bias.fill_(1.0)
        values = gate(experts)
        weights = torch.stack([layer(x) for layer, x in zip(gate.logits, experts)], 0).softmax(0) * 4
        torch.testing.assert_close(weights.sum(0), torch.full_like(weights[0], 4.0))
        assert (values[0] > experts[0]).all()
    try:
        gate(experts[:3])
    except ValueError:
        pass
    else:
        raise AssertionError('Missing expert accepted')
    candidate.train()
    trainable = [p for p in candidate.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg['lr'])
    optimizer_ids = [id(p) for group in opt.param_groups for p in group['params']]
    assert len(optimizer_ids) == len(set(optimizer_ids)) == len(trainable)
    before = {k: p.detach().clone() for k, p in candidate.fusion_gate.named_parameters()}
    x = torch.rand(2, 3, 13, 17, device=args.device)
    target = torch.rand(2, 3, 52, 68, device=args.device)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(candidate(x), target)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        opt.step()
    assert all(not torch.equal(before[k], p) for k, p in candidate.fusion_gate.named_parameters())
    candidate.eval()
    clone = copy.deepcopy(candidate)
    clone.load_state_dict(candidate.state_dict(), strict=True)
    deployed = fused_copy(candidate, arch.Conv3XC)
    with torch.no_grad():
        prediction = candidate(x)
        torch.testing.assert_close(prediction, clone(x), rtol=0, atol=0)
        torch.testing.assert_close(prediction, deployed(x), rtol=1e-5, atol=1e-6)
    assert candidate.extra_parameters == 11024
    assert sum(p.numel() for p in deployed.parameters()) == 509232
    print(json.dumps(dict(status='PASS', device=args.device, extra_parameters=11024,
                         deployed_parameters=509232,
                         checks=['baseline RNG/weights preserved', 'initial exact identity train/eval',
                                 'batch1/2 and rectangular shapes', 'expert softmax axis',
                                 'finite gradients and real gate updates', 'optimizer coverage',
                                 'strict reload', 'prefused deployment agreement'],
                         evidence='synthetic engineering only; no real PSNR measured'), indent=2))


if __name__ == '__main__':
    main()
