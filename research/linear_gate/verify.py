"""Synthetic engineering only: interfaces, affine formula, real updates, resume."""
import argparse
import copy
import io
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from entry import existing, ROOT
from lg_model import build_model
from deploy import fused_copy

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    if args.steps < 3:
        raise ValueError('Use at least three optimizer steps')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    arch, _ = existing.fs_runtime.components()
    reports = []
    for label in ('LG6-SiLU', 'LG6-ReLU'):
        cfg = json.loads((HERE / 'configs' / (label + '.json')).read_text())
        torch.manual_seed(cfg['seed'])
        baseline = arch.SPAN(**cfg['model']).to(args.device)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all()
        torch.manual_seed(cfg['seed'])
        model = build_model(arch, cfg).to(args.device)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in baseline.state_dict().items())
        gates = [getattr(model, f'block_{i}').local_gate for i in range(1, 7)]
        assert all(torch.count_nonzero(g.weight) == 0 and torch.all(g.bias == 0.1) for g in gates)
        # Direct block oracle, including original returned inplace-activated out1.
        block = model.block_6
        z = torch.randn(2, 52, 7, 9, device=args.device)
        block.eval()
        with torch.no_grad():
            f1 = block.act1(block.c1_r(z))
            f2 = block.act1(block.c2_r(f1))
            f3 = block.c3_r(f2)
            y, returned, m = block(z)
            torch.testing.assert_close(m, torch.full_like(m, 0.1), rtol=0, atol=0)
            torch.testing.assert_close(y, (z + f3) * m, rtol=0, atol=0)
            torch.testing.assert_close(returned, f1, rtol=0, atol=0)
            # Prove signed, cross-channel, unbounded affine gate, no sigmoid/softmax.
            block.local_gate.weight[0, 1, 0, 0] = 2
            block.local_gate.bias[0] = -1
            synthetic = torch.zeros_like(z)
            synthetic[:, 1] = 3
            torch.testing.assert_close(block.local_gate(synthetic)[:, 0], torch.full_like(z[:, 0], 5))
            block.local_gate.weight.zero_(); block.local_gate.bias.fill_(0.1)
        for shape in ((1, 3, 13, 17), (2, 3, 16, 20)):
            x = torch.rand(*shape, device=args.device)
            for training in (True, False):
                model.train(training)
                with torch.no_grad():
                    y = model(x)
                assert y.shape == (shape[0], 3, shape[2]*4, shape[3]*4)
                assert torch.isfinite(y).all()
        model.train()
        opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                               lr=cfg['lr'], betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'])
        ids = [id(p) for group in opt.param_groups for p in group['params']]
        assert len(ids) == len(set(ids)) == sum(p.requires_grad for p in model.parameters())
        before = {k: p.detach().clone() for k, p in model.named_parameters() if '.local_gate.' in k}
        x = torch.rand(2, 3, 16, 20, device=args.device)
        target = torch.rand(2, 3, 64, 80, device=args.device)
        losses = []

        def step(net, optimizer):
            net.train(); optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(net(x), target)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters() if p.requires_grad)
            assert all(g.weight.grad.abs().sum() > 0 and g.bias.grad.abs().sum() > 0
                       for g in (getattr(net, f'block_{i}').local_gate for i in range(1, 7)))
            optimizer.step()
            return float(loss.detach())

        for _ in range(args.steps):
            losses.append(step(model, opt))
        assert all(not torch.equal(before[k], p) for k, p in model.named_parameters() if k in before)
        # Actual serialization of model + optimizer state, followed by equal next update.
        stream = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), stream)
        stream.seek(0)
        state = torch.load(stream, map_location=args.device, weights_only=False)
        clone = build_model(arch, cfg).to(args.device)
        clone.load_state_dict(state['model'], strict=True)
        clone_opt = torch.optim.Adam((p for p in clone.parameters() if p.requires_grad), lr=cfg['lr'])
        clone_opt.load_state_dict(state['optimizer'])
        step(model, opt); step(clone, clone_opt)
        assert all(torch.equal(v, clone.state_dict()[k]) for k, v in model.state_dict().items())
        model.eval()
        deployed = fused_copy(model, arch.Conv3XC)
        with torch.no_grad():
            torch.testing.assert_close(model(x), deployed(x), rtol=1e-5, atol=1e-6)
        count = sum(p.numel() for p in deployed.parameters())
        assert model.extra_parameters == 16536 and count == 514744
        reports.append(dict(variant=label, steps=args.steps, first_loss=losses[0], last_loss=losses[-1],
                            extra_parameters=16536, deployed_parameters=count))
    print(json.dumps(dict(status='PASS', device=args.device, results=reports,
        checks=['shared initialization and RNG', 'signed cross-channel affine formula', 'constant nonzero initialization',
                'train/eval rectangular batch1/2', 'returned first feature semantics', 'optimizer coverage',
                'all six gate weight/bias gradients and updates', 'strict serialization and equal resumed next step',
                'deployment agreement'], evidence='Synthetic engineering only; no real PSNR or long-run stability claim'), indent=2))


if __name__ == '__main__':
    main()
