"""Compare adapted TAB outputs/gradients/state to the pinned official extract."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from tab import ContentTAB, IASA, reflected_tail

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('reference_tab', HERE / 'vendor/reference_tab.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    report = dict(device=args.device, torch=torch.__version__, checks=[])
    for h, w in ((13, 17), (16, 16), (64, 64)):
        torch.manual_seed(24)
        a = reference.TAB(52, 52, 104, 4, n_iter=5, num_tokens=16, group_size=128).to(args.device)
        b = ContentTAB(chunk_groups=3).to(args.device)
        a.load_state_dict(b.state_dict(), strict=True)
        x = torch.randn(2, 52, h, w, device=args.device)
        for training in (True, False):
            a.train(training)
            b.train(training)
            x1 = x.clone().requires_grad_()
            x2 = x.clone().requires_grad_()
            a.zero_grad(set_to_none=True)
            b.zero_grad(set_to_none=True)
            y1, y2 = a(x1), b(x2)
            torch.testing.assert_close(y1, y2, rtol=2e-5, atol=2e-6)
            target_gradient = torch.randn_like(y1) / y1.numel()
            (y1 * target_gradient).sum().backward()
            (y2 * target_gradient).sum().backward()
            torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-4, atol=1e-7)
            for name, parameter in a.named_parameters():
                other = dict(b.named_parameters())[name]
                torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-4, atol=1e-7)
            torch.testing.assert_close(a.means, b.means, rtol=1e-5, atol=1e-6)
            assert torch.equal(a.initted, b.initted)
            report['checks'].append(dict(hw=[h, w], training=training, status='PASS',
                                         output_max_error=float((y1-y2).detach().abs().max())))
    # Index round-trip is exact, independently of attention projections.
    for n in (1, 7, 16, 127, 128, 129, 130, 191, 192, 255, 256, 257):
        values = torch.arange(n, device=args.device).reshape(1, n, 1).expand(2, -1, 3)
        permutation = torch.stack([torch.randperm(n, device=args.device) for _ in range(2)]).unsqueeze(-1)
        restored = values.gather(1, permutation.expand_as(values)).gather(1, permutation.argsort(1).expand_as(values))
        assert torch.equal(values, restored)
        module = ContentTAB().to(args.device)
        x = torch.randn(1, 52, 1, n, device=args.device, requires_grad=True)
        y = module(x)
        y.square().mean().backward()
        assert y.shape == x.shape and torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    # Controlled direct remote-token path: query at position 0 receives token 7
    # through IASA's content-ordered KV, with global values held at zero.
    module = IASA(4, 4, 1, group_size=2, chunk_groups=1).to(args.device)
    with torch.no_grad():
        module.to_q.weight.zero_()
        module.to_k.weight.zero_()
        module.to_v.weight.copy_(torch.eye(4, device=args.device))
        module.proj.weight.copy_(torch.eye(4, device=args.device))
    order = torch.tensor([0, 7, 1, 6, 2, 5, 3, 4], device=args.device).reshape(1, 8, 1)
    values = torch.zeros(1, 8, 4, device=args.device, requires_grad=True)
    centers = torch.zeros(1, 2, 4, device=args.device)
    output = module(values, order, centers, centers)
    derivative = torch.autograd.grad(output[0, 0, 0], values)[0][0, 7, 0]
    torch.testing.assert_close(derivative, derivative.new_tensor(0.25), rtol=0, atol=0)
    # Sorting at the TAB level really uses content, not raster order.
    module = ContentTAB(dim=4, qk_dim=4, mlp_dim=8, heads=1,
                        num_tokens=2, group_size=2).to(args.device).eval()
    features = torch.tensor([[1., -1., 0., 0.], [-1., 1., 0., 0.]], device=args.device)
    raster = features[torch.tensor([0, 1, 1, 1, 1, 1, 1, 0], device=args.device)]
    with torch.no_grad():
        module.means.copy_(torch.nn.functional.normalize(features, dim=-1))
        module.initted.fill_(True)
    captured = {}
    hook = module.iasa_attn.register_forward_pre_hook(
        lambda _, inputs: captured.update(order=inputs[1].detach().clone()))
    module(raster.T.reshape(1, 4, 1, 8))
    hook.remove()
    assert set(captured['order'][0, :2, 0].tolist()) == {0, 7}
    report.update(status='PASS', edge_tokens=[1,7,16,127,128,129,130,191,192,255,256,257],
                  remote_value_derivative=float(derivative),
                  content_grouping_far_pair=[0,7],
                  parameters=sum(p.numel() for p in ContentTAB().parameters()),
                  scope='mechanism/implementation evidence only, not PSNR')
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
