"""Local synthetic checks. Real checkpoint/metric reproduction is a server gate."""
import argparse
import copy
import json
from pathlib import Path
import torch
import evaluate as evaluation


def main(device):
    torch.set_num_threads(2)
    torch.manual_seed(10)
    mix = evaluation.mix
    # Explicit interpolation arithmetic and immutable buffer contracts.
    a = dict(weight=torch.tensor([1., 3.]), counter=torch.tensor(2))
    b = dict(weight=torch.tensor([5., 7.]), counter=torch.tensor(2))
    assert torch.equal(mix([a, b], [.75, .25], {'weight'})['weight'], torch.tensor([2., 4.]))
    try:
        mix([a, dict(weight=b['weight'], counter=torch.tensor(3))], [.5, .5], {'weight'})
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatched buffers accepted')
    evaluation.check_scores({'x': dict(psnr=30., ssim=.9)}, {'x': dict(psnr=30., ssim=.9)})
    try:
        evaluation.check_scores({'x': dict(psnr=29.99, ssim=.9)}, {'x': dict(psnr=30., ssim=.9)})
    except AssertionError:
        pass
    else:
        raise AssertionError('Metric reproduction failure accepted')
    cfg = json.loads((evaluation.ROOT / 'research/multiscale_f0/template.json').read_text())
    cfg['arm'] = 'F0'
    cfg['model']['upscale'] = 2
    arch, _ = evaluation.run.fusion.components()
    source = evaluation.run.fusion.build_model(arch, cfg).to(device).eval()
    deployed = evaluation.fused_copy(source, arch.Conv3XC).eval()
    assert sum(p.numel() for p in deployed.parameters()) == 552003
    assert not any(isinstance(m, arch.Conv3XC) for m in deployed.modules())
    with torch.inference_mode():
        for h, w in ((1, 1), (17, 19), (63, 65)):
            x = torch.rand(1, 3, h, w, device=device)
            torch.testing.assert_close(source(x), deployed(x), atol=2e-5, rtol=1e-5)
    first = copy.deepcopy(deployed.state_dict())
    other = copy.deepcopy(deployed)
    with torch.no_grad():
        for p in other.parameters():
            p.add_(torch.randn_like(p) * 1e-5)
    second = copy.deepcopy(other.state_dict())
    names = set(dict(deployed.named_parameters()))
    for alpha in (0., .25, .5, .75, 1.):
        mixed = mix([first, second], [1-alpha, alpha], names)
        if alpha in (0., 1.):
            assert all(torch.equal(v, (first if alpha == 0 else second)[k]) for k, v in mixed.items())
        deployed.load_state_dict(mixed, strict=True)
        with torch.inference_mode():
            plain, x8 = evaluation.run.trainer.predict(deployed, torch.rand(1, 3, 17, 19, device=device))
            assert plain.shape == x8.shape == (1, 3, 34, 38)
            assert torch.isfinite(x8).all()
    average = mix([first] * 5, [.2] * 5, names)
    assert all(torch.equal(v, first[k]) for k, v in average.items())
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in second.items())
    print('PASS mix arithmetic, endpoints, buffers, native/deployed equivalence, x8, immutable sources', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    main(parser.parse_args().device)
