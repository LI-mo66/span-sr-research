"""Check objective semantics, optimizer reset, shapes and exact resume."""
import argparse
import copy
import tempfile
from pathlib import Path

import numpy as np
import random
import torch
import run


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    else:
        assert a == b


def test(device):
    # This fixture tests loss gradients, not just matching loss numbers.
    prediction = torch.tensor([0.2, -0.3], device=device, requires_grad=True)
    target = torch.zeros_like(prediction)
    l1, _, _ = run.objective(prediction, target, 'l1', 1, 10)
    mse, _, _ = run.objective(prediction, target, 'mse', 1, 10)
    torch.testing.assert_close(torch.autograd.grad(l1, prediction, retain_graph=True)[0],
                               torch.tensor([0.5, -0.5], device=device))
    torch.testing.assert_close(torch.autograd.grad(mse, prediction)[0], prediction.detach())
    assert run.objective(prediction, target, 'staged', 5, 10)[0] == l1
    assert run.objective(prediction, target, 'staged', 6, 10)[0] == mse
    for kind in ('l1', 'mse', 'staged'):
        torch.manual_seed(10)
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 12, 3, padding=1),
                                    torch.nn.PixelShuffle(2)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
        assert not optimizer.state
        def step(index):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(4):
                x = torch.rand(2, 3, 7, 9, device=device)
                y = torch.rand(2, 3, 14, 18, device=device)
                loss = run.objective(model(x), y, kind, index, 4)[0]
                (loss / 4).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
            optimizer.step()
        step(1)
        step(2)
        state = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   rng=run.rng_state()))
        step(3)
        step(4)
        expected = copy.deepcopy((model.state_dict(), optimizer.state_dict(), run.rng_state()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'resume.pt'
            torch.save(state, path)
            loaded = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(loaded['model'])
        optimizer.load_state_dict(loaded['optimizer'])
        run.restore_rng(loaded['rng'])
        step(3)
        step(4)
        equal(expected, (model.state_dict(), optimizer.state_dict(), run.rng_state()))
    print('PASS objectives and exact2+2 resume', device, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    test('cpu')
    if torch.cuda.is_available():
        test('cuda')
    if args.source_run:
        assert torch.cuda.is_available()
        metadata = __import__('json').loads((args.source_run / 'manifest.json').read_text())
        cfg = metadata['config']
        state = torch.load(args.source_run / 'epoch00982_u982000.pt',
                           map_location='cpu', weights_only=False)
        arch, _ = run.fusion.components()
        model = run.fusion.build_model(arch, cfg).cuda()
        model.load_state_dict(state['model'], strict=True)
        with torch.no_grad():
            for h, w in ((1, 1), (17, 19), (63, 65)):
                plain, average = run.trainer.predict(model, torch.rand(1, 3, h, w, device='cuda'))
                assert plain.shape == average.shape == (1, 3, h * 2, w * 2)
                assert torch.isfinite(average).all()
        run.trainer.SCALE = 2
        data = run.manifest_pairs(cfg, metadata['data'])['train']
        for arm, (kind, rate) in run.ARMS.items():
            model.load_state_dict(state['model'], strict=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=rate,
                betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'])
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            random.seed(10)
            for _ in range(4):
                samples = [run.trainer.sample(random.choice(data), 192) for _ in range(16)]
                x = torch.stack([s[0] for s in samples]).cuda()
                y = torch.stack([s[1] for s in samples]).cuda()
                loss = run.objective(model(x), y, kind, 1, 10000)[0]
                assert torch.isfinite(loss)
                (loss / 4).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.requires_grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
            optimizer.step()
            print('CUDA peak allocated GiB', torch.cuda.max_memory_allocated() / 1024**3, flush=True)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            print('PASS actual982K micro16 accum4', arm, flush=True)
