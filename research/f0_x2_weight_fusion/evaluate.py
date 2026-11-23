"""Evaluation only: fuse Conv3XC first, then average compatible deployed weights."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/f0_x2_finetune'))
import run
from deploy import fused_copy


def mix(states, weights, parameter_names):
    assert len(states) == len(weights) and states
    assert abs(sum(weights) - 1.) < 1e-12 and min(weights) >= 0
    assert all(s.keys() == states[0].keys() for s in states)
    result = {}
    for key, first in states[0].items():
        assert all(s[key].shape == first.shape and s[key].dtype == first.dtype for s in states)
        if key in parameter_names and first.is_floating_point():
            # Float64 accumulation avoids repeated float32 rounding.
            value = torch.zeros_like(first, dtype=torch.float64)
            for state, weight in zip(states, weights):
                value.add_(state[key].double(), alpha=weight)
            result[key] = value.to(first.dtype)
        else:
            assert all(torch.equal(first, s[key]) for s in states), key
            result[key] = first.clone()
        assert torch.isfinite(result[key]).all(), key
    return result


def check_scores(observed, expected):
    assert observed.keys() == expected.keys()
    differences = {}
    for name in observed:
        differences[name] = {}
        for key in ('psnr', 'ssim'):
            delta = observed[name][key] - expected[name][key]
            assert math.isfinite(delta)
            # Reparameterization may move a few uint8 quantization boundaries.
            limit = 2e-4 if key == 'psnr' else 2e-6
            assert abs(delta) <= limit, (name, key, delta, limit)
            differences[name][key] = delta
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--finetune-project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_dir() and not (out / 'protocol.json').exists(), 'Output already started'
    assert out.is_relative_to(Path('/home/featurize/work').resolve())
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(10)
    source = args.source_run.resolve()
    ft = args.finetune_project.resolve()
    metadata = json.loads((source / 'manifest.json').read_text())
    cfg = metadata['config']
    assert cfg['arm'] == 'F0' and cfg['model']['upscale'] == 2
    assert cfg['crop_border'] == 2 and cfg['self_ensemble'] is True
    paths = json.loads(Path(cfg['paths_file']).read_text(encoding='utf-8-sig'))
    run.trainer.SCALE = 2
    datasets = {}
    expected_counts = dict(DIV2K_valid=100, Set5=5, Set14=14, B100=100, Urban100=100, Manga109=109)
    for name, count in expected_counts.items():
        pairs = run.trainer.pairs(paths['validation'][name])
        records = metadata['data'][name]
        assert len(pairs) == len(records) == count
        for (hr, lr), row in zip(pairs, records):
            assert str(hr) == row['hr'] and str(lr) == row['lr']
            assert run.trainer.digest(hr) == row['hr_sha256'], hr
            assert run.trainer.digest(lr) == row['lr_sha256'], lr
            run.trainer.aligned(hr, lr)
        datasets[name] = pairs
        print('PASS evaluation data SHA256 and geometry', name, count, flush=True)
    receipt = json.loads((ft / 'logs/f0_x2_finetune/PATCH96-SWEEP-01/final-verification.json').read_text())
    assert receipt['status'] == 'PASS'
    recorded_hashes = {r['path']: r['sha256'] for r in receipt['files']}
    sources = {'original_982k': source / 'epoch00982_u982000.pt'}
    sources['l1_4k'] = ft / 'runs/FT-X2-982K-PC96-l1_lr5-10K-v1/step04000.pt'
    for step in (6000, 7000, 8000, 9000, 10000):
        sources[f'mse_{step // 1000}k'] = ft / f'runs/FT-X2-982K-PC96-mse_lr5-10K-v1/step{step:05d}.pt'
    source_hashes, expected, states = {}, {}, {}
    arch, metrics = run.fusion.components()
    template = None
    parameter_names = None
    equivalence = {}
    for label, path in sources.items():
        digest = run.trainer.digest(path)
        assert digest == (run.EXPECTED_SHA if label == 'original_982k' else recorded_hashes[str(path)])
        source_hashes[str(path)] = digest
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if label == 'original_982k':
            assert saved['update'] == 982000
            expected[label] = saved['epoch_record']['metrics']
        else:
            protocol = json.loads((path.parent / 'protocol.json').read_text())
            assert protocol['source_sha256'] == run.EXPECTED_SHA
            assert protocol['lr_patch'] == 96 and protocol['hr_patch'] == 192
            assert protocol['objective'] == ('l1' if label.startswith('l1') else 'mse')
            assert saved['fingerprint'] == hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            assert saved['step'] == int(label.split('_')[1][:-1]) * 1000
            expected[label] = saved['record']['metrics']
        native = run.fusion.build_model(arch, cfg).cuda().eval()
        native.load_state_dict(saved['model'], strict=True)
        deployed = fused_copy(native, arch.Conv3XC).eval()
        assert sum(p.numel() for p in deployed.parameters()) == 552003
        maximum = 0.
        with torch.inference_mode():
            for h, w in ((1, 1), (17, 19), (63, 65)):
                x = torch.rand(1, 3, h, w, device='cuda')
                a, b = native(x), deployed(x)
                assert a.shape == b.shape == (1, 3, h * 2, w * 2)
                torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-5)
                maximum = max(maximum, float((a - b).abs().max()))
        equivalence[label] = maximum
        states[label] = {k: v.detach().cpu().clone() for k, v in deployed.state_dict().items()}
        if template is None:
            template = copy.deepcopy(deployed).cpu()
            parameter_names = set(dict(template.named_parameters()))
        del saved, native, deployed, a, b, x
        print('PASS source and deployment conversion', label, maximum, flush=True)
    plans = {
        'original_982k': dict(original_982k=1.),
        'l1_4k': dict(l1_4k=1.),
        'mse_7k': dict(mse_7k=1.),
        'mse_mean_6to10k': {f'mse_{k}k': .2 for k in range(6, 11)},
    }
    for alpha in (.25, .5, .75):
        plans[f'l1_mse_alpha{int(alpha * 100):02d}'] = dict(l1_4k=1-alpha, mse_7k=alpha)
    protocol = dict(
        format='F0-x2-deployed-weight-fusion-v1', plans=plans,
        method='Fuse every Conv3XC into deployment convolution BEFORE weight averaging',
        source_hashes=source_hashes, equivalence_absmax=equivalence,
        expected_reference_scores={k: expected[k] for k in ('original_982k', 'l1_4k', 'mse_7k')},
        deployed_parameters=552003, cfg=cfg, evaluation_data={k: metadata['data'][k] for k in datasets},
        code_sha256=run.trainer.digest(Path(__file__)),
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=torch.cuda.get_device_name()),
        scope='No training. Exploratory selection using five test sets; no unseen-test or seed stability claim.'
    )
    run.json_write(out / 'protocol.json', protocol)
    scores, reference_differences = {}, {}
    for index, (label, coefficients) in enumerate(plans.items(), 1):
        print('EVALUATING', index, '/', len(plans), label, coefficients, flush=True)
        model = copy.deepcopy(template)
        mixed = mix([states[k] for k in coefficients], list(coefficients.values()), parameter_names)
        model.load_state_dict(mixed, strict=True)
        model = model.cuda().eval()
        folder = out / label
        folder.mkdir()
        with torch.inference_mode():
            p, e = run.trainer.predict(model, torch.rand(1, 3, 17, 19, device='cuda'))
            assert p.shape == e.shape == (1, 3, 34, 38)
            assert torch.isfinite(p).all() and torch.isfinite(e).all()
        torch.save(dict(format=protocol['format'], model=mixed, coefficients=coefficients,
                        source_hashes=source_hashes, cfg=cfg), folder / 'deployed.pt')
        scores[label] = run.trainer.evaluate(model, datasets, metrics, cfg, folder, 0, 0)
        assert all(math.isfinite(v) for s in scores[label].values() for v in s.values())
        if label in ('original_982k', 'l1_4k', 'mse_7k'):
            reference_differences[label] = check_scores(scores[label], expected[label])
        run.json_write(out / 'scores.json', scores)
        del model, mixed, p, e
    baseline = scores['original_982k']
    summary = {}
    for label, value in scores.items():
        deltas = {name: value[name]['psnr'] - baseline[name]['psnr'] for name in value}
        summary[label] = dict(metrics=value, delta_psnr_vs_982k= deltas,
            mean_five_x8_delta=sum(deltas[n+'_x8'] for n in run.TARGETS) / 5,
            minimum_five_x8_delta=min(deltas[n+'_x8'] for n in run.TARGETS),
            all_five_x8_improved=all(deltas[n+'_x8'] > 0 for n in run.TARGETS))
    comparison = dict(variants=summary, reference_reproduction_differences=reference_differences,
        scope=protocol['scope'], ranking=sorted(summary, key=lambda k: -summary[k]['mean_five_x8_delta']))
    run.json_write(out / 'comparison.json', comparison)
    for path, digest in source_hashes.items():
        assert run.trainer.digest(Path(path)) == digest, path
    files = []
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.suffix in ('.pt', '.json'):
            for attempt in range(3):
                try:
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
                    files.append(dict(path=str(path), sha256=run.trainer.digest(path)))
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(2)
    receipt = dict(status='PASS', source_unchanged=True, variants=len(scores), files=files)
    run.json_write(out / 'COMPLETE.json', receipt)
    with (out / 'COMPLETE.json').open('rb') as stream:
        os.fsync(stream.fileno())
    assert json.loads((out / 'COMPLETE.json').read_text()) == receipt
    print('COMPLETE', json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()
