"""Inference-only gate audit on fixed DIV2K validation checkpoints."""
import argparse
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(os.environ['SPAN_PROJECT']) / 'research/moe_fs' if 'SPAN_PROJECT' in os.environ else Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('fs_runtime', HERE / 'runtime.py')
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    arch, metrics = rt.components()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    runs = {}
    for label in ('B0', 'A1-G4'):
        matches = list((args.backup / 'runs').glob('*-' + label + '-s10'))
        if len(matches) != 1:
            raise ValueError(f'Expected one {label} run: {matches}')
        folder = matches[0]
        info = json.loads((folder / 'manifest.json').read_text())
        rows = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines() if line.strip()]
        rows = sorted((r for r in rows if r['update'] <= 50000), key=lambda r: r['update'])
        if rows[-1]['update'] != 50000:
            raise ValueError('Expected completed 50000-update run')
        best = max(rows, key=lambda r: r['metrics']['DIV2K_valid']['psnr'])
        checkpoint = folder / f'epoch{best["epoch"]:05d}_u{best["update"]}.pt'
        runs[label] = (info, best, checkpoint)
    data = rt.trainer.pairs(dict(hr=str(args.data_root / 'DIV2K/DIV2K_valid_HR'),
                                lr=str(args.data_root / 'DIV2K/DIV2K_valid_LR_bicubic/X4'), lr_suffix='x4'))
    if len(data) != 100:
        raise ValueError('Expected 100 DIV2K validation pairs')
    # Verify the exact source images against training provenance.
    reference = {Path(r['hr']).name: r for r in runs['A1-G4'][0]['data']['DIV2K_valid']}
    if set(reference) != {hp.name for hp, _ in data}:
        raise ValueError('Validation filenames differ from original experiment')
    for hp, lp in data:
        r = reference[hp.name]
        if rt.trainer.digest(hp) != r['hr_sha256'] or rt.trainer.digest(lp) != r['lr_sha256']:
            raise ValueError(f'Validation hash mismatch: {hp}')
    totals = np.zeros((4, 7), dtype=np.float64)
    count = 0
    entropy_sum = 0.0
    samples = [[] for _ in range(4)]
    summaries = {}
    kw = dict(crop_border=4, test_y_channel=True)
    for label in ('B0', 'A1-G4'):
        info, best, checkpoint = runs[label]
        cfg = info['config']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['fingerprint'] != info['fingerprint']:
            raise ValueError('Checkpoint fingerprint mismatch')
        model = build_model(arch, cfg).cuda().eval()
        model.load_state_dict(state['model'], strict=True)
        del state
        alpha = 1.0
        collect = False
        if label == 'A1-G4':
            gate = model.fusion_gate
            original_forward = gate.forward

            def forward(self, features):
                nonlocal count, entropy_sum
                prob = torch.stack([layer(x) for layer, x in zip(self.logits, features)], 0).softmax(dim=0)
                weights = 4.0 * prob
                if collect:
                    n = prob[0].numel()
                    count += n
                    entropy_sum += float((-(prob * prob.clamp_min(1e-12).log()).sum(0)).sum())
                    winners = prob.argmax(0)
                    for i, x in enumerate(features):
                        w = weights[i]
                        totals[i] += [float(prob[i].sum()), float((w-1).abs().sum()),
                                      float((winners == i).sum()), float(x.abs().sum()),
                                      float(x.square().sum()), float((x*w).abs().sum()),
                                      float((x*w).square().sum())]
                        flat = prob[i].flatten()
                        stride = max(1, math.ceil(flat.numel() / 2048))
                        samples[i].append(flat[::stride].cpu().numpy())
                if alpha == 1.0:
                    return [x * weights[i] for i, x in enumerate(features)]
                if alpha == 0.0:
                    return list(features)
                return [x * ((1.0-alpha) + alpha*weights[i]) for i, x in enumerate(features)]

            gate.forward = types.MethodType(forward, gate)
            # Check original vs diagnostic implementation before any evaluation.
            with torch.inference_mode():
                _, lr = rt.trainer.aligned(*data[0])
                x = rt.trainer.tensor(lr[:32, :32]).unsqueeze(0).cuda()
                gate.forward = original_forward
                expected = model(x)
                gate.forward = types.MethodType(forward, gate)
                actual = model(x)
                difference = float((expected-actual).abs().max())
                if not torch.allclose(expected, actual, rtol=1e-5, atol=1e-6):
                    raise ValueError(f'Alpha1 implementation differs: {difference}')
                print('Alpha1 implementation check:', difference, flush=True)
        for alpha in ([1.0] if label == 'B0' else [1.0, 0.75, 0.5, 0.25, 0.0]):
            collect = label == 'A1-G4' and alpha == 1.0
            rows = []
            with torch.inference_mode():
                for index, (hp, lp) in enumerate(data, 1):
                    hr, lr = rt.trainer.aligned(hp, lp)
                    prediction = model(rt.trainer.tensor(lr).unsqueeze(0).cuda())[0]
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError(f'Nonfinite prediction: {hp}')
                    sr = prediction.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
                    sr = np.rint(sr * 255).astype(np.uint8)
                    rows.append(dict(image=hp.name, psnr=metrics.calculate_psnr(sr, hr, **kw),
                                     ssim=metrics.calculate_ssim(sr, hr, **kw)))
                    if index % 20 == 0:
                        print(f'{label} alpha={alpha}: {index}/100', flush=True)
            key = f'{label}-alpha{alpha:g}'
            result = {m: float(np.mean([r[m] for r in rows])) for m in ('psnr', 'ssim')}
            result.update(checkpoint=str(checkpoint), update=best['update'], alpha=alpha)
            if alpha == 1.0:
                error = abs(result['psnr'] - best['metrics']['DIV2K_valid']['psnr'])
                if error > 1e-4:
                    raise ValueError(f'Original validation PSNR reproduction failed: {error}')
            summaries[key] = result
            (args.output / (key + '-images.json')).write_text(json.dumps(rows, indent=2))
            (args.output / 'partial-results.json').write_text(json.dumps(summaries, indent=2))
            print(key, json.dumps(result), flush=True)
        del model
        torch.cuda.empty_cache()
    stats = dict(normalized_entropy=entropy_sum / count / math.log(4),
                 aggregation='All validation spatial/channel sites pooled; quantiles use deterministic subsamples', features={})
    for i, name in enumerate(('F0-shallow', 'F6-conv2', 'B1-block1', 'T6-block6-first')):
        t = totals[i] / count
        stats['features'][name] = dict(probability_mean=t[0], gain_mean_abs_deviation_from_one=t[1],
            winner_fraction=t[2], feature_mean_abs=t[3], feature_rms=math.sqrt(t[4]),
            gated_mean_abs=t[5], gated_rms=math.sqrt(t[6]),
            probability_quantiles_05_50_95=np.quantile(np.concatenate(samples[i]), [.05, .5, .95]).tolist())
    reference = summaries['A1-G4-alpha1']['psnr']
    for result in summaries.values():
        result['delta_psnr_vs_original_A1'] = result['psnr'] - reference
        result['delta_psnr_vs_B0'] = result['psnr'] - summaries['B0-alpha1']['psnr']
    report = dict(gate_statistics=stats, validation_results=summaries,
                  note='Inference intervention on fixed learned weights; alpha0 retains A1 backbone and is not B0. No retraining or test-set selection.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print('DIAGNOSTIC COMPLETE\n' + json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
