"""Fixed GB checkpoint: local gate statistics and spatial-variation interventions."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import entry
from diagnose_cpu import sha
from gb_model import build_model


def intervention(alpha):
    def hook(module, inputs, gate):
        if alpha == 1:
            return gate
        mean = gate.mean(dim=(-2, -1), keepdim=True)
        return mean + alpha * (gate - mean)
    return hook


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backup', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runs = list((args.backup / 'runs').glob('*-LG6-SiLU-GB-s10'))
    if len(runs) != 1 or not (args.backup / 'BACKUP_COMPLETE').is_file():
        raise ValueError('Expected one complete GB backup')
    run = runs[0]
    info = json.loads((run / 'manifest.json').read_text())
    cfg = info['config']
    rows = sorted([json.loads(s) for s in (run / 'epochs.jsonl').read_text().splitlines() if s.strip()], key=lambda r: r['update'])
    rows = [r for r in rows if r['update'] <= 50000]
    if not rows or rows[-1]['update'] != 50000:
        raise ValueError('GB reference must complete 50K')
    best = max(rows, key=lambda r: r['metrics']['DIV2K_valid']['psnr'])
    checkpoint = run / f'epoch{best["epoch"]:05d}_u{best["update"]}.pt'
    relative = checkpoint.relative_to(args.backup).as_posix()
    inventory = json.loads((args.backup / 'backup-manifest.json').read_text())['files']
    expected = next(r['sha256'] for r in inventory if r['path'] == relative)
    if sha(checkpoint) != expected:
        raise ValueError('Checkpoint backup SHA mismatch')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['fingerprint'] != info['fingerprint']:
        raise ValueError('Checkpoint fingerprint mismatch')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = info['environment']['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32 = info['environment']['matmul_allow_tf32']
    if args.device == 'cuda':
        env = info['environment']
        if not torch.cuda.is_available() or str(torch.__version__) != env['torch'] or torch.version.cuda != env['cuda'] or torch.cuda.get_device_name() != env['device']:
            raise ValueError('CUDA reference environment mismatch')
    arch, metrics = entry.existing.fs_runtime.components()
    trainer = entry.existing.fs_runtime.trainer
    model = build_model(arch, cfg)
    model.load_state_dict(state['model'], strict=True)
    del state
    model.to(args.device).eval()
    files = []
    for rec in info['data']['DIV2K_valid']:
        hp = args.data_root / 'DIV2K/DIV2K_valid_HR' / Path(rec['hr']).name
        lp = args.data_root / 'DIV2K/DIV2K_valid_LR_bicubic/X4' / Path(rec['lr']).name
        if sha(hp) != rec['hr_sha256'] or sha(lp) != rec['lr_sha256']:
            raise ValueError(f'Validation image hash mismatch: {hp}')
        files.append((hp, lp))
    if len(files) != 100:
        raise ValueError('Expected 100 validation images')
    block_stats = {str(i): [] for i in range(1, 7)}
    handles = []
    current = {}
    def statistics(i):
        def hook(module, inputs, output):
            gate = output[2].detach().float()
            # Recover F3 with a c3 hook; do not divide by a possibly zero gate.
            branch = current[i]
            summed = branch + inputs[0]
            rms = lambda t: float(t.float().square().mean().sqrt())
            spatial = gate - gate.mean((-2, -1), keepdim=True)
            block_stats[str(i)].append(dict(
                image=current['image'], gate_mean=float(gate.mean()),
                gate_std=float(gate.std(unbiased=False)), gate_min=float(gate.min()),
                gate_max=float(gate.max()), negative_fraction=float((gate < 0).float().mean()),
                abs_above_one_fraction=float((gate.abs() > 1).float().mean()),
                spatial_rms=rms(spatial), input_rms=rms(inputs[0]), f3_rms=rms(branch),
                pre_gate_rms=rms(summed), output_rms=rms(output[0]),
                rms_gain=rms(output[0]) / max(rms(summed), 1e-12)))
        return hook
    def capture(i):
        def hook(module, inputs, output):
            current[i] = output.detach()
        return hook
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=expected,
                  selected_update=best['update'], source_commit=info.get('commit'),
                  device=args.device, scenarios={}, block_statistics=block_stats,
                  intervention='m_alpha=spatial_mean_per_image_channel(m)+alpha*(m-mean); one block at a time; image bypass unchanged',
                  limitation='Fixed-checkpoint sensitivity only; not retraining ablation or proof of causal training contribution.')
    scenarios = [('reference', None, 1.)] + [(f'block{i}_alpha{a}', i, a) for i in range(1, 7) for a in (.5, 0.)]
    for label, index, alpha in scenarios:
        if index is None:
            for i in range(1, 7):
                block = getattr(model, f'block_{i}')
                handles.extend([block.c3_r.register_forward_hook(capture(i)), block.register_forward_hook(statistics(i))])
        else:
            handles.append(getattr(model, f'block_{index}').local_gate.register_forward_hook(intervention(alpha)))
        values = []
        try:
            with torch.inference_mode():
                for j, (hp, lp) in enumerate(files, 1):
                    current['image'] = hp.name
                    hr, lr = trainer.aligned(hp, lp)
                    pred = model(trainer.tensor(lr).unsqueeze(0).to(args.device))[0]
                    if not torch.isfinite(pred).all():
                        raise FloatingPointError(label)
                    sr = np.rint(pred.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
                    if sr.shape != hr.shape:
                        raise ValueError('Output shape mismatch')
                    kw = dict(crop_border=cfg['crop_border'], test_y_channel=cfg['test_y_channel'])
                    values.append(dict(image=hp.name, psnr=metrics.calculate_psnr(sr, hr, **kw), ssim=metrics.calculate_ssim(sr, hr, **kw)))
                    if j % 20 == 0:
                        print(f'{label}: {j}/100', flush=True)
        finally:
            for handle in handles:
                handle.remove()
            handles.clear()
        (args.output / (label + '.json')).write_text(json.dumps(values, indent=2))
        score = {m: float(np.mean([r[m] for r in values])) for m in ('psnr', 'ssim')}
        if index is None:
            score['delta_vs_archived'] = {m: score[m] - best['metrics']['DIV2K_valid'][m] for m in ('psnr', 'ssim')}
            if args.device == 'cuda' and any(abs(v) > 1e-6 for v in score['delta_vs_archived'].values()):
                raise ValueError(f'Reference metric mismatch: {score}')
        else:
            score['delta_vs_reference'] = {m: score[m] - report['scenarios']['reference'][m] for m in ('psnr', 'ssim')}
        report['scenarios'][label] = score
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        print(label, json.dumps(score), flush=True)
    compact = {k: v for k, v in report.items() if k != 'block_statistics'}
    compact['block_statistics_image_means'] = {
        i: {k: float(np.mean([r[k] for r in records])) for k in records[0] if k != 'image'}
        for i, records in block_stats.items()}
    (args.output / 'summary.json').write_text(json.dumps(compact, indent=2))
    print('COMPLETE', json.dumps(compact, indent=2), flush=True)


if __name__ == '__main__':
    main()
