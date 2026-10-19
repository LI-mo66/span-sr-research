"""Frozen-checkpoint dependence audit. No optimizer, training, or release command."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from run import cata_runtime as runtime, validate_config
from model import build_model
import numpy as np
import torch

VARIANTS = ('full', 'group_disabled', 'center_disabled', 'both_disabled',
            'ffn_disabled', 'tab_identity')


@contextmanager
def intervention(model, name, token_count):
    """IASA calls local then center once per chunk; assert that exact contract."""
    handles = []
    calls = [0]
    original = torch.nn.functional.scaled_dot_product_attention
    def sdpa(*args, **kwargs):
        branch = 'group' if calls[0] % 2 == 0 else 'center'
        calls[0] += 1
        result = original(*args, **kwargs)
        disabled = name == 'both_disabled' or name == branch + '_disabled'
        return torch.zeros_like(result) if disabled else result
    if name == 'ffn_disabled':
        handles.append(model.content_tab.mlp.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output)))
    if name == 'tab_identity':
        handles.append(model.content_tab.register_forward_hook(
            lambda module, inputs, output: inputs[0]))
    try:
        with patch('torch.nn.functional.scaled_dot_product_attention', sdpa):
            yield
        attn = model.content_tab.iasa_attn
        gs = min(token_count, attn.group_size)
        expected = 2 * math.ceil(math.ceil(token_count / gs) / attn.chunk_groups)
        if calls[0] != expected:
            raise RuntimeError(f'SDPA routing changed: {calls[0]} != {expected}')
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    cfg = manifest['config']
    validate_config(cfg)
    if not cfg.get('cata_tab') or cfg['cata_settings']['position'] != 'tail':
        raise ValueError('Requires the explicit tail candidate')
    source_paths = ['research/cata_tab/model.py', 'research/cata_tab/tab.py',
                    'research/cata_tab/runtime.py', 'research/runtime.py',
                    'research/moe_fs/trainer.py', 'research/image_io.py', 'basicsr']
    changed = subprocess.check_output(
        ['git', 'diff', manifest['commit'], '--', *source_paths],
        cwd=runtime.ROOT, text=True).strip()
    if changed:
        raise ValueError('Forward/evaluation source differs from training source')
    torch.set_num_threads(cfg.get('cpu_threads', 4))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == 'cuda':
        env = dict(torch=torch.__version__, cuda=torch.version.cuda,
                   device=torch.cuda.get_device_name(), deterministic=True,
                   cudnn_allow_tf32=True, matmul_allow_tf32=False)
        if env != manifest['environment']:
            raise ValueError(f'Environment differs from trained run: {env}')
    elif not cfg.get('engineering_only'):
        raise ValueError('Real-run reproduction requires its original CUDA environment')
    history = [json.loads(line) for line in (args.run / 'epochs.jsonl').read_text().splitlines()]
    if not cfg.get('engineering_only') and sorted(r['update'] for r in history) != list(range(1000, 50001, 1000)):
        raise ValueError('Requires the completed 50k candidate, not a partial run')
    selected = max(history, key=lambda row: row['metrics'][cfg['selection_dataset']]['psnr'])
    checkpoint = args.run / f'epoch{selected["epoch"]:05d}_u{selected["update"]}.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['fingerprint'] != manifest['fingerprint'] or state['update'] != selected['update']:
        raise ValueError('Checkpoint/manifest/selected update mismatch')
    arch, metrics = runtime.components()
    model = build_model(arch, cfg).to(args.device).eval()
    model.load_state_dict(state['model'], strict=True)
    if not model.content_tab.initted.item():
        raise ValueError('Uninitialized trained centers')
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(checkpoint=checkpoint.name, checkpoint_sha256=runtime.trainer.digest(checkpoint),
                  training_commit=manifest['commit'], fingerprint=manifest['fingerprint'],
                  scope='same-checkpoint dependence only; cannot replace independently trained ablations',
                  variants=list(VARIANTS), datasets={})
    for dataset, expected_metrics in selected['metrics'].items():
        recorded = manifest['data'][dataset]
        reference_file = args.run / f'{dataset}_epoch{selected["epoch"]:05d}_u{selected["update"]}.json'
        reference = {r['image']: r for r in json.loads(reference_file.read_text())}
        if len(reference) != len(recorded):
            raise ValueError('Per-image reference count mismatch')
        rows = []
        for index, item in enumerate(recorded, 1):
            hp, lp = Path(item['hr']), Path(item['lr'])
            for path, key in ((hp, 'hr_sha256'), (lp, 'lr_sha256')):
                if runtime.trainer.digest(path) != item[key]:
                    raise ValueError(f'Evaluation data hash differs: {path}')
            hr, lr = runtime.trainer.aligned(hp, lp)
            value = runtime.trainer.tensor(lr).unsqueeze(0).to(args.device)
            row = dict(image=hp.name, variants={})
            full = None
            with torch.inference_mode():
                for name in VARIANTS:
                    with intervention(model, name, lr.shape[0] * lr.shape[1]):
                        output = model(value)
                    if not torch.isfinite(output).all():
                        raise ValueError(f'Nonfinite prediction: {dataset}/{hp.name}/{name}')
                    if name == 'full':
                        full = output
                    sr = output[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
                    sr = np.rint(sr * 255).astype(np.uint8)
                    kw = dict(crop_border=cfg['crop_border'], test_y_channel=cfg['test_y_channel'])
                    values = dict(psnr=metrics.calculate_psnr(sr, hr, **kw),
                                  ssim=metrics.calculate_ssim(sr, hr, **kw),
                                  output_change_ratio=float((output-full).norm()/full.norm().clamp_min(1e-12)))
                    row['variants'][name] = values
                    if name == 'full':
                        old = reference[hp.name]
                        if abs(values['psnr']-old['psnr']) > 1e-6 or abs(values['ssim']-old['ssim']) > 1e-8:
                            raise ValueError(f'Full model fails historical reproduction: {dataset}/{hp.name}: {values} vs {old}')
            rows.append(row)
            if index % 20 == 0 or index == len(recorded):
                print(f'{dataset}: {index}/{len(recorded)} frozen-image audits', flush=True)
        summary = {}
        for name in VARIANTS:
            delta = np.array([r['variants'][name]['psnr']-r['variants']['full']['psnr'] for r in rows])
            summary[name] = dict(
                psnr=float(np.mean([r['variants'][name]['psnr'] for r in rows])),
                ssim=float(np.mean([r['variants'][name]['ssim'] for r in rows])),
                delta_psnr=float(delta.mean()), median_delta_psnr=float(np.median(delta)),
                delta_ssim=float(np.mean([r['variants'][name]['ssim']-r['variants']['full']['ssim'] for r in rows])),
                images_improved=int((delta > 1e-12).sum()),
                images_worse=int((delta < -1e-12).sum()),
                images_tied=int((abs(delta) <= 1e-12).sum()),
                mean_output_change_ratio=float(np.mean([r['variants'][name]['output_change_ratio'] for r in rows])))
        if abs(summary['full']['psnr']-expected_metrics['psnr']) > 1e-6:
            raise ValueError('Full dataset mean does not reproduce selected checkpoint')
        if any(not torch.equal(v.detach().cpu(), frozen[k]) for k, v in model.state_dict().items()):
            raise ValueError('Model parameters or buffers changed during audit')
        (args.output / f'{dataset}-images.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
        report['datasets'][dataset] = summary
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        print(dataset, json.dumps(summary), flush=True)
    lines = [f'Checkpoint: {checkpoint.name}',
             'Frozen checkpoint dependence; delta is variant minus full, NOT versus B0.',
             'dataset variant PSNR delta_PSNR SSIM delta_SSIM improved/worse/tied output_change']
    for dataset, summary in report['datasets'].items():
        for name, values in summary.items():
            lines.append(f"{dataset} {name} {values['psnr']:.8f} {values['delta_psnr']:+.8f} "
                         f"{values['ssim']:.8f} {values['delta_ssim']:+.8f} "
                         f"{values['images_improved']}/{values['images_worse']}/{values['images_tied']} "
                         f"{values['mean_output_change_ratio']:.8g}")
    (args.output / 'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf8')
    (args.output / 'COMPLETE').write_text('Historical reproduction and frozen-state checks passed.\n')
    print('COMPLETE: frozen checkpoint diagnostics; no training or automatic release', flush=True)


if __name__ == '__main__':
    main()
