"""Linux candidate-only queue, baseline provenance audit and persistent backup."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VARIANTS = ('LG6-SiLU', 'LG6-ReLU')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def selected(folder):
    rows = sorted([json.loads(s) for s in (folder / 'epochs.jsonl').read_text().splitlines() if s.strip()], key=lambda r: r['update'])
    rows = [r for r in rows if r['update'] <= 50000]
    if not rows or rows[-1]['update'] != 50000:
        raise ValueError(f'Run has not reached 50000: {folder}')
    row = max(rows, key=lambda r: r['metrics']['DIV2K_valid']['psnr'])
    def weight(r):
        name = f'epoch{r["epoch"]:05d}_u{r["update"]}.pt'
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
        return name
    peaks = {}
    for name in row['metrics']:
        best = max(rows, key=lambda r: r['metrics'][name]['psnr'])
        peaks[name] = dict(update=best['update'], checkpoint=weight(best), metrics=best['metrics'][name])
    return dict(selected_update=row['update'], checkpoint=weight(row), metrics=row['metrics'], per_dataset_psnr_best=peaks)


def audit_baseline(args, group):
    from entry import existing
    baseline = args.baseline_backup / 'runs/FEAT-MOEFS-G4-LR2E4-20K-01-B0-s10'
    if not (args.baseline_backup / 'BACKUP_COMPLETE').is_file():
        raise ValueError('Baseline backup incomplete')
    info = json.loads((baseline / 'manifest.json').read_text())
    reference = info['config']
    keys = ('model', 'seed', 'lr', 'adam_betas', 'adam_eps', 'milestones', 'gamma', 'total_updates',
            'hr_patch', 'effective_batch', 'micro_batch', 'record_updates', 'image_backend',
            'crop_border', 'test_y_channel', 'self_ensemble', 'selection_dataset')
    for label in VARIANTS:
        cfg = json.loads((group / (label + '.json')).read_text())
        for key in keys:
            if cfg.get(key) != reference.get(key):
                raise ValueError(f'Baseline protocol differs: {label}/{key}')
    import torch
    env = info['environment']
    if not torch.cuda.is_available() or str(torch.__version__) != env['torch'] or torch.version.cuda != env['cuda'] or torch.cuda.get_device_name() != env['device']:
        raise ValueError('Baseline CUDA/PyTorch/GPU environment differs')
    paths = json.loads((group / 'paths.json').read_text())
    datasets = dict(train=paths['train'], **paths['validation'])
    for name, spec in datasets.items():
        pairs = existing.fs_runtime.trainer.pairs(spec)
        expected = {Path(r['hr']).name: r for r in info['data'][name]}
        if len(pairs) != len(expected):
            raise ValueError(f'Baseline dataset count differs: {name}')
        for i, (hp, lp) in enumerate(pairs, 1):
            r = expected.get(hp.name)
            if r is None or sha(hp) != r['hr_sha256'] or sha(lp) != r['lr_sha256']:
                raise ValueError(f'Baseline image hashes differ: {hp}')
            if i % 250 == 0:
                print(f'Baseline hash audit {name}: {i}/{len(pairs)}', flush=True)
    result = selected(baseline)
    result['checkpoint_sha256'] = sha(baseline / result['checkpoint'])
    result['source'] = str(baseline)
    (group / 'baseline-reuse.json').write_text(json.dumps(result, indent=2))
    print('Baseline protocol, data hashes and environment verified; B0 not scheduled.', flush=True)
    return result


def backup(args, group, code):
    sys.path.insert(0, str(ROOT / 'research/moe_fs'))
    from curves import generate
    dest = args.backup_root / (args.group + '-' + str(time.time_ns()))
    dest.mkdir(parents=True)
    folders = {label: ROOT / 'runs' / f'{args.group}-{label}-s10' for label in VARIANTS}
    sources = [(group, dest / 'logs')] + [(p, dest / 'runs' / p.name) for p in folders.values() if p.is_dir()]
    manifest = []
    for src, target in sources:
        for p in src.rglob('*'):
            if not p.is_file():
                continue
            out = target / p.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            digest = sha(p)
            if sha(out) != digest:
                raise IOError(f'Backup hash mismatch: {p}')
            manifest.append(dict(path=out.relative_to(dest).as_posix(), sha256=digest, bytes=out.stat().st_size))
    available = {label: dest / 'runs' / p.name for label, p in folders.items() if (dest / 'runs' / p.name / 'epochs.jsonl').is_file()}
    plotting_error = None
    if available:
        try:
            generate(available, dest / 'curves/seed-10')
        except Exception as exc:
            plotting_error = str(exc)
            if code == 0:
                raise
    (dest / 'backup-manifest.json').write_text(json.dumps(dict(files=manifest, training_exit_code=code, plotting_error=plotting_error), indent=2))
    (dest / 'BACKUP_COMPLETE').write_text('Copied files SHA256 verified; check training_exit_code for training success.\n')
    print('Verified backup:', dest, flush=True)
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--baseline-backup', type=Path, required=True)
    p.add_argument('--backup-root', type=Path, required=True)
    p.add_argument('--job', type=Path, required=True)
    args = p.parse_args()
    args.backup_root.mkdir(parents=True, exist_ok=True)
    args.job.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (ROOT / 'logs/server-gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            (args.job / 'status.json').write_text(json.dumps(dict(exit_code=75, reason='GPU lock held')))
            raise SystemExit(75)
        group = ROOT / 'logs/linear_gate' / args.group
        code, error = 0, None
        try:
            if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
                raise ValueError('Source must be committed and clean')
            subprocess.run([sys.executable, str(HERE / 'prepare.py'), '--group', args.group,
                            '--data-root', str(args.data_root)], cwd=ROOT, check=True)
            baseline = audit_baseline(args, group)
            results = {}
            for label in VARIANTS:
                run_id = f'{args.group}-{label}-s10'
                folder = ROOT / 'runs' / run_id
                command = [sys.executable, '-u', str(HERE / 'entry.py'), '--config', str(group / (label+'.json')),
                           '--id', run_id, '--screen', '--screen-updates', '50000']
                if folder.exists():
                    import torch
                    state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
                    if state['update'] > 50000:
                        raise ValueError('Checkpoint exceeds 50k cap')
                    del state
                    command.append('--resume')
                print('RUN', label, '-> 50000 updates', flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                result = selected(folder)
                result['validation_delta_vs_B0'] = {m: result['metrics']['DIV2K_valid'][m]-baseline['metrics']['DIV2K_valid'][m] for m in ('psnr', 'ssim')}
                results[label] = result
                (group / 'comparison-50000.json').write_text(json.dumps(dict(B0=baseline, candidates=results,
                    checkpoint_rule='Highest DIV2K validation PSNR; earliest tie; no temporal averaging'), indent=2))
                print(json.dumps({label: result}, indent=2), flush=True)
        except Exception as exc:
            code, error = 1, repr(exc)
            print('QUEUE ERROR:', error, flush=True)
        status = dict(exit_code=code, error=error, auto_release=False)
        (args.job / 'status.json').write_text(json.dumps(status, indent=2))
        try:
            dest = backup(args, group, code)
            status.update(backup_exit_code=0, backup=str(dest))
        except Exception as exc:
            status.update(backup_exit_code=1, backup_error=repr(exc))
            code = 74
        (args.job / 'status.json').write_text(json.dumps(status, indent=2))
        (args.backup_root / (args.group+'-'+args.job.name+'-status.json')).write_text(json.dumps(status, indent=2))
        print(json.dumps(status, indent=2), flush=True)
        print('Instance retained; no release command is called.', flush=True)
        raise SystemExit(code)


if __name__ == '__main__':
    main()
