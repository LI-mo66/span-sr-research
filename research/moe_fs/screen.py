"""Baseline + FusionGate4 only. Explicit caps, immutable configs, no auto admission."""
import argparse
import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fs_runtime', HERE / 'runtime.py')
fs_runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs_runtime)
ROOT = fs_runtime.ROOT


def immutable(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != value:
            raise ValueError(f'Frozen config differs: {path}; use a new group')
    else:
        path.write_text(json.dumps(value, indent=2), encoding='utf8')


def prepare(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.group):
        raise ValueError('Invalid group ID')
    if args.paths_file:
        paths = json.loads(Path(args.paths_file).read_text(encoding='utf-8-sig'))
    else:
        root = Path(args.data_root).resolve()
        def pair(hr, lr):
            return dict(hr=str(root / hr), lr=str(root / lr), lr_suffix='x4')
        paths = dict(train=[
            pair('DIV2K/DIV2K_train_HR', 'DIV2K/DIV2K_train_LR_bicubic/X4'),
            pair('Flickr2K/Flickr2K_HR', 'Flickr2K/Flickr2K_LR_bicubic/X4')],
            validation={'DIV2K_valid': pair('DIV2K/DIV2K_valid_HR',
                                           'DIV2K/DIV2K_valid_LR_bicubic/X4')})
        for name in ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109'):
            folder = name
            if name == 'Manga109' and not (root / 'benchmark' / name).is_dir():
                folder = 'manga109'
            paths['validation'][name] = pair(f'benchmark/{folder}/HR',
                                             f'benchmark/{folder}/LR_bicubic/X4')
    group = ROOT / 'logs/moe_fs' / args.group
    group.mkdir(parents=True, exist_ok=True)
    immutable(group / 'paths.json', paths)
    configs = {}
    for label, enabled in (('B0', False), ('A1-G4', True)):
        cfg = json.loads((HERE / 'configs' / f'{label}.json').read_text())
        cfg.update(seed=args.seed, micro_batch=args.micro_batch,
                   paths_file=str(group / 'paths.json'), image_backend=args.image_backend)
        cfg['record_updates'] = getattr(args, 'eval_every', 5000)
        path = group / f'{label}-s{args.seed}.json'
        immutable(path, cfg)
        configs[label] = path
    return group, configs


def compare(group, run_ids, updates):
    rows = {}
    for label, run_id in run_ids.items():
        history = ROOT / 'runs' / run_id / 'epochs.jsonl'
        records = [json.loads(line) for line in history.read_text().splitlines()]
        records = sorted((r for r in records if r['update'] <= updates), key=lambda r: r['update'])
        if not records or records[-1]['update'] != updates:
            raise ValueError(f'Checkpoint budget missing for {run_id}')
        tail = records[-3:]
        rows[label] = dict(run_id=run_id, updates=updates,
                           tail_updates=[r['update'] for r in tail],
                           metrics={name: {m: sum(r['metrics'][name][m] for r in tail) / len(tail)
                                          for m in ('psnr', 'ssim')}
                                    for name in tail[-1]['metrics']})
    report = dict(status='short-screen evidence only; review required before extension',
                  checkpoint_rule='mean of last up to 3 fixed recording points at common update cap',
                  rows=rows,
                  validation_delta={m: rows['A1-G4']['metrics']['DIV2K_valid'][m] -
                                     rows['B0']['metrics']['DIV2K_valid'][m]
                                     for m in ('psnr', 'ssim')})
    path = group / f'comparison-s{next(iter(run_ids.values())).rsplit("-s", 1)[1]}-u{updates}.json'
    immutable(path, report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', required=True)
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument('--data-root')
    data.add_argument('--paths-file')
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--micro-batch', type=int, choices=(4, 8, 16, 32, 64), default=16)
    p.add_argument('--image-backend', choices=('png', 'npy_mmap'), default='npy_mmap')
    p.add_argument('--updates', type=int, choices=(5000, 20000, 100000), default=5000)
    p.add_argument('--eval-every', type=int, choices=(1000, 5000), default=1000)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Source must be committed and clean')
    group, configs = prepare(args)
    if args.prepare_only:
        print('Prepared', group)
        return
    # Check BOTH configurations before spending training budget.
    for label, config in configs.items():
        subprocess.run([sys.executable, '-u', str(HERE / 'run.py'), '--config', str(config),
                        '--id', f'{args.group}-{label}-s{args.seed}', '--preflight'],
                       cwd=ROOT, check=True)
    if args.preflight_only:
        return
    run_ids = {}
    for label, config in configs.items():
        run_id = f'{args.group}-{label}-s{args.seed}'
        run_ids[label] = run_id
        resource = group / f'{label}-s{args.seed}-resource.json'
        if not resource.exists():
            subprocess.run([sys.executable, '-u', str(HERE / 'resources.py'), '--config',
                            str(config), '--output', str(resource)], cwd=ROOT, check=True)
        out = ROOT / 'runs' / run_id
        command = [sys.executable, '-u', str(HERE / 'run.py'), '--config', str(config),
                   '--id', run_id, '--screen', '--screen-updates', str(args.updates)]
        if (out / 'latest.pt').exists():
            # Refuse lowering the cap, even if training would silently do nothing.
            import torch
            state = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
            if state['update'] > args.updates:
                raise ValueError('Requested cap is below the existing checkpoint')
            command.append('--resume')
        elif out.exists():
            raise RuntimeError(f'Run directory exists without resumable checkpoint: {out}')
        print(f'RUN {run_id} -> {args.updates} updates', flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    compare(group, run_ids, args.updates)


if __name__ == '__main__':
    main()
