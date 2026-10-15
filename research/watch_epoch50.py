"""External Linux watcher: stop only after committed epoch-50 state, export reports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import zipfile


def read_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def matches(pid, root, run_id):
    try:
        argv = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        argv = [a.decode() for a in argv if a]
        return (any(a.endswith('research/train.py') for a in argv)
                and '--id' in argv and argv[argv.index('--id')+1] == run_id
                and (Path('/proc')/str(pid)/'cwd').resolve() == root)
    except (OSError, ValueError, IndexError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--id', default='SPAN52-X4-FULL-S10')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    if not args.id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.id):
        raise ValueError('Invalid run ID')
    out = root/'runs'/args.id
    history = out/'epochs.jsonl'
    snapshot = out/'epoch00050_u250000.pt'
    dest = Path(args.output).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    # Prevent multiple watchers from independently stopping/exporting this run.
    import fcntl
    lock = open(out/'epoch50-watch.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('Waiting for epoch=50, update=250000 and completed checkpoint/history write', flush=True)
    while True:
        rows = read_rows(history)
        target = next((r for r in rows if r['epoch']==50 and r['update']==250000), None)
        if target and snapshot.exists():
            break
        pids = [int(d.name) for d in Path('/proc').iterdir()
                if d.name.isdigit() and matches(int(d.name), root, args.id)]
        if not pids:
            raise RuntimeError('Training not running and epoch50 is unavailable; no signals sent')
        print(f'Latest saved update={rows[-1]["update"] if rows else 0}; watching {pids}', flush=True)
        time.sleep(30)
    latest_saved = max(r['update'] for r in rows)
    pids = [int(d.name) for d in Path('/proc').iterdir()
            if d.name.isdigit() and matches(int(d.name), root, args.id)]
    for pid in pids:
        if matches(pid, root, args.id):
            os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if not any(matches(pid, root, args.id) for pid in pids):
            break
        time.sleep(1)
    else:
        raise RuntimeError('Training did not exit after SIGTERM; refusing to archive changing logs')
    # Preserve the explicit epoch50 checkpoint; never relabel a later latest.pt.
    checkpoint_info = dict(path=str(snapshot), bytes=snapshot.stat().st_size)
    h = hashlib.sha256()
    with snapshot.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    checkpoint_info['sha256'] = h.hexdigest()
    rows = [r for r in read_rows(history) if r['update']<=250000]
    (dest/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for name in ['manifest.json', 'console.log']:
        if (out/name).exists():
            shutil.copy2(out/name, dest/name)
    for file in out.glob('*_epoch*.json'):
        try:
            update = int(file.stem.rsplit('_u', 1)[1])
        except (ValueError, IndexError):
            continue
        if update<=250000:
            shutil.copy2(file, dest/file.name)
    job = root/'logs/server_jobs'/args.id
    configs = dest/'job'; configs.mkdir()
    for file in job.glob('*'):
        if file.is_file() and file.suffix in ['.json', '.log', '.txt', '.sh']:
            shutil.copy2(file, configs/file.name)
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
    summary = dict(run_id=args.id, source_commit=commit, target_update=250000,
                   latest_saved_when_observed=latest_saved, checkpoint50=checkpoint_info,
                   epoch50=target, seed=10, stopped_training_pids=pids,
                   limitation='Single seed, 250K stage. Not completed official 1M reproduction. '
                   'May execute extra unsaved updates before watcher observes completion. '
                   'best_validation.pt may include later epochs if watcher was started late.')
    (dest/'summary.json').write_text(json.dumps(summary, indent=2))
    (dest/'README.md').write_text(
        '# SPAN52 x4 seed10: 250K stage\n\n'
        'Epoch50 checkpoint remains on server; see summary.json for path and SHA256.\n'
        'Reports contain epochs <=50. Raw console logs may include later work.\n'
        'Weights are deliberately excluded from this report archive.\n'
        'No automatic GitHub push or provider shutdown is performed.\n')
    archive = dest.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in dest.rglob('*'):
            if file.is_file():
                z.write(file, file.relative_to(dest))
    print(f'STOPPED. Epoch50 checkpoint: {snapshot}\nReport archive: {archive}', flush=True)


if __name__ == '__main__':
    main()
