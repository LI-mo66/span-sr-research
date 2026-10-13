"""Durable, hash-verified backup before any platform release command."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from curves import generate


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def backup(project, group, destination, success=True):
    source = project / 'logs/moe_fs' / group
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / (group + '-' + str(time.time_ns()))
    output.mkdir()
    runs = {}
    run_folders = []
    for path in source.glob('*-s*.json'):
        if path.name.endswith('-resource.json'):
            continue
        cfg = json.loads(path.read_text(encoding='utf8'))
        if 'moe_fs' not in cfg:
            continue
        label = 'A1-G4' if cfg['moe_fs'] else 'B0'
        folder = project / 'runs' / f'{group}-{label}-s{cfg["seed"]}'
        if folder.is_dir():
            # Preserve every seed even if this group was reused for confirmation.
            run_folders.append(folder)
            runs[f'{label}-s{cfg["seed"]}'] = folder
    if len(run_folders) != len(set(run_folders)):
        raise ValueError('Duplicate run configurations in backup group')
    entries = []
    for folder, relative in [(source, Path('logs'))] + [(f, Path('runs') / f.name) for f in run_folders]:
        for path in folder.rglob('*'):
            if not path.is_file():
                continue
            target = output / relative / path.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            original_hash = sha256(path)
            if original_hash != sha256(target):
                raise IOError(f'Backup verification failed: {path}')
            entries.append(dict(path=str(target.relative_to(output)), sha256=original_hash, bytes=target.stat().st_size))
    if success and (not any(k.startswith('B0-s') for k in runs) or not any(k.startswith('A1-G4-s') for k in runs)):
        raise ValueError('Successful batch requires both run artifacts')
    plot_error = None
    try:
        seeds = sorted({label.rsplit('-s', 1)[1] for label in runs})
        for seed in seeds:
            selected = {label.rsplit('-s', 1)[0]: output / 'runs' / folder.name
                        for label, folder in runs.items() if label.endswith('-s' + seed)}
            generate(selected, output / 'curves' / ('seed-' + seed))
    except Exception as exc:
        plot_error = str(exc)
        if success:
            raise
    manifest = dict(group=group, training_success=success, files=entries, plotting_error=plot_error,
                    note='Contains run checkpoints/logs/metrics; excludes original datasets and decoded cache')
    (output / 'backup-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    (output / 'BACKUP_COMPLETE').write_text('All copied source files SHA256 verified.\n', encoding='utf8')
    print('Verified backup:', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--group', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--training-exit-code', type=int, default=0)
    args = parser.parse_args()
    backup(args.project, args.group, args.destination, args.training_exit_code == 0)
