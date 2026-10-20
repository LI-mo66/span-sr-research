"""Hash-verified single-SG6-ATD backup; never releases the server itself."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def backup(project, group_dir, run_id, destination):
    project, group_dir = project.resolve(), group_dir.resolve()
    if not group_dir.is_relative_to(project / 'logs/sg6_atd'):
        raise ValueError('Group must be inside project/logs/sg6_atd')
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ValueError('Invalid run ID')
    run = project / 'runs' / run_id
    comparison = json.loads((group_dir / 'comparison-50000.json').read_text())
    required = ('manifest.json', 'epochs.jsonl', 'loss_steps.jsonl', 'latest.pt',
                'best_validation.pt', comparison['candidate_checkpoint'])
    for filename in required:
        if Path(filename).name != filename or not (run / filename).is_file():
            raise FileNotFoundError('Required completed artifact missing: ' + filename)
    sources = []
    for source, relative in ((group_dir, Path('logs')), (run, Path('runs') / run_id)):
        for path in source.rglob('*'):
            if path.is_symlink():
                raise ValueError('Backup source cannot contain symlinks: ' + str(path))
            if path.is_file():
                sources.append((path, relative / path.relative_to(source)))
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    needed = sum(path.stat().st_size for path, _ in sources)
    if shutil.disk_usage(destination).free < needed + 1024**3:
        raise OSError('Persistent backup space insufficient; instance retained')
    output = destination / (group_dir.name + '-' + str(time.time_ns()))
    output.mkdir()
    entries = []
    for source, relative in sources:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(source)
        shutil.copy2(source, target)
        if before != digest(target) or before != digest(source):
            raise IOError('Backup hash mismatch or changing source: ' + str(source))
        entries.append(dict(path=str(relative), sha256=before, bytes=target.stat().st_size))
    (output / 'backup-manifest.json').write_text(json.dumps(
        dict(run_id=run_id, group=group_dir.name, files=entries,
             evidence='Completed SG6-ATD only; archived SG6 referenced in reuse-audit.json',
             excluded='Original datasets and decoded cache'), indent=2), encoding='utf8')
    (output / 'BACKUP_COMPLETE').write_text('All copied files verified with SHA256.\n', encoding='utf8')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--group-dir', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    output = backup(args.project, args.group_dir, args.run_id, args.destination)
    args.receipt.write_text(str(output) + '\n', encoding='utf8')
    print('Verified SG6-ATD backup:', output, flush=True)


if __name__ == '__main__':
    main()
