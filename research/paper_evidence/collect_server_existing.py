"""Collect existing SPAN paper evidence with Python stdlib, without GPU use."""
import argparse
import hashlib
import json
import math
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/featurize/work/span-x4-research'))
    parser.add_argument('--max-mib', type=int, default=200)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.max_mib < 1:
        parser.error('--max-mib must be positive')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = root/'records'/f'PAPER-EXISTING-EVIDENCE-{stamp}-{os.getpid()}'
    output.mkdir(parents=True, exist_ok=False)
    limit = args.max_mib*1024*1024
    total = 0
    inventory, runs, errors = [], [], []
    excluded = {'.git', '__pycache__', 'datasets', 'image_cache', 'node_modules',
                'vendor', 'venv', '.venv', 'site-packages', 'tests'}
    relevant = ('sg6', 'f0', 'ucan', 'fusion', 'mid-window', 'mid_window', 'wu-', 'wu_', 'bound8')

    def keep(path, category, data=None):
        nonlocal total
        try:
            relative = path.relative_to(root)
            size = path.stat().st_size
            if size > 8*1024*1024 and data is None:
                inventory.append(dict(path=str(path), category=category, status='SIZE_LIMIT', bytes=size))
                return
            payload = path.read_bytes() if data is None else data
            if total+len(payload) > limit:
                inventory.append(dict(path=str(path), category=category, status='TOTAL_LIMIT', bytes=size))
                return
            destination = output/'files'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
            total += len(payload)
            inventory.append(dict(path=str(path), saved=str(destination.relative_to(output)),
                                  category=category, status='COPIED', bytes=len(payload),
                                  sha256=digest, scope='snapshot of potentially live file'))
        except (OSError, ValueError) as exc:
            errors.append(dict(path=str(path), error=str(exc)))

    for start in [root/'code', root/'records']:
        if not start.is_dir():
            errors.append(dict(path=str(start), error='NOT_FOUND'))
            continue
        for folder, dirs, filenames in os.walk(start, followlinks=False):
            current = Path(folder)
            dirs[:] = [d for d in dirs if d not in excluded
                       and not d.startswith('PAPER-EXISTING-EVIDENCE-')
                       and not any(s in d.lower() for s in ('fixture', 'smoke', 'engineering', 'tmp'))]
            if not any(t in str(current).lower() for t in relevant):
                continue
            for name in sorted(filenames):
                path = current/name
                if path.is_symlink():
                    continue
                lower = name.lower()
                if lower == 'epochs.jsonl':
                    keep(path, 'evaluation_history')
                    best = {}
                    count, last, invalid = 0, None, 0
                    try:
                        with path.open(encoding='utf-8-sig') as stream:
                            for line in stream:
                                if not line.strip():
                                    continue
                                try:
                                    row = json.loads(line)
                                    if not isinstance(row, dict):
                                        invalid += 1
                                        continue
                                    update = row.get('update', row.get('total_update'))
                                    if not isinstance(update, int):
                                        invalid += 1
                                        continue
                                    count += 1
                                    last = update
                                    for key in ('DIV2K_valid', 'DIV2K_valid_x8'):
                                        value = row.get('metrics', {}).get(key, {}).get('psnr')
                                        if not isinstance(value, (int, float)) or not math.isfinite(value):
                                            continue
                                        rank = (-value, update)
                                        if key not in best or rank < best[key][0]:
                                            best[key] = (rank, row)
                                except (ValueError, TypeError, AttributeError):
                                    invalid += 1
                        runs.append(dict(path=str(path), count=count, last_update=last,
                                         invalid_rows=invalid,
                                         selected={k:v[1] for k,v in best.items()},
                                         rule='Each DIV mode selected separately; all metric values from its same history row',
                                         status='existing-record snapshot; no checkpoint load or reevaluation'))
                    except OSError as exc:
                        errors.append(dict(path=str(path), error=str(exc)))
                elif lower.endswith('.png'):
                    if any(s in lower for s in ('img004', 'img062', 'img073', 'img092', 'aisazuniha')):
                        keep(path, 'existing_image')
                elif lower.endswith('.log'):
                    try:
                        with path.open('rb') as stream:
                            stream.seek(max(0, path.stat().st_size-65536))
                            keep(path, 'log_tail_last_64KiB', stream.read(65536))
                    except OSError as exc:
                        errors.append(dict(path=str(path), error=str(exc)))
                elif lower in ('manifest.json', 'readme.md', 'readme.zh.md', 'config.json',
                               'protocol.json', 'status.txt', 'complete', 'complete.json',
                               'best-checkpoint.json', 'comparison.json', 'scores.json',
                               'backup-path.txt') or (
                    lower.endswith(('.json', '.md', '.sh', '.py')) and any(s in lower for s in
                    ('comparison', 'finetune', 'protocol', 'resume', 'bound', 'replay', 'worker', 'queue', 'profile', 'summary'))
                ):
                    keep(path, 'protocol_or_existing_result')
    for name, value in [('file-inventory.json', inventory), ('run-summary.json', runs), ('errors.json', errors)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'README.txt').write_text(
        'Existing records only. No torch import, GPU use, training, evaluation, process control or shutdown.\n'
        'Files from live runs are snapshots, not an atomic experiment backup.\n'
        'No model weights collected. See SIZE_LIMIT/TOTAL_LIMIT and errors.json for omissions.\n'
        'History-best plain and x8 summaries may select different weights; do not combine them into one model row.\n', encoding='utf-8')
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.add(output, arcname=output.name)
    with tarfile.open(archive, 'r:gz') as bundle:
        for entry in bundle:
            if entry.isfile():
                with bundle.extractfile(entry) as stream:
                    while stream.read(1024*1024):
                        pass
    (output/'COMPLETE').write_text('COLLECTION_FINISHED_NO_SHUTDOWN\n', encoding='utf-8')
    print('OUTPUT='+str(output), flush=True)
    print('ARCHIVE='+str(archive), flush=True)
    print('Collected_bytes='+str(total)+' histories='+str(len(runs)), flush=True)
    print('Read errors='+str(len(errors))+'; inspect inventory for omissions.', flush=True)


if __name__ == '__main__':
    main()

