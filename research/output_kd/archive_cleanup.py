"""Archive this rejected experiment to GitHub, then remove only verified sources.

Linux server tool. No datasets, B0, shared cache, source checkout or environment
is removed. A failed upload/verification leaves all original files untouched.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

GROUP = 'FEAT-SPAN-KD-OUTPUT-50K-01'
RUN = GROUP + '-SPAN-KD-Output-s10'
REMOTE = 'https://github.com/LI-mo66/span-sr-research.git'
WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors', '.state'}
SMALL = {'.json', '.jsonl', '.log', '.txt', '.md', '.yaml', '.yml', '.png', '.svg', '.csv', '.sha256'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def live_guard():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            # Inspect script arguments, not shell -c/heredoc contents.
            scripts = [s.decode(errors='replace') for s in argv[1:3]]
            if any(s.endswith(('output_kd/entry.py', 'output_kd/server.py',
                               '/stop-kd-at-20k.py', '/stop_at_20k.py')) for s in scripts):
                raise RuntimeError(f'Experiment/watch process {proc.name} is still present; no cleanup')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def discover(work, runtime):
    records = work / 'records'
    found = []
    run = runtime / 'runs' / RUN
    logs = runtime / 'logs/output_kd' / GROUP
    for path in (run, logs):
        if path.exists():
            found.append(path)
    if records.is_dir():
        for p in records.iterdir():
            if p.is_dir() and re.fullmatch(re.escape(GROUP) + r'-\d+', p.name):
                found.append(p)
            elif p.is_file() and (p.name.startswith(GROUP + '-') or
                                  p.name in ('stop-kd-at-20k.py',)):
                if p.suffix in SMALL or p.suffix == '.py':
                    found.append(p)
    for path in found:
        validate_target(path, work, runtime)
    return sorted(set(found))


def validate_target(path, work, runtime):
    if path.is_symlink():
        raise ValueError(f'Symlink cleanup target refused: {path}')
    resolved = path.resolve()
    records = (work / 'records').resolve()
    permitted = resolved in ((runtime / 'runs' / RUN).resolve(),
                             (runtime / 'logs/output_kd' / GROUP).resolve())
    if resolved.parent == records:
        permitted = (bool(re.fullmatch(re.escape(GROUP) + r'-\d+', resolved.name))
                     or (resolved.is_file() and (resolved.name.startswith(GROUP + '-')
                                                or resolved.name == 'stop-kd-at-20k.py')))
    if not permitted:
        raise ValueError(f'Target outside the exact experiment allowlist: {path}')


def files_under(path):
    if path.is_file():
        return [path]
    files = []
    for p in path.rglob('*'):
        if p.is_symlink():
            raise ValueError(f'Symlink inside archive source refused: {p}')
        if p.is_file():
            files.append(p)
    return sorted(files)


def read_rows(folder):
    result = {}
    path = folder / 'epochs.jsonl'
    if not path.is_file():
        return []
    for line in path.read_text(encoding='utf8').splitlines():
        try:
            row = json.loads(line)
            result[int(row['update'])] = row
        except (ValueError, KeyError, TypeError):
            continue
    return [result[k] for k in sorted(result)]


def make_archive(targets, archive):
    archive.mkdir(parents=True)
    inventory = []
    candidates = []
    for index, source in enumerate(targets):
        destination = archive / 'sources' / f'{index:02d}'
        for path in files_under(source):
            relative = Path(path.name) if source.is_file() else path.relative_to(source)
            item = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
            if path.suffix in SMALL or path.name == 'BACKUP_COMPLETE' or path.suffix == '.py':
                out = destination / relative
                out.parent.mkdir(parents=True, exist_ok=True)
                if item['bytes'] > 40*1024**2:
                    out = out.with_name(out.name + '.gz')
                    with path.open('rb') as src, gzip.open(out, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    shutil.copy2(path, out)
                if out.stat().st_size > 45*1024**2:
                    raise ValueError(f'Archive file too large for ordinary Git; originals preserved: {path}')
                item['archived_as'] = out.relative_to(archive).as_posix()
                item['archive_sha256'] = sha(out)
            elif path.suffix in WEIGHTS:
                item['retention'] = 'inventory hash; intermediate optimizer/checkpoint removed after upload'
            elif path.suffix in ('.lock', '.tmp') or path.name in ('pid',):
                item['retention'] = 'ephemeral file; inventory only'
            else:
                # Never silently delete an unrecognized potentially useful artifact.
                raise ValueError(f'Unclassified artifact {path}; originals preserved')
            inventory.append(item)
            if path.name == 'epochs.jsonl':
                history = read_rows(path.parent)
                if history:
                    candidates.append((history[-1]['update'], path.parent, history))
    if not candidates:
        raise ValueError('No complete evaluation records found; originals preserved')
    _, folder, history = max(candidates, key=lambda entry: entry[0])
    best = max(history, key=lambda row: row['metrics']['DIV2K_valid']['psnr'])
    selected = folder / f'epoch{best["epoch"]:05d}_u{best["update"]}.pt'
    if not selected.is_file():
        selected = folder / 'best_validation.pt'
    if not selected.is_file():
        raise FileNotFoundError('Selected student checkpoint missing; originals preserved')
    import torch
    state = torch.load(selected, map_location='cpu', weights_only=False)
    if state['update'] != best['update'] or 'model' not in state:
        raise ValueError('Selected checkpoint/history mismatch')
    # Retain one inference model with config/provenance, excluding optimizer/RNG.
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf8'))
    buffer = io.BytesIO()
    torch.save(dict(model=state['model'], config=manifest['config'],
                    source_checkpoint_sha256=sha(selected), update=state['update']), buffer)
    model = archive / 'selected-student.pt.gz'
    model.write_bytes(gzip.compress(buffer.getvalue()))
    if model.stat().st_size > 45*1024**2:
        raise ValueError('Selected model too large for Git; originals preserved')
    # Check the retained student survives serialization before deleting full states.
    restored = torch.load(io.BytesIO(gzip.decompress(model.read_bytes())), map_location='cpu', weights_only=False)
    assert all(torch.equal(value, restored['model'][key]) for key,value in state['model'].items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, key, title in zip(axes, ('train_l1_mean', 'psnr', 'ssim'),
                                ('HR reconstruction L1', 'DIV2K validation PSNR', 'DIV2K validation SSIM')):
        if key == 'train_l1_mean':
            values = [(row['update'], row[key]) for row in history if key in row]
        else:
            values = [(row['update'], row['metrics']['DIV2K_valid'][key]) for row in history]
        if values:
            axis.plot([p[0] for p in values], [p[1] for p in values])
        axis.set(xlabel='Optimizer updates', title=title)
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(archive / 'overview.png', dpi=180)
    fig.savefig(archive / 'overview.svg')
    plt.close(fig)
    summary = dict(status='rejected_by_user', reason='online teacher compute cost exceeds acceptable budget',
                   actual_last_recorded_update=history[-1]['update'],
                   selected_update=best['update'], metrics=best['metrics'],
                   source_run=str(folder), retained_model=model.name,
                   caveat='Rejection is a resource decision; no claim that all distillation is ineffective')
    (archive / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    (archive / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf8')
    (archive / 'README.md').write_text(
        '# Rejected SPAN52 output-distillation experiment\n\n'
        'Archived at the user request after abandoning this online-teacher candidate. '
        'See summary.json for actual completed updates and validation-selected metrics; '
        'sources/ retains raw configuration, per-image scores, loss traces, plots and logs. '
        'One student model is retained as selected-student.pt.gz; optimizer/intermediate weights '
        'are inventoried by SHA256 and deliberately discarded after verified remote upload. '
        'B0, all image datasets, shared caches, teacher weights and other experiments are preserved.\n', encoding='utf8')
    return inventory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, default=Path('/home/featurize/work/span-x4-research'))
    p.add_argument('--runtime', type=Path, default=Path('/home/featurize/span-x4-runtime'))
    p.add_argument('--apply', action='store_true', help='Upload, verify remote, then delete exact inventoried sources')
    args = p.parse_args()
    work, runtime = args.work.resolve(), args.runtime.resolve()
    live_guard()
    targets = discover(work, runtime)
    if not targets:
        print('No matching experiment artifacts; nothing deleted.')
        return
    total = 0
    for target in targets:
        size = sum(p.stat().st_size for p in files_under(target))
        total += size
        print(f'{size/1024**3:.3f} GiB  {target}', flush=True)
    print(f'Exact scoped sources: {total/1024**3:.3f} GiB. Shared data/cache/B0/teacher preserved.', flush=True)
    if not args.apply:
        print('Inventory only. Use --apply to archive/upload/verify before cleanup.')
        return
    tag = time.strftime('%Y%m%dT%H%M%S') + '-' + format(time.time_ns() % 0x100000000, '08x')
    checkout = work / 'code' / ('kd-arc-' + tag)
    checkout.parent.mkdir(parents=True, exist_ok=True)
    print(command(['git', 'clone', '--depth', '1', '--single-branch', '--branch',
                   'research/span-kd-output', REMOTE, str(checkout)]), flush=True)
    branch = 'archive/span-kd-output-' + tag
    command(['git', 'switch', '-c', branch], checkout)
    relative = Path('research/docs/module-study/results') / ('KD-OUT-' + tag)
    archive = checkout / relative
    inventory = make_archive(targets, archive)
    export = work / 'records' / ('KD-RESULTS-EXPORT-' + tag + '.tar.gz')
    with tarfile.open(export, 'w:gz') as bundle:
        bundle.add(archive, arcname=archive.name)
    print(f'LOCAL_ARCHIVE: {archive}\nPORTABLE_EXPORT: {export}', flush=True)
    print(command(['git', '-c', 'user.name=LI-mo66', '-c', 'user.email=LI-mo66@users.noreply.github.com',
                   'add', '-f', str(relative)], checkout), flush=True)
    print(command(['git', '-c', 'user.name=LI-mo66', '-c', 'user.email=LI-mo66@users.noreply.github.com',
                   'commit', '-m', 'Archive rejected SPAN output KD results and selected student'], checkout), flush=True)
    print(command(['git', '-c', 'credential.interactive=never', 'push', 'origin', 'HEAD:refs/heads/' + branch], checkout), flush=True)
    commit = command(['git', 'rev-parse', 'HEAD'], checkout)
    remote = command(['git', 'ls-remote', 'origin', 'refs/heads/' + branch], checkout).split()[0]
    if remote != commit:
        raise RuntimeError('Remote verification failed; originals preserved')
    live_guard()
    # Confirm every source still matches what was archived before any deletion.
    for item in inventory:
        path = Path(item['path'])
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise RuntimeError(f'Source changed during archival; nothing deleted: {path}')
    for target in targets:
        validate_target(target, work, runtime)
        expected = {item['path'] for item in inventory if Path(item['path']) == target or target in Path(item['path']).parents}
        if {str(p) for p in files_under(target)} != expected:
            raise RuntimeError(f'Source contents changed; nothing deleted: {target}')
    url = f'https://github.com/LI-mo66/span-sr-research/tree/{commit}/{relative.as_posix()}'
    receipt = dict(commit=commit, branch=branch, url=url, targets=[str(t) for t in targets],
                   local_export=str(export),
                   archived_bytes=sum(p.stat().st_size for p in archive.rglob('*') if p.is_file()),
                   removed_source_bytes=total, completed=[])
    receipt_path = work / 'records' / ('KD-CLEANUP-RECEIPT-' + tag + '.json')
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf8')
    for target in targets:
        validate_target(target, work, runtime)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        receipt['completed'].append(str(target))
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(f'UPLOADED: {url}\nCLEANED: {total/1024**3:.3f} GiB of original experiment files\nReceipt: {receipt_path}', flush=True)


if __name__ == '__main__':
    main()
