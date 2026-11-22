"""Collect text evidence into this isolated archive checkout using CPU only."""
import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve(strict=True)
    run = project / 'runs/F0-X2-CLIP01-U1000K-21'
    group = project / 'logs/x2-clip01-600k-1000k-21'
    output = HERE / 'server'
    output.mkdir(exist_ok=False)
    assert (group / 'status.txt').read_text().strip() == 'COMPLETE_X2_CLIP01_1000K_NO_SHUTDOWN'
    history = [json.loads(line) for line in (run / 'epochs.jsonl').open() if line.strip()]
    assert [row['update'] for row in history] == list(range(1000, 1000001, 1000))
    selected = min(history, key=lambda row: (-row['metrics']['DIV2K_valid']['psnr'], row['update']))
    assert selected['update'] == 982000
    weights = {
        'latest.pt': '6a90e22b74412ab69154b797db737771c0bc38d483db75300f25f4fb8cc8ff65',
        'epoch00982_u982000.pt': '3dae5537ca672813dc4e9870f97276d173eda6f15ef8ffaf9aaf1c8406ef3068',
    }
    for name, expected in weights.items():
        assert sha(run / name) == expected, name
        print('PASS weight SHA256', name, flush=True)
    count = bad = 0
    rates = {}
    max_loss = (-math.inf, None)
    for line in (run / 'loss_steps.jsonl').open():
        if not line.strip():
            continue
        row = json.loads(line)
        count += 1
        assert row['update'] == count
        loss = row['train_l1']
        bad += not math.isfinite(loss)
        rates[str(row['lr'])] = rates.get(str(row['lr']), 0) + 1
        max_loss = max(max_loss, (loss, row['update']))
    assert count == 1000000 and bad == 0
    summary = dict(selected_update=selected['update'], selected_metrics=selected['metrics'],
                   fixed_1m_metrics=history[-1]['metrics'], checkpoint_sha256=weights,
                   loss_records=count, nonfinite_losses=bad, logged_lrs=rates,
                   max_training_l1=max_loss, scope='CPU file and hash audit; no model inference')
    (output / 'cpu-audit.json').write_text(json.dumps(summary, indent=2) + '\n')

    # Explicit paths and text formats keep checkpoints, caches and datasets out of Git.
    paths = [run / 'manifest.json', run / 'epochs.jsonl', group / 'result.json',
             group / 'to800k/stage-summary.json', group / 'to1000k/stage-summary.json']
    paths.extend(sorted(run.glob('*_epoch00982_u982000.json')))
    assert len(list(run.glob('*_epoch00982_u982000.json'))) == 12
    for stage_name in (
        'f0-x2x3-100k-01', 'f0-x2x3-300k-01',
        'x2-gradient-clip-screen-07/clip01', 'x2-clip01-200k-210k-08',
        'f0-x2clip-x3original-to300k-09', 'x2-clip01-300k-400k-13',
        'x2-clip01-400k-500k-19', 'x2-clip01-500k-600k-20',
        'x2-clip01-600k-1000k-21', 'x2-clip01-600k-1000k-21/to800k',
        'x2-clip01-600k-1000k-21/to1000k',
        'x3-clip01-recovery-295k-300k-17', 'x3-clip05-test-295k-300k-18',
    ):
        folder = project / 'logs' / stage_name
        for name in ('continue.py', 'replay.py', 'manager.py', 'worker.sh', 'prepare.py',
                     'check.py', 'provenance.json', 'stage-summary.json', 'status.txt'):
            path = folder / name
            if path.is_file():
                paths.append(path)
    for name in ('x2-F0.json', 'x2-B0.json', 'x2-paths.json', 'smoke-x2-F0.json',
                 'comparison-x2.json', 'comparison-x3.json'):
        path = project / 'logs/multiscale50k' / name
        if path.is_file():
            paths.append(path)
    entries = []
    total = 0
    for source in dict.fromkeys(paths):
        relative = source.relative_to(project)
        assert source.resolve().is_relative_to(project)
        assert source.suffix in ('.json', '.jsonl', '.py', '.sh', '.txt')
        size = source.stat().st_size
        assert size <= 20 * 1024**2, f'File exceeds20MiB: {relative}'
        total += size
        assert total <= 80 * 1024**2, 'Collection exceeds80MiB'
        expected = sha(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(target) == expected == sha(source), f'Source changed: {relative}'
        entries.append(dict(path=str(relative), bytes=size, sha256=expected))
    (output / 'collection-manifest.json').write_text(json.dumps(
        dict(files=entries, total_bytes=total, omitted=['weights', 'datasets', 'cache',
             'full per-step loss and clipping logs', 'large console logs']), indent=2) + '\n')
    (output / 'COMPLETE').write_text('CPU audit and text-file copy hashes passed.\n')
    print('COLLECTION_COMPLETE', len(entries), 'files', total, 'bytes', flush=True)


if __name__ == '__main__':
    main()
