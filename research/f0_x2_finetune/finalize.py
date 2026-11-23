"""Verify completed patch96 fine-tunes and flush persistent files before release."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import run


def finalize(group):
    torch.set_num_threads(1)
    comparison = json.loads((group / 'comparison.json').read_text())
    assert set(comparison['arms']) == set(run.ARMS)
    files = [group / name for name in (
        'comparison.json', 'baseline-verified.json', 'preflight.json'
    )]
    checkpoints = {}
    for arm, (kind, rate) in run.ARMS.items():
        folder = run.ROOT / 'runs' / ('FT-X2-982K-PC96-' + arm + '-10K-v1')
        assert not (folder / 'ANOMALY.json').exists()
        complete = json.loads((folder / 'COMPLETE.json').read_text())
        protocol = json.loads((folder / 'protocol.json').read_text())
        assert protocol['source_sha256'] == run.EXPECTED_SHA
        assert protocol['lr_patch'] == 96 and protocol['hr_patch'] == 192
        assert protocol['objective'] == kind and protocol['lr'] == rate
        state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
        assert state['step'] == complete['step'] == 10000
        assert state['fingerprint'] == complete['fingerprint']
        assert [g['lr'] for g in state['optimizer']['param_groups']] == [rate]
        for mapping in (state['model'], *state['optimizer']['state'].values()):
            assert all(bool(torch.isfinite(v).all()) for v in mapping.values()
                       if torch.is_tensor(v) and v.is_floating_point())
        del state
        epochs = [json.loads(line) for line in (folder / 'epochs.jsonl').open() if line.strip()]
        assert [r['step'] for r in epochs] == list(range(1000, 10001, 1000))
        assert all(math.isfinite(value) for row in epochs
                   for score in row['metrics'].values() for value in score.values())
        count = 0
        for line in (folder / 'steps.jsonl').open():
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            assert row['step'] == count and row['lr'] == rate
            assert all(math.isfinite(row[key]) for key in (
                'objective', 'rgb_l1', 'rgb_mse', 'gradient_norm_before_clip'
            ))
        assert count == 10000
        for step in range(1000, 10001, 1000):
            saved = torch.load(folder / f'step{step:05d}.pt',
                               map_location='cpu', weights_only=False)
            assert saved['step'] == step and saved['fingerprint'] == complete['fingerprint']
            assert saved['record'] == epochs[step // 1000 - 1]
            del saved
        checkpoints[arm] = str(folder / 'latest.pt')
        files.extend(path for path in folder.iterdir()
                     if path.is_file() and path.suffix in ('.pt', '.json', '.jsonl'))
    entries = []
    work = Path('/home/featurize/work').resolve()
    for path in files:
        assert path.resolve(strict=True).is_relative_to(work), path
        for attempt in range(3):
            try:
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                entries.append(dict(path=str(path), sha256=digest, bytes=path.stat().st_size))
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2)
    receipt = group / 'final-verification.json'
    payload = json.dumps(dict(status='PASS', checkpoints=checkpoints, files=entries), indent=2) + '\n'
    with receipt.open('w') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    assert receipt.read_text() == payload
    print('PASS both10K runs, all checkpoints and text files flushed and hashed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('group', type=Path)
    finalize(parser.parse_args().group.resolve())
