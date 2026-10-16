"""Two-update synthetic CUDA run verifies entry point, artifacts and exact resume."""
import importlib.util
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

HERE = Path(__file__).resolve().parent
from entry import existing
import cv2
import numpy as np
import torch
fs_runtime = existing.fs_runtime
os.environ['LINEAR_GATE_ENGINEERING'] = '1'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('LG6-SiLU-PB',), default='LG6-SiLU-PB')
    variant = parser.parse_args().variant
    fixture = fs_runtime.ROOT / 'tmp' / ('pb-engineering-' + uuid.uuid4().hex[:8])
    fixture.mkdir(parents=True)
    paths = {'validation': {}}
    rng = np.random.default_rng(20261012)
    for name in ('train', 'DIV2K_valid', 'Set5', 'Set14', 'B100', 'Urban100', 'Manga109'):
        hr, lr = fixture / name / 'HR', fixture / name / 'LR'
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        assert cv2.imwrite(str(hr / 'a.png'), rng.integers(0, 256, (64, 80, 3), dtype=np.uint8))
        assert cv2.imwrite(str(lr / 'a.png'), rng.integers(0, 256, (16, 20, 3), dtype=np.uint8))
        pair = dict(hr=str(hr), lr=str(lr))
        if name == 'train':
            paths['train'] = pair
        else:
            paths['validation'][name] = pair
    path_file = fixture / 'paths.json'
    path_file.write_text(json.dumps(paths))
    cfg = json.loads((HERE / 'configs' / (variant + '.json')).read_text())
    cfg.update(engineering_only=True, paths_file=str(path_file), effective_batch=2,
               micro_batch=1, hr_patch=32, total_updates=2, record_updates=1,
               screen_epochs=1, image_backend='npy_mmap')
    config = fixture / 'config.json'
    config.write_text(json.dumps(cfg))
    run_id = 'ENGINEERING-PB-' + uuid.uuid4().hex[:10]
    command = [sys.executable, '-u', str(HERE / 'entry.py'), '--config', str(config), '--id', run_id]
    subprocess.run(command + ['--screen'], cwd=fs_runtime.ROOT, check=True)
    out = fs_runtime.ROOT / 'runs' / run_id
    with open(out / 'epochs.jsonl', 'a') as f:
        f.write('{"interrupted":')
    with open(out / 'loss_steps.jsonl', 'a') as f:
        f.write(json.dumps(dict(update=999, train_l1=999))+'\n'+ '{"interrupted":')
    subprocess.run(command + ['--screen', '--screen-updates', '50000', '--resume'],
                   cwd=fs_runtime.ROOT, check=True)
    direct_id = run_id + '-continuous'
    direct_cfg = dict(cfg, record_updates=2)
    direct_config = fixture / 'direct-config.json'
    direct_config.write_text(json.dumps(direct_cfg))
    subprocess.run([sys.executable, '-u', str(HERE / 'entry.py'), '--config', str(direct_config),
                    '--id', direct_id, '--screen', '--screen-updates', '50000'],
                   cwd=fs_runtime.ROOT, check=True)
    resumed = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    direct = torch.load(fs_runtime.ROOT / 'runs' / direct_id / 'latest.pt',
                        map_location='cpu', weights_only=False)
    assert resumed['update'] == direct['update'] == 2
    assert all(torch.equal(v, direct['model'][k]) for k, v in resumed['model'].items())
    assert resumed['scheduler'] == direct['scheduler']
    assert len((out / 'epochs.jsonl').read_text().splitlines()) == 2
    step_rows = [json.loads(line) for line in (out / 'loss_steps.jsonl').read_text().splitlines()]
    assert [row['update'] for row in step_rows] == [1, 2]
    assert step_rows == [json.loads(line) for line in (fs_runtime.ROOT / 'runs' / direct_id / 'loss_steps.jsonl').read_text().splitlines()]
    assert (out / 'best_validation.pt').exists()
    assert list(out.glob('epochs_before_resume_*.jsonl'))
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['extra_deployed_parameters'] == 17880
    print('PASS synthetic accumulation/cache/checkpoint/resume and evaluation-cadence equality:', run_id)
    print('No real-data accuracy evidence. Fixture artifacts retained under tmp/ and runs/.')


if __name__ == '__main__':
    main()
