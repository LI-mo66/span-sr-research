"""Freeze the seed-10, from-scratch SPAN52 x4 full training protocol."""
import argparse
import json
from pathlib import Path
from runtime import ROOT


def prepare(data, run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ValueError('Invalid run ID')
    data = Path(data).resolve()
    job = ROOT / 'logs/server_jobs' / run_id
    def pair(base, hr, lr):
        return dict(hr=str(base/hr), lr=str(base/lr), lr_suffix='x4')
    paths = dict(train=[
        pair(data/'DIV2K', 'DIV2K_train_HR', 'DIV2K_train_LR_bicubic/X4'),
        pair(data/'downloads/Flickr2K', 'Flickr2K_HR', 'Flickr2K_LR_bicubic/X4')
    ], validation={'DIV2K_valid': pair(data/'DIV2K', 'DIV2K_valid_HR', 'DIV2K_valid_LR_bicubic/X4')})
    for name, folder in [('Set5','Set5'), ('Set14','Set14'), ('B100','B100'),
                         ('Urban100','Urban100'), ('Manga109','manga109')]:
        paths['validation'][name] = pair(data/'benchmark'/folder, 'HR', 'LR_bicubic/X4')
    for item in paths['train'] + list(paths['validation'].values()):
        for side in ['hr', 'lr']:
            if not Path(item[side]).is_dir():
                raise FileNotFoundError(item[side])
    cfg = json.loads((ROOT/'research/configs/baseline_x4.json').read_text())
    cfg.update(seed=10, variants=[], total_updates=1000000, effective_batch=64,
               micro_batch=16, record_updates=5000, self_ensemble=False,
               image_backend='png', loader_workers=8, loader_prefetch=2,
               cpu_threads=4, log_updates=10, loader_sampling='indexed-v1',
               paths_file=str(job/'paths.local.json'))
    job.mkdir(parents=True, exist_ok=True)
    for path, value in [(job/'paths.local.json', paths), (job/'baseline.local.json', cfg)]:
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError(f'Frozen configuration differs: {path}')
        else:
            path.write_text(json.dumps(value, indent=2), encoding='utf8')
    return job


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/home/featurize/datasets')
    parser.add_argument('--id', default='SPAN52-X4-FULL-S10')
    args = parser.parse_args()
    print('Prepared:', prepare(args.data_root, args.id))
