"""Freeze candidate-only configs and paths. Never schedules baseline training."""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def immutable(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen file differs: {path}')
    else:
        path.write_text(json.dumps(value, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', required=True)
    p.add_argument('--data-root', type=Path, required=True)
    args = p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+', args.group):
        raise ValueError('Invalid group')
    root = args.data_root.resolve()
    def pair(hr, lr):
        return dict(hr=str(root / hr), lr=str(root / lr), lr_suffix='x4')
    paths = dict(train=[pair('DIV2K/DIV2K_train_HR', 'DIV2K/DIV2K_train_LR_bicubic/X4'),
                        pair('Flickr2K/Flickr2K_HR', 'Flickr2K/Flickr2K_LR_bicubic/X4')],
                 validation={'DIV2K_valid': pair('DIV2K/DIV2K_valid_HR', 'DIV2K/DIV2K_valid_LR_bicubic/X4')})
    for name in ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109'):
        folder = 'manga109' if name == 'Manga109' and not (root / 'benchmark/Manga109').is_dir() else name
        paths['validation'][name] = pair(f'benchmark/{folder}/HR', f'benchmark/{folder}/LR_bicubic/X4')
    target = ROOT / 'logs/plain_bypass' / args.group
    target.mkdir(parents=True, exist_ok=True)
    immutable(target / 'paths.json', paths)
    for variant in ('LG6-SiLU-PB',):
        cfg = json.loads((HERE / 'configs' / (variant + '.json')).read_text())
        cfg['paths_file'] = str(target / 'paths.json')
        immutable(target / (variant + '.json'), cfg)
        print(variant, 'config=', target / (variant + '.json'), 'run_id=', args.group + '-' + variant + '-s10')


if __name__ == '__main__':
    main()
