"""Separate entry point using the shared trainer with an isolated model factory."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('easa_runtime', HERE / 'runtime.py')
easa_runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(easa_runtime)
from model import build_model


def validate_config(cfg):
    if cfg.get('variants') or cfg.get('moe_fs', False):
        raise ValueError('EASA cannot mix other candidates')
    if type(cfg.get('easa', False)) is not bool:
        raise ValueError('easa must be a boolean')
    settings = cfg.get('easa_settings', {})
    if set(settings) - {'position', 'down_scale', 'residual_init'}:
        raise ValueError('Unknown EASA setting')
    if cfg.get('easa'):
        if settings.get('position', 'mid') not in ('mid', 'tail'):
            raise ValueError('Invalid EASA position')
        # Primary experiment is frozen; new hyperparameters require a new version.
        if settings.get('down_scale', 8) != 8 or settings.get('residual_init', 0.01) != 0.01:
            raise ValueError('EASAResidual-v1 requires down_scale=8 and residual_init=0.01')
    if cfg.get('engineering_only'):
        return
    reference = json.loads((easa_runtime.RESEARCH / 'configs/baseline_x4.json').read_text())
    for key in ('model', 'total_updates', 'hr_patch', 'effective_batch',
                'adam_betas', 'adam_eps', 'milestones', 'gamma', 'crop_border',
                'test_y_channel', 'self_ensemble', 'selection_dataset'):
        if cfg.get(key) != reference[key]:
            raise ValueError(f'Protocol differs from locked baseline: {key}')
    protocol = cfg.get('training_protocol', 'original')
    expected_lr = {'original': 5e-4, 'multistep-lr2e4-v1': 2e-4}
    if protocol not in expected_lr or cfg.get('lr') != expected_lr[protocol]:
        raise ValueError('Learning rate does not match the declared training protocol')
    if cfg.get('record_updates') not in (1000, 5000):
        raise ValueError('Evaluation/recording interval must be 1000 or 5000 updates')
    if cfg['micro_batch'] not in (4, 8, 16, 32, 64):
        raise ValueError('Unsupported micro batch')


def validate_data(cfg):
    paths = json.loads(Path(cfg['paths_file']).read_text(encoding='utf-8-sig'))
    datasets = {'train': easa_runtime.trainer.pairs(paths['train'])}
    datasets.update({k: easa_runtime.trainer.pairs(v) for k, v in paths['validation'].items()})
    expected = {'train': 3450, 'DIV2K_valid': 100, 'Set5': 5, 'Set14': 14,
                'B100': 100, 'Urban100': 100, 'Manga109': 109}
    if not cfg.get('engineering_only'):
        for name, count in expected.items():
            if len(datasets.get(name, [])) != count:
                raise ValueError(f'{name}: expected {count} paired images')
    for name, pairs in datasets.items():
        print(f'Data audit: {name}, {len(pairs)} pairs', flush=True)
        for index, (hp, lp) in enumerate(pairs, 1):
            hr, lr = easa_runtime.trainer.aligned(hp, lp)
            if name == 'train' and min(lr.shape[:2]) < cfg['hr_patch'] // 4:
                raise ValueError(f'Training image too small: {lp}')
            if name != 'train' and min(hr.shape[:2]) - 2 * cfg['crop_border'] < 11:
                raise ValueError(f'Image too small for cropped SSIM: {hp}')
            if index % 250 == 0:
                print(f'Data audit: {name} decoded {index}/{len(pairs)} pairs', flush=True)
    print('All paired images decoded and x4 shapes checked.', flush=True)


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        easa_runtime.trainer.main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    args, _ = parser.parse_known_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf8'))
    validate_config(cfg)
    if cfg.get('variants'):
        raise ValueError('Mixed candidates are forbidden')
    validate_data(cfg)
    # Replace only the factory bound in this process, never source files.
    easa_runtime.trainer.build_model = build_model
    print('Shared trainer: hashing source images and checking train/evaluation overlap.', flush=True)
    easa_runtime.trainer.main()


if __name__ == '__main__':
    main()
