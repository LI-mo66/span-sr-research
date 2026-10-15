"""Separate entry point using the shared trainer with an isolated model factory."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('a6_runtime', HERE / 'runtime.py')
a6_runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a6_runtime)
from model import build_model


def train_main():
    # Extend only the stopping-cap CLI; retain the shared 1M-update schedule.
    original = argparse.ArgumentParser.add_argument
    def add_argument(parser, *args, **kwargs):
        if '--screen-updates' in args:
            kwargs['choices'] = [5000, 20000, 50000, 100000]
        return original(parser, *args, **kwargs)
    argparse.ArgumentParser.add_argument = add_argument
    try:
        a6_runtime.trainer.main()
    finally:
        argparse.ArgumentParser.add_argument = original


def validate_config(cfg):
    reference = json.loads((HERE / 'configs/A6-lr2e4.json').read_text())
    if cfg.get('a6') is not True or cfg.get('easa') or cfg.get('moe_fs') or cfg.get('variants'):
        raise ValueError('Only A6 training is permitted by this entry')
    keys = ('a6_settings', 'experiment_version', 'model', 'seed', 'total_updates',
            'hr_patch', 'effective_batch', 'micro_batch', 'lr', 'adam_betas', 'adam_eps',
            'milestones', 'gamma', 'crop_border', 'test_y_channel', 'self_ensemble',
            'selection_dataset', 'record_updates', 'image_backend', 'training_protocol',
            'deployed_baseline_parameters', 'max_extra_deployed_parameters')
    if cfg.get('engineering_only'):
        keys = ('a6_settings', 'experiment_version', 'model', 'lr')
    for key in keys:
        if cfg.get(key) != reference[key]:
            raise ValueError('Frozen A6/B0 protocol mismatch: ' + key)
    if cfg.get('engineering_only'):
        return
    if not cfg.get('reuse_audit'):
        raise ValueError('Run prepare_reuse.py against archived B0 before training')
    import subprocess
    import hashlib
    audit = json.loads(Path(cfg['reuse_audit']).read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a6_runtime.ROOT, text=True).strip()
    if (audit.get('status') != 'PASS' or audit['candidate_source'] != commit or
            audit['candidate_config'] != cfg):
        raise ValueError('A6 configuration/source not covered by successful B0 audit')
    common = ['basicsr', 'research/moe_fs/trainer.py', 'research/runtime.py',
              'research/image_io.py', 'research/parallel_loader.py']
    changed = subprocess.check_output(['git', 'diff', '--name-only', audit['baseline_source'],
                                      'HEAD', '--', *common], cwd=a6_runtime.ROOT, text=True).strip()
    if changed:
        raise ValueError('Common B0 source changed: ' + changed)
    torch = a6_runtime.trainer.torch
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    environment = dict(torch=torch.__version__, cuda=torch.version.cuda,
                       device=torch.cuda.get_device_name(), deterministic=True,
                       cudnn_allow_tf32=True, matmul_allow_tf32=False)
    if environment != audit['environment']:
        raise ValueError('Runtime environment differs from audited B0')
    import platform
    import numpy
    import cv2
    import scipy
    if (platform.python_version(), numpy.__version__, cv2.__version__, scipy.__version__) != (
            '3.11.8', '1.26.4', '4.10.0', '1.13.1'):
        raise ValueError('Python/NumPy/OpenCV/SciPy differ from archived B0')
    baseline = Path(audit['baseline_run'])
    if a6_runtime.trainer.digest(baseline / 'manifest.json') != audit['baseline_manifest_sha256']:
        raise ValueError('Archived B0 manifest changed')
    paths = json.loads(Path(cfg['paths_file']).read_text())
    datasets = dict(train=a6_runtime.trainer.pairs(paths['train']))
    datasets.update({name: a6_runtime.trainer.pairs(spec) for name, spec in paths['validation'].items()})
    data = {name: [dict(hr=str(h), lr=str(l), hr_sha256=a6_runtime.trainer.digest(h),
                        lr_sha256=a6_runtime.trainer.digest(l)) for h, l in pairs]
            for name, pairs in datasets.items()}
    signature = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    if signature != audit['data_fingerprint']:
        raise ValueError('Current ordered source images differ from B0 audit')


def validate_data(cfg):
    paths = json.loads(Path(cfg['paths_file']).read_text(encoding='utf-8-sig'))
    datasets = {'train': a6_runtime.trainer.pairs(paths['train'])}
    datasets.update({k: a6_runtime.trainer.pairs(v) for k, v in paths['validation'].items()})
    expected = {'train': 3450, 'DIV2K_valid': 100, 'Set5': 5, 'Set14': 14,
                'B100': 100, 'Urban100': 100, 'Manga109': 109}
    if not cfg.get('engineering_only'):
        for name, count in expected.items():
            if len(datasets.get(name, [])) != count:
                raise ValueError(f'{name}: expected {count} paired images')
    for name, pairs in datasets.items():
        print(f'Data audit: {name}, {len(pairs)} pairs', flush=True)
        for index, (hp, lp) in enumerate(pairs, 1):
            hr, lr = a6_runtime.trainer.aligned(hp, lp)
            if name == 'train' and min(lr.shape[:2]) < cfg['hr_patch'] // 4:
                raise ValueError(f'Training image too small: {lp}')
            if name != 'train' and min(hr.shape[:2]) - 2 * cfg['crop_border'] < 11:
                raise ValueError(f'Image too small for cropped SSIM: {hp}')
            if index % 250 == 0:
                print(f'Data audit: {name} decoded {index}/{len(pairs)} pairs', flush=True)
    print('All paired images decoded and x4 shapes checked.', flush=True)


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        train_main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    args, _ = parser.parse_known_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf8'))
    validate_config(cfg)
    cli = sys.argv[1:]
    if not cfg.get('engineering_only') and '--preflight' not in cli:
        cap_parser = argparse.ArgumentParser(add_help=False)
        cap_parser.add_argument('--screen', action='store_true')
        cap_parser.add_argument('--screen-updates', type=int)
        caps, _ = cap_parser.parse_known_args(cli)
        if not caps.screen or caps.screen_updates != 50000:
            raise ValueError('This experiment requires --screen --screen-updates 50000')
    validate_data(cfg)
    # Replace only the factory bound in this process, never source files.
    a6_runtime.trainer.build_model = build_model
    print('Shared trainer: hashing source images and checking train/evaluation overlap.', flush=True)
    train_main()


if __name__ == '__main__':
    main()
