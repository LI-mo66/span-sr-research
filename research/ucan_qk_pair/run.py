"""Reuse unchanged SG6 data/optimizer/metric trainer; real runs are fixed 50K."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / 'spatial_gate6'))
from entry import existing
sys.path.insert(0, str(HERE))
from pair_model import VERSION, build_model, validate_variant


def validate_config(cfg):
    mode = validate_variant(cfg)
    if cfg.get('engineering_only'):
        if os.environ.get('UCAN_PAIR_ENGINEERING') != '1':
            raise ValueError('Synthetic fixtures require UCAN_PAIR_ENGINEERING=1')
        return mode
    reference = json.loads((HERE.parent / 'spatial_gate6/configs/LG6-SiLU-SG6.json').read_text())
    allowed = {'paths_file', 'experiment_version'}
    for key, value in reference.items():
        if key not in allowed and cfg.get(key) != value:
            raise ValueError('Fixed SG6 protocol mismatch: ' + key)
    if set(cfg) != set(reference) | {'ucan_pair'}:
        raise ValueError('Unknown config fields; protocol drift forbidden')
    if cfg['experiment_version'] != VERSION + '-' + mode:
        raise ValueError('Version must identify the specific arm')
    return mode


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--screen-updates', type=int)
    args, _ = parser.parse_known_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf8'))
    mode = validate_config(cfg)
    if not cfg.get('engineering_only'):
        import torch
        # Match the archived SG6 precision flags explicitly in this process.
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        prefix = {'off': 'FEAT-SG6-UCAN-OFF-', 'independent': 'FEAT-SG6-UCAN-IND-',
                  'reuse': 'FEAT-SG6-UCAN-REUSE-'}[mode]
        if not args.id.startswith(prefix):
            raise ValueError('Use the arm-specific experiment ID prefix: ' + prefix)
        if not args.screen or args.screen_updates != 50000:
            raise ValueError('v2 real runs require --screen --screen-updates 50000')
        if mode == 'reuse':
            raise ValueError('Reuse arm is engineering-only until the independent screen is reviewed')
    existing.validate_config(cfg)
    existing.validate_data(cfg)
    existing.fs_runtime.trainer.build_model = build_model
    original = argparse.ArgumentParser.add_argument
    def add_argument(parser, *flags, **kwargs):
        if '--screen-updates' in flags:
            kwargs['choices'] = (1000, 5000, 20000, 50000)
        return original(parser, *flags, **kwargs)
    argparse.ArgumentParser.add_argument = add_argument
    try:
        existing.fs_runtime.trainer.main()
    finally:
        argparse.ArgumentParser.add_argument = original


if __name__ == '__main__':
    main()
