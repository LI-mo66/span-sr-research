"""Reuse the archived trainer without changing optimizer, data or metric code."""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / 'research/spatial_gate6'))
sys.path.insert(1, str(ROOT / 'research/linear_gate'))
sys.path.insert(1, str(ROOT / 'research/gated_bypass'))
sys.path.insert(1, str(ROOT / 'research/moe_fs'))
spec = importlib.util.spec_from_file_location('lg_existing_run', ROOT / 'research/moe_fs/run.py')
existing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(existing)
from fe_model import build_model, validate_variant


def main():
    import json
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--resume', action='store_true')
    args, _ = parser.parse_known_args()
    cfg = json.loads(Path(args.config).read_text())
    if os.environ.get('FE_REFERENCE_AUDIT'):
        import torch
        reference_env=json.loads(Path(os.environ['FE_REFERENCE_AUDIT']).read_text())['environment']
        torch.backends.cudnn.allow_tf32=reference_env['cudnn_allow_tf32']
        torch.backends.cuda.matmul.allow_tf32=reference_env['matmul_allow_tf32']
    validate_variant(cfg)
    existing.validate_config(cfg)
    if cfg.get('engineering_only'):
        if os.environ.get('LINEAR_GATE_ENGINEERING') != '1':
            raise ValueError('Engineering fixture requires explicit environment flag')
    elif cfg.get('image_backend') != 'npy_mmap' or cfg['micro_batch'] != 16 or cfg['seed'] != 10:
        raise ValueError('Fixed reference reuse requires seed10, micro16, npy_mmap')
    existing.validate_data(cfg)
    def factory(arch, config):
        model=build_model(arch, config)
        if os.environ.get('FE_OBSERVE') == '1':
            from observe import Observer
            initial_update=0
            if args.resume:
                import torch
                saved=torch.load(ROOT/'runs'/args.id/'latest.pt',map_location='cpu',weights_only=False)
                initial_update=saved['update']
                del saved
            # Keep observer outside module state; hooks do not change the optimizer.
            model._fe_observer=Observer(model, ROOT/'runs'/args.id/'feature-observations.jsonl',
                                        config['effective_batch']//config['micro_batch'],initial_update)
        return model
    existing.fs_runtime.trainer.build_model = factory
    # Extend only CLI stop-cap choices. The 1M scheduler horizon is unchanged.
    original = argparse.ArgumentParser.add_argument
    def add_argument(self, *flags, **kwargs):
        if '--screen-updates' in flags and kwargs.get('choices') is not None:
            kwargs['choices'] = tuple(sorted(set(kwargs['choices']) | {50000}))
        return original(self, *flags, **kwargs)
    argparse.ArgumentParser.add_argument = add_argument
    try:
        existing.fs_runtime.trainer.main()
    finally:
        argparse.ArgumentParser.add_argument = original


if __name__ == '__main__':
    main()

