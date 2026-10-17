"""Locked B0 protocol plus one auxiliary output loss."""
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHT_SHA256 = '4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215'


def validate_config(cfg):
    reference = json.loads((HERE / 'configs/SPAN-KD-Output.json').read_text())
    engineering = cfg.get('engineering_only', False)
    if engineering and os.environ.get('OUTPUT_KD_ENGINEERING') != '1':
        raise ValueError('Synthetic fixture needs OUTPUT_KD_ENGINEERING=1')
    keys = ('model', 'seed', 'lr', 'adam_betas', 'adam_eps', 'milestones', 'gamma',
            'crop_border', 'test_y_channel', 'self_ensemble', 'selection_dataset', 'moe_fs')
    if not engineering:
        keys += ('total_updates', 'hr_patch', 'effective_batch', 'micro_batch',
                 'record_updates', 'image_backend', 'training_protocol')
    for key in keys:
        if cfg.get(key) != reference[key]:
            raise ValueError(f'Locked B0 protocol differs: {key}')
    if cfg.get('variants') or cfg.get('local_gate') or cfg.get('image_bypass'):
        raise ValueError('Only original SPAN52 is supported')
    if cfg.get('loader_workers', 0) != 0:
        raise ValueError('B0 uses the serial deterministic sampler')
    kd = cfg['distillation']
    expected = {'teacher', 'checkpoint', 'checkpoint_sha256', 'weight', 'teacher_chunk'}
    if set(kd) != expected or kd['teacher'] != 'SwinIR-M-DF2K-x4':
        raise ValueError('Unknown teacher specification')
    if kd['checkpoint_sha256'] != WEIGHT_SHA256 or not Path(kd['checkpoint']).is_file():
        raise ValueError('Missing or unpinned official teacher')
    if not math.isfinite(kd['weight']) or kd['weight'] != 0.1 or kd['teacher_chunk'] != 1:
        raise ValueError('v1 fixes output L1 weight=0.1 and teacher chunk=1')
    if not engineering and (cfg['micro_batch'] != 16 or cfg['effective_batch'] != 64):
        raise ValueError('B0 batch protocol mismatch')
