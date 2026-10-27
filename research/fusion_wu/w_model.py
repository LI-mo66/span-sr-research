"""Wrap only block_3 main output; inherit fixed SG6 forward without copying it."""
import sys
from pathlib import Path
import torch
from torch import nn
from adapter import PartialInteraction

RESEARCH = Path(__file__).resolve().parents[1]
for folder in ('spatial_gate6', 'gated_bypass', 'linear_gate'):
    sys.path.insert(1, str(RESEARCH / folder))
from sg_model import build_model as build_sg6, validate_variant as validate_sg6

COUNTS = {'A': 4808, 'C': 4784}


def validate_variant(cfg):
    validate_sg6(cfg)
    kind = cfg.get('mid_interaction')
    if kind not in COUNTS or cfg.get('experiment_version') != 'SG6-Mid-Partial-' + kind + '-v2':
        raise ValueError('Unknown mid interaction variant')
    expected = dict(position='block3-output', channels=26, window=16, heads=2,
                    window_chunk=128, initial_scale=.03, normalization='branch-channel-LN',
                    initialization='native-core-small-output',
                    amplitude_restore=False, padding='zero-with-key-mask', shifted=False)
    if cfg.get('mid_settings') != expected:
        raise ValueError('Unregistered structure, execution, or initialization')
    if cfg.get('deployed_reference_parameters') != 535772:
        raise ValueError('Direct comparator must be fixed SG6')
    if cfg.get('max_extra_deployed_parameters') != 37564 + COUNTS[kind]:
        raise ValueError('Unexpected cumulative parameter budget')
    if cfg.get('max_new_deployed_parameters') != COUNTS[kind]:
        raise ValueError('Unexpected candidate parameter budget')
    if any(cfg.get(k) for k in ('assm','atd_attention','cata_tab','cata_ffn','easa','variants')):
        raise ValueError('No other research modules permitted')


class MidBlock(nn.Module):
    def __init__(self, original, interaction):
        super().__init__()
        self.original = original
        self.interaction = interaction

    def forward(self, x):
        out, first, gate = self.original(x)
        return self.interaction(out), first, gate


def build_model(arch, cfg):
    validate_variant(cfg)
    model = build_sg6(arch, cfg)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed'] + 70001)
        interaction = PartialInteraction(cfg['mid_interaction'], cfg['mid_settings']['initial_scale'],
                                         cfg['mid_settings']['window_chunk'])
    count = sum(p.numel() for p in interaction.parameters())
    if count != COUNTS[cfg['mid_interaction']]:
        raise ValueError('Unexpected module parameter count')
    model.block_3 = MidBlock(model.block_3, interaction)
    model.extra_parameters += count
    model.mid_extra_parameters = count
    return model
