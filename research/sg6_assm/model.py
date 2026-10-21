"""Wrap block3 output, inherit all fixed SG6 forward and RGB bypass logic."""
import sys
from pathlib import Path
import torch
from torch import nn
from adapter import ASSMAdapter

RESEARCH = Path(__file__).resolve().parents[1]
for folder in ('spatial_gate6', 'gated_bypass', 'linear_gate'):
    sys.path.insert(1, str(RESEARCH/folder))
from sg_model import build_model as build_sg6, validate_variant as validate_sg6


class BlockWithASSM(nn.Module):
    def __init__(self, block, adapter):
        super().__init__()
        self.block, self.adapter = block, adapter

    def forward(self, x):
        value, first, gate = self.block(x)
        return self.adapter(value), first, gate


def build_model(arch, cfg):
    validate_sg6(cfg)
    settings = cfg['assm_settings']
    expected = dict(position='after_block3',channels=52,d_state=8,expand=1.0,
                    num_tokens=64,inner_rank=32,initial_scale=.01,routing='deterministic-st',
                    prompt_alignment='sorted-with-values',stable_sort=True)
    if cfg.get('assm') is not True or cfg.get('experiment_version') != 'SG6-ASSM-mid3-aligned-ST-v1':
        raise ValueError('Only SG6+ASSM candidate supported')
    if {k:v for k,v in settings.items() if k!='backend'} != expected:
        raise ValueError('Unexpected ASSM structure/settings')
    if settings['backend']=='reference' and not cfg.get('engineering_only'):
        raise ValueError('Reference scan cannot train a research candidate')
    if settings['backend'] not in ('reference','cuda'):
        raise ValueError('Unknown scan backend')
    model = build_sg6(arch,cfg)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed']+60001)
        adapter = ASSMAdapter(settings['backend'])
    model.block_3 = BlockWithASSM(model.block_3, adapter)
    model.assm_extra_parameters = sum(p.numel() for p in adapter.parameters())
    if model.assm_extra_parameters != 12417:
        raise ValueError('Unexpected ASSM parameter count')
    model.extra_parameters += model.assm_extra_parameters
    return model
