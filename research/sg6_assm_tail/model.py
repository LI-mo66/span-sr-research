"""Fixed SG6, wrapped post-fusion ASSM with locally amplitude-matched writeback."""
import sys
from pathlib import Path
import torch
from torch import nn
from adapter import ASSMAdapter

RESEARCH=Path(__file__).resolve().parents[1]
for folder in ('spatial_gate6','gated_bypass','linear_gate'):
    sys.path.insert(1,str(RESEARCH/folder))
from sg_model import build_model as build_sg6, validate_variant as validate_sg6


class FusionWithASSM(nn.Module):
    def __init__(self,fusion,adapter):
        super().__init__()
        self.fusion,self.adapter=fusion,adapter

    def forward(self,x):
        return self.adapter(self.fusion(x))


def build_model(arch,cfg):
    validate_sg6(cfg)
    expected=dict(position='post_conv_cat',channels=52,d_state=8,expand=1.0,
                  num_tokens=64,inner_rank=32,initial_scale=.01,routing='deterministic-st',
                  prompt_alignment='sorted-with-values',stable_sort=True,
                  prompt_init='variance-preserving-rank-token',writeback='per-pixel-detached-rms')
    settings=cfg['assm_settings']
    if cfg.get('assm') is not True or cfg.get('experiment_version')!='SG6-ASSM-tail-RMS-v2':
        raise ValueError('Only this SG6+ASSM variant is supported')
    if {k:v for k,v in settings.items() if k!='backend'}!=expected:
        raise ValueError('Unknown structure or initialization')
    if settings['backend'] not in ('cuda','reference'):
        raise ValueError('Unknown backend')
    if settings['backend']=='reference' and not cfg.get('engineering_only'):
        raise ValueError('Reference scan is diagnostic only')
    model=build_sg6(arch,cfg)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed']+60001)
        adapter=ASSMAdapter(settings['backend'])
    model.conv_cat=FusionWithASSM(model.conv_cat,adapter)
    model.assm_extra_parameters=sum(p.numel() for p in adapter.parameters())
    if model.assm_extra_parameters!=12417:
        raise ValueError('Unexpected parameter count')
    model.extra_parameters+=model.assm_extra_parameters
    return model
