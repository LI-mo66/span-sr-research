"""Wrap fixed SG6's fusion only; inherit its entire forward and RGB bypass."""
from pathlib import Path
import sys
import torch
from torch import nn
from adapter import AttentionAdapter

RESEARCH=Path(__file__).resolve().parents[1]
for folder in ('spatial_gate6','gated_bypass','linear_gate'):
    sys.path.insert(1,str(RESEARCH/folder))
from sg_model import build_model as build_sg6, validate_variant as validate_sg6


def validate_variant(cfg):
    validate_sg6(cfg)
    if cfg.get('atd_attention') is not True or cfg.get('variants') or cfg.get('easa') or cfg.get('cata_tab') or cfg.get('cata_ffn'):
        raise ValueError('Requires fixed SG6 plus ATD only')
    if cfg.get('experiment_version')!='SG6-ATD-Attention-tail-v1':
        raise ValueError('Unknown SG6 ATD version')
    if cfg.get('deployed_reference_parameters')!=535772:
        raise ValueError('Direct reference must be SG6')


class FusionWithATD(nn.Module):
    def __init__(self,fusion,attention):
        super().__init__()
        self.fusion=fusion
        self.attention=attention

    def forward(self,x):
        return self.attention(self.fusion(x))


def build_model(arch,cfg):
    if not cfg.get('atd_attention',False):
        return build_sg6(arch,cfg)
    validate_variant(cfg)
    settings=cfg.get('atd_settings',{})
    if set(settings)!={'position','query_chunk','group_chunk'} or settings['position']!='tail':
        raise ValueError('Only explicit post-conv_cat attention is supported')
    model=build_sg6(arch,cfg)
    class SG6Attention(type(model)):
        @property
        def content_tab(self):
            # A property, not a second module registration/state_dict alias.
            return self.conv_cat.attention
    model.__class__=SG6Attention
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed']+50001)
        attention=AttentionAdapter(query_chunk=settings['query_chunk'],group_chunk=settings['group_chunk'])
    model.conv_cat=FusionWithATD(model.conv_cat,attention)
    model.atd_extra_parameters=sum(p.numel() for p in attention.parameters())
    if model.atd_extra_parameters!=21813 or model.atd_extra_parameters>cfg['max_new_deployed_parameters']:
        raise ValueError('Unexpected new parameter budget')
    model.extra_parameters+=model.atd_extra_parameters
    if model.extra_parameters!=59377:raise ValueError('Unexpected cumulative parameter count')
    return model
