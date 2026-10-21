"""Attach one explicit tuple-preserving adapter; inherit the actual SG6 forward."""
from pathlib import Path
import sys
import torch
from torch import nn
from attention import Adapter

RESEARCH = Path(__file__).resolve().parents[1]
for directory in ('spatial_gate6', 'gated_bypass', 'linear_gate'):
    sys.path.insert(1, str(RESEARCH / directory))
from sg_model import build_model as build_sg6, validate_variant as validate_sg6

SETTINGS = dict(mode='context', position=4, heads=1, version='SG6-SGLA-QKDW-v1')

def validate_variant(cfg):
    validate_sg6(cfg)
    if cfg.get('max_total_deployed_parameters',1000000)!=1000000:
        raise ValueError('User deployment budget is fixed at 1000K')
    settings = cfg.get('sgla', dict(SETTINGS, mode='off'))
    if set(settings) != set(SETTINGS) or settings['mode'] not in ('off','original','context','local'):
        raise ValueError('Unknown SGLA configuration')
    if any(settings[k] != SETTINGS[k] for k in ('position','heads','version')):
        raise ValueError('Position/head/version fixed for this experiment')
    for key in ('atd_attention','easa','cata_tab','cata_ffn','variants'):
        if cfg.get(key): raise ValueError('Mixed candidates are forbidden')
    return settings

class AdaptedSPAB(nn.Module):
    def __init__(self, block, adapter):
        super().__init__()
        self.block, self.adapter = block, adapter

    def forward(self, x):
        output, first, gate = self.block(x)
        return self.adapter(output), first, gate

def build_model(arch, cfg):
    settings = validate_variant(cfg)
    model = build_sg6(arch, cfg)
    if settings['mode'] == 'off': return model
    # Preserve SG6 initial tensors AND the random stream used by the trainer.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed']+60001)
        adapter = Adapter(settings['mode'])
    # Keep shared weights identical for original/context ablation as well.
    if settings['mode'] == 'context':
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(cfg['seed']+60001)
            reference = Adapter('original')
        state = adapter.state_dict()
        state.update(reference.state_dict())
        adapter.load_state_dict(state, strict=True)
    model.block_4 = AdaptedSPAB(model.block_4, adapter)
    model.sgla_extra_parameters = sum(p.numel() for p in adapter.parameters())
    if 535772+model.sgla_extra_parameters>cfg.get('max_total_deployed_parameters',1000000):
        raise ValueError('Total deployed parameters exceed 1000K')
    model.extra_parameters += model.sgla_extra_parameters
    return model
