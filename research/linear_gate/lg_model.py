"""Six local affine gates; no edits to upstream SPAN or its reconstruction head."""
import torch
from torch import nn


def validate_variant(cfg):
    variant = cfg.get('local_gate', {})
    expected = {'activation', 'initial_bias', 'initial_weight', 'blocks'}
    if set(variant) != expected:
        raise ValueError('Incomplete or unknown local_gate configuration')
    if variant['activation'] not in ('silu', 'relu'):
        raise ValueError('Unsupported activation')
    if variant['initial_bias'] != 0.1 or variant['initial_weight'] != 'zero':
        raise ValueError('v1 requires zero weights and constant 0.1 bias')
    if variant['blocks'] != [1, 2, 3, 4, 5, 6]:
        raise ValueError('v1 changes exactly six blocks')
    if cfg.get('moe_fs') is not False or cfg.get('variants'):
        raise ValueError('Do not combine with FusionGate4 or other candidates')
    if cfg['model']['feature_channels'] != 52 or cfg['model']['upscale'] != 4:
        raise ValueError('Only SPAN52 x4 is supported')
    return variant


def build_model(arch, cfg):
    variant = validate_variant(cfg)
    model = arch.SPAN(**cfg['model'])

    class LinearGateSPAB(arch.SPAB):
        def forward(self, x):
            # Preserve inplace activation semantics of the returned out1.
            out1 = self.c1_r(x)
            out1_act = self.act1(out1)
            out2 = self.c2_r(out1_act)
            out2_act = self.act1(out2)
            out3 = self.c3_r(out2_act)
            gate = self.local_gate(out3)
            out = (out3 + x) * gate
            return out, out1, gate

    # Construct the entire baseline first, then isolate added-layer RNG usage.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed'] + 20001)
        for i in variant['blocks']:
            block = getattr(model, f'block_{i}')
            block.__class__ = LinearGateSPAB
            block.local_gate = nn.Conv2d(52, 52, 1, bias=True)
            nn.init.zeros_(block.local_gate.weight)
            nn.init.constant_(block.local_gate.bias, variant['initial_bias'])
            if variant['activation'] == 'relu':
                block.act1 = nn.ReLU(inplace=True)
    model.extra_parameters = sum(p.numel() for name, p in model.named_parameters()
                                 if '.local_gate.' in name)
    return model
