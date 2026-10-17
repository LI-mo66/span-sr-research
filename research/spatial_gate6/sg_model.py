"""GB plus six zero-projected spatial corrections to signed local gates."""
import torch
from torch import nn
from gb_model import build_model as build_gb, validate_variant as validate_gb


def validate_variant(cfg):
    validate_gb(cfg)
    expected = dict(blocks=[1, 2, 3, 4, 5, 6], kernel=3, channels=52,
                    activation='silu', projection_init='zero')
    if cfg.get('spatial_gate') != expected:
        raise ValueError('Unknown spatial gate configuration')


class SpatialCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = nn.Conv2d(52, 52, 3, padding=1, groups=52, bias=True)
        self.act = nn.SiLU(inplace=False)
        self.project = nn.Conv2d(52, 52, 1, bias=True)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, feature):
        return self.project(self.act(self.depth(feature)))


def build_model(arch, cfg):
    validate_variant(cfg)
    model = build_gb(arch, cfg)

    class SpatialGateSPAB(arch.SPAB):
        def forward(self, x):
            # Retain GB's inplace activation and returned out1 semantics.
            out1 = self.c1_r(x)
            out1_act = self.act1(out1)
            out2 = self.c2_r(out1_act)
            out2_act = self.act1(out2)
            out3 = self.c3_r(out2_act)
            gate = self.local_gate(out3) + self.spatial_gate(out3)
            out = (out3 + x) * gate
            return out, out1, gate

    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed'] + 40001)
        for i in cfg['spatial_gate']['blocks']:
            block = getattr(model, f'block_{i}')
            block.__class__ = SpatialGateSPAB
            block.spatial_gate = SpatialCorrection()
    model.extra_parameters += 19656
    return model
