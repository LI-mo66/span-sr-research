"""Single B3 evolution experiment; no changes to the archived SG6 builder."""
import torch
from torch import nn
from sg_model import build_model as build_sg6, validate_variant as validate_sg6


def validate_variant(cfg):
    validate_sg6(cfg)
    spec = cfg.get('feature_evolution')
    if spec not in ({'block': 3, 'width': 13, 'sign': 1},
                    {'block': 3, 'width': 13, 'sign': -1}):
        raise ValueError('Only fixed B3/width13/add or difference is supported')
    return spec


class StageFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(52, 13, 1) for _ in range(3)])
        self.activation = nn.SiLU(inplace=False)
        self.output = nn.Conv2d(39, 52, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, a, b, c):
        if a.shape != b.shape or b.shape != c.shape or c.shape[1] != 52:
            raise ValueError('Stage fusion requires aligned NCHW/52 feature maps')
        q = torch.cat([p(v) for p, v in zip(self.projections, (a, b, c))], dim=1)
        return self.output(self.activation(q))


def build_model(arch, cfg):
    spec = validate_variant(cfg)
    model = build_sg6(arch, cfg)

    class EvolutionSPAB(arch.SPAB):
        def forward(self, x):
            # Original out1 return denotes the activated first stage.
            a = self.act1(self.c1_r(x))
            b = self.act1(self.c2_r(a) + self.evolution_sign * a)
            c = self.c3_r(b)
            feature = c + self.stage_fusion(a, b, c)
            gate = self.local_gate(feature) + self.spatial_gate(feature)
            return (x + feature) * gate, a, gate

    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed'] + 50001)
        block = model.block_3
        block.__class__ = EvolutionSPAB
        # Sign is frozen by the run configuration, not an optimizer parameter.
        block.evolution_sign = spec['sign']
        block.stage_fusion = StageFusion()
    model.extra_parameters += 4147
    return model
