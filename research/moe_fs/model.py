"""Independent MoE-FS-inspired adapter; upstream SPAN remains untouched."""
import torch
from torch import nn


class FusionGate4(nn.Module):
    """Dense per-pixel/channel expert selection, followed by original concat."""
    def __init__(self, channels=52):
        super().__init__()
        self.channels = channels
        self.logits = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(4)])
        for layer in self.logits:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        if len(features) != 4:
            raise ValueError('FusionGate4 requires exactly four experts')
        shape = features[0].shape
        if len(shape) != 4 or shape[1] != self.channels:
            raise ValueError('Expected NCHW features with configured channels')
        if any(x.shape != shape or x.device != features[0].device or
               x.dtype != features[0].dtype for x in features):
            raise ValueError('Experts must have identical shape, dtype and device')
        weights = torch.stack([layer(x) for layer, x in zip(self.logits, features)], 0)
        weights = weights.softmax(dim=0) * 4.0
        return [x * weights[i] for i, x in enumerate(features)]


def build_model(arch, cfg):
    if cfg.get('variants'):
        raise ValueError('This experiment admits only baseline or FusionGate4')
    if cfg['model']['feature_channels'] != 52 or cfg['model']['upscale'] != 4:
        raise ValueError('This experiment is locked to SPAN52 x4')
    if not isinstance(cfg.get('moe_fs', False), bool):
        raise ValueError('moe_fs must be a boolean')
    if not cfg.get('moe_fs', False):
        return arch.SPAN(**cfg['model'])

    class FusionSPAN(arch.SPAN):
        def __init__(self):
            super().__init__(**cfg['model'])
            # Initialize baseline first. Adapter construction must not consume
            # the subsequent main CPU/CUDA random streams.
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(cfg['seed'] + 10001)
                self.fusion_gate = FusionGate4(52)
            self.extra_parameters = sum(p.numel() for p in self.fusion_gate.parameters())
            if self.extra_parameters > cfg['max_extra_deployed_parameters']:
                raise ValueError('Deployment parameter budget exceeded')

        def forward(self, x):
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range
            feature = self.conv_1(x)
            b1, _, _ = self.block_1(feature)
            b2, _, _ = self.block_2(b1)
            b3, _, _ = self.block_3(b2)
            b4, _, _ = self.block_4(b3)
            b5, _, _ = self.block_5(b4)
            b6, b6_first, _ = self.block_6(b5)
            b6 = self.conv_2(b6)
            gated = self.fusion_gate([feature, b6, b1, b6_first])
            out = self.conv_cat(torch.cat(gated, dim=1))
            return self.upsampler(out)

    return FusionSPAN()
