"""Spatially gated RGB correction on top of the unchanged LG6-SiLU backbone."""
import torch
from torch import nn
from lg_model import build_model as build_a, validate_variant as validate_a


def validate_variant(cfg):
    validate_a(cfg)
    if cfg['local_gate']['activation'] != 'silu':
        raise ValueError('Only archived SiLU candidate A is supported')
    expected = dict(residual_kernel=3, gate_kernel=3, gate_channels=1,
                    gate_gain=2.0, initial_weight='zero', initial_bias=0.0)
    if cfg.get('image_bypass') != expected:
        raise ValueError('Unknown gated image bypass version')


def build_model(arch, cfg):
    validate_variant(cfg)
    model = build_a(arch, cfg)

    class GatedImageSPAN(arch.SPAN):
        def forward(self, image):
            # RGB [0,1] bypass; original main-branch preprocessing unchanged.
            residual = self.image_residual(image)
            gate = 2.0 * torch.sigmoid(self.image_gate(image))
            self.mean = self.mean.type_as(image)
            x = (image-self.mean) * self.img_range
            feature = self.conv_1(x)
            b1, _, _ = self.block_1(feature)
            b2, _, _ = self.block_2(b1)
            b3, _, _ = self.block_3(b2)
            b4, _, _ = self.block_4(b3)
            b5, _, _ = self.block_5(b4)
            b6, first, _ = self.block_6(b5)
            b6 = self.conv_2(b6)
            fused = self.conv_cat(torch.cat((feature, b6, b1, first), dim=1))
            pixels = self.upsampler[0](fused)
            # gate N,1,H,W broadcasts only across the 48 RGB/phase channels.
            return self.upsampler[1](pixels + gate * residual)

    model.__class__ = GatedImageSPAN
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed'] + 30001)
        model.image_residual = nn.Conv2d(3, 48, 3, padding=1, bias=True)
        model.image_gate = nn.Conv2d(3, 1, 3, padding=1, bias=True)
        for module in (model.image_residual, model.image_gate):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
    model.extra_parameters += 1372
    return model
