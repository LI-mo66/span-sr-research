"""Nonlinear multiscale RGB correction on top of the unchanged LG6-SiLU backbone."""
import torch
from torch import nn
from lg_model import build_model as build_a, validate_variant as validate_a


class MultiscaleResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=True)
        self.local = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=True)
        self.wide = nn.Conv2d(16, 16, 5, padding=2, groups=16, bias=True)
        self.project = nn.Conv2d(32, 48, 1, bias=True)
        self.activation = nn.SiLU(inplace=False)

    def forward(self, image):
        features = self.activation(self.stem(image))
        local = self.activation(self.local(features))
        wide = self.activation(self.wide(features))
        return self.project(torch.cat((local, wide), dim=1))


def validate_variant(cfg):
    validate_a(cfg)
    if cfg['local_gate']['activation'] != 'silu':
        raise ValueError('Only archived SiLU candidate A is supported')
    expected = dict(hidden_channels=16, spatial_kernels=[3,5], activation='silu', gate_kernel=3,
                    gate_channels=1, gate_gain=2.0, projection_init='zero', gate_init='zero')
    if cfg.get('image_bypass') != expected:
        raise ValueError('Unknown gated image bypass version')


def build_model(arch, cfg):
    validate_variant(cfg)
    model = build_a(arch, cfg)

    class MultiscaleImageSPAN(arch.SPAN):
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

    model.__class__ = MultiscaleImageSPAN
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed'] + 30001)
        model.image_residual = MultiscaleResidual()
        model.image_gate = nn.Conv2d(3, 1, 3, padding=1, bias=True)
        for module in (model.image_residual.project, model.image_gate):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
    model.extra_parameters += 2636
    return model
