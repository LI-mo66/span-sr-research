"""Residual ConvFFN only; original SPAN routes remain intact."""
import torch

import importlib.util
from pathlib import Path
from torch import nn

_spec = importlib.util.spec_from_file_location('ffn_initialization_reference', Path(__file__).resolve().parents[1] / 'cata_tab/tab.py')
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)
ContentTAB = _reference.ContentTAB


class ResidualConvFFN(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 52 or min(image.shape) < 1:
            raise ValueError('Expected nonempty N52HW features')
        tokens = image.flatten(2).transpose(1, 2).contiguous()
        output = tokens + self.mlp(tokens, image.shape[2:])
        return output.transpose(1, 2).reshape_as(image)



def build_model(arch, cfg):
    if cfg.get('variants') or cfg.get('moe_fs') or cfg.get('easa') or cfg.get('cata_tab'):
        raise ValueError('ConvFFN experiment cannot mix other candidates')
    if (cfg['model']['feature_channels'] != 52 or cfg['model']['upscale'] != 4 or
            cfg['model']['num_in_ch'] != 3 or cfg['model']['num_out_ch'] != 3):
        raise ValueError('Experiment requires RGB SPAN52 x4')
    enabled = cfg.get('cata_ffn', False)
    if type(enabled) is not bool:
        raise ValueError('cata_ffn must be boolean')
    if not enabled:
        return arch.SPAN(**cfg['model'])
    settings = cfg.get('ffn_settings', {})
    if set(settings) - {'position'}:
        raise ValueError('Structural settings are fixed for ConvFFN-v1')
    position = settings.get('position')
    if position != 'tail':
        raise ValueError('ConvFFN control requires tail position')

    class ContentSPAN(arch.SPAN):
        def __init__(self):
            super().__init__(**cfg['model'])
            self.content_position = position
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(cfg['seed'] + 30001)
                reference = ContentTAB(chunk_groups=128)
                self.content_tab = ResidualConvFFN(reference.mlp)
            self.extra_parameters = sum(p.numel() for p in self.content_tab.parameters())
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
            fused = self.conv_cat(torch.cat([feature, b6, b1, b6_first], 1))
            fused = self.content_tab(fused)  # Exactly one FFN residual.
            return self.upsampler(fused)

    return ContentSPAN()
