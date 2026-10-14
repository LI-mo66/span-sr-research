"""EASA-inspired residual adapter, independently implemented for SPAN52 x4."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def adaptive_structure_pool(x, size):
    """Exact adaptive-max bins with deterministic CUDA backward when required.

    Native adaptive_max_pool2d lacks deterministic CUDA backward in PyTorch
    2.9.1. Divisible grids use a vectorized max; irregular grids use explicit
    floor/ceil bins and first-index max, preserving values and tie semantics.
    Native pooling remains the fast inference/non-deterministic path.
    """
    if not (x.is_cuda and torch.is_grad_enabled() and
            torch.are_deterministic_algorithms_enabled()):
        return F.adaptive_max_pool2d(x, size)
    h, w = x.shape[-2:]
    oh, ow = size
    if h % oh == 0 and w % ow == 0:
        cells = x.reshape(x.shape[0], x.shape[1], oh, h // oh, ow, w // ow)
        cells = cells.permute(0, 1, 2, 4, 3, 5).flatten(-2)
        return cells.max(dim=-1).values
    rows = []
    for i in range(oh):
        top, bottom = i * h // oh, ((i + 1) * h + oh - 1) // oh
        columns = []
        for j in range(ow):
            left, right = j * w // ow, ((j + 1) * w + ow - 1) // ow
            columns.append(x[:, :, top:bottom, left:right].flatten(2).max(-1).values)
        rows.append(torch.stack(columns, dim=-1))
    return torch.stack(rows, dim=-2)


class EASAResidual(nn.Module):
    """Return input plus a signed correction; never normalize the skip path.

    There is no LDE/PCFN or token matching. Pooling describes coarse structure;
    spatial variance provides per-image, per-channel global statistics.
    """

    def __init__(self, channels=52, down_scale=8, residual_init=0.01):
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError('channels must be a positive integer')
        if type(down_scale) is not int or down_scale < 1:
            raise ValueError('down_scale must be a positive integer')
        if not math.isfinite(residual_init) or residual_init <= 0:
            raise ValueError('residual_init must be finite and positive')
        self.channels = channels
        self.down_scale = down_scale
        self.input_projection = nn.Conv2d(channels, channels, 1)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels)
        self.modulation_projection = nn.Conv2d(channels, channels, 1)
        self.output_projection = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), residual_init))

    def correction(self, features):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError('Expected NCHW with configured channels')
        if not features.is_floating_point() or min(features.shape) < 1:
            raise ValueError('Expected nonempty floating-point features')
        h, w = features.shape[-2:]
        normalized = F.normalize(features, p=2, dim=1, eps=1e-12)
        x = self.input_projection(normalized)
        pooled = adaptive_structure_pool(
            x, (max(1, h // self.down_scale), max(1, w // self.down_scale)))
        structure = self.depthwise(pooled)
        # Explicit population variance, including the 1x1 spatial case.
        variance = torch.var(x, dim=(-2, -1), keepdim=True, correction=0)
        modulation = self.activation(self.modulation_projection(
            self.alpha * structure + self.beta * variance))
        modulation = F.interpolate(modulation, size=(h, w), mode='nearest')
        return self.output_projection(x * modulation)

    def forward(self, features):
        return features + self.gamma * self.correction(features)


def build_model(arch, cfg):
    if cfg.get('variants') or cfg.get('moe_fs', False):
        raise ValueError('EASA experiment cannot mix other candidates')
    model_cfg = cfg['model']
    if (model_cfg['feature_channels'] != 52 or model_cfg['upscale'] != 4 or
            model_cfg['num_in_ch'] != 3 or model_cfg['num_out_ch'] != 3):
        raise ValueError('This experiment is locked to RGB SPAN52 x4')
    enabled = cfg.get('easa', False)
    if type(enabled) is not bool:
        raise ValueError('easa must be a boolean')
    if not enabled:
        return arch.SPAN(**model_cfg)
    settings = cfg.get('easa_settings', {})
    if set(settings) - {'position', 'down_scale', 'residual_init'}:
        raise ValueError('Unknown EASA setting')
    position = settings.get('position', 'mid')
    if position not in ('mid', 'tail'):
        raise ValueError('EASA position must be mid or tail')

    class EASASPAN(arch.SPAN):
        def __init__(self):
            super().__init__(**model_cfg)
            self.easa_position = position
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(cfg['seed'] + 20001)
                self.easa_adapter = EASAResidual(
                    channels=52, down_scale=settings.get('down_scale', 8),
                    residual_init=settings.get('residual_init', 0.01))
            self.extra_parameters = sum(p.numel() for p in self.easa_adapter.parameters())
            if self.extra_parameters > cfg['max_extra_deployed_parameters']:
                raise ValueError('Deployment parameter budget exceeded')

        def forward(self, x):
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range
            feature = self.conv_1(x)
            b1, _, _ = self.block_1(feature)
            b2, _, _ = self.block_2(b1)
            b3, _, _ = self.block_3(b2)
            if self.easa_position == 'mid':
                b3 = self.easa_adapter(b3)
            b4, _, _ = self.block_4(b3)
            b5, _, _ = self.block_5(b4)
            b6, b6_first, _ = self.block_6(b5)
            b6 = self.conv_2(b6)
            # Preserve original expert order and block6's actual returned tensor.
            out = self.conv_cat(torch.cat([feature, b6, b1, b6_first], dim=1))
            if self.easa_position == 'tail':
                out = self.easa_adapter(out)
            return self.upsampler(out)

    return EASASPAN()
