"""A6: six independent partial-channel grid-attention residual adapters.

Mechanism: HAESR 6943f17f544ba9adbd9375db37122c6b9d2eaa85,
models/team06_HAESR.py, GridCenterUnfold / SelfAttentionA_GridSample.
Independent implementation; sampling and bias shapes follow that source.
Insertion sites, C26 and residual initialization are unvalidated adaptations.
"""
import torch
from torch import nn
from torch.nn import functional as F


class GridAttention(nn.Module):
    def __init__(self, channels=26, window=8):
        super().__init__()
        self.channels, self.window = channels, window
        self.q = nn.Linear(channels, channels, bias=False)
        self.kv = nn.Linear(channels, 2 * channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.relative_bias = nn.Embedding((2 * window - 1) ** 2, 1)
        self.extra_bias = nn.Parameter(torch.zeros(1, window**2, (window - 1)**2))
        nn.init.trunc_normal_(self.extra_bias, std=0.02)
        yy, xx = torch.meshgrid(torch.arange(window), torch.arange(window), indexing='ij')
        coords = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
        delta = coords[:, None] - coords[None, :] + window - 1
        self.register_buffer('relative_index', delta[..., 0] * (2 * window - 1) + delta[..., 1],
                             persistent=False)

    def tokens(self, x):
        b, c, H, W = x.shape
        p = self.window
        h, w = H // p, W // p
        return x.reshape(b, c, p, h, p, w).permute(0, 3, 5, 2, 4, 1).reshape(b*h*w, p, p, c)

    @staticmethod
    def reflected_indices(length, offset, device):
        # Same values as right/bottom reflect-padding and an offset crop.
        # index_select avoids reflection_pad2d's unsupported deterministic CUDA
        # backward on some PyTorch versions. Padded dimensions are >= 8 here.
        index = torch.arange(length, device=device) + offset
        period = 2 * (length - 1)
        index = index.remainder(period)
        return torch.minimum(index, period - index)

    def forward(self, x):
        b, c, H, W = x.shape
        p = self.window
        if c != self.channels or H % p or W % p:
            raise ValueError('GridAttention requires configured channels and padded H/W')
        h, w = H // p, W // p
        base = self.tokens(x).flatten(1, 2)
        yi = self.reflected_indices(H, h // 2, x.device)
        xi = self.reflected_indices(W, w // 2, x.device)
        offset = x.index_select(2, yi).index_select(3, xi)
        centers = self.tokens(offset)[:, :p-1, :p-1].reshape(b*h*w, (p-1)**2, c)
        kv_tokens = torch.cat((base, centers), dim=1)
        q = self.q(base) * (c ** -0.5)
        k, v = self.kv(kv_tokens).chunk(2, dim=-1)
        logits = q @ k.transpose(-2, -1)
        base_bias = self.relative_bias(self.relative_index).squeeze(-1).unsqueeze(0)
        logits = logits + torch.cat((base_bias, self.extra_bias), dim=-1)
        # Fresh QK for every adapter and image. No persistent attention cache.
        out = self.proj(logits.softmax(dim=-1) @ v)
        return out.reshape(b, h, w, p, p, c).permute(0, 5, 3, 1, 4, 2).reshape(b, c, H, W)


class GridResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(26, eps=1e-6)
        self.attention = GridAttention()
        self.output = nn.Conv2d(26, 52, 1)
        self.gamma = nn.Parameter(torch.full((1, 52, 1, 1), 0.01))

    def correction(self, features):
        if features.ndim != 4 or features.shape[1] != 52 or min(features.shape) < 1:
            raise ValueError('Expected nonempty Bx52xHxW features')
        h, w = features.shape[-2:]
        x = self.norm(features[:, :26].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.pad(x, (0, (-w) % 8, 0, (-h) % 8))
        x = self.attention(x)[:, :, :h, :w]
        return self.output(x)

    def forward(self, features):
        return features + self.gamma * self.correction(features)


def build_model(arch, cfg):
    if cfg.get('a6') is not True or cfg.get('easa') or cfg.get('moe_fs') or cfg.get('variants'):
        raise ValueError('This entry builds A6 only; baseline/EASA/mixed training forbidden')
    expected = dict(active_channels=26, window_size=8, sample_mode='center',
                    residual_init=0.01, share_attention=False)
    if cfg.get('a6_settings') != expected:
        raise ValueError('A6-Grid26-fresh-v1 settings are frozen')
    if cfg['model'] != dict(num_in_ch=3, num_out_ch=3, feature_channels=52, upscale=4, img_range=255.0):
        raise ValueError('A6 requires original RGB SPAN52 x4 configuration')

    class A6SPAN(arch.SPAN):
        def __init__(self):
            super().__init__(**cfg['model'])
            # Preserve baseline weights AND all post-construction RNG streams.
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(cfg['seed'] + 30001)
                self.grid_adapters = nn.ModuleList([GridResidual() for _ in range(6)])
            self.extra_parameters = sum(p.numel() for p in self.grid_adapters.parameters())
            if self.extra_parameters != 45438 or self.extra_parameters > cfg['max_extra_deployed_parameters']:
                raise ValueError('A6 parameter count/budget mismatch')

        def forward(self, x):
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range
            shallow = self.conv_1(x)
            features, b1, b6_first = shallow, None, None
            for i, adapter in enumerate(self.grid_adapters, 1):
                features, first, _ = getattr(self, f'block_{i}')(features)
                features = adapter(features)
                if i == 1:
                    b1 = features
                if i == 6:
                    b6_first = first
            deep = self.conv_2(features)
            # Preserve the four original sources and order, including SPAB6's
            # actual returned first feature (upstream in-place SiLU semantics).
            fused = self.conv_cat(torch.cat((shallow, deep, b1, b6_first), dim=1))
            return self.upsampler(fused)

    return A6SPAN()
