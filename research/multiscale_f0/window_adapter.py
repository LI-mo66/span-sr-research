"""Partial-channel spatial interaction; independently implemented window attention.

Spatial RPE is inspired by ESC's WindowAttention, not its ConvAttn or ASSM.
No sampling, semantic sorting, state scan, BN, or image-shared statistics.
"""
import torch
from torch import nn
from torch.nn import functional as F


class WindowAttention(nn.Module):
    def __init__(self, channels=26, window=16, heads=2, chunk=128):
        super().__init__()
        if channels % heads or min(channels, window, heads, chunk) < 1:
            raise ValueError('Invalid attention dimensions')
        self.channels, self.window, self.heads, self.chunk = channels, window, heads, chunk
        self.qkv = nn.Linear(channels, 3 * channels)
        self.project = nn.Linear(channels, channels)
        self.relative_bias = nn.Parameter(torch.zeros(heads, (2 * window - 1) ** 2))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing='ij')).flatten(1)
        relative = coords[:, None, :] - coords[:, :, None]  # key minus query
        relative = relative + window - 1
        index = relative[0] * (2 * window - 1) + relative[1]
        self.register_buffer('relative_index', index, persistent=False)

    def partition(self, x):
        b, c, h, w = x.shape
        m = self.window
        return x.reshape(b, c, h // m, m, w // m, m).permute(0, 2, 4, 3, 5, 1).reshape(-1, m * m, c)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.channels or min(b, h, w) < 1:
            raise ValueError('Nonempty NCHW attention input required')
        m, heads = self.window, self.heads
        ph, pw = (-h) % m, (-w) % m
        padded = F.pad(x, (0, pw, 0, ph))
        hp, wp = h + ph, w + pw
        tokens = self.partition(padded)
        valid = F.pad(torch.ones((b, 1, h, w), dtype=torch.bool, device=x.device), (0, pw, 0, ph))
        valid = self.partition(valid).squeeze(-1)
        qkv = self.qkv(tokens).reshape(-1, m*m, 3, heads, c // heads).permute(2, 0, 3, 1, 4)
        bias = self.relative_bias[:, self.relative_index]
        outputs = []
        for start in range(0, tokens.shape[0], self.chunk):
            q, k, v = qkv[:, start:start+self.chunk].unbind(0)
            score = (q @ k.transpose(-2, -1)) * ((c // heads) ** -.5)
            score = score + bias[None]
            score = score.masked_fill(~valid[start:start+self.chunk, None, None, :], float('-inf'))
            outputs.append((score.softmax(-1) @ v).transpose(1, 2).reshape(-1, m*m, c))
        out = self.project(torch.cat(outputs, dim=0))
        out = out.reshape(b, hp//m, wp//m, m, m, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, hp, wp)
        return out[:, :, :h, :w].contiguous()


class LocalConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(26, 20, 1), nn.GELU(),
                                 nn.Conv2d(20, 20, 3, padding=1), nn.GELU(), nn.Conv2d(20, 26, 1))

    def forward(self, x):
        return self.net(x)


class PartialInteraction(nn.Module):
    def __init__(self, kind, initial_scale=.03, chunk=128):
        super().__init__()
        if kind not in ('A', 'C'):
            raise ValueError('Expected A=window or C=local convolution')
        self.kind = kind
        self.norm = nn.LayerNorm(26)
        self.core = WindowAttention(chunk=chunk) if kind == 'A' else LocalConv()
        self.scale = nn.Parameter(torch.full((26,), initial_scale))
        # Keep native fan-in initialization of the interaction core so Q/K and
        # the local bottleneck are not uniformly tiny. Only the writeback
        # projection is small; apply the same rule to A and C.
        for module in self.core.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        output = self.core.project if kind == 'A' else self.core.net[-1]
        nn.init.trunc_normal_(output.weight, std=.02)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 52 or min(x.shape) < 1:
            raise ValueError('Expected nonempty N52HW feature')
        active, bypass = x[:, :26], x[:, 26:]
        normalized = self.norm(active.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        correction = self.core(normalized) * self.scale[None, :, None, None]
        return torch.cat((active + correction, bypass), dim=1)
