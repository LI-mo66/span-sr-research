"""SeeAll-derived SGLA; explicit Q/K neighborhood adaptation for SG6.

Reference: sxdyyds/SeeAll 34175a665a4ff6223ceab5bcd449b04458ed57c6,
basicsr/archs/seemore16_arch.py. See SOURCES.md for scope/differences.
"""
import torch
from torch import nn


class ChannelNorm(nn.Module):
    def __init__(self, channels=52):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).square().mean(1, keepdim=True)
        return (x - mean) / (variance + 1e-6).sqrt() * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SGLA(nn.Module):
    def __init__(self, context=False, channels=52, heads=1):
        super().__init__()
        if channels % heads or (channels // heads) % 4:
            raise ValueError('2D RoPE needs channels/heads divisible by four')
        self.channels, self.heads = channels, heads
        self.dim = channels // heads
        self.qkvo = nn.Conv2d(channels, channels * 4, 1)
        self.qk_context = nn.Conv2d(2*channels, 2*channels, 3, padding=1, groups=2*channels) if context else nn.Identity()
        self.lepe = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        angle = 1 / (10000 ** torch.linspace(0, 1, self.dim // 4))
        self.register_buffer('angle', angle[:, None].repeat(1, 2).flatten())
        if context:
            nn.init.dirac_(self.qk_context.weight, groups=2*channels)
            nn.init.zeros_(self.qk_context.bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def positions(self, h, w):
        ih = torch.arange(h, device=self.angle.device, dtype=self.angle.dtype)
        iw = torch.arange(w, device=self.angle.device, dtype=self.angle.dtype)
        ah, aw = ih[:, None] * self.angle, iw[:, None] * self.angle
        def join(a, b):
            return torch.cat((a[:, None].expand(h, w, -1), b[None].expand(h, w, -1)), -1).reshape(h*w, self.dim)
        return join(ah.sin(), aw.sin()), join(ah.cos(), aw.cos())

    @staticmethod
    def rotate(x, sin, cos):
        rotated = torch.stack((-x[..., 1::2], x[..., ::2]), -1).flatten(-2)
        return x * cos + rotated * sin

    def forward(self, x):
        b, c, h, w = x.shape
        n = h*w
        raw = self.qkvo(x)
        qk = self.qk_context(raw[:, :2*c])
        q, k = qk.reshape(b, 2, self.heads, self.dim, n).permute(1, 0, 2, 4, 3).unbind(0)
        v = raw[:, 2*c:3*c].reshape(b, self.heads, self.dim, n).transpose(-1, -2)
        q, k = torch.nn.functional.elu(q)+1, torch.nn.functional.elu(k)+1
        relevance = ((self.dim ** -.5) * q.mean(-2, keepdim=True) @ k.transpose(-1, -2)).softmax(-1).transpose(-1, -2)
        k = k * relevance * n
        sin, cos = self.positions(h, w)
        qr, kr = self.rotate(q, sin, cos), self.rotate(k, sin, cos)
        inverse = 1 / (q @ k.mean(-2, keepdim=True).transpose(-1, -2) + 1e-6)
        kv = (kr.transpose(-1, -2) * n**-.5) @ (v * n**-.5)
        aggregate = (qr @ kv * inverse).transpose(-1, -2).reshape(b, c, h, w)
        return self.proj((aggregate + self.lepe(raw[:, 2*c:3*c])) * raw[:, 3*c:])


class Adapter(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.norm = ChannelNorm()
        if mode in ('original', 'context'):
            self.core = SGLA(context=mode == 'context')
        elif mode == 'local':
            self.core = nn.Sequential(nn.Conv2d(52,124,1), nn.GELU(), nn.Conv2d(124,124,5,padding=2,groups=124), nn.GELU(), nn.Conv2d(124,52,1))
            nn.init.zeros_(self.core[-1].weight)
            nn.init.zeros_(self.core[-1].bias)
        else:
            raise ValueError('Unknown adapter mode')

    def forward(self, x):
        return x + self.core(self.norm(x))
