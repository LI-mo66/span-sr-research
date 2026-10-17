"""CATANet TAB adaptation (Apache-2.0; provenance in vendor/NOTICE.md).

Keep content grouping, adjacent subgroups, global centers and both residuals.
Changes: torch-only layout operations, sufficient reflected-tail padding, 4D
chunked SDPA, inverse gather and explicit, autograd-safe buffer updates.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def reflected_tail(x, count):
    """Original flipped tail when enough tokens; repeat for tiny sequences."""
    if count == 0:
        return x[:, :0]
    return x.flip(1).repeat(1, math.ceil(count / x.shape[1]), 1)[:, :count]


def center_iter(x, means):
    buckets = (x @ means.T).argmax(-1)
    bins = torch.zeros(x.shape[0], means.shape[0], dtype=torch.long, device=x.device)
    bins.scatter_add_(1, buckets, torch.ones_like(buckets))
    totals = x.new_zeros(x.shape[0], means.shape[0], x.shape[-1])
    totals.scatter_add_(1, buckets.unsqueeze(-1).expand_as(x), x)
    updated = F.normalize(totals.sum(0, keepdim=True), dim=-1)
    return torch.where((bins.sum(0, keepdim=True) == 0).unsqueeze(-1),
                       means, updated).squeeze(0)


class DepthwiseFFN(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden), nn.GELU())

    def forward(self, x, x_size):
        image = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], *x_size)
        return self.depthwise_conv(image).flatten(2).transpose(1, 2).contiguous()


class ConvFFN(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.dwconv = DepthwiseFFN(hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x, x_size):
        x = self.act(self.fc1(x))
        return self.fc2(x + self.dwconv(x, x_size))


class PreNorm(nn.Module):
    def __init__(self, channels, fn):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.fn = fn

    def forward(self, x, x_size):
        return self.fn(self.norm(x), x_size)


class IRCA(nn.Module):
    def __init__(self, channels, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(channels, qk_dim, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)

    def forward(self, x, means):
        centers = center_iter(F.normalize(x, dim=-1), F.normalize(means, dim=-1)) if self.training else means
        k, v = self.to_k(centers), self.to_v(centers)
        k = k.reshape(k.shape[0], self.heads, -1).permute(1, 0, 2)
        v = v.reshape(v.shape[0], self.heads, -1).permute(1, 0, 2)
        return k, v, centers.detach()


class IASA(nn.Module):
    """Two attention paths share query, as in the official implementation."""
    def __init__(self, channels, qk_dim, heads, group_size, chunk_groups=128):
        super().__init__()
        self.heads, self.group_size = heads, group_size
        self.chunk_groups = chunk_groups
        self.to_q = nn.Linear(channels, qk_dim, bias=False)
        self.to_k = nn.Linear(channels, qk_dim, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)

    def forward(self, x, indices, k_global, v_global):
        b, n, _ = x.shape
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = q.gather(1, indices.expand_as(q))
        k = k.gather(1, indices.expand_as(k))
        v = v.gather(1, indices.expand_as(v))
        gs = min(n, self.group_size)
        ng = (n + gs - 1) // gs
        padding = ng * gs - n
        q = torch.cat((q, reflected_tail(q, padding)), 1)
        q = q.reshape(b, ng, gs, self.heads, -1).permute(0, 1, 3, 2, 4)

        def key_windows(value):
            value = torch.cat((value, reflected_tail(value, padding + gs)), 1)
            value = value.unfold(1, 2 * gs, gs)
            # B,ng,C,2gs -> B,ng,heads,2gs,head_dim
            return value.reshape(b, ng, self.heads, -1, 2 * gs).permute(0, 1, 2, 4, 3)

        k, v = key_windows(k), key_windows(v)
        q = q.reshape(b * ng, self.heads, gs, -1)
        k = k.reshape(b * ng, self.heads, 2 * gs, -1)
        v = v.reshape(b * ng, self.heads, 2 * gs, -1)
        results = []
        # Chunking bounds inference score workspace. Training may retain all
        # chunks for backward; this is not a claimed training-memory bound.
        for start in range(0, b * ng, self.chunk_groups):
            stop = min(start + self.chunk_groups, b * ng)
            queries = q[start:stop]
            local = F.scaled_dot_product_attention(queries, k[start:stop], v[start:stop])
            global_out = F.scaled_dot_product_attention(
                queries, k_global.unsqueeze(0).expand(stop - start, -1, -1, -1),
                v_global.unsqueeze(0).expand(stop - start, -1, -1, -1))
            results.append(local + global_out)
        out = torch.cat(results, 0).reshape(b, ng, self.heads, gs, -1)
        out = out.permute(0, 1, 3, 2, 4).reshape(b, ng * gs, -1)[:, :n]
        inverse = indices.argsort(dim=1)
        return self.proj(out.gather(1, inverse.expand_as(out)))


class ContentTAB(nn.Module):
    def __init__(self, dim=52, qk_dim=52, mlp_dim=104, heads=4,
                 n_iter=5, num_tokens=16, group_size=128,
                 ema_decay=0.999, chunk_groups=128):
        super().__init__()
        for name, value in dict(dim=dim, qk_dim=qk_dim, mlp_dim=mlp_dim,
                                heads=heads, n_iter=n_iter, num_tokens=num_tokens,
                                group_size=group_size, chunk_groups=chunk_groups).items():
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if dim % heads or qk_dim % heads:
            raise ValueError('QK and value channels must be divisible by heads')
        if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError('Invalid EMA decay')
        self.dim, self.n_iter = dim, n_iter
        self.num_tokens, self.ema_decay = num_tokens, ema_decay
        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim, mlp_dim))
        self.irca_attn = IRCA(dim, qk_dim, heads)
        self.iasa_attn = IASA(dim, qk_dim, heads, group_size, chunk_groups)
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _forward(self, image, diagnostics):
        if image.ndim != 4 or image.shape[1] != self.dim or min(image.shape) < 1:
            raise ValueError('Expected nonempty NCHW features with configured channels')
        if not image.is_floating_point():
            raise ValueError('Expected floating-point features')
        b, _, h, w = image.shape
        residual = image.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(residual)
        n = x.shape[1]
        if not self.initted.item():
            # Preserve original initialization on supported sizes, including
            # its extra center-count padding when n is divisible by M.
            padding = self.num_tokens - n % self.num_tokens
            padded = torch.cat((x, reflected_tail(x, padding)), 1)
            means = padded.reshape(b, self.num_tokens, -1, self.dim).permute(1, 0, 2, 3)
            means = means.reshape(self.num_tokens, -1, self.dim).mean(1).detach()
        else:
            means = self.means.detach()
        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    means = center_iter(F.normalize(x, dim=-1), F.normalize(means, dim=-1))
        kg, vg, updated = self.irca_attn(x, means)
        with torch.no_grad():
            labels = (F.normalize(x, dim=-1) @ F.normalize(updated, dim=-1).T).argmax(-1)
            indices = labels.argsort(-1).unsqueeze(-1)
        attended = self.iasa_attn(x, indices, kg, vg)
        attended = self.conv1x1(attended.transpose(1, 2).reshape(b, self.dim, h, w))
        attention_delta = attended.flatten(2).transpose(1, 2)
        intermediate = residual + attention_delta
        ffn_delta = self.mlp(intermediate, (h, w))
        output = (intermediate + ffn_delta).transpose(1, 2).reshape(b, self.dim, h, w)
        if self.training:
            with torch.no_grad():
                if self.initted.item():
                    self.means.mul_(self.ema_decay).add_(updated, alpha=1 - self.ema_decay)
                else:
                    self.means.copy_(updated)
                    self.initted.fill_(True)
        if not diagnostics:
            return output
        with torch.no_grad():
            norm = residual.norm().clamp_min(1e-12)
            stats = dict(attention_delta_ratio=float(attention_delta.norm() / norm),
                         ffn_delta_ratio=float(ffn_delta.norm() / norm),
                         total_delta_ratio=float((output - image).norm() / norm),
                         center_norm=float(self.means.norm()),
                         occupied_centers=int(labels.unique().numel()))
        return output, stats

    def forward(self, image):
        return self._forward(image, diagnostics=False)

    def forward_with_diagnostics(self, image):
        """One real forward, with detached scalars; not a second hidden update."""
        return self._forward(image, diagnostics=True)
