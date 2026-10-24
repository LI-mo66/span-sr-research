"""UCAN-derived dual interaction, adapted for SG6. See SOURCES.md.

Modified from hokiyoshi/UCAN b812c0a (Apache-2.0): explicit state/shape
contracts, device-safe frequencies, bounded Fourier mixing, one branch LN,
feature-unit writeback, and zero projection. See the recorded v1 scale failure.
"""
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class QKState(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    height: int
    width: int


class HedgehogFeatureMap(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.layer = nn.Linear(head_dim, head_dim)
        nn.init.eye_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        x = self.layer(x)
        return torch.cat((2 * x, -2 * x), dim=-1).softmax(-1)


class FourierEmbedding(nn.Module):
    """Source 1D flattened-position semantics, without implicit CUDA allocation.

    Coefficients retain source initialization. Production uses abs-sum/L1
    normalization, preventing cancellation amplification. source_signed exists
    only for source-equivalence and retrospective synthetic diagnostics.
    """
    def __init__(self, dim=52, heads=1, floor=1e-4, normalization='abs_sum'):
        super().__init__()
        if dim % heads or (dim // heads) % 2:
            raise ValueError('Fourier rotation requires an even per-head dimension')
        self.dim = dim // heads
        self.floor = floor
        if normalization not in ('abs_sum','source_signed'):
            raise ValueError('Unknown Fourier normalization')
        self.normalization = normalization
        frequency = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        frequency = frequency[frequency >= 2 * torch.pi / 4096][None]
        self.register_buffer('inv_freq', frequency)
        inputs = frequency.size(-1)
        outputs = min(inputs, self.dim // 4)
        if outputs < 1:
            raise ValueError('Fourier bottleneck must have at least one component')
        self.sin_coef = nn.Parameter(torch.randn(heads, inputs, outputs))
        self.cos_coef = nn.Parameter(torch.randn(heads, inputs, outputs))
        nn.init.xavier_normal_(self.sin_coef, gain=0.5)
        nn.init.xavier_normal_(self.cos_coef, gain=0.5)
        identity = torch.zeros_like(self.sin_coef)
        step = (inputs + outputs - 1) // outputs
        for i in range(outputs):
            if i * step < inputs:
                identity[..., i * step, i] = 1
        with torch.no_grad():
            self.sin_coef.add_(identity)
            self.cos_coef.add_(identity)

    def normalized(self, coefficient):
        if self.normalization == 'abs_sum':
            return coefficient / coefficient.abs().sum(dim=-2, keepdim=True).clamp_min(self.floor)
        denominator = coefficient.sum(dim=-2, keepdim=True)
        sign = torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator))
        return coefficient / (sign * denominator.abs().clamp_min(self.floor))

    def forward(self, x):
        # Follow input dtype/device, including .double(), .cpu(), and model.to().
        n = x.shape[-2]
        positions = torch.arange(n, device=x.device, dtype=self.inv_freq.dtype)
        phase = torch.einsum('t,hd->htd', positions, self.inv_freq).unsqueeze(0)
        sine = torch.einsum('bhtD,hDd->bhtd', phase.sin(), self.normalized(self.sin_coef))
        cosine = torch.einsum('bhtD,hDd->bhtd', phase.cos(), self.normalized(self.cos_coef))
        sine = F.pad(sine, (0, self.dim // 2 - sine.size(-1)), value=1)
        cosine = F.pad(cosine, (0, self.dim // 2 - cosine.size(-1)), value=1)
        sine = torch.cat((sine, sine), -1).to(x.dtype)
        cosine = torch.cat((cosine, cosine), -1).to(x.dtype)
        first, second = x.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return x * cosine - rotated * sine


class DualInteraction(nn.Module):
    """SDFL/DFRL core: raw Q/K reuse, current V, separate receiver transforms."""
    def __init__(self, reuse=False, channels=52, heads=1, fourier_normalization='abs_sum'):
        super().__init__()
        # Only the preregistered C52/h1 configuration is admitted in v2.
        if channels != 52 or heads != 1:
            raise ValueError('v2 fixes channels=52 and heads=1')
        self.channels, self.heads, self.half = channels, heads, channels // 2
        self.reuse = reuse
        if not reuse:
            self.q = nn.Linear(channels, self.half)
            self.k = nn.Linear(channels, self.half)
        self.v = nn.Linear(channels, self.half)
        self.ln_q = HedgehogFeatureMap(self.half)
        self.ln_k = HedgehogFeatureMap(self.half)
        self.lepe = nn.Conv2d(self.half, self.half, 3, padding=1, groups=self.half)
        self.proj = nn.Linear(channels, channels)
        self.temperature = nn.Parameter(torch.ones(self.half, 1))
        self.pos_emb = FourierEmbedding(channels, heads, normalization=fourier_normalization)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _validate(self, x, h, w, state):
        if x.ndim != 3 or x.shape[1:] != (h * w, self.channels) or min(h, w) < 1:
            raise ValueError('Expected B, H*W, 52 tokens with positive H/W')
        if self.reuse:
            if not isinstance(state, QKState) or (state.height, state.width) != (h, w):
                raise ValueError('Reused Q/K requires matching H and W, not just token count')
            for t in (state.q, state.k):
                if t.shape != (x.shape[0], h * w, self.half):
                    raise ValueError('Reused Q/K batch/token/channel mismatch')
                if t.dtype != x.dtype or t.device != x.device:
                    raise ValueError('Reused Q/K dtype/device mismatch')
        elif state is not None:
            raise ValueError('Independent unit must generate its own Q/K')

    def branches(self, q, k, v, h, w):
        b, n, c = v.shape
        qh, kh, vh = (t.reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
                      for t in (q, k, v))
        qm, km = self.ln_q(qh), self.ln_k(kh)
        qr, kr = self.pos_emb(qm), self.pos_emb(km)
        # Positive Hedgehog maps are used for the normalization denominator.
        inverse = 1.0 / (qm @ km.mean(-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (kr.transpose(-2, -1) * n ** -0.5) @ (vh * n ** -0.5)
        spatial = (qr @ kv * inverse).transpose(1, 2).reshape(b, n, c)
        # Same row-major pixels as v; avoid a silent transpose of H/W.
        v_image = vh.transpose(-1, -2).reshape(b, c, h, w)
        spatial = spatial + self.lepe(v_image).flatten(2).transpose(1, 2)
        qc, kc = F.normalize(q.transpose(1, 2), dim=-1), F.normalize(k.transpose(1, 2), dim=-1)
        affinity = ((qc @ kc.transpose(-2, -1)) * self.temperature).softmax(-1)
        channel = (affinity @ v.transpose(1, 2)).transpose(1, 2)
        return spatial, channel

    def forward(self, x, h, w, state=None):
        self._validate(x, h, w, state)
        if self.reuse:
            q, k = state.q, state.k
        else:
            q, k = self.q(x), self.k(x)
        v = self.v(x)
        spatial, channel = self.branches(q, k, v, h, w)
        return self.proj(torch.cat((spatial, channel), dim=-1)), QKState(q, k, h, w)


class ContextAdapter(nn.Module):
    def __init__(self, reuse=False):
        super().__init__()
        self.scale_writes = True
        self.norm = nn.LayerNorm(52, eps=1e-5)
        self.core = DualInteraction(reuse=reuse)

    def forward(self, x, state=None):
        if x.ndim != 4 or x.shape[1] != 52:
            raise ValueError('Expected B,52,H,W feature map')
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        self.core._validate(tokens, h, w, state)
        correction, new_state = self.core(self.norm(tokens), h, w, state)
        delta = correction.transpose(1, 2).reshape(b, c, h, w)
        if self.scale_writes:
            # Per-image, not per-batch. Detached so this unit conversion does
            # not add a global magnitude gradient path to SG6's identity path.
            scale = x.detach().square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
            delta = delta * scale
        return x + delta, new_state
