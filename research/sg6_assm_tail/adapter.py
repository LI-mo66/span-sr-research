"""SG6 adaptation of MambaIRv2 ASSM, not a full-network reproduction.

Changes: deterministic straight-through routing, aligned prompt permutation,
stable category ties, independent small residual scale. See vendor/NOTICE.md.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from scan import scan


def gather_tokens(x, index):
    return x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))


def policy(logits):
    soft = logits.softmax(-1)
    hard = F.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
    return hard - soft.detach() + soft


class SelectiveScan(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.d_state, self.dt_rank = 8, 4
        self.x_proj_weight = nn.Parameter(nn.Linear(52, 20, bias=False).weight[None].clone())
        dt = nn.Linear(4, 52, bias=True)
        nn.init.uniform_(dt.weight, -0.5, 0.5)
        times = torch.exp(torch.rand(52) * (math.log(.1)-math.log(.001)) + math.log(.001)).clamp_min(1e-4)
        with torch.no_grad():
            dt.bias.copy_(times + torch.log(-torch.expm1(-times)))
        self.dt_projs_weight = nn.Parameter(dt.weight[None].clone())
        self.dt_projs_bias = nn.Parameter(dt.bias[None].clone())
        self.A_logs = nn.Parameter(torch.arange(1, 9).float().log().repeat(52, 1))
        self.Ds = nn.Parameter(torch.ones(52))

    def forward(self, x, prompt):
        if x.dtype != torch.float32 or prompt.dtype != torch.float32:
            raise ValueError('SG6 screen uses FP32 only')
        u = x.transpose(1, 2).contiguous()
        projected = torch.einsum('bdl,kcd->bkcl', u, self.x_proj_weight)
        dt, B, C = projected.split((4, 8, 8), dim=2)
        dt = torch.einsum('bkrl,kdr->bkdl', dt, self.dt_projs_weight)[:, 0].contiguous()
        C = C + prompt.transpose(1, 2)[:, None]
        y = scan(self.backend, u, dt, -self.A_logs.exp(), B.contiguous(), C.contiguous(),
                 self.Ds, z=None, delta_bias=self.dt_projs_bias.flatten(),
                 delta_softplus=True, return_last_state=False)
        return y.transpose(1, 2).contiguous()


class ASSMCore(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.selectiveScan = SelectiveScan(backend)
        self.out_norm = nn.LayerNorm(52)
        self.out_proj = nn.Linear(52, 52)
        self.in_proj = nn.Sequential(nn.Conv2d(52, 52, 1))
        self.CPE = nn.Sequential(nn.Conv2d(52, 52, 3, padding=1, groups=52))
        self.embeddingB = nn.Embedding(64, 32)
        nn.init.uniform_(self.embeddingB.weight, -math.sqrt(3/64), math.sqrt(3/64))
        self.route = nn.Sequential(nn.Linear(52, 17), nn.GELU(), nn.Linear(17, 64), nn.LogSoftmax(-1))
        self.embeddingA = nn.Embedding(32, 8)
        nn.init.uniform_(self.embeddingA.weight, -math.sqrt(3/32), math.sqrt(3/32))
        # Match outer official network's Linear/LN initialization, while keeping
        # stacked dt projection parameters and their inverse-softplus bias intact.
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, tokens, size):
        b, n, c = tokens.shape
        h, w = size
        if c != 52 or n != h*w:
            raise ValueError('N52HW token alignment required')
        selected = policy(self.route(tokens))
        prompt = selected @ (self.embeddingB.weight @ self.embeddingA.weight)
        indices = selected.detach().argmax(-1).argsort(dim=-1, stable=True)
        # Match the official NCHW convolution input, including physical layout.
        image = self.in_proj(tokens.transpose(1, 2).reshape(b, c, h, w).contiguous())
        image = image * torch.sigmoid(self.CPE(image))
        values = image.flatten(2).transpose(1, 2)
        # Both values and their pixel-specific readout prompts use ONE ordering.
        y = self.selectiveScan(gather_tokens(values, indices), gather_tokens(prompt, indices))
        y = self.out_proj(self.out_norm(y))
        return gather_tokens(y, indices.argsort(-1))


class ASSMAdapter(nn.Module):
    def __init__(self, backend='cuda'):
        super().__init__()
        self.norm = nn.LayerNorm(52)
        self.core = ASSMCore(backend)
        self.scale = nn.Parameter(torch.full((52,), .01))

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 52 or min(image.shape) < 1:
            raise ValueError('Expected nonempty N52HW feature')
        b, c, h, w = image.shape
        tokens = image.flatten(2).transpose(1, 2)
        correction = self.core(self.norm(tokens), (h, w)) * self.scale
        # Restore local feature amplitude without a third normalization layer.
        # Only amplitude measurement is detached; LN(F), scan and identity paths
        # remain differentiable. Never borrow statistics from another image.
        amplitude = image.detach().square().mean(1, keepdim=True).clamp_min(1e-12).sqrt()
        return image + amplitude * correction.transpose(1, 2).reshape(b, c, h, w)
