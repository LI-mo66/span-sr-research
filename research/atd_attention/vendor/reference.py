import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def index_reverse(index):
    index_r = torch.zeros_like(index)

    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def feature_shuffle(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    # match the shape of x and index
    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim-1, index=index)
    return shuffled_x

class ATD_CA(nn.Module):
    def __init__(self, dim, num_tokens=64, reducted_dim=10, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.dr = reducted_dim
        self.qkv_bias = qkv_bias

        self.wq = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = nn.Parameter(torch.ones([1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, td, x_size):
        h, w = x_size
        b, n, c = x.shape
        b, m, c = td.shape

        # Q: b, n, c
        q = self.wq(x)
        # K: b, m, c
        k = self.wk(td)
        # V: b, m, c
        v = self.wv(td)

        # Q @ K^T
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # b, n, m
        scale = 1 + torch.clamp(self.scale, 0, 3) * np.log(self.num_tokens)
        attn = attn * scale
        attn = self.softmax(attn)

        # Attn * V
        x = (attn @ v).reshape(b, n, c)

        return x, attn

    def flops(self, n):
        flops = 0

        # qkv = self.wq(x)
        flops += n * self.dim * self.dr
        # k = self.wk(td)
        flops += self.num_tokens * self.dim * self.dr
        # v = self.wv(td)
        flops += self.num_tokens * self.dim * self.dim

        # attn = (q @ k.transpose(-2, -1))
        flops += n * self.dim * self.dr

        # x = (attn @ v)
        flops += n * self.num_tokens * self.dim

        return flops

class AC_MSA(nn.Module):
    def __init__(self, dim, num_heads=4, category_size=128, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.category_size = category_size
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = (dim // self.num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, tk_id, x_size):
        b, n, c3 = qkv.shape
        c = c3 // 3
        gs = min(n, self.category_size)  # group size
        ng = (n + gs - 1) // gs

        # sort features by type
        x_sort_values, x_sort_indices = torch.sort(tk_id, dim=-1, stable=False)
        tk_id_inv = index_reverse(x_sort_indices)

        # feature categorization
        shuffled_qkv = feature_shuffle(qkv, x_sort_indices)  # b, n, c3
        pad_n = ng * gs - n
        paded_qkv = torch.cat((shuffled_qkv, torch.flip(shuffled_qkv[:, n-pad_n:n, :], dims=[1])), dim=1)
        y = paded_qkv.reshape(b, -1, gs, c3)  # b, ng, gs, c*3

        qkv = y.reshape(b, ng, gs, 3, self.num_heads, c//self.num_heads).permute(3, 0, 1, 4, 2, 5)  # 3, b, ng, nh, gs, c//nh
        q, k, v = qkv[0], qkv[1], qkv[2]    # b, ng, nh, gs, c//nh

        attn = (q @ k.transpose(-2, -1))  # b, ng, nh, gs, gs
        attn = attn * self.scale
        attn = self.softmax(attn)  # b, ng, nh, gs, gs

        y = (attn @ v).permute(0, 1, 3, 2, 4).reshape(b, n+pad_n, c)[:, :n, :]
        x = feature_shuffle(y, tk_id_inv)
        x = self.proj(x)

        return x


    def flops(self, n):
        flops = 0

        # attn = (q @ k.transpose(-2, -1))
        flops += n * self.dim * self.category_size

        # y = (attn @ v)
        flops += n * self.dim * self.category_size

        # x = self.proj(x)
        flops += n * self.dim * self.dim

        return flops
