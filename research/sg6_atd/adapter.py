"""ATD 2026 attention-only adaptation; Apache-2.0 provenance in vendor/NOTICE.md."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TDCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(52,12)
        self.wk = nn.Linear(52,12)
        self.wv = nn.Linear(52,52)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, dictionary):
        q, k, v = self.wq(x), self.wk(dictionary), self.wv(dictionary)
        scores = F.normalize(q,dim=-1) @ F.normalize(k,dim=-1).transpose(-2,-1)
        scores = scores * (1 + self.scale.clamp(0,3) * math.log(128))
        weights = scores.softmax(-1)
        return weights @ v, weights.argmax(-1)


class ACMSA(nn.Module):
    def __init__(self, chunk):
        super().__init__()
        self.proj = nn.Linear(52,52)
        self.chunk = chunk

    def forward(self, qkv, labels):
        b,n,_ = qkv.shape
        gs = min(n,128)
        ng = (n+gs-1)//gs
        indices = labels.sort(dim=-1,stable=False).indices
        ordered = qkv.gather(1,indices.unsqueeze(-1).expand_as(qkv))
        padding = ng*gs-n
        ordered = torch.cat((ordered,ordered[:,n-padding:n].flip(1)),1)
        packed = ordered.reshape(b,ng,gs,3,4,13).permute(3,0,1,4,2,5)
        q,k,v = [part.reshape(b*ng,4,gs,13) for part in packed]
        pieces = []
        for start in range(0,b*ng,self.chunk):
            scores = (q[start:start+self.chunk] @ k[start:start+self.chunk].transpose(-2,-1)) * (13**-0.5)
            pieces.append(scores.softmax(-1) @ v[start:start+self.chunk])
        result = torch.cat(pieces).reshape(b,ng,4,gs,13).permute(0,1,3,2,4).reshape(b,ng*gs,52)[:,:n]
        inverse = indices.argsort(-1)
        return self.proj(result.gather(1,inverse.unsqueeze(-1).expand_as(result)))


class AttentionAdapter(nn.Module):
    def __init__(self, query_chunk=1000000000, group_chunk=1000000000):
        super().__init__()
        if any(type(v) is not int or v<1 for v in (query_chunk,group_chunk)):
            raise ValueError('Chunk sizes must be positive integers')
        self.query_chunk = query_chunk
        self.norm = nn.LayerNorm(52)
        self.wqkv = nn.Linear(52,156)
        self.tdca = TDCA()
        self.acmsa = ACMSA(group_chunk)
        self.dictionary = nn.Parameter(torch.randn(128,52))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module,nn.Linear):
            nn.init.trunc_normal_(module.weight,std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module,nn.LayerNorm):
            nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

    def forward(self, image):
        if image.ndim!=4 or image.shape[1]!=52 or min(image.shape)<1 or not image.is_floating_point():
            raise ValueError('Expected nonempty floating-point N52HW')
        b,c,h,w = image.shape
        x = self.norm(image.flatten(2).transpose(1,2).contiguous())
        td = self.dictionary.unsqueeze(0).expand(b,-1,-1)
        pieces, labels = [], []
        for start in range(0,h*w,self.query_chunk):
            value,category = self.tdca(x[:,start:start+self.query_chunk],td)
            pieces.append(value); labels.append(category)
        direct = torch.cat(pieces,1)
        grouped = self.acmsa(self.wqkv(x),torch.cat(labels,1))
        return image + (direct+grouped).transpose(1,2).reshape(b,c,h,w)
