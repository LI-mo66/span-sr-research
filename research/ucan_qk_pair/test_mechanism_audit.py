"""Synthetic hook, residual, gradient and no-mutation tests for the audit."""
import copy
import tempfile
from pathlib import Path
import zipfile
import numpy as np
import torch
from torch import nn
from attention import ContextAdapter
from mechanism_audit import intervene, parameter_digest, patch_probe, extract_source
from run import existing, HERE
from pair_model import build_model
import json


class Pair(nn.Module):
    def __init__(self):
        super().__init__();self.context_2=ContextAdapter();self.context_4=ContextAdapter()
    def forward(self,x):
        return self.context_4(self.context_2(x)[0])[0]


def main():
    torch.set_num_threads(2);torch.manual_seed(77)
    m=Pair().double()
    with torch.no_grad():
        for unit in (m.context_2,m.context_4):
            unit.core.proj.weight.normal_(0,.01);unit.core.proj.bias.fill_(.03)
    x=torch.randn(2,52,7,11,dtype=torch.float64)
    before=parameter_digest(m);full=m(x)
    with intervene(m,{2:'off',4:'off'}):assert torch.equal(m(x),x)
    with intervene(m,{2:'spatial',4:'channel'}):changed=m(x)
    assert not torch.equal(changed,full) and torch.equal(m(x),full)
    try:
        with intervene(m,{2:'off'}):raise RuntimeError('test cleanup')
    except RuntimeError:pass
    assert torch.equal(m(x),full) and parameter_digest(m)==before
    # Validate the half-mask algebra including projection bias exactly once.
    a=m.context_2;u=x[:1];b,c,h,w=u.shape
    normalized=a.norm(u.flatten(2).transpose(1,2))
    q,k,v=a.core.q(normalized),a.core.k(normalized),a.core.v(normalized)
    spatial,channel=a.core.branches(q,k,v,h,w)
    expected_delta=torch.nn.functional.linear(torch.cat((torch.zeros_like(spatial),channel),-1),a.core.proj.weight,a.core.proj.bias)
    scale=u.square().mean((1,2,3),keepdim=True).sqrt()
    expected=u+expected_delta.transpose(1,2).reshape_as(u)*scale
    with intervene(m,{2:'spatial'}):actual=a(u)[0]
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    lepe=a.core.lepe(v.transpose(1,2).reshape(b,26,h,w)).flatten(2).transpose(1,2)
    expected_delta=a.core.proj(torch.cat((lepe,channel),-1))
    expected=u+expected_delta.transpose(1,2).reshape_as(u)*scale
    with intervene(m,{2:'global'}):actual=a(u)[0]
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    assert torch.equal(m(x),full)
    arch,_=existing.fs_runtime.components()
    cfg=json.loads((HERE/'configs/SG6-UCAN-independent.json').read_text())
    real=build_model(arch,cfg).eval()
    with torch.no_grad():
        real.context_2.core.proj.weight.normal_(0,.002)
        real.context_4.core.proj.weight.normal_(0,.002)
    image=torch.rand(1,3,7,11)
    hr=np.random.default_rng(77).integers(0,256,(28,44,3),dtype=np.uint8)
    old=parameter_digest(real)
    result=patch_probe(real,image,hr,existing.fs_runtime.trainer)
    assert result['status']=='PASS',result
    assert old==parameter_digest(real)
    assert all(p.grad is None for p in real.parameters())
    scratch=HERE.parents[1]/'tmp';scratch.mkdir(exist_ok=True)
    root=Path(tempfile.mkdtemp(dir=scratch,prefix='mechanism-test-'))
    z=root/'bad.zip'
    with zipfile.ZipFile(z,'w') as handle:handle.writestr('../escape.py','x')
    try:extract_source(z,root/'out')
    except ValueError:pass
    else:raise AssertionError('Unsafe path accepted')
    print('PASS: hook cleanup, identity bypass, half-mask+bias algebra, original weights/grad untouched, FP64 gradient probe, safe source extraction')


if __name__=='__main__':main()
