"""Paper-inspired partial strip replacement, not an original LKMN/LKFMixer module."""
import math
import torch
from torch import nn
from sg_model import build_model as build_sg6, validate_variant as validate_sg6

BASE='e731483d4ce4e325f3d64dbbc80add342b46f7f1'
VERSIONS={15:'SG6-PartialStrip-B3-K15-v1',31:'SG6-PartialStrip-B3-K31-v1'}

def validate_variant(cfg):
    validate_sg6(cfg)
    s=cfg.get('partial_strip',{})
    k=s.get('kernel')
    expected=dict(block=3,channels=13,kernel=k,init='matched-kernel-energy-v1',seed=70031)
    if k not in VERSIONS or s!=expected or cfg.get('experiment_version')!=VERSIONS[k]:
        raise ValueError('Only the frozen B3 K15/K31 variants are supported')
    return k

class PartialStripDepth(nn.Module):
    def __init__(self,original,kernel):
        super().__init__()
        if original.weight.shape!=(52,1,3,3) or original.groups!=52 or kernel not in VERSIONS:
            raise ValueError('Unexpected depthwise source or strip kernel')
        self.kernel=kernel
        self.horizontal=nn.Conv2d(13,13,(1,kernel),padding=(0,kernel//2),groups=13,bias=True)
        self.vertical=nn.Conv2d(13,13,(kernel,1),padding=(kernel//2,0),groups=13,bias=True)
        self.local=nn.Conv2d(39,39,3,padding=1,groups=39,bias=True)
        # Same random 31-tap parent directions for both variants, central crop for K15.
        # Match the interior IID-input response energy of each original selected DW3.
        # This does not guarantee matching responses on correlated natural-image features.
        generator=torch.Generator(device='cpu').manual_seed(70031)
        h=torch.randn(13,31,generator=generator)
        v=torch.randn(13,31,generator=generator)
        start=(31-kernel)//2
        h=h[:,start:start+kernel]
        v=v[:,start:start+kernel]
        target=original.weight[:13].detach().cpu().flatten(1).norm(dim=1).clamp_min(1e-12)
        h=h/h.norm(dim=1,keepdim=True)*target.sqrt().unsqueeze(1)
        v=v/v.norm(dim=1,keepdim=True)*target.sqrt().unsqueeze(1)
        with torch.no_grad():
            self.horizontal.weight.copy_(h[:,None,None,:])
            self.vertical.weight.copy_(v[:,None,:,None])
            self.horizontal.bias.zero_()
            self.vertical.bias.copy_(original.bias[:13])
            self.local.weight.copy_(original.weight[13:])
            self.local.bias.copy_(original.bias[13:])

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=52:
            raise ValueError('PartialStripDepth needs N,52,H,W')
        selected,local=x.split((13,39),dim=1)
        return torch.cat((self.vertical(self.horizontal(selected)),self.local(local)),dim=1)

def build_model(arch,cfg):
    k=validate_variant(cfg)
    model=build_sg6(arch,cfg)
    branch=model.block_3.spatial_gate
    # Build on CPU, as the unchanged trainer moves the entire network afterwards.
    # Isolate both module constructors and custom initialization from the sampler RNG.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['partial_strip']['seed'])
        branch.depth=PartialStripDepth(branch.depth,k)
    model.extra_parameters+=13*(2*k-8)
    return model
