"""Own SG6 adaptation: retain full local DW3 and add a zero-output partial strip."""
import torch
from torch import nn
from sg_model import build_model as build_sg6,validate_variant as validate_sg6

BASE='e731483d4ce4e325f3d64dbbc80add342b46f7f1'
VERSION='SG6-StripAdd-B3-K31-v1'

def validate_variant(cfg):
    validate_sg6(cfg)
    if cfg.get('experiment_version')!=VERSION or cfg.get('strip_add')!=dict(block=3,channels=13,kernel=31,seed=70031,init='zero-output-vertical-v1'):
        raise ValueError('Only frozen SG6 StripAdd B3 K31 v1 is supported')
    return 31

class AdditiveStripDepth(nn.Module):
    def __init__(self,original,kernel=31):
        super().__init__()
        if original.weight.shape!=(52,1,3,3) or original.groups!=52 or kernel!=31 or original.bias is None:
            raise ValueError('Unexpected original depthwise layer')
        self.kernel=kernel
        self.local=original
        self.horizontal=nn.Conv2d(13,13,(1,31),padding=(0,15),groups=13,bias=True)
        self.vertical=nn.Conv2d(13,13,(31,1),padding=(15,0),groups=13,bias=True)
        generator=torch.Generator(device='cpu').manual_seed(70031)
        h=torch.randn(13,31,generator=generator)
        target=original.weight[:13].detach().cpu().flatten(1).norm(dim=1).clamp_min(1e-12)
        with torch.no_grad():
            self.horizontal.weight.copy_((h/h.norm(dim=1,keepdim=True)*target.sqrt()[:,None])[:,None,None,:])
            self.horizontal.bias.zero_()
            self.vertical.weight.zero_()
            self.vertical.bias.zero_()

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=52:raise ValueError('Expected N,52,H,W')
        local=self.local(x)
        delta=self.vertical(self.horizontal(x[:,:13]))
        return torch.cat((local[:,:13]+delta,local[:,13:]),dim=1)

def build_model(arch,cfg):
    validate_variant(cfg)
    model=build_sg6(arch,cfg)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(70031)
        branch=model.block_3.spatial_gate
        branch.depth=AdditiveStripDepth(branch.depth)
    model.extra_parameters+=832
    return model
