"""Independent-context LKSA expansion; no protected source modification."""
import json,sys
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
for directory in ('spatial_gate6','gated_bypass','linear_gate'):
    sys.path.insert(1,str(HERE.parent/directory))
from sg_model import build_model as build_sg6,validate_variant as validate_sg6
from attention import ContextAdapter
from lksa import LKSA

def settings():return json.loads((HERE/'config.json').read_text())['ucan_sweep']
def validate_variant(cfg):
    validate_sg6(cfg)
    if cfg.get('ucan_sweep')!=settings() or cfg['ucan_sweep']['qk_mode']!='independent':raise ValueError('Uncommitted arm settings')
    if any(cfg.get(k) for k in ('ucan_pair','sgla','variants','easa','atd_attention','cata_tab','cata_ffn')):raise ValueError('Undeclared mixed candidate')

class PlainSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(52,25,1),nn.GELU(),nn.Conv2d(25,25,3,padding=1),nn.GELU(),nn.Conv2d(25,52,1))
    def forward(self,x):return self.body(x)

class SpatialAdapter(nn.Module):
    def __init__(self,core='lksa',k=35,ratio=4):
        super().__init__()
        self.core=LKSA(52,k=k,ratio=ratio) if core=='lksa' else PlainSpatial()
        self.project=nn.Conv2d(52,52,1)
        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=52:raise ValueError('Expected B,52,H,W')
        scale=x.detach().square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6)
        return x+scale*self.project(self.core(x/scale))

def adapter_name(s,pos):return ('plain_' if s['spatial_core']=='plain' else 'lksa_')+str(pos)
def expected_extra(cfg):
    s=cfg['ucan_sweep'];core=s['spatial_core']
    if core=='none':spatial=0
    elif core=='plain':spatial=8327+2756
    else:spatial={(35,4):8203,(53,4):8827,(35,2):14144}[(s['lksa_kernel'],s['lksa_ratio'])]+2756
    return len(s['positions'])*9178+len(s['spatial_positions'])*spatial

def build_model(arch,cfg):
    validate_variant(cfg);model=build_sg6(arch,cfg);s=cfg['ucan_sweep']
    devices=list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg['seed']+80001)
        for pos in s['positions']:setattr(model,f'context_{pos}',ContextAdapter())
    if s['spatial_positions']:
        # Reproduce old anchor5 first, then preserve every semantically matching
        # same-shape tensor. Changed shapes must not shift unrelated initializers.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg['seed']+90001);anchor=SpatialAdapter()
        reference=anchor.state_dict()
        for pos in s['spatial_positions']:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(cfg['seed']+(93001 if pos==3 else 91001+pos))
                unit=SpatialAdapter(s['spatial_core'],s['lksa_kernel'],s['lksa_ratio'])
            if pos==5:
                with torch.no_grad():
                    for name,value in unit.state_dict().items():
                        if name in reference and value.shape==reference[name].shape:value.copy_(reference[name])
            setattr(model,adapter_name(s,pos),unit)
    class ExpandSPAN(type(model)):
        def forward(self,image):
            residual=self.image_residual(image);gate=2.0*torch.sigmoid(self.image_gate(image))
            self.mean=self.mean.type_as(image)
            feature=self.conv_1((image-self.mean)*self.img_range);x=feature
            for pos in range(1,7):
                x,first,_=getattr(self,f'block_{pos}')(x)
                if pos in s['positions']:x,_=getattr(self,f'context_{pos}')(x)
                if pos in s['spatial_positions']:x=getattr(self,adapter_name(s,pos))(x)
                if pos==1:b1=x
            b6=self.conv_2(x);fused=self.conv_cat(torch.cat((feature,b6,b1,first),dim=1))
            return self.upsampler[1](self.upsampler[0](fused)+gate*residual)
    model.__class__=ExpandSPAN
    extra=sum(p.numel() for name,p in model.named_parameters() if name.startswith(('context_','lksa_','plain_')))
    if extra!=expected_extra(cfg):raise ValueError(f'Unexpected extra parameter count: {extra}')
    model.extra_parameters+=extra;model.ucan_extra_parameters=extra
    return model
