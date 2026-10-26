"""Isolated SG6 placement/capacity/LKSA screen; no shared source modification."""
import json
from pathlib import Path
import sys
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
for directory in ('spatial_gate6','gated_bypass','linear_gate'):
    sys.path.insert(1,str(HERE.parent/directory))
from sg_model import build_model as build_sg6, validate_variant as validate_sg6
from attention import ContextAdapter
from lksa import LKSA

def settings():
    return json.loads((HERE/'config.json').read_text())['ucan_sweep']

def validate_variant(cfg):
    validate_sg6(cfg)
    if cfg.get('ucan_sweep')!=settings():raise ValueError('Arm settings differ from committed config')
    if any(cfg.get(k) for k in ('ucan_pair','sgla','variants','easa','atd_attention','cata_tab','cata_ffn')):
        raise ValueError('Undeclared mixed candidate')

class LKSAAdapter(nn.Module):
    """Full source LKSA on RMS-normalized input; residual starts at identity.

    No new LayerNorm. Detached per-image scale preserves feature units and
    does not create an extra gradient through the scale estimator.
    """
    def __init__(self):
        super().__init__()
        self.core=LKSA(52,k=35,ratio=4)
        self.project=nn.Conv2d(52,52,1)
        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=52:raise ValueError('Expected B,52,H,W')
        scale=x.detach().square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6)
        return x+scale*self.project(self.core(x/scale))

def expected_extra(cfg):
    s=cfg['ucan_sweep']
    return len(s['positions'])*9178+10959-(2756 if s['qk_mode']=='reuse' else 0)

def build_model(arch,cfg):
    validate_variant(cfg)
    model=build_sg6(arch,cfg)
    s=cfg['ucan_sweep'];positions=s['positions']
    # Slot order exactly matches prior pair initialization. The isolated RNG
    # scope leaves backbone initialization and training random stream intact.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed']+80001)
        for pos in positions:setattr(model,f'context_{pos}',ContextAdapter())
    if s['qk_mode']=='reuse':
        receiver=getattr(model,f'context_{positions[1]}').core
        del receiver.q
        del receiver.k
        receiver.reuse=True
    if s['lksa_position']:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(cfg['seed']+90001)
            setattr(model,f'lksa_{s["lksa_position"]}',LKSAAdapter())

    class SweepSPAN(type(model)):
        def forward(self,image):
            residual=self.image_residual(image)
            gate=2.0*torch.sigmoid(self.image_gate(image))
            self.mean=self.mean.type_as(image)
            feature=self.conv_1((image-self.mean)*self.img_range)
            x=feature
            state=None  # Per-forward raw Q/K only; never stored on a module.
            for pos in range(1,7):
                x,first,_=getattr(self,f'block_{pos}')(x)
                if pos in positions:
                    if s['qk_mode']=='reuse':
                        reused=state if pos==positions[1] else None
                        x,new_state=getattr(self,f'context_{pos}')(x,reused)
                        if pos==positions[0]:state=new_state
                    else:
                        # Match prior independent state lifetime: no retained
                        # raw-QK variable between adapters during inference.
                        x,_=getattr(self,f'context_{pos}')(x)
                if pos==s['lksa_position']:x=getattr(self,f'lksa_{pos}')(x)
                if pos==1:b1=x
            b6=self.conv_2(x)
            fused=self.conv_cat(torch.cat((feature,b6,b1,first),dim=1))
            return self.upsampler[1](self.upsampler[0](fused)+gate*residual)

    model.__class__=SweepSPAN
    extra=sum(p.numel() for n,p in model.named_parameters() if n.startswith(('context_','lksa_')))
    if extra!=expected_extra(cfg):raise ValueError(f'Unexpected extra parameters: {extra}')
    model.extra_parameters+=extra
    model.ucan_extra_parameters=extra
    return model
