"""Independently written adaptations of cited mechanisms; no measured gains yet."""
import torch
from torch import nn
import torch.nn.functional as F

class GlobalMean(nn.Module):
    def forward(self,x):return x.mean(dim=(2,3),keepdim=True)

class FusionGate(nn.Module):
    def __init__(self,c):
        super().__init__();self.logits=nn.ModuleList([nn.Conv2d(c,c,1) for _ in range(4)])
        for layer in self.logits:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self,features):
        weights=torch.stack([layer(x) for layer,x in zip(self.logits,features)],dim=0).softmax(dim=0)*4
        return [x*weights[i] for i,x in enumerate(features)]

class PartialContext(nn.Module):
    def __init__(self,c):
        super().__init__();self.p=c//4;hidden=self.p//2
        self.large=nn.Conv2d(self.p,self.p,13,padding=6,bias=False)
        self.dynamic=nn.Sequential(GlobalMean(),nn.Conv2d(self.p,hidden,1),nn.GELU(),nn.Conv2d(hidden,self.p*9,1))
        nn.init.zeros_(self.dynamic[-1].weight);nn.init.zeros_(self.dynamic[-1].bias)
        self.mix=nn.Conv2d(c,c,1);self.scale=nn.Parameter(torch.full((1,c,1,1),0.01))
    def forward(self,x):
        b,_,h,w=x.shape;part=x[:,:self.p]
        kernel=self.dynamic(part).reshape(b*self.p,1,3,3)
        dynamic=F.conv2d(part.reshape(1,b*self.p,h,w),kernel,padding=1,groups=b*self.p).reshape(b,self.p,h,w)
        correction=self.mix(torch.cat([self.large(part)+dynamic,x[:,self.p:]],dim=1))
        return x+self.scale*correction

class FrequencyCorrection(nn.Module):
    def __init__(self,c):
        super().__init__();hidden=max(1,c//16)
        self.channel=nn.Sequential(nn.Conv2d(c,hidden,1,bias=False),nn.ReLU(),nn.Conv2d(hidden,c,1,bias=False))
        self.spatial=nn.Conv2d(2,1,7,padding=3,bias=False)
        self.mix=nn.Conv2d(c,c,1);self.scale=nn.Parameter(torch.full((1,c,1,1),0.01))
    def forward(self,x):
        low=F.avg_pool2d(x,5,stride=1,padding=2,count_include_pad=False);high=x-low
        channel=(self.channel(low.mean(dim=(2,3),keepdim=True))+self.channel(low.amax(dim=(2,3),keepdim=True))).sigmoid()
        spatial=self.spatial(torch.cat([high.amax(dim=1,keepdim=True),high.mean(dim=1,keepdim=True)],dim=1)).sigmoid()
        return x+self.scale*self.mix(high*channel+low*spatial)

def build_model(arch,cfg):
    specifications=cfg.get('variants',[])
    if not specifications:return arch.SPAN(**cfg['model'])
    class AdaptedSPAN(arch.SPAN):
        def __init__(self):
            super().__init__(**cfg['model']);c=cfg['model']['feature_channels']
            self.adapters=nn.ModuleDict();self.specifications=specifications
            # Preserve all original parameter initialization AND subsequent RNG stream.
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(cfg['seed']+10001)
                for spec in specifications:
                    name,position=spec['name'],spec['position']
                    if name not in ('fusion','context','frequency'):raise ValueError(name)
                    if (name=='fusion' and position!='fusion') or (name!='fusion' and position not in ('mid','tail')):raise ValueError(spec)
                    if name in self.adapters:raise ValueError('Only one instance per mechanism permitted')
                    cls={'fusion':FusionGate,'context':PartialContext,'frequency':FrequencyCorrection}[name]
                    self.adapters[name]=cls(c)
            self.extra_parameters=sum(p.numel() for p in self.adapters.parameters())
            if self.extra_parameters>cfg['max_extra_deployed_parameters']:raise ValueError('Extra deployed parameter budget exceeded')
        def apply_at(self,x,position):
            for spec in self.specifications:
                if spec['position']==position:x=self.adapters[spec['name']](x)
            return x
        def forward(self,x):
            self.mean=self.mean.type_as(x);x=(x-self.mean)*self.img_range
            feature=self.conv_1(x)
            b1,_,_=self.block_1(feature);b2,_,_=self.block_2(b1);b3,_,_=self.block_3(b2)
            b3=self.apply_at(b3,'mid')
            b4,_,_=self.block_4(b3);b5,_,_=self.block_5(b4);b6,b6_first,_=self.block_6(b5)
            b6=self.conv_2(b6);features=[feature,b6,b1,b6_first]
            if 'fusion' in self.adapters:features=self.adapters['fusion'](features)
            out=self.conv_cat(torch.cat(features,dim=1));out=self.apply_at(out,'tail')
            return self.upsampler(out)
    return AdaptedSPAN()
