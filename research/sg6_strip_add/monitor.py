"""Sparse observational hooks. No loss, RNG, optimizer math, or model-state changes."""
import json
from pathlib import Path
import torch

def stat(x):
    x=x.detach().float()
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite monitored activation')
    return dict(rms=float(x.square().mean().sqrt()),mean=float(x.mean()),
                absmax=float(x.abs().max()),negative_fraction=float((x<0).float().mean()))

class Monitor:
    def __init__(self,path,start=0):
        self.path=Path(path);self.update=start;self.features={};self.parameters={}
    def due(self):return self.update<8 or (self.update+1)%1000==0
    def bind(self,model):
        self.parameters={n:p for n,p in model.named_parameters() if p.requires_grad and
            ('block_3.spatial_gate.' in n or 'block_3.local_gate.' in n)}
        modules=dict(F=model.block_3.c3_r,depth=model.block_3.spatial_gate.depth,
            spatial=model.block_3.spatial_gate,local=model.block_3.local_gate,
            original_local_depth=model.block_3.spatial_gate.depth.local,
            additive_delta=model.block_3.spatial_gate.depth.vertical)
        for name,module in modules.items():
            def capture(m,inputs,out,name=name):
                if m.training and self.due():
                    self.features[name]=stat(out)
                    if name=='depth':
                        self.features['combined_first13']=stat(out[:,:13])
                        self.features['local_channels']=stat(out[:,13:])
            module.register_forward_hook(capture)
    def optimizer(self,original):
        monitor=self
        class ObservedAdam(original):
            def step(self,closure=None):
                due=monitor.due()
                previous={n:p.detach().clone() for n,p in monitor.parameters.items()} if due else {}
                if due:
                    for n,p in monitor.parameters.items():
                        if p.grad is None or not torch.isfinite(p.grad).all():
                            raise FloatingPointError('Missing/nonfinite monitored gradient: '+n)
                result=super().step(closure)
                monitor.update+=1
                if due:
                    values={n:dict(parameter=stat(previous[n]),gradient=stat(p.grad),
                        actual_update=stat(p.detach()-previous[n])) for n,p in monitor.parameters.items()}
                    monitor.path.parent.mkdir(parents=True,exist_ok=True)
                    with monitor.path.open('a',encoding='utf8') as f:
                        f.write(json.dumps(dict(update=monitor.update,features_last_microbatch=monitor.features,
                            parameters=values,observational_only=True))+'\n')
                return result
        return ObservedAdam
