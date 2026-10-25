"""Read-only sampled observations. No optimizer, parameter or RNG mutation."""
import json
import math
from pathlib import Path
import torch


class Observer:
    def __init__(self, model, path, accumulation, initial_update=0):
        self.path=Path(path)
        self.accumulation=accumulation
        self.calls=initial_update*accumulation
        self.active=False
        self.pending=None
        self.stats={}
        self.fusion=model.block_3.stage_fusion
        self.handles=[model.register_forward_pre_hook(self.before)]
        for name,module in [('a',model.block_3.stage_fusion.projections[0]),
                            ('b',model.block_3.stage_fusion.projections[1]),
                            ('c',model.block_3.stage_fusion.projections[2])]:
            self.handles.append(module.register_forward_pre_hook(self.input_hook(name)))
        for name,module in [('delta',model.block_3.stage_fusion),
                            ('local_gate',model.block_3.local_gate),
                            ('spatial_gate',model.block_3.spatial_gate)]:
            self.handles.append(module.register_forward_hook(self.output_hook(name)))
        self.handles.append(model.register_forward_hook(self.after))

    @property
    def parameters(self):
        # Resolve after .cuda()/.to(), which can replace Parameter objects.
        return dict(self.fusion.named_parameters())

    def emit(self, row):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')

    def before(self, model, inputs):
        # Eval immediately after an optimizer update also settles a pending sample.
        boundary=not model.training or self.calls % self.accumulation==0
        if boundary and self.pending is not None:
            step,before=self.pending
            updates={name:float((p.detach()-before[name]).norm()) for name,p in self.parameters.items()}
            if not all(math.isfinite(v) for v in updates.values()):
                raise FloatingPointError('Nonfinite observed parameter update')
            self.emit(dict(kind='actual_update',update=step,norms=updates))
            self.pending=None
        self.active=False
        if not model.training:return
        step=self.calls//self.accumulation+1
        first=self.calls % self.accumulation==0
        self.calls+=1
        self.active=first and (step<=12 or step%1000==0)
        if self.active:
            self.step=step
            self.stats={}
            self.pending=(step,{n:p.detach().clone() for n,p in self.parameters.items()})

    def record(self,name,value):
        if not self.active:return
        value=value.detach().float()
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite feature '+name)
        self.stats[name]=dict(shape=list(value.shape),rms=float(value.square().mean().sqrt()),
                              mean=float(value.mean()),absmax=float(value.abs().max()))

    def input_hook(self,name):
        def hook(module,inputs):self.record(name,inputs[0])
        return hook

    def output_hook(self,name):
        def hook(module,inputs,output):self.record(name,output)
        return hook

    def after(self,model,inputs,output):
        if self.active:
            self.emit(dict(kind='features_first_microbatch',update=self.step,statistics=self.stats))
        self.active=False
