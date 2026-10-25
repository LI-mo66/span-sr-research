"""Read-only diagnostics at fixed updates; no loss, gradient, RNG or state changes."""
import json
from pathlib import Path
import torch

NODES={1,1000,10000,25000,50000}


def rms(value):
    return float(value.detach().double().square().mean().sqrt())


def attach(model, cfg, output):
    interaction=model.block_3.interaction
    accum=cfg['effective_batch']//cfg['micro_batch']
    state=dict(micro=0,pending=None)

    def complete(module, inputs):
        pending=state['pending']
        if pending is not None:
            before=pending.pop('before')
            updates={name:rms(p-before[name]) for name,p in interaction.named_parameters()}
            if any(not torch.isfinite(p).all() for p in interaction.parameters()):
                raise FloatingPointError('Nonfinite interaction parameters after optimizer step')
            pending['actual_adam_update_rms']=updates
            pending['zero_update_tensors']=[k for k,v in updates.items() if v==0]
            with Path(output).open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(pending)+'\n')
            print('MID_DIAGNOSTIC:',json.dumps(pending),flush=True)
            state['pending']=None
        if module.training:
            state['micro']+=1

    def observe(module, inputs, result):
        micro=state['micro']
        update=(micro+accum-1)//accum
        if not model.training or micro%accum or update not in NODES:
            return
        feature=inputs[0].detach();value=result.detach();delta=value-feature
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite interaction output')
        state['pending']=dict(update=update,variant=cfg['mid_interaction'],
            feature_rms=rms(feature),correction_rms=rms(delta),
            correction_relative_rms=rms(delta)/max(rms(feature),1e-12),
            scale_rms=rms(interaction.scale),normalization_layers=1,
            before={name:p.detach().clone() for name,p in interaction.named_parameters()},
            scope='Last microbatch feature statistics and next-forward observation of the actual optimizer update; no region/PSNR attribution')

    model.register_forward_pre_hook(complete)
    interaction.register_forward_hook(observe)
    return model
