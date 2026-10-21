"""Synthetic scale/route/actual Adam-update diagnostics, not PSNR evidence."""
import argparse
import copy
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from adapter import policy
from model import build_model
import runtime


def rms(x):
    return float(x.detach().square().mean().sqrt())


def summary(x):
    v=x.detach()
    return dict(mean=float(v.mean()),std=float(v.std(unbiased=False)),rms=rms(v),
                maximum=float(v.abs().max()))


def inspect(model,x):
    a=model.block_3.adapter
    snapshots={}
    def capture(name):
        def hook(m,i,o):snapshots[name]=o.detach().clone()
        return hook
    handles=[a.register_forward_pre_hook(lambda m,i:snapshots.update(F=i[0].detach().clone())),
             a.norm.register_forward_hook(capture('input_LN')),
             a.core.selectiveScan.register_forward_hook(capture('scan_output')),
             a.core.out_norm.register_forward_hook(capture('output_LN')),
             a.core.out_proj.register_forward_hook(capture('raw_correction')),
             a.register_forward_hook(capture('fused_F'))]
    try:
        with torch.no_grad():full=model(x)
    finally:
        for h in handles:h.remove()
    f=snapshots['F'];delta=snapshots['fused_F']-f
    with torch.no_grad():
        z=a.norm(f.flatten(2).transpose(1,2))
        logits=a.core.route(z);soft=logits.softmax(-1)
        labels=soft.argmax(-1)
        counts=torch.bincount(labels.flatten(),minlength=64).float()
        probabilities=counts/counts.sum()
        active=probabilities[probabilities>0]
        prompt=policy(logits) @ (a.core.embeddingB.weight@a.core.embeddingA.weight)
    row=dict(tensors={k:summary(v) for k,v in snapshots.items()},
             correction_rms=rms(delta),correction_to_F_rms=rms(delta)/max(rms(f),1e-12),
             fused_to_F_rms=rms(snapshots['fused_F'])/max(rms(f),1e-12),
             scale_min=float(a.scale.min().detach()),scale_max=float(a.scale.max().detach()),
             route_active=int((counts>0).sum()),route_top_fraction=float(probabilities.max()),
             route_effective_categories=float((-(active*active.log()).sum()).exp()),
             route_soft_entropy=float((-(soft*soft.clamp_min(1e-12).log()).sum(-1)).mean()),
             prompt_rms=rms(prompt))
    with torch.no_grad():
        h=a.core.selectiveScan.register_forward_hook(lambda m,i,o:i[0]*m.Ds)
        try:without_state=model(x)
        finally:h.remove()
        h=a.core.selectiveScan.register_forward_pre_hook(lambda m,i:(i[0],torch.zeros_like(i[1])))
        try:without_prompt=model(x)
        finally:h.remove()
        h=a.register_forward_hook(lambda m,i,o:i[0])
        try:without_adapter=model(x)
        finally:h.remove()
    total=rms(full-without_adapter)
    row.update(final_rgb_adapter_change_rms=total,
               final_rgb_state_dependence_rms=rms(full-without_state),
               final_rgb_prompt_dependence_rms=rms(full-without_prompt),
               state_dependence_over_adapter_change=rms(full-without_state)/max(total,1e-12),
               prompt_dependence_over_adapter_change=rms(full-without_prompt)/max(total,1e-12))
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((Path(__file__).parent/'configs/M1-mid3-lr2e4.json').read_text())
    cfg['engineering_only']=True;cfg['assm_settings']['backend']='reference'
    arch,_=runtime.components();torch.manual_seed(10)
    model=build_model(arch,cfg).to(args.device)
    x=torch.rand(2,3,13,17,device=args.device)
    target=torch.rand(2,3,52,68,device=args.device)
    report=dict(device=args.device,backend='reference',
                scope='Scratch initialization and 8 synthetic updates only; not trained SG6 distributions or PSNR',
                normalization_layers=[dict(name=k,type=type(m).__name__,axis='last/channel',eps=m.eps)
                                      for k,m in model.named_modules() if isinstance(m,nn.LayerNorm)],
                initialization=inspect(model.eval(),x),updates=[])
    model.train();opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],
                                      betas=cfg['adam_betas'],eps=cfg['adam_eps'],weight_decay=0)
    for step in range(1,9):
        before={k:v.detach().clone() for k,v in model.block_3.adapter.named_parameters()}
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.l1_loss(model(x),target)
        loss.backward();opt.step()
        updates={}
        for k,v in model.block_3.adapter.named_parameters():
            delta=v.detach()-before[k]
            updates[k]=dict(weight_rms=rms(before[k]),gradient_rms=rms(v.grad),
                            actual_update_rms=rms(delta),
                            relative_update_rms=rms(delta)/max(rms(before[k]),1e-12))
        report['updates'].append(dict(step=step,loss=float(loss.detach()),parameters=updates))
    report['after_8_updates']=inspect(model.eval(),x)
    # Input contrast/offset probes isolate adapter invariance, not trained SR.
    a=model.block_3.adapter
    f=torch.randn(2,52,3,5,device=args.device)
    with torch.no_grad():
        delta=a(f)-f;offset=a(f+2)-(f+2);gain=a(2*f)-2*f
    report['normalization_invariance_probe']=dict(offset_correction_difference_rms=rms(offset-delta),
                                                 gain_correction_difference_rms=rms(gain-delta),
                                                 correction_rms=rms(delta))
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='updates'},indent=2))


if __name__=='__main__':main()
