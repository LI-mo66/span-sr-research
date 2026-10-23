"""Inspect synthetic startup checkpoint scales; no real-data inference or updates."""
import argparse
import json
from pathlib import Path

import torch
from run import HERE, existing
from pair_model import build_model


def describe(tensor):
    x=tensor.detach().double()
    return dict(mean=float(x.mean()),rms=float(x.square().mean().sqrt()),
                absmax=float(x.abs().max()),finite=bool(torch.isfinite(x).all()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay-v1',action='store_true',help='Replay v1 math on a synthetic v1 checkpoint, then compare fixes at identical unit inputs')
    a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text())
    cfg=manifest['config']
    if not cfg.get('engineering_only'):
        raise ValueError('This diagnostic is restricted to our synthetic fixture checkpoints')
    state=torch.load(a.run/'latest.pt',map_location='cpu',weights_only=False)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    arch,_=existing.fs_runtime.components()
    current=json.loads((HERE/'configs'/('SG6-UCAN-'+cfg['ucan_pair']['mode']+'.json')).read_text())
    model=build_model(arch,current).cuda().eval()
    model.load_state_dict(state['model'],strict=True)
    if a.replay_v1:
        for name in ('context_2','context_4'):
            unit=getattr(model,name)
            unit.scale_writes=False
            unit.core.pos_emb.normalization='source_signed'
    report=dict(scope='synthetic checkpoint and random RGB inputs; descriptive, not PSNR evidence',
                run=str(a.run),updates=state['update'],replay_v1=a.replay_v1,
                torch=torch.__version__,sizes={})
    for h,w in ((13,17),(64,64),(63,65)):
        rows={}
        handles=[]
        for name in ('context_2','context_4'):
            unit=getattr(model,name)
            row=rows[name]={}
            def norm_hook(module,args,out,row=row):
                row['before_norm']=describe(args[0])
                row['after_norm']=describe(out)
            def adapter_hook(module,args,out,row=row):
                x,y=args[0],out[0]
                row['input']=describe(x)
                row['correction']=describe(y-x)
                row['output']=describe(y)
                row['correction_to_input_rms']=row['correction']['rms']/max(row['input']['rms'],1e-30)
                if a.replay_v1:
                    # Fixed local input and identical checkpoint parameters;
                    # these are numerical interventions, not PSNR ablations.
                    tokens=module.norm(x.flatten(2).transpose(1,2))
                    norm=module.core.pos_emb.normalization
                    comparison={}
                    for alternative in ('source_signed','abs_sum'):
                        module.core.pos_emb.normalization=alternative
                        delta,_=module.core(tokens,h,w)
                        rms=describe(delta)['rms']
                        comparison[alternative]=dict(unscaled_correction_to_input_rms=rms/max(row['input']['rms'],1e-30),
                                                     rms_writeback_correction_to_input_rms=rms)
                    module.core.pos_emb.normalization=norm
                    row['fixed_input_interventions']=comparison
            def q_hook(module,args,out,row=row):
                row['_mapped_q']=out
            def k_hook(module,args,out,row=row):
                q=row.pop('_mapped_q')
                denominator=q@out.mean(-2,keepdim=True).transpose(-2,-1)+1e-6
                row['spatial_denominator_min']=float(denominator.min())
                row['spatial_inverse_max']=float(denominator.reciprocal().max())
            handles.extend((unit.norm.register_forward_hook(norm_hook),
                            unit.register_forward_hook(adapter_hook),
                            unit.core.ln_q.register_forward_hook(q_hook),
                            unit.core.ln_k.register_forward_hook(k_hook)))
            row['fourier']={}
            for key in ('sin_coef','cos_coef'):
                sums=getattr(unit.core.pos_emb,key).detach().sum(-2).abs()
                row['fourier'][key]=dict(min_abs_sum=float(sums.min()),
                    floored_columns=int((sums<unit.core.pos_emb.floor).sum()))
        torch.manual_seed(10)
        with torch.inference_mode():
            y=model(torch.rand(1,3,h,w,device='cuda'))
        assert torch.isfinite(y).all()
        for handle in handles:
            handle.remove()
        report['sizes'][f'{h}x{w}']=rows
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
