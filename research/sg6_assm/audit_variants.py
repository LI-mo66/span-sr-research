"""One matched synthetic step; diagnostic ablations, no production changes."""
import copy
import json
from pathlib import Path
import torch
from torch import nn
from model import build_model, build_sg6, BlockWithASSM
import runtime


def rms(x):return float(x.detach().square().mean().sqrt())


def main():
    torch.set_num_threads(2)
    cfg=json.loads((Path(__file__).parent/'configs/M1-mid3-lr2e4.json').read_text())
    cfg.update(engineering_only=True);cfg['assm_settings']['backend']='reference'
    arch,_=runtime.components();torch.manual_seed(10)
    original=build_sg6(arch,cfg);torch.manual_seed(10);prototype=build_model(arch,cfg)
    x=torch.rand(2,3,13,17);target=torch.rand(2,3,52,68)
    with torch.no_grad():baseline=original(x)
    variants=[('mid3_current','mid',.01,True,True),
              ('mid3_no_input_LN','mid',.01,False,True),
              ('mid3_no_output_LN','mid',.01,True,False),
              ('mid3_scale_0p1','mid',.1,True,True),
              ('tail_scale_0p01','tail',.01,True,True),
              ('tail_scale_0p1','tail',.1,True,True)]
    results=[]
    for name,position,scale,input_ln,output_ln in variants:
        model=copy.deepcopy(original)
        a=copy.deepcopy(prototype.block_3.adapter)
        with torch.no_grad():a.scale.fill_(scale)
        if not input_ln:a.norm=nn.Identity()
        if not output_ln:a.core.out_norm=nn.Identity()
        if position=='mid':model.block_3=BlockWithASSM(model.block_3,a)
        else:model.conv_cat=nn.Sequential(model.conv_cat,a)
        captures={}
        h1=a.register_forward_pre_hook(lambda m,i:captures.update(F=i[0].detach().clone()))
        h2=a.register_forward_hook(lambda m,i,o:captures.update(Y=o.detach().clone()))
        try:y=model(x)
        finally:h1.remove();h2.remove()
        f=captures['F']
        with torch.no_grad():
            probabilities=a.core.route(a.norm(f.flatten(2).transpose(1,2))).softmax(-1)
            counts=torch.bincount(probabilities.argmax(-1).flatten(),minlength=64).float()
        loss=torch.nn.functional.l1_loss(y,target);loss.backward()
        before={k:p.detach().clone() for k,p in a.named_parameters()}
        grads={k:rms(p.grad) for k,p in a.named_parameters()}
        opt=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(.9,.999),eps=1e-8)
        opt.step()
        updates={k:dict(gradient_rms=grads[k],actual_update_rms=rms(p-before[k]),
                        relative_update_rms=rms(p-before[k])/max(rms(before[k]),1e-12))
                 for k,p in a.named_parameters()}
        results.append(dict(name=name,loss=float(loss.detach()),F_rms=rms(f),
                            correction_to_F_rms=rms(captures['Y']-f)/max(rms(f),1e-12),
                            final_rgb_adapter_change_rms=rms(y-baseline),
                            route_active=int((counts>0).sum()),route_top_fraction=float(counts.max()/counts.sum()),
                            parameter_updates=updates))
    report=dict(scope='Matched scratch weights, one synthetic step; no real images or PSNR, no architecture changed',results=results)
    path=Path(__file__).parent/'normalization-placement-diagnostic.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    for row in results:
        print(row['name'],{k:row[k] for k in ('correction_to_F_rms','final_rgb_adapter_change_rms','route_active')},
              'route0',row['parameter_updates']['core.route.0.weight'],
              'A_logs',row['parameter_updates']['core.selectiveScan.A_logs'])


if __name__=='__main__':main()
