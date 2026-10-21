"""Matched small diagnostics for distribution, optimization and normalization.

Run after any candidate change. PASS is not a PSNR or convergence claim.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from model import build_model
import runtime


def rms(x):return float(x.detach().square().mean().sqrt())


def gradient_comparison(loss,other,parameters):
    g=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    h=torch.autograd.grad(other,parameters,allow_unused=True)
    dot=aa=bb=diff=0.
    for p,x,y in zip(parameters,g,h):
        x=torch.zeros_like(p) if x is None else x
        y=torch.zeros_like(p) if y is None else y
        x=x.detach().double();y=y.detach().double()
        dot+=float((x*y).sum());aa+=float(x.square().sum());bb+=float(y.square().sum())
        diff+=float((x-y).square().sum())
    return dict(cosine=dot/max((aa*bb)**.5,1e-30),
                relative_gradient_change=(diff/max(bb,1e-30))**.5,
                scope='Same current weights and batch; auxiliary backward does not change optimizer gradients')


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--backend',choices=('reference','cuda','cuda_isolated'),default='reference')
    p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((Path(__file__).parent/'configs/M2-tail-rms-lr2e4.json').read_text())
    cfg.update(engineering_only=True);cfg['assm_settings']['backend']=args.backend
    arch,_=runtime.components();torch.manual_seed(10)
    model=build_model(arch,cfg).to(args.device).train();a=model.conv_cat.adapter
    norms=[name for name,m in model.named_modules() if isinstance(m,nn.LayerNorm)]
    assert norms==['conv_cat.adapter.norm','conv_cat.adapter.core.out_norm']
    x=torch.rand(2,3,13,17,device=args.device)
    target=torch.nn.functional.interpolate(x,scale_factor=4,mode='bicubic',align_corners=False).clamp(0,1)
    parameters=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('conv_cat.adapter.')]
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=2e-4,betas=(.9,.999),eps=1e-8)
    rows=[]
    for step in range(1,9):
        captured={}
        h=a.register_forward_pre_hook(lambda m,i:captured.update(F=i[0].detach().clone()))
        j=a.register_forward_hook(lambda m,i,o:captured.update(Y=o.detach().clone()))
        opt.zero_grad(set_to_none=True)
        try:y=model(x)
        finally:h.remove();j.remove()
        loss=torch.nn.functional.l1_loss(y,target)
        assert torch.isfinite(loss)
        comparison=None
        if step in (1,8):
            handle=a.register_forward_hook(lambda m,i,o:i[0])
            try:other=torch.nn.functional.l1_loss(model(x),target)
            finally:handle.remove()
            comparison=gradient_comparison(loss,other,parameters)
        loss.backward()
        before={name:p.detach().clone() for name,p in a.named_parameters()}
        for name,p in a.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0,name
        opt.step()
        updates={name:dict(actual_rms=rms(p-before[name]),
                           relative_rms=rms(p-before[name])/max(rms(before[name]),1e-12))
                 for name,p in a.named_parameters()}
        f=captured['F'];ratio=rms(captured['Y']-f)/max(rms(f),1e-12)
        assert ratio<.05,'Synthetic residual exceeds 5% screening guard'
        route_updates=[updates[k]['relative_rms'] for k in ('core.route.0.weight','core.route.2.weight')]
        assert min(route_updates)>1e-5,'Synthetic route update remains too weak'
        rows.append(dict(step=step,loss=float(loss.detach()),correction_over_F_rms=ratio,
                         min_route_relative_update=min(route_updates),main_gradient_comparison=comparison,
                         parameter_updates=updates))
    # Stress feature amplitude separately; no shared image/batch statistics.
    feature=torch.randn(2,52,3,5,device=args.device)
    contrasts=[]
    with torch.no_grad():
        for gain in (.1,1.,10.):
            value=feature*gain;delta=a(value)-value
            ratio=rms(delta)/rms(value)
            assert torch.isfinite(delta).all() and ratio<.05
            contrasts.append(dict(input_gain=gain,correction_over_F_rms=ratio))
    report=dict(status='SMALL_DIAGNOSTICS_PASS',device=args.device,backend=args.backend,
                scope='8 synthetic bicubic updates only; diagnostics do not establish PSNR or real-image stability',
                guards=dict(max_residual_ratio=.05,min_route_relative_update=1e-5,
                            meaning='Local engineering screens, not universal stability or PSNR thresholds'),
                normalization_layers=norms,updates=rows,feature_contrast_probe=contrasts)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],contrast=contrasts,
                         progress=[{k:r[k] for k in ('step','correction_over_F_rms','min_route_relative_update','main_gradient_comparison')} for r in rows]),indent=2))


if __name__=='__main__':main()
