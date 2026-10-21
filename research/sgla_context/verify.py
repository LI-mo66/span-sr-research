"""Meaningful synthetic correctness/resource checks; never launches real training."""
import argparse
import ast
import copy
import io
import json
from pathlib import Path
from typing import Tuple
import torch
from torch import nn
from einops import rearrange
from run import existing, validate_config
from sgla_model import build_model, build_sg6
from attention import Adapter, SGLA

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2); torch.manual_seed(10)
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((HERE/'configs/SG6-context.json').read_text())
    validate_config(cfg)
    arch,_=existing.fs_runtime.components()
    from deploy import fused_copy
    # Exact source class extraction without executing unrelated imports/plugins.
    tree=ast.parse((HERE/'vendor/reference.py').read_text(encoding='utf8'))
    names={'RoPE','GateLinearAttentionNoSilu','theta_shift','rotate_every_two'}
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    scope=dict(torch=torch,nn=nn,F=torch.nn.functional,Tuple=Tuple,rearrange=rearrange)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'reference.py','exec'),scope)
    equivalent=[]
    for shape in [(1,52,1,1),(1,52,1,7),(2,52,13,17),(1,52,7,31)]:
        m=SGLA(context=True).double().to(args.device)
        with torch.no_grad(): m.proj.weight.normal_(0,.03); m.proj.bias.normal_(0,.01)
        ref=scope['GateLinearAttentionNoSilu'](52,1).double().to(args.device)
        ref.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith(('angle','qk_context.'))},strict=True)
        rope=scope['RoPE'](52,1).double().to(args.device)
        x=torch.randn(shape,dtype=torch.float64,device=args.device,requires_grad=True)
        xr=x.detach().clone().requires_grad_()
        y,yr=m(x),ref(xr,*rope(shape[-2:]))
        torch.testing.assert_close(y,yr,atol=1e-12,rtol=1e-12)
        y.square().mean().backward(); yr.square().mean().backward()
        torch.testing.assert_close(x.grad,xr.grad,atol=1e-12,rtol=1e-12)
        for name,p in ref.named_parameters():
            torch.testing.assert_close(dict(m.named_parameters())[name].grad,p.grad,atol=1e-12,rtol=1e-12)
        equivalent.append(dict(shape=shape,absmax=float((y-yr).abs().max().detach())))
    torch.manual_seed(10); base=build_sg6(arch,cfg).to(args.device)
    # eval_conv is an upstream nontrainable cache refreshed by forward;
    # compare initialization before any reference forward mutates that cache.
    base_initial={k:v.clone() for k,v in base.state_dict().items()}
    cpu_rng=torch.get_rng_state().clone(); gpu_rng=torch.cuda.get_rng_state_all()
    models={}
    for mode in ('off','original','context','local'):
        torch.manual_seed(10)
        settings=dict(cfg,sgla=dict(cfg['sgla'],mode=mode))
        models[mode]=build_model(arch,settings).to(args.device)
        assert torch.equal(cpu_rng,torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(gpu_rng,torch.cuda.get_rng_state_all()))
        m=models[mode]
        for k,v in base_initial.items():
            mapped=k.replace('block_4.','block_4.block.') if mode!='off' else k
            assert torch.equal(v,m.state_dict()[mapped]),k
        for shape in ((1,3,13,17),(2,3,7,11)):
            x=torch.rand(shape,device=args.device)
            for train in (True,False):
                m.train(train); base.train(train)
                with torch.no_grad(): assert torch.equal(m(x),base(x))
    # Identity-initialized context and original share every common attention tensor.
    orig=models['original'].block_4.adapter.state_dict()
    assert all(torch.equal(v,models['context'].block_4.adapter.state_dict()[k]) for k,v in orig.items())
    m=models['context'].train()
    params=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=cfg['lr'],betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'])
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(params)
    x=torch.rand(2,3,13,17,device=args.device); target=torch.rand(2,3,52,68,device=args.device)
    before={k:v.detach().clone() for k,v in m.block_4.adapter.named_parameters()}
    gradients=[]
    for step in range(4):
        opt.zero_grad(); loss=(m(x)-target).abs().mean(); loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        g={k:float(v.grad.abs().sum()) for k,v in m.block_4.adapter.named_parameters()}
        assert g['core.proj.weight']>0
        assert (g['core.qk_context.weight']==0) if step==0 else (g['core.qk_context.weight']>0)
        gradients.append(g); opt.step()
    assert all(not torch.equal(v,before[k]) for k,v in m.block_4.adapter.named_parameters())
    f=io.BytesIO(); torch.save(m.state_dict(),f); f.seek(0)
    restored=build_model(arch,cfg).to(args.device); restored.load_state_dict(torch.load(f,weights_only=True),strict=True)
    m.eval(); restored.eval()
    with torch.no_grad(): assert torch.equal(m(x),restored(x))
    # Deployment conversion checked in FP64 to isolate accumulation rounding.
    double=copy.deepcopy(m).double(); fused=fused_copy(double,arch.Conv3XC)
    with torch.no_grad():
        delta=float((double(x.double())-fused(x.double())).abs().max())
        assert delta<1e-9
    # Nonzero output projection: context must actually affect output, even far away.
    a=Adapter('context').double().to(args.device)
    with torch.no_grad(): a.core.proj.weight.normal_(0,.03)
    u=torch.randn(1,52,13,17,dtype=torch.float64,device=args.device,requires_grad=True)
    a(u)[0,0,0,0].backward()
    remote=float(u.grad[0,:,12,16].abs().sum()); assert remote>0
    report=dict(status='PASS',device=args.device,torch=torch.__version__,reference_equivalence=equivalent,base_initialization_identical=True,rng_identical=True,zero_projection_identity_train_eval=True,gradient_steps=gradients,all_adapter_parameters_updated=True,strict_reload=True,deployment_fp64_absmax=delta,remote_derivative=remote,deployed_parameters={mode:sum(p.numel() for p in fused_copy(v,arch.Conv3XC).parameters()) for mode,v in models.items()})
    path=args.output or HERE/('engineering-'+args.device+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf8')
    print('PASS',args.device,'parameters',report['deployed_parameters'])

if __name__=='__main__': main()
