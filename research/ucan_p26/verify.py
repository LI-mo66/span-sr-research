"""Placement/fusion oracle, full LKSA equivalence and exact Adam reload checks."""
import argparse,ast,copy,hashlib,io,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch import nn
from run import HERE,existing
from candidate import build_model,build_sg6,expected_extra

def recursive_equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:recursive_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):recursive_equal(x,y)
    else:assert a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    arch,_=existing.fs_runtime.components()
    from deploy import fused_copy
    cfg=json.loads((HERE/'config.json').read_text());s=cfg['ucan_sweep']
    vendor=json.loads((HERE/'vendor/manifest.json').read_text())
    for n,v in vendor['files'].items():assert hashlib.sha256((HERE/'vendor'/n).read_bytes()).hexdigest()==v
    tree=ast.parse((HERE/'vendor/ucan_arch.py').read_text(encoding='utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LKSA')
    scope={'torch':torch,'nn':nn};exec(compile(ast.Module(body=[node],type_ignores=[]),'locked-LKSA','exec'),scope)
    from lksa import LKSA
    core=LKSA(52).double().to(a.device);ref=scope['LKSA'](52).double().to(a.device)
    ref.load_state_dict(core.state_dict(),strict=True)
    x=torch.randn(2,52,13,17,device=a.device,dtype=torch.float64,requires_grad=True)
    xr=x.detach().clone().requires_grad_()
    y=core(x);yr=ref(xr);torch.testing.assert_close(y,yr,atol=0,rtol=0)
    y.square().mean().backward();yr.square().mean().backward()
    torch.testing.assert_close(x.grad,xr.grad,atol=0,rtol=0)
    for pp,qq in zip(core.parameters(),ref.parameters()):torch.testing.assert_close(pp.grad,qq.grad,atol=0,rtol=0)
    del core,ref,x,xr,y,yr
    torch.manual_seed(10);model=build_model(arch,cfg)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(10);base=build_sg6(arch,cfg)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in base.state_dict().items():assert torch.equal(v,model.state_dict()[k]),k
    model=model.double().to(a.device);base=base.double().to(a.device)
    errors=[]
    for b,h,w in ((1,1,1),(2,7,11),(1,13,9)):
        image=torch.rand(b,3,h,w,device=a.device,dtype=torch.float64)
        with torch.no_grad():
            model.eval();base.eval();out=model(image);old=base(image)
        assert out.shape==(b,3,h*4,w*4);torch.testing.assert_close(out,old,atol=0,rtol=0)
    # Make all insertions nontrivial before checking placement against original
    # SG6 forward with tuple-preserving hooks, including both original taps.
    with torch.no_grad():
        for name,pv in model.named_parameters():
            if name.startswith(('context_','lksa_')) and ('proj.' in name or 'project.' in name):pv.normal_(0,.01)
    handles=[]
    def oracle(pos):
        def hook(module,args,out):
            z,first,gate=out
            if pos in s['positions']:z,_=getattr(model,f'context_{pos}')(z)
            if pos==s['lksa_position']:z=model.lksa_5(z)
            return z,first,gate
        return hook
    for pos in range(1,7):handles.append(getattr(base,f'block_{pos}').register_forward_hook(oracle(pos)))
    with torch.no_grad():
        for b,h,w in ((2,7,11),(1,13,9)):
            image=torch.rand(b,3,h,w,device=a.device,dtype=torch.float64)
            torch.testing.assert_close(model(image),base(image),atol=1e-11,rtol=1e-11)
    for handle in handles:handle.remove()
    image=torch.rand(1,3,7,9,device=a.device,dtype=torch.float64)
    model.train();yt=model(image)
    model.eval();ye=model(image)
    torch.testing.assert_close(yt,ye,atol=1e-9,rtol=1e-9)
    fused=fused_copy(model,arch.Conv3XC).eval()
    with torch.no_grad():torch.testing.assert_close(fused(image),ye,atol=1e-9,rtol=1e-9)
    assert sum(p.numel() for p in fused.parameters())==535772+expected_extra(cfg)
    # Actual zero-projection startup: internal paths wake after projection moves.
    torch.manual_seed(10);model=build_model(arch,cfg).double().to(a.device).train()
    opt=torch.optim.Adam(model.parameters(),lr=2e-4)
    schedule=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[2,4],gamma=.5)
    target=torch.rand(1,3,28,36,device=a.device,dtype=torch.float64)
    startup=[]
    initial={n:pv.detach().clone() for n,pv in model.named_parameters() if n.startswith(('context_','lksa_'))}
    for step in range(3):
        opt.zero_grad(set_to_none=True);loss=(model(image)-target).abs().mean();loss.backward()
        norms={n:float(pv.grad.norm()) if pv.grad is not None else None for n,pv in model.named_parameters() if n.startswith(('context_','lksa_'))}
        assert all(v is not None and torch.isfinite(torch.tensor(v)) for v in norms.values())
        startup.append(norms);opt.step();schedule.step()
    assert all(v>0 for v in startup[-1].values()),startup[-1]
    assert all(not torch.equal(pv,initial[n]) for n,pv in model.named_parameters() if n in initial)
    stream=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict()},stream)
    stream.seek(0)
    # This buffer was generated immediately above by this process, not supplied externally.
    saved=torch.load(stream,map_location='cpu',weights_only=False)
    other=build_model(arch,cfg).double().to(a.device).train();other.load_state_dict(saved['model'],strict=True)
    opt2=torch.optim.Adam(other.parameters(),lr=2e-4);opt2.load_state_dict(saved['optimizer'])
    sch2=torch.optim.lr_scheduler.MultiStepLR(opt2,milestones=[2,4],gamma=.5);sch2.load_state_dict(saved['scheduler'])
    for m,o,sc in ((model,opt,schedule),(other,opt2,sch2)):
        o.zero_grad(set_to_none=True);(m(image)-target).abs().mean().backward();o.step();sc.step()
    recursive_equal(model.state_dict(),other.state_dict());recursive_equal(opt.state_dict(),opt2.state_dict());recursive_equal(schedule.state_dict(),sch2.state_dict())
    report={'status':'PASS','device':a.device,'arm':s,'deployed_parameters':535772+expected_extra(cfg),
        'source_LKSA_output_input_parameter_gradient_equivalence':True,'zero_adapter_identity':True,
        'original_forward_hook_oracle':True,'backbone_and_RNG_preserved':True,
        'odd_rectangular_batch_shapes':True,'fp64_train_eval_and_deploy':True,
        'exact_Adam_scheduler_continuation':True,'startup_gradient_norms':startup,
        'scope':'synthetic engineering only; no real-data gain claim'}
    from pathlib import Path
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print('PASS engineering',s['arm'],flush=True)

if __name__=='__main__':main()
