"""Engineering checks, not a PSNR experiment or proof of distribution matching."""
import argparse
import copy
import io
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch import nn
from bootstrap import ROOT,components
from strip_model import build_model,PartialStripDepth
from sg_model import build_model as sg6
from run import validate_config

HERE=Path(__file__).resolve().parent

def rms(x):return float(x.detach().double().square().mean().sqrt())

def oracle(layer,x):
    # Independent scalar-offset oracle. Includes horizontal bias and vertical zero padding.
    b,c,h,w=x.shape
    z=torch.zeros_like(x[:,:13])
    y=torch.zeros_like(z)
    pad=layer.kernel//2
    for col in range(w):
        z[:,:,:,col]=layer.horizontal.bias[None,:,None]
        for offset in range(layer.kernel):
            source=col+offset-pad
            if 0<=source<w:
                z[:,:,:,col]+=x[:,:13,:,source]*layer.horizontal.weight[:,0,0,offset][None,:,None]
    for row in range(h):
        y[:,:,row,:]=layer.vertical.bias[None,:,None]
        for offset in range(layer.kernel):
            source=row+offset-pad
            if 0<=source<h:
                y[:,:,row,:]+=z[:,:,source,:]*layer.vertical.weight[:,0,offset,0][None,:,None]
    return y

def inspect(model,x):
    data={}
    def capture(name):
        def hook(m,inp,out):data[name]=dict(rms=rms(out),mean=float(out.detach().mean()),absmax=float(out.detach().abs().max()))
        return hook
    handles=[]
    branch=model.block_3.spatial_gate
    for name,module in [('F',model.block_3.c3_r),('depth',branch.depth),('spatial',branch),('local',model.block_3.local_gate)]:
        handles.append(module.register_forward_hook(capture(name)))
    try:
        with torch.no_grad():model(x)
    finally:
        for h in handles:h.remove()
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--kernel',type=int,choices=[15,31],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    # Keep normal archived flags in production; FP64 validates algebra independently of TF32.
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((HERE/f'configs/K{args.kernel}.json').read_text())
    validate_config(cfg)
    arch,_=components()
    torch.manual_seed(10)
    baseline=sg6(arch,cfg)
    rng=torch.get_rng_state().clone()
    gpu_rng=torch.cuda.get_rng_state_all()
    torch.manual_seed(10)
    model=build_model(arch,cfg)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(gpu_rng,torch.cuda.get_rng_state_all()))
    expected=13*(2*args.kernel-8)
    before=baseline.state_dict()
    assert all(torch.equal(v,before[n]) for n,v in model.state_dict().items() if 'block_3.spatial_gate.depth.' not in n)
    depth=model.block_3.spatial_gate.depth
    assert torch.equal(depth.local.weight,baseline.block_3.spatial_gate.depth.weight[13:])
    assert torch.equal(depth.local.bias,baseline.block_3.spatial_gate.depth.bias[13:])
    outer=depth.vertical.weight[:,0,:,0].norm(dim=1)*depth.horizontal.weight[:,0,0,:].norm(dim=1)
    torch.testing.assert_close(outer,baseline.block_3.spatial_gate.depth.weight[:13].flatten(1).norm(dim=1),rtol=1e-6,atol=1e-7)
    small=copy.deepcopy(depth).double()
    with torch.no_grad():small.horizontal.bias.normal_(0,.1)
    for shape in [(1,52,1,1),(2,52,3,5),(1,52,1,7),(1,52,7,1)]:
        x=torch.randn(*shape,dtype=torch.float64)
        torch.testing.assert_close(small(x)[:,:13],oracle(small,x),rtol=1e-10,atol=1e-10)
    model.to(args.device);baseline.to(args.device)
    for shape in [(1,3,1,1),(2,3,13,17),(1,3,17,11)]:
        x=torch.rand(*shape,device=args.device)
        for train in (True,False):
            model.train(train);baseline.train(train)
            with torch.no_grad():
                y=model(x)
                torch.testing.assert_close(y,baseline(x),rtol=0,atol=0)
                assert y.shape==(shape[0],3,shape[2]*4,shape[3]*4)
    norms=[n for n,m in model.named_modules() if isinstance(m,(nn.LayerNorm,nn.GroupNorm,nn.modules.batchnorm._BatchNorm,nn.modules.instancenorm._InstanceNorm))]
    assert not norms
    model.train()
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=2e-4,betas=(.9,.999),eps=1e-8)
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(params)
    x=torch.rand(2,3,13,17,device=args.device)
    target=torch.rand(2,3,52,68,device=args.device)
    measurements=[]
    for step in range(8):
        opt.zero_grad(set_to_none=True)
        previous={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        loss=(model(x)-target).abs().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        d=model.block_3.spatial_gate.depth
        if step==0:
            assert d.horizontal.weight.grad.abs().sum()==0
            assert d.vertical.weight.grad.abs().sum()==0
        elif step==1:
            assert d.horizontal.weight.grad.abs().sum()>0 and d.vertical.weight.grad.abs().sum()>0
        opt.step()
        updated={n:dict(parameter_rms=rms(previous[n]),gradient_rms=rms(p.grad),
            actual_update_rms=rms(p.detach()-previous[n]),relative_update=rms(p.detach()-previous[n])/max(rms(previous[n]),1e-30))
            for n,p in model.named_parameters() if p.requires_grad and ('block_3.spatial_gate.' in n)}
        measurements.append(dict(step=step+1,loss=float(loss.detach()),parameters=updated))
    for n in ('horizontal.weight','vertical.weight','local.weight'):
        assert any(r['parameters']['block_3.spatial_gate.depth.'+n]['actual_update_rms']>0 for r in measurements[1:])
    model.eval()
    dist=inspect(model,x)
    stream=io.BytesIO()
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),stream)
    stream.seek(0)
    state=torch.load(stream,map_location=args.device,weights_only=False)
    clone=build_model(arch,cfg).to(args.device)
    clone.load_state_dict(state['model'],strict=True)
    from monitor import Monitor
    observed=Monitor(args.output.with_suffix('.monitor.jsonl'));observed.bind(clone)
    co=observed.optimizer(torch.optim.Adam)((p for p in clone.parameters() if p.requires_grad),lr=2e-4)
    co.load_state_dict(state['optimizer'])
    for net,optimizer in ((model,opt),(clone,co)):
        net.train();optimizer.zero_grad(set_to_none=True)
        (net(x)-target).abs().mean().backward();optimizer.step()
    assert all(torch.equal(v,clone.state_dict()[n]) for n,v in model.state_dict().items())
    # FP64: same trained parameters, train Conv3XC paths vs refreshed fused eval paths.
    double=copy.deepcopy(model).double()
    with torch.no_grad():
        double.train();a=double(x.double())
        double.eval();b=double(x.double())
    torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-9)
    from deploy import fused_copy
    model.eval();deployed=fused_copy(model,arch.Conv3XC)
    with torch.no_grad():torch.testing.assert_close(model(x),deployed(x),rtol=1e-5,atol=1e-6)
    count=sum(p.numel() for p in deployed.parameters())
    assert count==535772+expected
    report=dict(status='PASS',kernel=args.kernel,device=args.device,torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,synthetic_only=True,
        parameters_deployed=count,parameters_new=expected,parameters_trainable=sum(p.numel() for p in params),
        checks=['shared initialization and RNG exact','initial SG6 mapping exact','39 local channels unchanged',
        'per-channel kernel energy matched','independent offset oracle with bias and boundaries',
        'tiny odd rectangular batch train/eval','no normalization added','optimizer no duplicate/missing',
        'zero-projection staged startup','actual Adam updates measured','strict model/optimizer resume',
        'observational hooks and Adam subclass preserve exact next update',
        'FP64 train/eval equivalence','FP32 deployment export agreement'],
        normalization_layers=norms,fp64_max_abs=float((a-b).abs().max()),
        post8step_feature_statistics=dist,updates=measurements,
        limitations='Synthetic engineering only. IID kernel energy matching is not real-feature distribution matching; no real-data PSNR or long-run optimization claim.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('updates',)},indent=2))

if __name__=='__main__':main()
