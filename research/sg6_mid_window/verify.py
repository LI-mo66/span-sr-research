"""Interface, equations, distribution, optimizer and exact-resume engineering gates."""
import argparse
import copy
import io
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from adapter import WindowAttention, PartialInteraction
from model import build_model, build_sg6, COUNTS
import runtime
from deploy import fused_copy


def rms(value):
    return float(value.detach().double().square().mean().sqrt())


def stats(value):
    value=value.detach().double()
    return dict(mean=float(value.mean()),std=float(value.std(unbiased=False)),rms=rms(value),max_abs=float(value.abs().max()))


def backbone_gradient_comparison(loss, other, parameters):
    actual=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    bypass=torch.autograd.grad(other,parameters,allow_unused=True)
    dot=aa=bb=delta=0.
    for p,g,h in zip(parameters,actual,bypass):
        g=torch.zeros_like(p) if g is None else g
        h=torch.zeros_like(p) if h is None else h
        g=g.detach().double();h=h.detach().double()
        if not torch.isfinite(g).all() or not torch.isfinite(h).all():
            raise FloatingPointError('Nonfinite backbone gradient')
        dot+=float((g*h).sum());aa+=float(g.square().sum());bb+=float(h.square().sum())
        delta+=float((g-h).square().sum())
    return dict(cosine=dot/max((aa*bb)**.5,1e-30),relative_change=(delta/max(bb,1e-30))**.5,
                scope='Same current weights and batch, correction on/off; diagnostic only, no same-direction requirement or PSNR inference')


def independent_window(core, x):
    # Enumerate only real positions: no padding or vectorized partition helpers.
    b,c,h,w=x.shape
    m,heads=core.window,core.heads
    strips=[]
    for y in range(0,h,m):
        tiles=[]
        for z in range(0,w,m):
            tile=x[:,:,y:y+m,z:z+m]
            hh,ww=tile.shape[-2:]
            token=tile.permute(0,2,3,1).reshape(b,hh*ww,c)
            q,k,v=core.qkv(token).reshape(b,hh*ww,3,heads,c//heads).permute(2,0,3,1,4).unbind(0)
            cy,cx=torch.meshgrid(torch.arange(hh,device=x.device),torch.arange(ww,device=x.device),indexing='ij')
            dy=cy.flatten()[None,:]-cy.flatten()[:,None]+m-1
            dx=cx.flatten()[None,:]-cx.flatten()[:,None]+m-1
            bias=core.relative_bias[:,dy*(2*m-1)+dx]
            score=(q@k.transpose(-2,-1))/((c//heads)**.5)+bias[None]
            out=(score.softmax(-1)@v).transpose(1,2).reshape(b,hh*ww,c)
            tiles.append(core.project(out).reshape(b,hh,ww,c).permute(0,3,1,2))
        strips.append(torch.cat(tiles,dim=3))
    return torch.cat(strips,dim=2)


def check_equations():
    torch.manual_seed(101)
    core=WindowAttention(chunk=2).double()
    with torch.no_grad():core.relative_bias.normal_(std=.03)
    x=torch.randn(2,26,17,19,dtype=torch.float64,requires_grad=True)
    xr=x.detach().clone().requires_grad_()
    reference=copy.deepcopy(core)
    actual,expected=core(x),independent_window(reference,xr)
    torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-10)
    target=torch.randn_like(actual)
    (actual*target).mean().backward();(expected*target).mean().backward()
    torch.testing.assert_close(x.grad,xr.grad,atol=1e-11,rtol=1e-10)
    error=0.
    for (name,p),(other,q) in zip(core.named_parameters(),reference.named_parameters()):
        assert name==other and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,atol=1e-11,rtol=1e-10)
        error=max(error,float((p.grad-q.grad).abs().max()))
    return dict(fp64_output_max_error=float((actual-expected).detach().abs().max()),fp64_gradient_max_error=error,
                reference='Independent real-position window enumeration, including RPE; tests padded key exclusion')


def gate(device, kind):
    cfg=json.loads((Path(__file__).parent/'configs'/f'{kind}.json').read_text())
    arch,_=runtime.components()
    torch.manual_seed(10)
    baseline=build_sg6(arch,cfg).to(device)
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state_all()
    torch.manual_seed(10)
    model=build_model(arch,cfg).to(device)
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    state=model.state_dict()
    for key,value in baseline.state_dict().items():
        mapped='block_3.original.'+key[len('block_3.'):] if key.startswith('block_3.') else key
        assert torch.equal(value,state[mapped]),key
    a=model.block_3.interaction
    norms=[name for name,m in model.named_modules() if isinstance(m,nn.LayerNorm)]
    assert norms==['block_3.interaction.norm'],norms
    model.eval();baseline.eval()
    shapes=[]
    for h,w in ((1,1),(1,17),(3,5),(16,16),(17,19),(31,33)):
        x=torch.randn(2,52,h,w,device=device)
        untouched=x.clone()
        with torch.no_grad():
            y=a(x);again=a(x)
            assert y.shape==x.shape and torch.isfinite(y).all()
            assert torch.equal(y,again) and torch.equal(x,untouched)
            assert torch.equal(y[:,26:],x[:,26:])
            changed=x.clone();changed[1].normal_()
            assert torch.equal(a(changed)[0],y[0])
        shapes.append([2,52,h,w])
    # Noncontiguous feature layout must retain spatial meanings.
    x=torch.randn(2,52,7,11,device=device).transpose(-1,-2)
    with torch.no_grad():torch.testing.assert_close(a(x),a(x.contiguous()),atol=1e-7,rtol=1e-6)
    x=torch.rand(2,3,13,17,device=device)
    # Exercise a nonzero GB bypass, not only its zero initialization.
    with torch.no_grad():
        baseline.image_residual.weight.fill_(.001)
        model.image_residual.weight.copy_(baseline.image_residual.weight)
        scale=a.scale.clone();a.scale.zero_()
        expected=baseline(x);actual=model(x)
        assert torch.equal(actual,expected),'Zero correction must restore complete SG6 exactly'
        a.scale.copy_(scale)
        baseline.image_residual.weight.zero_();model.image_residual.weight.zero_()
    rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state_all()
    with torch.no_grad():
        y=model(x);assert torch.equal(y,model(x))
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(t,u) for t,u in zip(crng,torch.cuda.get_rng_state_all()))
    model.train()
    parameters=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(parameters,lr=2e-4,betas=(.9,.999),eps=1e-8)
    optimizer_ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(optimizer_ids)==len(set(optimizer_ids)) and set(optimizer_ids)=={id(p) for p in parameters}
    target=torch.nn.functional.interpolate(x,scale_factor=4,mode='bicubic',align_corners=False).clamp(0,1)
    depth={i:getattr(getattr(model,f'block_{i}').original if i==3 else getattr(model,f'block_{i}'),'spatial_gate').depth.weight.detach().clone() for i in range(1,7)}
    initial={k:p.detach().clone() for k,p in a.named_parameters()}
    backbone=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('block_3.interaction.')]
    rows=[]
    for step in range(4):
        captures={}
        first=a.register_forward_pre_hook(lambda m,v:captures.update(F=v[0].detach().clone()))
        last=a.register_forward_hook(lambda m,v,o:captures.update(Y=o.detach().clone()))
        opt.zero_grad(set_to_none=True)
        try:prediction=model(x)
        finally:first.remove();last.remove()
        loss=torch.nn.functional.l1_loss(prediction,target)
        assert torch.isfinite(loss)
        comparison=None
        if step in (0,3):
            handle=a.register_forward_hook(lambda module,inputs,output:inputs[0])
            try:other=torch.nn.functional.l1_loss(model(x),target)
            finally:handle.remove()
            comparison=backbone_gradient_comparison(loss,other,backbone)
        loss.backward()
        before={k:p.detach().clone() for k,p in a.named_parameters()}
        gradients={}
        for name,p in a.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0,name
            gradients[name]=rms(p.grad)
        opt.step()
        updates={name:rms(p-before[name]) for name,p in a.named_parameters()}
        assert all(v>0 for v in updates.values()),updates
        f,y=captures['F'],captures['Y']
        ratio=rms(y-f)/max(rms(f),1e-12)
        assert ratio<.05,'Correction exceeds local 5% engineering guard (not a PSNR criterion)'
        rows.append(dict(step=step+1,loss=float(loss.detach()),feature=stats(f),correction=stats(y-f),
                         block4_input=stats(y),correction_relative_rms=ratio,gradient_rms=gradients,
                         actual_adam_update_rms=updates,backbone_gradient_comparison=comparison))
        if step==1:
            saved=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),saved)
    assert all(not torch.equal(v,dict(a.named_parameters())[k]) for k,v in initial.items())
    for i,old in depth.items():
        block=model.block_3.original if i==3 else getattr(model,f'block_{i}')
        assert not torch.equal(old,block.spatial_gate.depth.weight),i
    saved.seek(0);checkpoint=torch.load(saved,map_location=device,weights_only=True)
    replay=build_model(arch,cfg).to(device).train()
    replay.load_state_dict(checkpoint['model'],strict=True)
    replay_opt=torch.optim.Adam((p for p in replay.parameters() if p.requires_grad),lr=2e-4)
    replay_opt.load_state_dict(checkpoint['optimizer'])
    for _ in range(2):
        replay_opt.zero_grad(set_to_none=True)
        torch.nn.functional.l1_loss(replay(x),target).backward();replay_opt.step()
    assert all(torch.equal(value,replay.state_dict()[key]) for key,value in model.state_dict().items()),'Strict resume differs'
    model.eval();deployed=fused_copy(model,arch.Conv3XC)
    with torch.no_grad():torch.testing.assert_close(model(x),deployed(x),atol=2e-6,rtol=1e-5)
    count=sum(p.numel() for p in deployed.parameters())
    assert count==535772+COUNTS[kind]
    return dict(status='PASS',variant=kind,new_parameters=COUNTS[kind],deployed_parameters=count,
                shapes=shapes,normalization_layers=norms,exact_sg6_identity=True,sg6_rng_and_parameters_preserved=True,
                nonzero_gb_preserved=True,optimizer_coverage=True,all_six_spatial_branches_started=True,
                save_resume_exact=True,repeat_exact=True,image_isolation_exact=True,steps=rows,
                scope='Disposable synthetic engineering checks; no PSNR or long-run stability claim')


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    report=dict(status='CHECKING',torch=torch.__version__,device=args.device)
    report['equations']=check_equations()
    report['variants']={}
    for kind in ('A','C'):
        report['variants'][kind]=gate(args.device,kind)
        print('PASS engineering:',kind,flush=True)
    report['status']='PASS'
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='variants'},indent=2))


if __name__=='__main__':main()
