"""Exact initialization, original forward equivalence, finite updates and resume."""
import copy
import io
import json
import sys
from pathlib import Path
import torch
from torch import nn
from fusion import HERE,ROOT,shared,build_model,build_sg6,original_backbone_state,load,window,attention,lksa
from deploy import fused_copy

def main(device):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    arch,_=shared.components()
    cfg=json.loads((ROOT/'logs/fusion-WU/frozen/config.json').read_text())
    # Load fixed source W and U factories for independent comparison.
    sys.modules['attention']=attention;sys.modules['lksa']=lksa
    sys.modules['adapter']=window
    u=load('fusion_original_u',HERE/'u_candidate.py')
    w=load('fusion_original_w',HERE/'w_model.py')
    uc=json.loads((HERE/'config.json').read_text())
    wc=json.loads((HERE/'w_config.json').read_text())
    def construct(factory,c):
        torch.manual_seed(10)
        model=factory(arch,c)
        return model.to(device)
    baseline=construct(build_sg6,cfg)
    model=construct(build_model,cfg)
    expected=baseline.state_dict();actual=original_backbone_state(model)
    assert expected.keys()==actual.keys()
    assert all(torch.equal(v,actual[k]) for k,v in expected.items())
    assert sum(isinstance(m,nn.LayerNorm) for m in model.modules())==3
    directw=construct(w.build_model,wc)
    directu=construct(u.build_model,uc)
    assert sum(p.numel() for p in fused_copy(model,arch.Conv3XC).parameters())==569895
    assert sum(p.numel() for p in fused_copy(directw,arch.Conv3XC).parameters())==540580
    assert sum(p.numel() for p in fused_copy(directu,arch.Conv3XC).parameters())==565087
    for pos in (2,4):
        aa=getattr(model,f'block_{pos}').adapter.state_dict()
        bb=getattr(directu,f'context_{pos}').state_dict()
        assert aa.keys()==bb.keys() and all(torch.equal(v,bb[k]) for k,v in aa.items())
    aa=model.block_3.adapter.spatial.state_dict();bb=directu.lksa_5.state_dict()
    assert aa.keys()==bb.keys() and all(torch.equal(v,bb[k]) for k,v in aa.items())
    aa=model.block_3.adapter.window.state_dict();bb=directw.block_3.interaction.state_dict()
    assert aa.keys()==bb.keys() and all(torch.equal(v,bb[k]) for k,v in aa.items())
    # Zero U projections -> exact W forward; includes nonzero GB image path.
    with torch.no_grad():
        for m in (model,directw,directu):
            m.image_residual.weight.fill_(.0001);m.image_residual.bias.fill_(.0002)
        for h,width in ((1,1),(3,5),(16,16),(17,19),(31,33)):
            x=torch.rand(1,3,h,width,device=device)
            result=model(x);target=directw(x)
            torch.testing.assert_close(result,target,rtol=0,atol=0)
            assert result.shape==(1,3,h*4,width*4) and torch.isfinite(result).all()
        # Learned-like nonzero U + bypass W -> independently implemented U forward.
        for pos in (2,4):
            module=getattr(model,f'block_{pos}').adapter
            module.core.proj.weight.normal_(std=.002)
            getattr(directu,f'context_{pos}').load_state_dict(module.state_dict())
        model.block_3.adapter.spatial.project.weight.normal_(std=.002)
        directu.lksa_5.load_state_dict(model.block_3.adapter.spatial.state_dict())
        model.block_3.adapter.window.scale.zero_()
        moved=directu.lksa_5
        directu.lksa_5=nn.Identity()
        def move_lksa(module,inputs,result):
            main,first,gate=result
            return moved(main),first,gate
        relocation=directu.block_3.register_forward_hook(move_lksa)
        x=torch.rand(2,3,17,19,device=device)
        torch.testing.assert_close(model(x),directu(x),rtol=0,atol=0)
        relocation.remove()
        # Both nontrivial W and U: original SG6 forward with explicit-order hooks.
        model.block_3.adapter.window.scale.fill_(.03)
        oracle=construct(build_sg6,cfg)
        oracle.load_state_dict(original_backbone_state(model),strict=True)
        handles=[]
        def correction(position):
            def hook(module,inputs,result):
                main,first,gate=result
                if position in (2,4):main=getattr(model,f'block_{position}').adapter(main)[0]
                elif cfg['fusion_order']=='F1':
                    main=model.block_3.adapter.spatial(model.block_3.adapter.window(main))
                else:main=model.block_3.adapter.window(model.block_3.adapter.spatial(main))
                return main,first,gate
            return hook
        for pos in (2,3,4):handles.append(getattr(oracle,f'block_{pos}').register_forward_hook(correction(pos)))
        torch.testing.assert_close(model(x),oracle(x),rtol=0,atol=0)
        for handle in handles:handle.remove()
        del oracle
    del baseline,directw,directu,model
    if device=='cuda':torch.cuda.empty_cache()
    model=construct(build_model,cfg)
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=2e-4,betas=(.9,.999),eps=1e-8)
    parameters={k:p for k,p in model.named_parameters() if p.requires_grad}
    assert len({id(p) for g in opt.param_groups for p in g['params']})==len(parameters)
    initial={k:p.detach().clone() for k,p in parameters.items() if '.adapter.' in k}
    statistics=[];handles=[]
    for pos in (2,3,4):
        def observer(module,inputs,result,pos=pos):
            if isinstance(result,tuple):result=result[0]
            x=inputs[0].detach();delta=result.detach()-x
            assert torch.isfinite(result).all()
            statistics.append(dict(position=pos,input_rms=float(x.square().mean().sqrt()),
                correction_rms=float(delta.square().mean().sqrt())))
        handles.append(getattr(model,f'block_{pos}').adapter.register_forward_hook(observer))
    x=torch.rand(1,3,16,20,device=device);target=torch.rand(1,3,64,80,device=device)
    saved=None
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.l1_loss(model(x),target)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        opt.step()
        if step==1:
            saved=copy.deepcopy((model.state_dict(),opt.state_dict()))
    updates={k:float((parameters[k].detach()-v).abs().max()) for k,v in initial.items()}
    for pos in (2,3,4):
        prefix=f'block_{pos}.adapter.'
        assert any(v>0 for k,v in updates.items() if k.startswith(prefix))
        assert any(v>0 for k,v in updates.items() if k.startswith(prefix) and '.core.' in k),prefix
    for handle in handles:handle.remove()
    final=copy.deepcopy(model.state_dict())
    # In-memory exact state reload and replay; shared trainer integration is separate.
    stream=io.BytesIO();torch.save(saved,stream);stream.seek(0)
    restored=torch.load(stream,map_location=device,weights_only=False)
    model.load_state_dict(restored[0],strict=True);opt.load_state_dict(restored[1])
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.l1_loss(model(x),target).backward();opt.step()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in final.items())
    report=dict(status='PASS',device=device,parameters=569895,normalization_LN=3,
        original_W_U_parameter_initialization_exact=True,backbone_initialization_exact=True,
        W_only_forward_exact=True,U_only_nonzero_forward_exact=True,
        four_step_actual_updates=updates,branch_statistics=statistics,
        exact_order_oracle=True,arm=cfg['fusion_order'],exact_save_reload_replay=True,scope='Engineering only; no PSNR or optimality claim')
    path=ROOT/f'logs/fusion-WU/verify-{device}.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    print('PASS:',path,flush=True)

if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'cpu')
