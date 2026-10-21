"""Small diagnostic checks only. Reference backend is NOT a training backend."""
import argparse
import copy
import io
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from adapter import ASSMAdapter, gather_tokens, policy
from scan import reference_scan
from model import build_model, build_sg6
import runtime
from deploy import fused_copy
from vendor import reference


def check_reference(device):
    actual = ASSMAdapter('reference').to(device)
    official = reference.ASSM(52,8,(3,5),num_tokens=64,inner_rank=32,mlp_ratio=1.).to(device)
    state = actual.core.state_dict()
    official.load_state_dict({k:v for k,v in state.items() if not k.startswith('embeddingA.')},strict=True)
    token = copy.deepcopy(actual.core.embeddingA)
    official.selectiveScan.selective_scan = reference_scan
    x = torch.randn(2,15,52,device=device,requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    # Verify the declared SG6 adaptations against retained official equations.
    # This is intentionally not called equivalence to the untouched official ASSM.
    indices = []
    original_neighbor, original_sort = reference.semantic_neighbor, torch.sort
    def neighbor(value,index):
        indices.append(index.clone())
        return original_neighbor(value,index)
    def stable_sort(*args,**kwargs):
        kwargs['stable']=True
        return original_sort(*args,**kwargs)
    def align(module,inputs):
        value,prompt=inputs
        return value,gather_tokens(prompt,indices[0])
    handle=official.selectiveScan.register_forward_pre_hook(align)
    try:
        with patch.object(reference.F,'gumbel_softmax',lambda v,**kw:policy(v)), \
             patch.object(reference,'semantic_neighbor',neighbor), \
             patch.object(reference.torch,'sort',stable_sort):
            expected=official(xr,(3,5),token)
    finally:
        handle.remove()
    observed=actual.core(x,(3,5))
    torch.testing.assert_close(observed,expected,atol=2e-6,rtol=1e-5)
    target=torch.randn_like(expected)
    ((observed-target)**2).mean().backward()
    ((expected-target)**2).mean().backward()
    torch.testing.assert_close(x.grad,xr.grad,atol=1e-7,rtol=1e-4)
    max_gradient=0.
    expected_parameters=dict(official.named_parameters())
    expected_parameters['embeddingA.weight']=token.weight
    for name,p in actual.core.named_parameters():
        q=expected_parameters[name]
        assert p.grad is not None and q.grad is not None,name
        torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=2e-3)
        max_gradient=max(max_gradient,float((p.grad-q.grad).abs().max()))
    return dict(output_max_error=float((observed-expected).abs().max().detach()),
                parameter_gradient_max_error=max_gradient,
                scope='official ASSM equations plus explicitly adapted routing/permutation')


def check_alignment(device):
    core=ASSMAdapter('reference').to(device).core
    tokens=torch.randn(2,15,52,device=device)
    chosen=policy(core.route(tokens))
    index=chosen.detach().argmax(-1).argsort(dim=-1,stable=True)
    prompt=chosen @ (core.embeddingB.weight @ core.embeddingA.weight)
    v=core.in_proj(tokens.transpose(1,2).reshape(2,52,3,5))
    v=v*core.CPE(v).sigmoid()
    expected_value=gather_tokens(v.flatten(2).transpose(1,2),index)
    expected_prompt=gather_tokens(prompt,index)
    captures=[]
    handle=core.selectiveScan.register_forward_pre_hook(lambda m,i: captures.append(i))
    try:
        core(tokens,(3,5))
    finally:
        handle.remove()
    assert torch.equal(captures[0][0],expected_value)
    assert torch.equal(captures[0][1],expected_prompt)
    # Explicit labelled toy makes an identity/class-collapsed permutation insufficient.
    order=torch.tensor([[2,0,3,1]],device=device)
    labels=torch.arange(4,device=device)[None,:,None]
    assert torch.equal(gather_tokens(gather_tokens(labels,order),order.argsort(-1)),labels)
    assert not torch.equal(gather_tokens(labels,order),labels)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',required=True)
    parser.add_argument('--backend',choices=('reference','cuda'),default='reference')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(10)
    equivalence=check_reference(args.device)
    check_alignment(args.device)
    cfg=json.loads((Path(__file__).parent/'configs/M2-tail-rms-lr2e4.json').read_text())
    cfg.update(engineering_only=True)
    cfg['assm_settings']['backend']=args.backend
    arch,_=runtime.components()
    torch.manual_seed(10)
    baseline=build_sg6(arch,cfg).to(args.device)
    saved_rng=torch.get_rng_state().clone();saved_cuda=torch.cuda.get_rng_state_all()
    torch.manual_seed(10)
    model=build_model(arch,cfg).to(args.device)
    assert torch.equal(saved_rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(saved_cuda,torch.cuda.get_rng_state_all()))
    def mapped(key):
        return 'conv_cat.fusion.'+key[len('conv_cat.'):] if key.startswith('conv_cat.') else key
    assert all(torch.equal(v,model.state_dict()[mapped(k)]) for k,v in baseline.state_dict().items())
    shapes=((1,3,1,1),(2,3,3,5),(1,3,3,43),(2,3,13,17))
    for shape in shapes:
        x=torch.rand(*shape,device=args.device);snapshot=x.clone()
        for training in (True,False):
            model.train(training)
            with torch.no_grad(): y=model(x)
            assert y.shape==(*shape[:2],shape[2]*4,shape[3]*4) and torch.isfinite(y).all()
        assert torch.equal(x,snapshot)
    x=torch.rand(2,3,7,9,device=args.device)
    model.eval();baseline.eval()
    with torch.no_grad(),patch.object(model.conv_cat,'adapter',nn.Identity()):
        assert torch.equal(model(x),baseline(x))
    base_probe=copy.deepcopy(baseline);probe=copy.deepcopy(model)
    with torch.no_grad():
        for name,p in base_probe.named_parameters():
            if name.startswith(('image_residual.','image_gate.')) or '.spatial_gate.' in name:
                p.normal_(0,.001)
        state=probe.state_dict()
        for k,v in base_probe.state_dict().items():state[mapped(k)]=v
        probe.load_state_dict(state,strict=True)
        with patch.object(probe.conv_cat,'adapter',nn.Identity()):
            assert torch.equal(base_probe(x),probe(x))
    del probe,base_probe
    before_cpu=torch.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state_all()
    with torch.no_grad():
        first=model(x);second=model(x)
        separate=torch.cat([model(v[None]) for v in x])
        changed=x.clone();changed[1]=1-changed[1]
        isolated=model(changed)
        base_batch=baseline(x)
        base_separate=torch.cat([baseline(v[None]) for v in x])
    assert torch.equal(first,second)
    # Keep batch shape fixed to distinguish cross-image leakage from common
    # cuDNN/TF32 algorithm changes caused by changing the batch shape.
    assert torch.equal(first[0],isolated[0])
    batch_shape_errors=dict(SG6=float((base_batch-base_separate).abs().max()),
                            SG6_ASSM=float((first-separate).abs().max()))
    assert torch.equal(before_cpu,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(before_cuda,torch.cuda.get_rng_state_all()))
    params=list(model.parameters())
    optimizer=torch.optim.Adam(params,lr=2e-4)
    ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(params)
    before={k:p.clone() for k,p in model.conv_cat.adapter.named_parameters()}
    old_depth={i:getattr(getattr(model,f'block_{i}'),'spatial_gate').depth.weight.clone() for i in range(1,7)}
    target=torch.rand(2,3,28,36,device=args.device)
    losses=[];model.train()
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.l1_loss(model(x),target);loss.backward()
        assert torch.isfinite(loss)
        for name,p in model.conv_cat.adapter.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0,name
        optimizer.step();losses.append(float(loss.detach()))
        if step==1:
            checkpoint=io.BytesIO()
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),checkpoint)
    assert all(not torch.equal(p,dict(model.conv_cat.adapter.named_parameters())[k]) for k,p in before.items())
    for i,p in old_depth.items():
        block=getattr(model,f'block_{i}')
        assert not torch.equal(p,block.spatial_gate.depth.weight),i
    checkpoint.seek(0);state=torch.load(checkpoint,map_location=args.device,weights_only=True)
    replay=build_model(arch,cfg).to(args.device).train();replay.load_state_dict(state['model'],strict=True)
    opt=torch.optim.Adam(replay.parameters(),lr=2e-4);opt.load_state_dict(state['optimizer'])
    for _ in range(2):
        opt.zero_grad(set_to_none=True);torch.nn.functional.l1_loss(replay(x),target).backward();opt.step()
    assert all(torch.equal(v,replay.state_dict()[k]) for k,v in model.state_dict().items())
    model.eval();deployed=fused_copy(model,arch.Conv3XC)
    with torch.no_grad():torch.testing.assert_close(model(x),deployed(x),atol=2e-6,rtol=1e-5)
    report=dict(status='PASS',device=args.device,scan_backend=args.backend,
                official_core_adaptation_check=equivalence,prompt_value_alignment=True,
                retained_sg6_state_and_rng_exact=True,nonzero_rgb_bypass_preserved=True,
                all_new_parameter_tensors_updated=len(before),all_six_spatial_branches_started=True,
                optimizer_coverage=True,save_resume_exact=True,eval_repeat_exact=True,
                forward_preserves_rng=True,same_shape_batch_isolation_exact=True,
                full_batch_shape_change_diagnostic=batch_shape_errors,new_parameters=model.assm_extra_parameters,
                deployed_parameters=sum(p.numel() for p in deployed.parameters()),synthetic_loss=losses,
                scope='Small engineering diagnostic only; CUDA extension and PSNR remain unverified')
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
