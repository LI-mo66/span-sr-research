"""Verify retained initialization, residual semantics and real FFN learning."""
import argparse
import io
import json
import os
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from run import atd_runtime as runtime
from model import build_model, AttentionAdapter
from deploy import fused_copy
import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=('cpu','cuda'))
    parser.add_argument('--config', default=str(Path(__file__).parent/'configs/A1-tail-lr2e4.json'))
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cfg = json.loads(Path(args.config).read_text())
    arch, _ = runtime.components()
    torch.manual_seed(cfg['seed'])
    baseline = arch.SPAN(**cfg['model']).to(args.device)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    torch.manual_seed(cfg['seed'])
    model = build_model(arch,cfg).to(args.device)
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in baseline.state_dict().items())
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(cfg['seed']+30001)
        original = AttentionAdapter().to(args.device)
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in model.content_tab.state_dict().items())
    assert not list(model.content_tab.buffers())
    assert 'dictionary' in model.content_tab.state_dict() and not any('means' in k for k in model.content_tab.state_dict())
    for shape in ((1,3,1,1),(2,3,3,5),(1,3,3,43),(2,3,13,17),(1,3,64,64),(1,3,65,71)):
        x = torch.rand(*shape,device=args.device)
        snapshot = x.clone()
        for training in (True,False):
            model.train(training)
            with torch.no_grad():
                y = model(x)
            assert y.shape == (shape[0],3,shape[2]*4,shape[3]*4) and torch.isfinite(y).all()
        assert torch.equal(x,snapshot)
    model.eval(); baseline.eval()
    x = torch.rand(2,3,13,17,device=args.device)
    with torch.no_grad(), patch.object(model,'content_tab',nn.Identity()):
        assert torch.equal(model(x),baseline(x))
    features = torch.randn(2,52,13,17,device=args.device)
    expected = original(features)
    torch.testing.assert_close(model.content_tab(features),expected,atol=0,rtol=0)
    trainable = list(model.parameters())
    opt = torch.optim.Adam(trainable,lr=cfg['lr'])
    ids = [id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(trainable)
    before = {k:p.clone() for k,p in model.content_tab.named_parameters()}
    target = torch.rand(2,3,52,68,device=args.device)
    losses = []
    branch_gradients = {'tdca':[], 'acmsa':[]}
    def observe(name):
        def hook(module, inputs, output):
            value = output[0] if name == 'tdca' else output
            if value.requires_grad:
                value.register_hook(lambda g: branch_gradients[name].append(float(g.abs().max())))
        return hook
    handles = [model.content_tab.tdca.register_forward_hook(observe('tdca')), model.content_tab.acmsa.register_forward_hook(observe('acmsa'))]
    model.train()
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(model(x),target)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in model.content_tab.parameters())
        opt.step(); losses.append(float(loss.detach()))
        if step == 1:
            saved = io.BytesIO()
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),saved)
    for handle in handles: handle.remove()
    assert all(v and min(v)>0 for v in branch_gradients.values())
    assert losses[-1] < losses[0]
    assert all(not torch.equal(v,dict(model.content_tab.named_parameters())[k]) for k,v in before.items())
    saved.seek(0); state = torch.load(saved,map_location=args.device,weights_only=True)
    replay = build_model(arch,cfg).to(args.device).train()
    replay.load_state_dict(state['model'],strict=True)
    replay_opt = torch.optim.Adam(replay.parameters(),lr=cfg['lr'])
    replay_opt.load_state_dict(state['optimizer'])
    for _ in range(2):
        replay_opt.zero_grad(set_to_none=True)
        torch.nn.functional.l1_loss(replay(x),target).backward(); replay_opt.step()
    assert all(torch.equal(v,replay.state_dict()[k]) for k,v in model.state_dict().items())
    model.eval(); replay.eval()
    deployed = fused_copy(model,arch.Conv3XC)
    with torch.no_grad():
        torch.testing.assert_close(model(x),replay(x),atol=0,rtol=0)
        torch.testing.assert_close(model(x),deployed(x),atol=1e-6,rtol=1e-5)
        full=model(x)
        branch_changes={}
        for name in ('tdca','acmsa'):
            def bypass(module,inputs,output):
                # Preserve TDCA routing labels while disabling its direct value path.
                return (torch.zeros_like(output[0]),output[1]) if isinstance(output,tuple) else torch.zeros_like(output)
            handle=getattr(model.content_tab,name).register_forward_hook(bypass)
            try:
                changed=model(x)
            finally:
                handle.remove()
            branch_changes[name]=float((changed-full).norm()/full.norm().clamp_min(1e-12))
            assert branch_changes[name]>0
    assert model.extra_parameters == 21813
    assert sum(p.numel() for p in deployed.parameters()) == 520021
    report = dict(status='PASS',device=args.device,extra_parameters=21813,deployed_parameters=520021,
                  adapter_initialization_exact=True,baseline_initialization_and_rng_exact=True,
                  all_adapter_tensors_updated=True,resume_exact=True,synthetic_loss=losses,branch_output_gradient_min={k:min(v) for k,v in branch_gradients.items()},
                  branch_bypass_output_change_ratio=branch_changes,
                  no_center_buffers=True,evidence='Engineering only; no real PSNR evidence')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
