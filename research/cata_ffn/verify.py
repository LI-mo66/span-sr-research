"""Verify retained initialization, residual semantics and real FFN learning."""
import argparse
import io
import json
import os
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from run import ffn_runtime as runtime
from model import build_model, ContentTAB
from deploy import fused_copy
import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=('cpu','cuda'))
    parser.add_argument('--config', default=str(Path(__file__).parent/'configs/F1-tail-lr2e4.json'))
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
        original = ContentTAB().to(args.device)
    assert all(torch.equal(v,original.mlp.state_dict()[k]) for k,v in model.content_tab.mlp.state_dict().items())
    assert not list(model.content_tab.buffers())
    assert not any('attn' in k or 'means' in k for k in model.content_tab.state_dict())
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
    tokens = features.flatten(2).transpose(1,2).contiguous()
    expected = (tokens+original.mlp(tokens,(13,17))).transpose(1,2).reshape_as(features)
    torch.testing.assert_close(model.content_tab(features),expected,atol=0,rtol=0)
    trainable = list(model.parameters())
    opt = torch.optim.Adam(trainable,lr=cfg['lr'])
    ids = [id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(trainable)
    before = {k:p.clone() for k,p in model.content_tab.named_parameters()}
    target = torch.rand(2,3,52,68,device=args.device)
    losses = []
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
    assert model.extra_parameters == 13780
    assert sum(p.numel() for p in deployed.parameters()) == 511988
    report = dict(status='PASS',device=args.device,extra_parameters=13780,deployed_parameters=511988,
                  retained_ffn_initialization_exact=True,baseline_initialization_and_rng_exact=True,
                  all_ffn_tensors_updated=True,resume_exact=True,synthetic_loss=losses,
                  no_attention_or_centers=True,evidence='Engineering only; no real PSNR evidence')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
