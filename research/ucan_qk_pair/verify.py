"""Synthetic mechanism/interface/optimizer checks, never real-data training."""
import argparse
import ast
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn

from run import HERE, existing, validate_config
from attention import ContextAdapter, DualInteraction, FourierEmbedding, QKState
from pair_model import build_model, build_sg6


def config(mode):
    cfg = json.loads((HERE / 'configs' / ('SG6-UCAN-' + mode + '.json')).read_text())
    validate_config(cfg)
    return cfg


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Invalid interface/config was accepted')


def reference_transpose(tensor, pattern):
    # The AST-extracted SDFL/DFRL classes use only this pure permutation.
    # Resolve it explicitly without importing the rest of UCAN or einops.
    if pattern != 'b l c -> b c l' or tensor.ndim != 3:
        raise ValueError('Unexpected layout expression in the locked reference')
    return tensor.transpose(1, 2)


def source_scope():
    manifest = json.loads((HERE / 'vendor/manifest.json').read_text())
    for filename, sha in manifest['files'].items():
        assert hashlib.sha256((HERE / 'vendor' / filename).read_bytes()).hexdigest() == sha
    tree = ast.parse((HERE / 'vendor/ucan_arch.py').read_text(encoding='utf8'))
    names = {'HedgehogFeatureMap', 'RotaryEmbedding', 'FourierEmbedding', 'SDFL', 'DFRL'}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert {n.name for n in nodes} == names
    scope = dict(torch=torch, nn=nn, F=torch.nn.functional, math=math, rearrange=reference_transpose)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'locked-UCAN-reference', 'exec'), scope)
    return scope


def equivalence(device):
    scope = source_scope()
    rows = []
    for reuse in (False, True):
        for h, w in ((1, 1), (1, 7), (13, 17), (17, 13)):
            torch.manual_seed(41)
            core = DualInteraction(reuse=reuse, fourier_normalization='source_signed').double().to(device)
            with torch.no_grad():
                core.proj.weight.normal_(0, .025)
                core.proj.bias.normal_(0, .01)
            # The source has an unregistered inv_freq: move/cast that constant
            # explicitly for the reference only, without modifying equations.
            reference = scope['DFRL' if reuse else 'SDFL'](52, (h, w), 1).double().to(device)
            state = {k: v for k, v in core.state_dict().items() if k != 'pos_emb.inv_freq'}
            if reuse:
                state.update({'dwc.' + k: v for k, v in core.lepe.state_dict().items()})
            reference.load_state_dict(state, strict=True)
            reference.pos_emb.inv_freq = core.pos_emb.inv_freq.detach().clone()
            for coefficient in (core.pos_emb.sin_coef, core.pos_emb.cos_coef):
                assert coefficient.sum(-2).abs().min() > core.pos_emb.floor
            x = torch.randn(2, h*w, 52, device=device, dtype=torch.float64, requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            q = torch.randn(2, h*w, 26, device=device, dtype=torch.float64, requires_grad=True)
            k = torch.randn_like(q, requires_grad=True)
            qr, kr = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
            y, _ = core(x, h, w, QKState(q, k, h, w) if reuse else None)
            yr = reference(xr, (qr, kr), (h, w)) if reuse else reference(xr, (h, w))[0]
            torch.testing.assert_close(y, yr, atol=2e-11, rtol=2e-11)
            y.square().mean().backward()
            yr.square().mean().backward()
            torch.testing.assert_close(x.grad, xr.grad, atol=2e-11, rtol=2e-11)
            if reuse:
                torch.testing.assert_close(q.grad, qr.grad, atol=2e-11, rtol=2e-11)
                torch.testing.assert_close(k.grad, kr.grad, atol=2e-11, rtol=2e-11)
            actual = dict(core.named_parameters())
            for name, param in reference.named_parameters():
                torch.testing.assert_close(actual[name].grad, param.grad, atol=2e-11, rtol=2e-11)
            rows.append(dict(reuse=reuse, shape=[2, h, w], forward_absmax=float((y-yr).detach().abs().max())))
    return rows


def check_interfaces(device):
    a, b = ContextAdapter().double().to(device), ContextAdapter(reuse=True).double().to(device)
    with torch.no_grad():
        a.core.proj.weight.normal_(0, .01)
        b.core.proj.weight.normal_(0, .01)
    rows = []
    for shape in ((1,52,1,1), (2,52,1,7), (2,52,13,17), (1,52,63,65), (1,52,65,63)):
        x = torch.randn(shape, device=device, dtype=torch.float64)
        before = x.clone()
        y, state = a(x)
        z, _ = b(y, state)
        assert z.shape == x.shape and torch.equal(x, before) and torch.isfinite(z).all()
        delta,_=a.core(a.norm(x.flatten(2).transpose(1,2)),shape[-2],shape[-1])
        scale=x.detach().square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6)
        expected=x+delta.transpose(1,2).reshape_as(x)*scale
        torch.testing.assert_close(y,expected,atol=0,rtol=0)
        if shape[0] == 2:
            separate = torch.cat([b(a(v[None])[0], a(v[None])[1])[0] for v in x])
            torch.testing.assert_close(z, separate, atol=1e-10, rtol=1e-10)
        rows.append(list(shape))
    # Equal HW does not permit a different spatial layout or batch.
    x = torch.randn(2, 52, 13, 17, device=device, dtype=torch.float64)
    _, state = a(x)
    rejects(lambda: b(x.transpose(-1, -2), state))
    rejects(lambda: b(x[:1], state))
    rejects(lambda: b(x.float(), state))
    rejects(lambda: b(x))
    rejects(lambda: a(x, state))
    rejects(lambda: DualInteraction(heads=2))
    # No persistent per-image cache: interleaved calls leave output unchanged.
    y1 = b(a(x)[0], a(x)[1])[0]
    _ = a(torch.randn(1, 52, 7, 11, device=device, dtype=torch.float64))
    y2 = b(a(x)[0], a(x)[1])[0]
    assert torch.equal(y1, y2)
    assert not any(isinstance(v, QKState) for m in (a, b) for v in vars(m).values())
    # Near-zero signed sums: source is undefined for a zero column; adaptation
    # must remain finite and preserve ordinary source denominators exactly.
    f = FourierEmbedding().double().to(device)
    with torch.no_grad():
        f.sin_coef.zero_()
        f.sin_coef[:, 0, :] = 1
        f.sin_coef[:, 1, :] = -1
        f.cos_coef.zero_()
    u = torch.randn(1,1,9,52,device=device,dtype=torch.float64,requires_grad=True)
    f(u).square().mean().backward()
    assert torch.isfinite(u.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters())
    assert not torch.isfinite(f.sin_coef / f.sin_coef.sum(-2,keepdim=True)).all()
    assert f.normalized(f.sin_coef).abs().sum(-2).max() <= 1 + 1e-12
    # The declared formula change is checked directly, independently of the
    # source-signed equivalence path above, including cancellation columns.
    for coef in (f.sin_coef, f.cos_coef):
        expected=coef/coef.abs().sum(-2,keepdim=True).clamp_min(f.floor)
        torch.testing.assert_close(f.normalized(coef),expected,atol=0,rtol=0)
    return dict(shapes=rows,invalid_state_rejected=True,batch_independent=True,no_cache=True,
                signed_zero_sum_stress_finite=True)


def optimize(model, x, target, opt, scheduler):
    opt.zero_grad(set_to_none=True)
    loss = (model(x)-target).abs().mean()
    assert torch.isfinite(loss)
    loss.backward()
    parameters = [(k,p) for k,p in model.named_parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in parameters)
    grads = {k:float(p.grad.detach().abs().sum()) for k,p in parameters if k.startswith('context_')}
    opt.step()
    scheduler.step()
    return float(loss.detach()), grads


def recursive_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            recursive_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for u,v in zip(a,b):
            recursive_equal(u,v)
    else:
        assert a == b


def check_models(device):
    cfg = config('off')
    arch, _ = existing.fs_runtime.components()
    from deploy import fused_copy
    torch.manual_seed(10)
    base = build_sg6(arch,cfg).to(device)
    initial = {k:v.detach().clone() for k,v in base.state_dict().items()}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    models, counts = {}, {}
    for mode in ('off','independent','reuse'):
        torch.manual_seed(10)
        m = build_model(arch,config(mode)).to(device)
        assert torch.equal(cpu_rng,torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items())
        models[mode] = m
        counts[mode] = dict(trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),
                            serialized_parameters=sum(p.numel() for p in m.parameters()),
                            deployed=sum(p.numel() for p in fused_copy(m,arch.Conv3XC).parameters()))
        for shape in ((1,3,1,1),(2,3,13,17),(1,3,17,13)):
            x = torch.rand(shape,device=device)
            for training in (True,False):
                m.train(training)
                base.train(training)
                with torch.no_grad():
                    assert torch.equal(m(x),base(x))
    # Nontrivial existing gates/spatial correction/image bypass must also survive
    # insertion; startup zeros alone cannot validate an inherited forward.
    with torch.no_grad():
        for name,p in base.named_parameters():
            if any(s in name for s in ('local_gate','spatial_gate.project','image_residual','image_gate')):
                p.add_(torch.randn_like(p)*.005)
    for m in models.values():
        state = m.state_dict()
        state.update(base.state_dict())
        m.load_state_dict(state,strict=True)
        m.eval();base.eval()
        with torch.no_grad():
            x = torch.rand(2,3,13,17,device=device)
            assert torch.equal(m(x),base(x))
    shared = models['reuse'].state_dict()
    for k,v in models['independent'].state_dict().items():
        if k in shared:
            assert torch.equal(v,shared[k]), k
    checks = {}
    for mode in ('independent','reuse'):
        torch.manual_seed(10)
        m = build_model(arch,config(mode)).to(device).train()
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params,lr=cfg['lr'],betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'])
        sched = torch.optim.lr_scheduler.MultiStepLR(opt,cfg['milestones'],gamma=cfg['gamma'])
        ids = [id(p) for group in opt.param_groups for p in group['params']]
        assert len(ids) == len(set(ids)) == len(params)
        x = torch.rand(2,3,13,17,device=device)
        target = torch.rand(2,3,52,68,device=device)
        before = {k:p.detach().clone() for k,p in m.named_parameters() if k.startswith('context_')}
        steps = []
        for step in range(4):
            loss, grads = optimize(m,x,target,opt,sched)
            assert all(grads[f'context_{i}.core.proj.weight'] > 0 for i in (2,4))
            internal = {k:v for k,v in grads.items() if '.core.proj.' not in k}
            if step == 0:
                assert all(v == 0 for v in internal.values())
            elif step >= 2:
                assert all(v > 0 for v in internal.values()), [k for k,v in internal.items() if v == 0]
            steps.append(dict(step=step+1,loss=loss,gradient_l1=grads))
        assert all(not torch.equal(p,before[k]) for k,p in m.named_parameters() if k in before)
        numerical = {}
        for name in ('context_2','context_4'):
            unit = getattr(m,name)
            for which in ('sin_coef','cos_coef'):
                sums = getattr(unit.core.pos_emb,which).detach().sum(-2).abs()
                numerical[name+'.'+which] = dict(min_abs_sum=float(sums.min()),
                    floored_columns=int((sums<unit.core.pos_emb.floor).sum()))
        # Strict checkpoint and exact model/Adam/scheduler continuation.
        stream = io.BytesIO()
        torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict()),stream)
        stream.seek(0)
        # Match the real trainer: Adam step counters stay on CPU; load_state_dict
        # moves moment tensors to their parameter devices.
        saved = torch.load(stream,map_location='cpu',weights_only=True)
        restored = build_model(arch,config(mode)).to(device).train()
        restored.load_state_dict(saved['model'],strict=True)
        opt2 = torch.optim.Adam((p for p in restored.parameters() if p.requires_grad),lr=cfg['lr'])
        opt2.load_state_dict(saved['optimizer'])
        sched2 = torch.optim.lr_scheduler.MultiStepLR(opt2,cfg['milestones'],gamma=cfg['gamma'])
        sched2.load_state_dict(saved['scheduler'])
        l1,_ = optimize(m,x,target,opt,sched)
        l2,_ = optimize(restored,x,target,opt2,sched2)
        assert l1 == l2
        recursive_equal(m.state_dict(),restored.state_dict())
        recursive_equal(opt.state_dict(),opt2.state_dict())
        recursive_equal(sched.state_dict(),sched2.state_dict())
        # Receiver-only objective must reach producer Q/K, even with its own
        # first correction projection zero. This detects accidental detach.
        if mode == 'reuse':
            a,b = m.context_2,m.context_4
            a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
            with torch.no_grad():
                a.core.proj.weight.zero_();a.core.proj.bias.zero_()
            u = torch.randn(2,52,13,17,device=device)
            _,state = a(u)
            receiver,_ = b(torch.randn_like(u),state)
            receiver.square().mean().backward()
            assert a.core.q.weight.grad.abs().sum() > 0
            assert a.core.k.weight.grad.abs().sum() > 0
        double = copy.deepcopy(restored).double().eval()
        fused = fused_copy(double,arch.Conv3XC)
        with torch.no_grad():
            error = float((double(x.double())-fused(x.double())).abs().max())
            assert error < 1e-9
            double.train()
            train64 = double(x.double())
            double.eval()
            train_eval64 = float((train64-double(x.double())).abs().max())
            assert train_eval64 < 1e-9
            restored.train();train32=restored(x)
            restored.eval()
            train_eval32 = float((train32-restored(x)).abs().max())
        norms = sum(isinstance(v,nn.LayerNorm) for v in m.modules())
        assert norms == 2
        assert not any(isinstance(v,(nn.BatchNorm2d,nn.GroupNorm)) for v in m.modules())
        checks[mode] = dict(steps=steps,all_new_parameters_changed=True,exact_resume=True,
                            strict_reload=True,fused_fp64_absmax=error,layer_norm_count=norms,
                            train_eval_fp64_absmax=train_eval64,train_eval_fp32_absmax=train_eval32,
                            fourier_denominators=numerical)
    return dict(counts=counts,base_initialization_identical=True,rng_identical=True,
                identity_with_nonzero_base_branches=True,common_arm_tensors_identical=True,
                receiver_loss_reaches_producer_qk=True,arms=checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    report = dict(status='RUNNING',device=args.device,torch=torch.__version__,
                  scope='synthetic engineering only; no PSNR or real-data claims',
                  tf32=False,deterministic=True)
    try:
        for name, check in (('reference',equivalence),('interfaces',check_interfaces),('models',check_models)):
            print('CHECK',name,flush=True)
            report[name] = check(args.device)
        report['status'] = 'PASS'
    except Exception:
        report['status'] = 'FAIL'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('PASS',args.device,report['models']['counts'],flush=True)


if __name__ == '__main__':
    main()
