import ast,hashlib,json,math,torch
from torch import nn
from run import HERE
from attention import ContextAdapter,DualInteraction,FourierEmbedding,QKState

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
