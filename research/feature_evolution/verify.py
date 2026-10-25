"""Synthetic engineering evidence only. Run from any directory."""
import argparse
import copy
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for relative in ('research', 'research/linear_gate', 'research/gated_bypass', 'research/spatial_gate6',
                 'research/modules'):
    sys.path.insert(0, str(ROOT / relative))
from fe_model import build_model
from sg_model import build_model as baseline
from deploy import fused_copy


def load_arch():
    spec = importlib.util.spec_from_file_location('fe_runtime', ROOT/'research/runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.components()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    original_precision = dict(cudnn_tf32=torch.backends.cudnn.allow_tf32,
                              matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    # Diagnostic process only. Do not alter the archived trainer's precision policy.
    if args.device == 'cuda':
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(91)
    arch = load_arch()
    cfg = json.loads((ROOT/'research/spatial_gate6/configs/LG6-SiLU-SG6.json').read_text())
    def make(sign):
        torch.manual_seed(10)
        return build_model(arch, dict(cfg, feature_evolution=dict(block=3,width=13,sign=sign))).to(args.device)
    torch.manual_seed(10)
    reference = baseline(arch, cfg).to(args.device)
    rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    add, difference = make(1), make(-1)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(gpu_rng, torch.cuda.get_rng_state_all()))
    common = reference.state_dict()
    assert all(torch.equal(v,add.state_dict()[k]) and torch.equal(v,difference.state_dict()[k])
               for k,v in common.items())
    assert all(torch.equal(v,difference.state_dict()[k]) for k,v in add.state_dict().items())
    norm_types = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm,
                  torch.nn.GroupNorm, torch.nn.modules.instancenorm._InstanceNorm)
    assert not any(isinstance(m,norm_types) for m in add.modules())
    # Exact function-class counterexample in FP64, independent of initial SG6 equality.
    plus = copy.deepcopy(add.block_3).cpu().double().train()
    minus = copy.deepcopy(difference.block_3).cpu().double().train()
    with torch.no_grad():
        minus.c2_r.sk.weight.add_(2*torch.eye(52,dtype=torch.float64).reshape(52,52,1,1))
        z = torch.randn(2,52,7,11,dtype=torch.float64)
        for actual, expected in zip(minus(z),plus(z)):
            torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    result = dict(device=args.device,torch=torch.__version__,evidence='synthetic only',
                  original_precision=original_precision,
                  diagnostic_tf32_disabled=args.device=='cuda',variants={})
    for sign, model in ((1,add),(-1,difference)):
        for shape in ((1,3,1,1),(1,3,13,17),(2,3,16,20),(1,3,17,13)):
            x = torch.rand(*shape,device=args.device)
            for training in (True,False):
                model.train(training)
                with torch.no_grad():
                    y = model(x)
                    assert y.shape == (shape[0],3,shape[2]*4,shape[3]*4)
                    assert torch.isfinite(y).all()
            # Numerical agreement, not a requirement for bitwise FP32 convolution equality.
            with torch.no_grad():
                model.train(); train_y=model(x)
                model.eval(); eval_y=model(x)
                torch.testing.assert_close(train_y,eval_y,rtol=2e-4,atol=2e-5)
        model.train()
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(params,lr=2e-4,betas=(.9,.999),eps=1e-8)
        ids=[id(p) for g in optimizer.param_groups for p in g['params']]
        assert len(ids)==len(set(ids))==len(params)
        x=torch.rand(2,3,13,17,device=args.device)
        target=torch.rand(2,3,52,68,device=args.device)
        records=[]
        for step in range(12):
            optimizer.zero_grad(set_to_none=True)
            before={n:p.detach().clone() for n,p in model.block_3.stage_fusion.named_parameters()}
            prediction=model(x)
            loss=torch.nn.functional.l1_loss(prediction,target)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            grads={n:float(p.grad.norm()) for n,p in model.block_3.stage_fusion.named_parameters()}
            if step==0:
                assert all(v==0 for n,v in grads.items() if n.startswith('projections.'))
                assert grads['output.weight']>0
            optimizer.step()
            updates={n:float((p.detach()-before[n]).norm())
                     for n,p in model.block_3.stage_fusion.named_parameters()}
            assert all(torch.isfinite(p).all() for p in params)
            records.append(dict(step=step+1,loss=float(loss.detach()),grad_norms=grads,update_norms=updates))
        assert all(any(row['update_norms'][n]>0 for row in records[1:]) for n in before)
        # Report scale diagnostics with synthetic input, never label real features matched.
        stats={}
        hooks=[]
        def hook(name):
            def capture(module, inputs, output):
                value=output.detach()
                stats[name]=dict(rms=float(value.square().mean().sqrt()),absmax=float(value.abs().max()))
            return capture
        for name,module in (('fusion_delta',model.block_3.stage_fusion),
                            ('local_gate',model.block_3.local_gate),('spatial_gate',model.block_3.spatial_gate)):
            hooks.append(module.register_forward_hook(hook(name)))
        model.eval()
        with torch.no_grad(): expected=model(x)
        for h in hooks:h.remove()
        buffer=io.BytesIO()
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),buffer)
        buffer.seek(0)
        saved=torch.load(buffer,map_location=args.device,weights_only=False)
        clone=make(sign)
        incompatible=clone.load_state_dict(saved['model'],strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        clone_opt=torch.optim.Adam((p for p in clone.parameters() if p.requires_grad),lr=2e-4)
        clone_opt.load_state_dict(saved['optimizer'])
        clone.eval()
        with torch.no_grad():torch.testing.assert_close(clone(x),expected,rtol=0,atol=0)
        for net,opt in ((model,optimizer),(clone,clone_opt)):
            net.train();opt.zero_grad(set_to_none=True)
            torch.nn.functional.l1_loss(net(x),target).backward();opt.step()
        assert all(torch.equal(v,clone.state_dict()[k]) for k,v in model.state_dict().items())
        model.eval()
        deployed=fused_copy(model,arch.Conv3XC)
        with torch.no_grad():torch.testing.assert_close(model(x),deployed(x),rtol=1e-5,atol=1e-6)
        count=sum(p.numel() for p in deployed.parameters())
        assert count==539919 and model.extra_parameters==41711
        result['variants'][str(sign)]=dict(deployed_parameters=count,updates=records,
            synthetic_scales=stats,strict_reload=dict(missing=[],unexpected=[]))
    result.update(status='PASS',checks=['shared initial state and isolated RNG',
        'zero added normalization layers','FP64 sign-equivalent counterexample',
        'odd/rectangular/tiny shapes and batch1/2','train/eval numerical agreement',
        'finite L1 gradients and actual staged updates','optimizer exact coverage',
        'strict model/optimizer save and equal resumed update','deployment fusion'])
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',device=args.device,output=args.output)))


if __name__=='__main__':main()
