"""Bounded diagnosis, not a production change or relaxed determinism gate.

Compare raw CUDA scan with a mathematically equivalent execution experiment:
one image per call and explicit per-channel B/C copies. The latter aims to avoid
colliding atomic gradient accumulation; actual repeatability/cost must be tested.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from adapter import ASSMAdapter
import adapter as adapter_module


def isolated_scan(native,u,delta,A,B,C,D,**kwargs):
    if B.ndim!=4 or C.ndim!=4 or B.shape[1]!=1 or C.shape[1]!=1:
        raise ValueError('Diagnostic expects single-group variable B/C')
    channels=u.shape[1]
    outputs=[]
    for i in range(u.shape[0]):
        b=B[i:i+1].expand(-1,channels,-1,-1).contiguous()
        c=C[i:i+1].expand(-1,channels,-1,-1).contiguous()
        outputs.append(native(u[i:i+1],delta[i:i+1],A,b,c,D,**kwargs))
    return torch.cat(outputs,0)


def simulate_adam(model,gradients):
    replay=copy.deepcopy(model)
    opt=torch.optim.Adam(replay.parameters(),lr=2e-4,betas=(.9,.999),eps=1e-8)
    for name,p in replay.named_parameters():p.grad=gradients[name].clone()
    opt.step()
    return {name:p.detach().clone() for name,p in replay.named_parameters()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=dict(scope='Disposable fixed-weight gradients and one-step Adam replay; no real training/PSNR',cases=[])
    original=adapter_module.scan
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=True
        torch.backends.cuda.matmul.allow_tf32=False
        report['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name())
        torch.manual_seed(10);prototype=ASSMAdapter('cuda').cuda()
        def alternate(backend,*a,**kw):
            return isolated_scan(selective_scan_fn,*a,**kw) if backend=='cuda' else original(backend,*a,**kw)
        for batch,h,w,repeats in ((2,3,5,8),(16,3,5,8),(16,64,64,3)):
            torch.manual_seed(1000+batch+h)
            x=torch.randn(batch,52,h,w,device='cuda');target=torch.randn_like(x)
            first_native=None;first_native_grad=None
            for execution in ('native','isolated_experiment'):
                adapter_module.scan=original if execution=='native' else alternate
                model=copy.deepcopy(prototype)
                reference_output=reference_grad=reference_weights=None
                row=dict(shape=list(x.shape),execution=execution,repeats=repeats,
                         forward_exact=True,gradients_exact=True,adam_weights_exact=True,parameters={})
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                for iteration in range(repeats):
                    model.zero_grad(set_to_none=True)
                    y=model(x);loss=(y-target).square().mean();loss.backward()
                    gradients={k:p.grad.detach().clone() for k,p in model.named_parameters()}
                    assert all(torch.isfinite(v).all() for v in gradients.values())
                    weights=simulate_adam(model,gradients)
                    if iteration==0:
                        reference_output=y.detach().clone();reference_grad=gradients;reference_weights=weights
                        if execution=='native':
                            first_native=reference_output;first_native_grad=gradients
                        else:
                            row['forward_matches_native_exact']=torch.equal(reference_output,first_native)
                            row['forward_vs_native_max_abs']=float((reference_output-first_native).abs().max())
                            torch.testing.assert_close(reference_output,first_native,atol=2e-6,rtol=1e-5)
                            for k,g in gradients.items():
                                torch.testing.assert_close(g,first_native_grad[k],atol=2e-7,rtol=2e-3)
                    else:
                        row['forward_exact'] &= torch.equal(y,reference_output)
                        for k,g in gradients.items():
                            ref=reference_grad[k];change=(g-ref).abs();wdiff=(weights[k]-reference_weights[k]).abs()
                            row['gradients_exact'] &= torch.equal(g,ref)
                            row['adam_weights_exact'] &= torch.equal(weights[k],reference_weights[k])
                            p=row['parameters'].setdefault(k,dict(gradient_rms=float(ref.square().mean().sqrt()),
                                max_gradient_abs_difference=0.,max_gradient_relative_l2=0.,
                                max_changed_gradient_elements=0,max_weight_abs_difference=0.,max_changed_weight_elements=0))
                            p['max_gradient_abs_difference']=max(p['max_gradient_abs_difference'],float(change.max()))
                            p['max_gradient_relative_l2']=max(p['max_gradient_relative_l2'],float(change.norm()/ref.norm().clamp_min(1e-30)))
                            p['max_changed_gradient_elements']=max(p['max_changed_gradient_elements'],int((change!=0).sum()))
                            p['max_weight_abs_difference']=max(p['max_weight_abs_difference'],float(wdiff.max()))
                            p['max_changed_weight_elements']=max(p['max_changed_weight_elements'],int((wdiff!=0).sum()))
                    del y,loss,gradients,weights
                torch.cuda.synchronize()
                row['elapsed_seconds']=time.perf_counter()-start
                row['peak_allocated']=torch.cuda.max_memory_allocated()
                report['cases'].append(row)
                Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k!='parameters'}),flush=True)
                del model,reference_output,reference_grad,reference_weights
                torch.cuda.empty_cache()
        isolated=[r for r in report['cases'] if r['execution']=='isolated_experiment']
        report['status']='ISOLATED_EXACT_IN_TESTED_CASES' if all(r['forward_exact'] and r['gradients_exact'] and r['adam_weights_exact'] for r in isolated) else 'REPEAT_DIFFERENCE_REMAINS'
    except Exception as exc:
        import traceback
        report.update(status='REVIEW_FAILED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        adapter_module.scan=original
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print('REVIEW_STATUS:',report['status'],flush=True)


if __name__=='__main__':main()
