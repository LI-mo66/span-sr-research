"""No installs/training/shutdown. Inspect ABI and test an existing scan kernel."""
import argparse
import copy
import importlib.metadata
import json
import os
import platform
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from adapter import ASSMAdapter


def probe(backend='cuda_isolated'):
    report=dict(status='INSPECTING',backend=backend,python=platform.python_version(),torch=torch.__version__,
                cuda=torch.version.cuda,cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),packages={})
    for name in ('mamba-ssm','causal-conv1d','einops','triton','ninja','packaging'):
        try:report['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:report['packages'][name]=None
    if not torch.cuda.is_available():
        report.update(status='NEEDS_CUDA',scope='No model test performed')
        return report
    report['device']=torch.cuda.get_device_name()
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except Exception as exc:
        report.update(status='NEEDS_COMPATIBLE_KERNEL',import_error=f'{type(exc).__name__}: {exc}',
                      scope='No installation attempted; this is not a model accuracy failure')
        return report
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(10)
    reference=ASSMAdapter('reference').cuda()
    production=ASSMAdapter(backend).cuda()
    production.load_state_dict(reference.state_dict(),strict=True)
    x=torch.randn(2,52,3,5,device='cuda',requires_grad=True)
    xr=x.detach().clone().requires_grad_()
    actual=production(x);expected=reference(xr)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5)
    target=torch.randn_like(actual)
    ((actual-target)**2).mean().backward();((expected-target)**2).mean().backward()
    torch.testing.assert_close(x.grad,xr.grad,atol=1e-7,rtol=1e-4)
    first_grad={k:p.grad.clone() for k,p in production.named_parameters()}
    max_gradient_error=0.
    for (name,p),(other,q) in zip(production.named_parameters(),reference.named_parameters()):
        assert name==other and p.grad is not None and q.grad is not None
        assert torch.isfinite(p.grad).all() and p.grad.abs().max()>0,name
        torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=2e-3)
        max_gradient_error=max(max_gradient_error,float((p.grad-q.grad).abs().max()))
    production.zero_grad(set_to_none=True)
    y=production(x.detach());((y-target)**2).mean().backward()
    repeat_diff=max(float((p.grad-first_grad[k]).abs().max()) for k,p in production.named_parameters())
    report.update(status='PASS_SMALL_KERNEL' if repeat_diff==0 else 'NEEDS_DETERMINISM_REVIEW',
                  output_max_error=float((actual-expected).abs().max().detach()),
                  parameter_gradient_max_error=max_gradient_error,
                  repeated_gradient_max_difference=repeat_diff,
                  scope='Small CUDA-kernel check only; micro16, largest image, trainer integration and PSNR still pending')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--backend',choices=('cuda','cuda_isolated'),default='cuda_isolated')
    args=parser.parse_args()
    try:report=probe(args.backend)
    except Exception as exc:
        import traceback
        report=dict(status='KERNEL_CHECK_FAILED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
