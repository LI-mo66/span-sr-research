"""Disposable production-shape scratch updates and inference timings; never save weights."""
import argparse
import gc
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import statistics
import time
import torch
from bootstrap import components,trainer
from strip_model import build_model
from sg_model import build_model as sg6

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--group',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    audit=json.loads((args.group/'reuse-audit.json').read_text())
    env=audit['environment']
    torch.backends.cudnn.allow_tf32=env['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32=env['matmul_allow_tf32']
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    arch,_=components()
    first=json.loads((args.group/f'K{audit["kernels"][0]}.json').read_text())
    paths=json.loads(Path(first['paths_file']).read_text())
    largest=None
    for name,spec in paths['validation'].items():
        for hp,lp in trainer.pairs(spec):
            _,lr=trainer.aligned(hp,lp)
            if largest is None or lr.shape[0]*lr.shape[1]>largest[0]:largest=(lr.shape[0]*lr.shape[1],lr.shape[:2],name,hp.name)
    report=dict(status='RUNNING',device=torch.cuda.get_device_name(),torch=str(torch.__version__),shape=dict(micro=16,accumulation=4,lr_patch=64),
        largest_evaluation=dict(shape=list(largest[1]),dataset=largest[2],image=largest[3]),results={},synthetic_only=True)
    for label in ['SG6']+[f'K{k}' for k in audit['kernels']]:
        cfg=first if label=='SG6' else json.loads((args.group/(label+'.json')).read_text())
        torch.manual_seed(10)
        model=(sg6 if label=='SG6' else build_model)(arch,cfg).cuda().train()
        opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=2e-4,betas=(.9,.999),eps=1e-8)
        gen=torch.Generator(device='cuda').manual_seed(90001)
        x=torch.rand(16,3,64,64,device='cuda',generator=gen)
        target=torch.rand(16,3,256,256,device='cuda',generator=gen)
        def update():
            opt.zero_grad(set_to_none=True)
            for _ in range(4):
                loss=(model(x)-target).abs().mean()/4
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite resource smoke loss '+label)
                loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise FloatingPointError('Nonfinite resource gradient')
            opt.step()
        for _ in range(2):update()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(3):
            start=time.perf_counter();update();torch.cuda.synchronize();times.append(time.perf_counter()-start)
        trainmem=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        opt.zero_grad(set_to_none=True);model.eval()
        big=torch.rand(1,3,*largest[1],device='cuda',generator=gen)
        with torch.inference_mode():
            y=model(big)
            if not torch.isfinite(y).all() or tuple(y.shape[-2:])!=tuple(v*4 for v in largest[1]):raise ValueError('Largest full-image smoke failed')
            del y
            probe=torch.rand(1,3,180,320,device='cuda',generator=gen)
            for _ in range(3):model(probe)
            torch.cuda.synchronize();evaltimes=[]
            for _ in range(5):
                start=time.perf_counter();model(probe);torch.cuda.synchronize();evaltimes.append(time.perf_counter()-start)
        report['results'][label]=dict(update_seconds_median=statistics.median(times),update_seconds_samples=times,
            train_peak_bytes=trainmem,eval180x320_seconds_median=statistics.median(evaltimes),largest_image_finite=True)
        with args.output.open('w') as f:json.dump(report,f,indent=2)
        del model,opt,x,target,big,probe;gc.collect();torch.cuda.empty_cache()
    base=report['results']['SG6']['update_seconds_median']
    for label,row in report['results'].items():row['update_time_ratio_vs_SG6']=row['update_seconds_median']/base
    report.update(status='PASS',note='Five disposable synthetic Adam updates per network, no checkpoint or production training. Timings include original per-forward Conv3XC fusion in eval; no claim of fused latency or PSNR. Real data/feature optimization still requires short screening.')
    with args.output.open('w') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
