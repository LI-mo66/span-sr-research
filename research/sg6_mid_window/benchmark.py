"""Same-device, same-shape, FP32 prefused random-weight resource comparison."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from model import build_model, build_sg6
import runtime
from deploy import fused_copy


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--repeats',type=int,default=10);args=p.parse_args()
    if args.repeats<3:raise ValueError('At least three repeats required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    arch,_=runtime.components()
    configs={k:json.loads((Path(__file__).parent/'configs'/f'{k}.json').read_text()) for k in ('A','C')}
    models={}
    for key in ('SG6','A','C'):
        torch.manual_seed(10)
        model=build_sg6(arch,configs['A']) if key=='SG6' else build_model(arch,configs[key])
        models[key]=fused_copy(model.cuda(),arch.Conv3XC)
    x=torch.rand(1,3,180,320,device='cuda')
    times={k:[] for k in models};peaks={k:0 for k in models}
    reserved={k:0 for k in models}
    with torch.inference_mode():
        for _ in range(3):
            for model in models.values():
                output=model(x);del output
        torch.cuda.synchronize()
        keys=list(models)
        for iteration in range(args.repeats):
            for offset in range(len(keys)):
                key=keys[(iteration+offset)%len(keys)]
                torch.cuda.synchronize();resident=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                output=models[key](x);torch.cuda.synchronize()
                times[key].append((time.perf_counter()-start)*1000)
                peaks[key]=max(peaks[key],torch.cuda.max_memory_allocated()-resident)
                reserved[key]=max(reserved[key],torch.cuda.max_memory_reserved())
                del output
    report=dict(status='PASS',torch=torch.__version__,device=torch.cuda.get_device_name(),
                precision='FP32',shape=list(x.shape),warmup_each=3,repeats_each=args.repeats,
                models={k:dict(deployed_parameters=sum(p.numel() for p in m.parameters()),
                               median_ms=statistics.median(times[k]),min_ms=min(times[k]),
                               incremental_peak_allocated=peaks[k],process_peak_reserved=reserved[k]) for k,m in models.items()},
                scope='Random-weight prefused inference; models co-resident. Reserved memory is process-wide. Not training throughput, trained PSNR, or complete FLOPs coverage.')
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
