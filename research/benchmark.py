"""Measure upstream eval path; do not silently remove its per-forward fusion."""
import argparse
import json
import statistics
import time
import torch
from runtime import components

def main():
    p=argparse.ArgumentParser(); p.add_argument('--height',type=int,default=256); p.add_argument('--width',type=int,default=256); p.add_argument('--warmup',type=int,default=20); p.add_argument('--repeats',type=int,default=100); p.add_argument('--output',required=True); a=p.parse_args()
    arch,_=components(); model=arch.SPAN(3,3,feature_channels=48,upscale=4).cuda().eval()
    x=torch.rand(1,3,a.height,a.width,device='cuda')
    times=[]; macs=[0]; handles=[]
    def hook(m, inputs, output): macs[0] += output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
    with torch.inference_mode():
        for _ in range(a.warmup): model(x)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        for _ in range(a.repeats):
            torch.cuda.synchronize(); start=time.perf_counter(); model(x); torch.cuda.synchronize(); times.append((time.perf_counter()-start)*1000)
        for m in model.modules():
            if isinstance(m,torch.nn.Conv2d): handles.append(m.register_forward_hook(hook))
        model(x); torch.cuda.synchronize()
    for h in handles: h.remove()
    result=dict(path='unmodified upstream eval including per-forward fusion', checkpoint='random initialization: efficiency only', device=torch.cuda.get_device_name(), torch=torch.__version__, precision='FP32', lr_shape=list(x.shape), warmup=a.warmup, repeats=a.repeats, median_ms=statistics.median(times), mean_ms=statistics.mean(times), p95_ms=sorted(times)[min(len(times)-1,int(.95*len(times)))], peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(), parameters_total=sum(p.numel() for p in model.parameters()), parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad), forward_conv_MACs=macs[0], forward_conv_FLOPs_2_per_MAC=macs[0]*2, coverage='Conv2d module calls only; excludes functional kernel fusion, pointwise ops and layout. Not total FLOPs.')
    from pathlib import Path
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
