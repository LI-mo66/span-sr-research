"""Same-device sequential prefused inference timing; no PSNR conclusion."""
import gc
import json
import statistics
import sys
import time
import torch
from fusion import HERE,ROOT,shared,build_model,build_sg6,load,window,attention,lksa
from deploy import fused_copy

def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    arch,_=shared.components();cfg=json.loads((ROOT/'logs/fusion-WU/frozen/config.json').read_text())
    sys.modules.update(attention=attention,lksa=lksa,adapter=window)
    u=load('bench_u',HERE/'u_candidate.py');w=load('bench_w',HERE/'w_model.py')
    inputs=torch.rand(1,3,180,320,device='cuda');results={}
    for name,factory,config in (
        ('SG6',build_sg6,cfg),('W',w.build_model,json.loads((HERE/'w_config.json').read_text())),
        ('U',u.build_model,json.loads((HERE/'config.json').read_text())),('FUSION',build_model,cfg)):
        torch.manual_seed(10)
        model=fused_copy(factory(arch,config).cuda().eval(),arch.Conv3XC).eval()
        with torch.inference_mode():
            for _ in range(3):model(inputs)
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            durations=[]
            for _ in range(10):
                torch.cuda.synchronize();start=time.perf_counter();value=model(inputs)
                torch.cuda.synchronize();durations.append((time.perf_counter()-start)*1000)
        results[name]=dict(deployed_parameters=sum(p.numel() for p in model.parameters()),
            median_ms=statistics.median(durations),incremental_peak_allocated=torch.cuda.max_memory_allocated()-base)
        del model,value;gc.collect();torch.cuda.empty_cache()
    report=dict(models=results,device=torch.cuda.get_device_name(),torch=torch.__version__,
        input_shape=[1,3,180,320],precision='FP32',warmup=3,repeats=10,
        scope='Random weights, prefused inference only, sequential models; not training throughput')
    (ROOT/'logs/fusion-WU/resources.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
