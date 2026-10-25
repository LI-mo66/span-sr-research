"""Small local inference profile; not a server throughput or training benchmark."""
import gc
import argparse
import json
import statistics
import time
import torch
from verify import ROOT, baseline, build_model, fused_copy, load_arch

torch.set_num_threads(2)
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.benchmark=False
arch=load_arch()
cfg=json.loads((ROOT/'research/spatial_gate6/configs/LG6-SiLU-SG6.json').read_text())
records=[]
for sign in (0,1,-1):
    gc.collect();torch.cuda.empty_cache()
    torch.manual_seed(10)
    model=baseline(arch,cfg) if sign==0 else build_model(
        arch,dict(cfg,feature_evolution=dict(block=3,width=13,sign=sign)))
    model=fused_copy(model,arch.Conv3XC).cuda().eval()
    x=torch.rand(1,3,64,64,device='cuda')
    with torch.inference_mode():
        for _ in range(5):model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples=[]
        for _ in range(20):
            torch.cuda.synchronize();start=time.perf_counter()
            y=model(x)
            torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
    records.append(dict(sign=sign,deployed_parameters=sum(p.numel() for p in model.parameters()),
        median_ms=statistics.median(samples),samples_ms=samples,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    del model,x,y
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=str,default=str(ROOT/'research/feature_evolution/profile-cuda.json'))
args=parser.parse_args()
output=__import__('pathlib').Path(args.output)
output.write_text(json.dumps(dict(device=torch.cuda.get_device_name(),torch=torch.__version__,
    shape=[1,3,64,64],dtype='float32',tf32=False,warmup=5,repeats=20,records=records,
    limitations='Local untrained fused inference, sequential order. No training memory/throughput or server latency claim. MAC estimate covers only added convolutions: 4056*H*W; no full FLOP coverage.'),indent=2),encoding='utf-8')
print(output)
