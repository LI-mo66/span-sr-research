"""Compute-only train update timing; excludes real image loading and evaluation."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from runtime import components

def main():
    p=argparse.ArgumentParser();p.add_argument('--micro-batch',type=int,choices=[4,8,16,32,64],required=True);p.add_argument('--warmup',type=int,default=20);p.add_argument('--updates',type=int,default=100);p.add_argument('--output',required=True);a=p.parse_args()
    arch,_=components();torch.manual_seed(10)
    model=arch.SPAN(3,3,feature_channels=52,upscale=4,img_range=255.).cuda().train()
    optimizer=torch.optim.Adam((v for v in model.parameters() if v.requires_grad),lr=5e-4)
    lr=torch.rand(a.micro_batch,3,64,64,device='cuda');hr=torch.rand(a.micro_batch,3,256,256,device='cuda')
    accumulation=64//a.micro_batch
    def update():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            loss=torch.nn.functional.l1_loss(model(lr),hr)/accumulation
            loss.backward()
        optimizer.step()
    for _ in range(a.warmup):update()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for _ in range(a.updates):update()
    torch.cuda.synchronize();elapsed=time.perf_counter()-start;s=elapsed/a.updates
    report=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,micro_batch=a.micro_batch,effective_batch=64,accumulation=accumulation,warmup=a.warmup,measured_updates=a.updates,seconds_per_update=s,compute_only_hours_1M=s*1e6/3600,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),precision='FP32',note='synthetic inputs; excludes real data decoding, five benchmark/validation evaluation and checkpoint saving; not a complete ETA')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
