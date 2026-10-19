"""Exploratory full-network FP32 timing; randomized weights, not trained PSNR."""
import argparse
import json
import statistics
import time
from pathlib import Path
from run import atd_runtime as runtime
from model import build_model
from deploy import fused_copy
import torch


def benchmark(cfg,height,width,repeats=20,warmup=1.0):
    arch,_=runtime.components()
    torch.manual_seed(cfg['seed'])
    baseline=fused_copy(build_model(arch,dict(cfg,atd_attention=False)).cuda(),arch.Conv3XC)
    torch.manual_seed(cfg['seed'])
    candidate=build_model(arch,cfg).cuda().eval()
    native_cfg=dict(cfg,atd_settings=dict(position='tail',query_chunk=10**9,group_chunk=10**9))
    unchunked=build_model(arch,native_cfg).cuda().eval()
    unchunked.load_state_dict(candidate.state_dict(),strict=True)
    shared_state={k:v.clone() for k,v in candidate.state_dict().items()}
    candidate=fused_copy(candidate,arch.Conv3XC)
    unchunked=fused_copy(unchunked,arch.Conv3XC)
    models=dict(B0=baseline,ATD_chunked=candidate,ATD_unchunked=unchunked)
    for name,qc,gc in (('ATD_q16k_g128',16384,128),('ATD_q4k_g128',4096,128)):
        variant=build_model(arch,dict(cfg,atd_settings=dict(position='tail',query_chunk=qc,group_chunk=gc))).cuda().eval()
        variant.load_state_dict(shared_state,strict=True)
        models[name]=fused_copy(variant,arch.Conv3XC)
        del variant
    del shared_state
    x=torch.rand(1,3,height,width,device='cuda')
    values={k:[] for k in models}; peaks={k:0 for k in models}; reserved={k:0 for k in models}
    with torch.inference_mode():
        reference_output=unchunked(x)
        for name,model in models.items():
            if name!='B0':
                torch.testing.assert_close(model(x),reference_output,atol=2e-6,rtol=1e-5)
        del reference_output
        start=time.monotonic()
        while time.monotonic()-start<warmup:
            for model in models.values():
                output=model(x); del output
            torch.cuda.synchronize()
        keys=list(models)
        for iteration in range(repeats):
            for offset in range(len(keys)):
                name=keys[(iteration+offset)%len(keys)]
                torch.cuda.synchronize()
                resident=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter(); output=models[name](x); torch.cuda.synchronize()
                values[name].append((time.perf_counter()-start)*1000)
                peaks[name]=max(peaks[name],torch.cuda.max_memory_allocated()-resident)
                reserved[name]=max(reserved[name],torch.cuda.max_memory_reserved())
                del output
    return dict(input=[1,3,height,width],device=torch.cuda.get_device_name(),torch=torch.__version__,
                repeats=repeats,warmup_seconds=warmup,precision='FP32',prefused=True,
                results={name:dict(median_ms=statistics.median(values[name]),
                                  incremental_peak_allocated=peaks[name],
                                  process_peak_reserved=reserved[name]) for name in models},
                caveat='All five models co-resident. Reserved is process-wide; random-weight exploration, not trained deployment or full FLOPs.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(Path(__file__).parent/'configs/A1-tail-lr2e4.json'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--height',type=int,default=180)
    parser.add_argument('--width',type=int,default=320)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads(Path(args.config).read_text())
    result=benchmark(cfg,args.height,args.width)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
