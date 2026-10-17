"""Matched full-SPAN resource benchmark including the unoptimized TAB control."""
import argparse
import copy
import importlib.util
import json
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cata_runtime', HERE / 'runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
import torch
from model import build_model
from deploy import fused_copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    arch,_ = runtime.components()
    cfg = json.loads((HERE / 'configs/C1-tail-lr2e4.json').read_text())
    torch.manual_seed(cfg['seed'])
    b0 = build_model(arch,dict(cfg,cata_tab=False)).cuda()
    torch.manual_seed(cfg['seed'])
    c1 = build_model(arch,cfg).cuda()
    c1.content_tab.iasa_attn.chunk_groups = 32  # explicit control, independent of chosen default
    with torch.no_grad():
        c1(torch.rand(1,3,64,64,device='cuda'))  # initialize centers before eval
    ref_spec = importlib.util.spec_from_file_location('reference_tab',HERE / 'vendor/reference_tab.py')
    reference = importlib.util.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(reference)
    raw = copy.deepcopy(c1)
    raw.content_tab = reference.TAB(52,52,104,4,n_iter=5,num_tokens=16,group_size=128).cuda()
    raw.content_tab.load_state_dict(c1.content_tab.state_dict(),strict=True)
    models = {name:fused_copy(model,arch.Conv3XC) for name,model in
              (('B0',b0),('C1_chunked4D',c1),('C1_official5D',raw))}
    for chunk in (128, 512):
        extra = copy.deepcopy(models['C1_chunked4D'])
        extra.content_tab.iasa_attn.chunk_groups = chunk
        models[f'C1_chunk{chunk}'] = extra
    del b0,c1,raw
    report = dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),precision='FP32',
                  warmup='2 seconds interleaved per input size',repeats=20,
                  measurement_order='rotate across all models every repeat',
                  all_five_models_resident=True,results=[],
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  evidence='random weights, full prefused SPAN, not PSNR/4090 results')
    with torch.inference_mode():
        for h,w in ((64,64),(180,320)):
            x = torch.rand(1,3,h,w,device='cuda')
            warm_start=time.perf_counter()
            while time.perf_counter()-warm_start < 2:
                for model in models.values():
                    model(x)
                torch.cuda.synchronize()
            timings = {name:[] for name in models}
            peaks = {name:0 for name in models}
            reserves = {name:0 for name in models}
            names=list(models)
            for repeat in range(20):
                order=names[repeat % len(names):]+names[:repeat % len(names)]
                for name in order:
                    model=models[name]
                    torch.cuda.synchronize()
                    resident = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter()
                    model(x)
                    torch.cuda.synchronize()
                    timings[name].append((time.perf_counter()-start)*1000)
                    peaks[name]=max(peaks[name],torch.cuda.max_memory_allocated()-resident)
                    reserves[name]=max(reserves[name],torch.cuda.max_memory_reserved())
            for name,model in models.items():
                report['results'].append(dict(name=name,lr_hw=[h,w],
                    deployed_parameters=sum(p.numel() for p in model.parameters()),
                    median_ms=statistics.median(timings[name]),
                    peak_extra_allocated_bytes=peaks[name],
                    peak_reserved_bytes=reserves[name]))
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
