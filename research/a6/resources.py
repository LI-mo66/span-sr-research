"""Synthetic A6 FP32 training memory/latency check at the actual microbatch."""
import argparse
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('a6_runtime', HERE/'runtime.py')
runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
import torch
from model import build_model


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(10)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((HERE/'configs/A6-lr2e4.json').read_text())
    arch, _ = runtime.components(); model = build_model(arch, cfg).cuda().train()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    x = torch.rand(16, 3, 64, 64, device='cuda'); target = torch.rand(16, 3, 256, 256, device='cuda')
    def update():
        opt.zero_grad(set_to_none=True)
        for _ in range(4):
            loss = torch.nn.functional.l1_loss(model(x), target)/4
            loss.backward()
        opt.step()
    update(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(); times = []
    for _ in range(3):
        start = time.perf_counter(); update(); torch.cuda.synchronize(); times.append(time.perf_counter()-start)
    training = dict(microbatch=16, accumulation=4, effective_batch=64, seconds_per_update=times,
                    median_seconds_per_update=statistics.median(times),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    del opt, x, target
    model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); model.eval()
    inference = []
    with torch.inference_mode():
        for h, w in ((64, 64), (256, 256), (65, 71)):
            x = torch.rand(1, 3, h, w, device='cuda')
            for _ in range(5): model(x)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); times = []
            for _ in range(15):
                start=time.perf_counter(); y=model(x); torch.cuda.synchronize(); times.append((time.perf_counter()-start)*1000)
            inference.append(dict(LR=[h,w], median_ms=statistics.median(times), min_ms=min(times), max_ms=max(times),
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    result = dict(evidence='synthetic A6 only; no real images or PSNR evidence', torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), precision='FP32; cudnn TF32=true/matmul TF32=false',
                  path='original Conv3XC eval path, including per-forward weight fusion', training=training, inference=inference,
                  extra_parameters=45438, deployed_parameters=543646,
                  extra_64_MACs_linear_matmul_only=269528064,
                  excluded_MACs='softmax/LN/elementwise/padding/indexing and memory traffic; not full FLOPs')
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
