"""Same-device complete-model FP32 cost probe; no image quality evidence."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from run import HERE, existing
from pair_model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--repeats',type=int,default=20)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    arch,_ = existing.fs_runtime.components()
    from deploy import fused_copy
    report = dict(torch=torch.__version__,device=a.device,
                  hardware=torch.cuda.get_device_name() if a.device=='cuda' else 'CPU',
                  precision='FP32 TF32 off',batch=1,warmup=a.warmup,repeats=a.repeats,
                  scope='random inputs; complete fused model; no PSNR',
                  mac_coverage='Conv2d, Linear, linear-attention matmuls, channel-affinity matmuls, Fourier mixing einsums; excludes LN, softmax, activation, reductions, trig, elementwise, layouts',
                  sizes={})
    def sync():
        if a.device == 'cuda':
            torch.cuda.synchronize()
    for h,w in ((64,64),(180,320)):
        rows = {}
        for mode in ('off','independent','reuse'):
            cfg = json.loads((HERE/'configs'/('SG6-UCAN-'+mode+'.json')).read_text())
            torch.manual_seed(10)
            original = build_model(arch,cfg)
            # Zero output projection does NOT bypass any computation in forward.
            model = fused_copy(original,arch.Conv3XC).to(a.device).eval()
            row = dict(deployed_parameters=sum(p.numel() for p in model.parameters()),
                       trainable_parameters=sum(p.numel() for p in original.parameters() if p.requires_grad),
                       serialized_parameter_elements=sum(p.numel() for p in original.parameters()),
                       buffer_elements=sum(p.numel() for p in original.buffers()))
            del original
            x = torch.rand(1,3,h,w,device=a.device)
            conv_linear = [0]
            def count(module,args,y):
                if isinstance(module,nn.Conv2d):
                    conv_linear[0] += y.numel() * (module.in_channels//module.groups) * math_prod(module.kernel_size)
                else:
                    conv_linear[0] += y.numel() * module.in_features
            handles = [m.register_forward_hook(count) for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
            with torch.inference_mode():
                model(x)
                for handle in handles:
                    handle.remove()
                for _ in range(a.warmup):
                    model(x)
                sync()
                if a.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    allocated = torch.cuda.memory_allocated()
                    model(x);sync()
                    row.update(resident_allocated_bytes=allocated,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_allocated_increment_bytes=torch.cuda.max_memory_allocated()-allocated,
                               peak_reserved_bytes=torch.cuda.max_memory_reserved())
                times = []
                for _ in range(a.repeats):
                    sync();start=time.perf_counter()
                    model(x);sync()
                    times.append((time.perf_counter()-start)*1000)
            n,c = h*w,26
            units = 0 if mode=='off' else 2
            # Each spatial: K'V + Q'(K'V) = 4*N*c*c (mapped dim=2c),
            # plus Q Kmean = 2*N*c. Channel: two N*c*c matmuls.
            row['conv_linear_MACs'] = conv_linear[0]
            row['attention_matmul_MACs'] = units*n*(6*c*c+2*c)
            fourier = 0
            for name in ('context_2','context_4'):
                if hasattr(model,name):
                    pe = getattr(model,name).core.pos_emb
                    fourier += 4*n*pe.sin_coef.numel()  # Q/K, sine/cosine
            row['fourier_mix_MACs'] = fourier
            row['covered_MACs'] = conv_linear[0] + row['attention_matmul_MACs'] + fourier
            row['raw_ms'] = times
            row['median_ms'] = statistics.median(times)
            row['min_ms'] = min(times)
            row['max_ms'] = max(times)
            # Both raw tensors remain live across blocks in reuse; this is
            # tensor storage, not total autograd memory or a savings guarantee.
            row['raw_qk_payload_bytes'] = 2*n*c*4 if mode=='reuse' else 0
            rows[mode] = row
            print(h,w,mode,row['deployed_parameters'],round(row['median_ms'],3),'ms',flush=True)
            del model,x
            gc.collect()
            if a.device=='cuda':
                torch.cuda.empty_cache()
        report['sizes'][f'{h}x{w}'] = rows
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


def math_prod(values):
    result = 1
    for value in values:
        result *= value
    return result


if __name__=='__main__':
    main()
