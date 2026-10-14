"""Matched FP32 inference benchmark; random weights are not accuracy evidence."""
import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('easa_runtime', HERE / 'runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
import torch
from model import build_model
from deploy import fused_copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--height', type=int, default=180)
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads((HERE / 'configs/E1-mid-lr2e4.json').read_text(encoding='utf-8-sig'))
    arch, _ = runtime.components()
    models = {}
    for label, enabled in (('B0', False), ('E1', True)):
        torch.manual_seed(cfg['seed'])
        model = build_model(arch, dict(cfg, easa=enabled)).cuda().eval()
        models[label + '_original'] = model
        models[label + '_prefused'] = fused_copy(model, arch.Conv3XC)
    x = torch.rand(1, 3, args.height, args.width, device='cuda')
    report = dict(torch=torch.__version__, device=torch.cuda.get_device_name(),
                  precision='FP32', input_shape=list(x.shape), warmup=5, repeats=20,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  memory_scope='All four models resident for all measurements; no training activations.',
                  evidence='random initialization: resource evidence only', models={})
    with torch.inference_mode():
        for label, model in models.items():
            for _ in range(5):
                model(x)
            torch.cuda.synchronize()
            resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            report['models'][label] = dict(
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                median_ms=statistics.median(times), mean_ms=statistics.mean(times),
                resident_allocated_bytes=resident,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_extra_allocated_bytes=torch.cuda.max_memory_allocated() - resident,
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
        for label in ('B0_prefused', 'E1_prefused'):
            macs = [0]
            def count(module, inputs, output):
                macs[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            hooks = [m.register_forward_hook(count) for m in models[label].modules()
                     if isinstance(m, torch.nn.Conv2d)]
            models[label](x)
            for hook in hooks:
                hook.remove()
            report['models'][label]['convolution_MACs'] = macs[0]
    report['MAC_coverage'] = 'Conv2d only, including EASA coarse grid. Excludes normalization, pooling, variance, GELU, sigmoid, interpolation, products and layout. Not full FLOPs.'
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
