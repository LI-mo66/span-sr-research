"""CPU deployment graph profile. Needs PyTorch; no checkpoint or data required."""
import argparse
import json
from pathlib import Path
import sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--height', type=int, default=360)
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert min(args.height, args.width) > 0
root = args.repo.resolve()
sys.path.insert(0, str(root / 'research/f0_x2_finetune'))
import run
from deploy import fused_copy

torch.set_num_threads(4)
torch.manual_seed(10)
cfg = json.loads((root / 'research/multiscale_f0/template.json').read_text())
cfg['arm'] = 'F0'
cfg['model']['upscale'] = 2
arch, _ = run.fusion.components()
original = run.fusion.build_model(arch, cfg).eval()
stored_parameters = sum(p.numel() for p in original.parameters())
model = fused_copy(original, arch.Conv3XC).eval()
assert sum(p.numel() for p in model.parameters()) == 552003
x = torch.rand(1, 3, args.height, args.width)
with torch.inference_mode():
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True, record_shapes=True,
    ) as prof:
        y = model(x)
operations = [dict(op=e.key, flops=e.flops, calls=e.count)
              for e in prof.key_averages() if e.flops]
result = dict(
    input_shape=list(x.shape), output_shape=list(y.shape),
    training_parameters=stored_parameters, deployed_parameters=552003,
    torch=str(torch.__version__), device='CPU', operations=operations,
    core_flops=sum(e['flops'] for e in operations
                   if e['op'] in ('aten::conv2d', 'aten::addmm', 'aten::bmm')),
    profiler_supported_flops=sum(e['flops'] for e in operations),
    scope='Supported operator counts only. Multiply/add separate. Not full all-operator FLOPs. Random weights, single forward, no x8 or latency claim.',
)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
