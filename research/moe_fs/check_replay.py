"""Validate crop RNG equivalence and instrumented replay on the synthetic fixture."""
import json
import random
import subprocess
import sys
import uuid
from pathlib import Path
from replay import crop, runtime
import torch

ROOT = runtime.ROOT
runs = sorted(ROOT.glob('runs/ENGINEERING-MOEFS-*'))
run = None
# Pick a fixture built by the current independent trainer, with per-step records.
for candidate in reversed(runs):
    if candidate.name.endswith('-continuous') or not (candidate / 'epoch00001_u1.pt').exists():
        continue
    if not (candidate / 'loss_steps.jsonl').exists():
        continue
    info = json.loads((candidate / 'manifest.json').read_text())
    if info['environment']['torch'] == torch.__version__:
        run = candidate
        break
if run is None:
    raise RuntimeError('Run integration.py first to create a synthetic fixture')
row = info['data']['train'][0]
pair = (Path(row['hr']), Path(row['lr']))
random.seed(2026)
saved = random.getstate()
a, b = runtime.trainer.sample(pair, info['config']['hr_patch'])
after = random.getstate()
random.setstate(saved)
x, y, details = crop(pair, info['config']['hr_patch'])
assert torch.equal(a, x) and torch.equal(b, y) and random.getstate() == after
# Baseline fixture factory: remove candidate-only adapter state and reset optimizer
# cannot represent the saved candidate. Instead create a baseline fixture through
# the real independent training entry point, then compare checkpoint 2 replay.
fixture = ROOT / 'tmp' / ('baseline-replay-' + uuid.uuid4().hex[:8])
fixture.mkdir(parents=True)
cfg = dict(info['config'], moe_fs=False)
config = fixture / 'config.json'
config.write_text(json.dumps(cfg))
baseline_id = 'ENGINEERING-BASELINE-REPLAY-' + uuid.uuid4().hex[:8]
entry = Path(__file__).parent / 'run.py'
base_command = [sys.executable, str(entry), '--config', str(config), '--id', baseline_id]
subprocess.run(base_command + ['--screen', '--screen-updates', '5000'], cwd=ROOT, check=True)
baseline = ROOT / 'runs' / baseline_id
original_root = pair[0].parents[2]
output = fixture / 'diagnostic'
subprocess.run([sys.executable, str(Path(__file__).parent / 'replay.py'), '--run', str(baseline),
                '--checkpoint', 'epoch00001_u1.pt', '--data-root', str(original_root),
                '--original-data-root', str(original_root), '--until', '2', '--output', str(output)], cwd=ROOT, check=True)
replayed = torch.load(output / 'diagnostic-final.pt', map_location='cpu', weights_only=False)
reference = torch.load(baseline / 'latest.pt', map_location='cpu', weights_only=False)
assert all(torch.equal(v, reference['model'][key]) for key, v in replayed['model'].items())
assert replayed['scheduler'] == reference['scheduler']
print('PASS crop pixels/RNG and instrumented baseline replay state exactly match normal training.')
