import json
import math
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run

group = Path(sys.argv[1])
baseline = json.loads((group / 'baseline-verified.json').read_text())
report = dict(scope='Exploratory five-benchmark comparison, seed10; all arms fresh Adam',
              ranking='Mean plain PSNR delta over five sets, using last5 evaluations', arms={})
for arm in run.ARMS:
    folder = run.ROOT / 'runs' / ('FT-X2-982K-' + arm + '-10K-v1')
    if not (folder / 'COMPLETE.json').is_file():
        continue
    epochs = [json.loads(line) for line in (folder / 'epochs.jsonl').open() if line.strip()]
    assert [row['step'] for row in epochs] == list(range(1000, 10001, 1000))
    tail = epochs[-5:]
    averages = {name: {key: mean(row['metrics'][name][key] for row in tail)
                       for key in ('psnr', 'ssim')} for name in baseline}
    final = epochs[-1]['metrics']
    deltas = {name: averages[name]['psnr'] - baseline[name]['psnr'] for name in baseline}
    assert all(math.isfinite(value) for value in deltas.values())
    report['arms'][arm] = dict(final_10k=final, last5_mean=averages,
        last5_delta_vs_982k=deltas,
        target_mean_delta=mean(deltas[name] for name in run.TARGETS),
        all_five_plain_improved=all(deltas[name] > 0 for name in run.TARGETS),
        all_five_x8_improved=all(deltas[name + '_x8'] > 0 for name in run.TARGETS))
run.json_write(group / 'comparison.json', report)
print(json.dumps(report, indent=2), flush=True)
