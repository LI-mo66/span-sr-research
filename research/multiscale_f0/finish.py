"""Compare validation-selected checkpoints; keep test-set selection out of decisions."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
scale=int(sys.argv[1]);report={}
for arm in ('B0','F0'):
    run=ROOT/'runs'/f'MS-X{scale}-{arm}-S10-50K-01'
    history=[json.loads(s) for s in (run/'epochs.jsonl').open() if s.strip()]
    assert [r['update'] for r in history]==list(range(1000,50001,1000))
    best=min(history,key=lambda r:(-r['metrics']['DIV2K_valid']['psnr'],r['update']))
    weight=f'epoch{best["epoch"]:05d}_u{best["update"]}.pt'
    assert (run/weight).is_file()
    report[arm]=dict(update=best['update'],weight=weight,metrics=best['metrics'])
report['delta']={name:{metric:report['F0']['metrics'][name][metric]-values[metric] for metric in ('psnr','ssim')} for name,values in report['B0']['metrics'].items()}
report['scope']='Single-seed50K; each model selected by plain DIV2K PSNR, same checkpoint on all sets; x8 secondary'
out=ROOT/f'logs/multiscale50k/comparison-x{scale}.json'
out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
