"""Difference versus equal-capacity addition; selected weights and curve tails."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare import rows


def compare(add,difference,output):
    ah,ar=rows(add);dh,dr=rows(difference)
    result=dict(add_selected_update=ar['update'],difference_selected_update=dr['update'],datasets={},
        limitation='Single seed10/50K. Sign changes initialization/parameterization; it need not expand the function class.')
    for name in ar['metrics']:
        ai=json.loads((add/f'{name}_epoch{ar["epoch"]:05d}_u{ar["update"]}.json').read_text())
        di=json.loads((difference/f'{name}_epoch{dr["epoch"]:05d}_u{dr["update"]}.json').read_text())
        if [r['image'] for r in ai]!=[r['image'] for r in di]:raise ValueError('Image order mismatch')
        delta=np.array([d['psnr']-a['psnr'] for a,d in zip(ai,di)])
        result['datasets'][name]=dict(selected_delta_psnr=float(delta.mean()),
            selected_delta_ssim=dr['metrics'][name]['ssim']-ar['metrics'][name]['ssim'],
            final_delta_psnr=dh[-1]['metrics'][name]['psnr']-ah[-1]['metrics'][name]['psnr'],
            last3_mean_delta_psnr=float(np.mean([d['metrics'][name]['psnr']-a['metrics'][name]['psnr'] for a,d in zip(ah[-3:],dh[-3:])])),
            last10_mean_delta_psnr=float(np.mean([d['metrics'][name]['psnr']-a['metrics'][name]['psnr'] for a,d in zip(ah[-10:],dh[-10:])])),
            paired_median=float(np.median(delta)),win_fraction=float((delta>0).mean()))
    output.write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--add',type=Path,required=True)
    p.add_argument('--difference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(compare(a.add,a.difference,a.output),indent=2))
