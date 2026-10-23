"""Validation-selected single-weight six-dataset report plus prespecified tail."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
from prepare import history

def main():
    p=argparse.ArgumentParser(); p.add_argument('--audit',type=Path,required=True); p.add_argument('--run',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); audit=json.loads(a.audit.read_text()); manifest=json.loads((a.run/'manifest.json').read_text())
    if manifest['commit']!=audit['candidate_source'] or manifest['config']!=audit['candidate_config']:
        raise ValueError('Candidate source/config differs from audit')
    if manifest['environment']!=audit['environment']: raise ValueError('Environment differs from reference audit')
    rows,best=history(a.run); base_rows,_=history(Path(audit['reference_run']))
    checkpoint=a.run/f'epoch{best["epoch"]:05d}_u{best["update"]}.pt'
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    def scores(base,candidate):
        return {name:dict(SG6=metrics,candidate=candidate['metrics'][name],delta={m:candidate['metrics'][name][m]-metrics[m] for m in ('psnr','ssim')}) for name,metrics in base['metrics'].items()}
    tail=[dict(update=r['update'],delta_psnr=r['metrics']['DIV2K_valid']['psnr']-b['metrics']['DIV2K_valid']['psnr']) for r,b in zip(rows,base_rows) if r['update']>=40000]
    result=dict(reference='SG6',candidate_mode=manifest['config']['ucan_pair']['mode'],selected_checkpoint=str(checkpoint),selection='DIV2K_valid best in 50K; earliest tie; one weight for all datasets',validation_selected=scores(audit['reference_selected'],best),fixed_50000=scores(audit['reference_final'],rows[-1]),validation_tail_40k_50k=tail,tail_mean_delta=statistics.mean(v['delta_psnr'] for v in tail),tail_median_delta=statistics.median(v['delta_psnr'] for v in tail),scope='single seed10; no cross-seed or causal claim')
    paired={}
    reference_best=audit['reference_selected']
    for name in best['metrics']:
        def load(folder,row):
            file=folder/f'{name}_epoch{row["epoch"]:05d}_u{row["update"]}.json'
            data=json.loads(file.read_text())
            return {r['image']:r['psnr'] for r in data}
        old=load(Path(audit['reference_run']),reference_best); new=load(a.run,best)
        if old.keys()!=new.keys(): raise ValueError('Paired image set mismatch: '+name)
        delta=np.array([new[k]-old[k] for k in sorted(old)])
        rng=np.random.default_rng(10)
        boot=np.mean(rng.choice(delta,size=(2000,len(delta)),replace=True),axis=1)
        paired[name]=dict(images=len(delta),mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),win_fraction=float((delta>0).mean()),worst_delta=float(delta.min()),image_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),scope='image uncertainty only; not cross-training-seed stability')
    result['paired_images_selected_weight']=paired
    a.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
