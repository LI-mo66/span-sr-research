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
    result=dict(reference='SG6',candidate_mode=manifest['config']['ucan_sweep']['arm'],selected_checkpoint=str(checkpoint),selection='DIV2K_valid best in 50K; earliest tie; one weight for all datasets',validation_selected=scores(audit['reference_selected'],best),fixed_50000=scores(audit['reference_final'],rows[-1]),validation_tail_40k_50k=tail,tail_mean_delta=statistics.mean(v['delta_psnr'] for v in tail),tail_median_delta=statistics.median(v['delta_psnr'] for v in tail),scope='single seed10; no cross-seed or causal claim')
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
    add_context_comparison(result, a.run, best, rows[-1], audit)
    add_combo_comparison(result,a.run,best,rows,audit)
    a.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

def add_context_comparison(result, run, best, final, audit):
    """Reuse verified prior independent-p24 archive; compare chosen weights."""
    context=Path(audit['context_reference_run'])
    rows,selected=history(context)
    def compare(old,new):
        return {name:{'context_p24':v,'candidate':new['metrics'][name],
                     'delta':{m:new['metrics'][name][m]-v[m] for m in ('psnr','ssim')}}
                for name,v in old['metrics'].items()}
    result['vs_context_p24_selected']=compare(selected,best)
    result['vs_context_p24_fixed_50000']=compare(rows[-1],final)
    result['context_reference_selected_update']=selected['update']

def add_combo_comparison(result,run,best,rows,audit):
    reference=Path(audit['combo_reference_run']);old_rows,old_best=history(reference)
    def compare(old,new):
        return {name:{'prior_combo':v,'candidate':new['metrics'][name],
            'delta':{m:new['metrics'][name][m]-v[m] for m in ('psnr','ssim')}} for name,v in old['metrics'].items()}
    result['vs_prior_combo_selected']=compare(old_best,best)
    result['vs_prior_combo_fixed_50000']=compare(old_rows[-1],rows[-1])
    result['tail_mean_delta_vs_prior_combo']=statistics.mean(r['metrics']['DIV2K_valid']['psnr']-b['metrics']['DIV2K_valid']['psnr'] for r,b in zip(rows,old_rows) if r['update']>=40000)
    paired={}
    for name in best['metrics']:
        def load(folder,row):
            return {v['image']:v['psnr'] for v in json.loads((folder/f'{name}_epoch{row["epoch"]:05d}_u{row["update"]}.json').read_text())}
        old,new=load(reference,old_best),load(run,best)
        if old.keys()!=new.keys():raise ValueError('Prior combo paired image mismatch')
        delta=np.array([new[k]-old[k] for k in sorted(old)])
        boot=np.random.default_rng(10).choice(delta,size=(2000,len(delta)),replace=True).mean(1)
        paired[name]={'mean_delta':float(delta.mean()),'median_delta':float(np.median(delta)),
            'win_fraction':float((delta>0).mean()),'image_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
            'worst':sorted([{'image':k,'delta':float(new[k]-old[k])} for k in old],key=lambda v:v['delta'])[:5],
            'scope':'selected checkpoints; image uncertainty only; no cross-seed or mechanism claim'}
    result['paired_images_vs_prior_combo']=paired

if __name__=='__main__': main()
