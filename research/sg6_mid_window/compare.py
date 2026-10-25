"""One DIV2K-selected checkpoint per model, with paired-image descriptive evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from prepare import history


def difference(reference, candidate, label):
    if set(reference['metrics'])!=set(candidate['metrics']):
        raise ValueError('Dataset set differs')
    return {name:dict(SG6=scores,**{label:candidate['metrics'][name]},
                     delta_vs_SG6={m:candidate['metrics'][name][m]-scores[m] for m in ('psnr','ssim')})
            for name,scores in reference['metrics'].items()}


def selected_file(folder,name,row):
    return folder/f'{name}_epoch{row["epoch"]:05d}_u{row["update"]}.json'


def paired(reference_folder, run, old, selected):
    report={}
    rng=np.random.default_rng(20261025)
    for name in selected['metrics']:
        a=json.loads(selected_file(reference_folder,name,old).read_text())
        b=json.loads(selected_file(run,name,selected).read_text())
        if len({r['image'] for r in a})!=len(a) or len({r['image'] for r in b})!=len(b):
            raise ValueError('Duplicate image identifiers')
        am,bm={r['image']:r for r in a},{r['image']:r for r in b}
        if set(am)!=set(bm):raise ValueError('Paired image IDs differ: '+name)
        for metrics,summary in ((a,old),(b,selected)):
            for metric in ('psnr','ssim'):
                if abs(float(np.mean([r[metric] for r in metrics]))-summary['metrics'][name][metric])>1e-9:
                    raise ValueError('Per-image average differs from history: '+name)
        rows=[dict(image=k,delta_psnr=bm[k]['psnr']-am[k]['psnr'],
                   delta_ssim=bm[k]['ssim']-am[k]['ssim']) for k in sorted(am)]
        values=np.array([r['delta_psnr'] for r in rows])
        boot=values[rng.integers(0,len(values),size=(5000,len(values)))].mean(axis=1)
        report[name]=dict(mean_delta_psnr=float(values.mean()),median_delta_psnr=float(np.median(values)),
                          improved=int((values>0).sum()),worse=int((values<0).sum()),tied=int((values==0).sum()),
                          image_bootstrap_mean_95=np.quantile(boot,[.025,.975]).tolist(),
                          rows=sorted(rows,key=lambda r:r['delta_psnr']),
                          scope='Paired whole-image differences for this seed; image bootstrap is not training-seed stability or region-level evidence')
    return report


def compare_one(audit, run, variant, include_paired=True):
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['commit']!=audit['candidate_source'] or manifest['config']!=audit['candidate_configs'][variant]:
        raise ValueError('Candidate source/config differs from frozen audit')
    rows,best=history(run)
    weight=f'epoch{best["epoch"]:05d}_u{best["update"]}.pt'
    if not (run/weight).is_file():raise FileNotFoundError(weight)
    label='SG6_'+variant
    result=dict(direct_reference='SG6',variant=variant,fixed_base=audit['fixed_base'],seed=10,budget=50000,
                reference_checkpoint=audit['reference_weight'],candidate_checkpoint=weight,
                rule='DIV2K_valid best within 50K, earliest tie; same selected weight on all test sets',
                validation_selected=difference(audit['reference_selected'],best,label),
                fixed_50000=difference(audit['reference_final'],rows[-1],label),
                scope='Single-seed 50K screen; no convergence/stability or failure-cause attribution claim')
    pairs=paired(Path(audit['reference_run']),run,audit['reference_selected'],best) if include_paired else None
    return result,pairs


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--variant',choices=('A','C'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result,pairs=compare_one(json.loads(args.audit.read_text()),args.run,args.variant)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    args.output.with_name(f'paired-{args.variant}-selected.json').write_text(json.dumps(pairs,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
