"""Same-budget, validation-selected comparison, including paired image diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
from bootstrap import trainer

def rows(run):
    values=[json.loads(s) for s in (run/'epochs.jsonl').read_text().splitlines() if s.strip()]
    if [r['update'] for r in values]!=list(range(1000,50001,1000)):
        raise ValueError('Complete 1K..50K ordered history required: '+str(run))
    return values,max(values,key=lambda r:r['metrics']['DIV2K_valid']['psnr'])

def compare(frozen,run,output):
    audit=json.loads((frozen/'reuse-audit.json').read_text())
    reference=Path(audit['reference_run'])
    for file,key in [('manifest.json','reference_manifest_sha256'),('epochs.jsonl','reference_history_sha256')]:
        if trainer.digest(reference/file)!=audit[key]:raise ValueError('Protected SG6 archive changed')
    base,bs=rows(reference);candidate,cs=rows(run)
    info=json.loads((run/'manifest.json').read_text())
    if info['commit']!=audit['candidate_commit']:raise ValueError('Candidate source differs')
    result=dict(status='COMPLETE_SINGLE_SEED',reference_selected_update=bs['update'],candidate_selected_update=cs['update'],
        candidate_source=info['commit'],seed=10,updates=50000,selection='DIV2K_valid PSNR only; earliest tie',datasets={})
    for name in bs['metrics']:
        def images(directory,record):
            return json.loads((directory/f'{name}_epoch{record["epoch"]:05d}_u{record["update"]}.json').read_text())
        br,cr=images(reference,bs),images(run,cs)
        if [r['image'] for r in br]!=[r['image'] for r in cr]:raise ValueError('Paired evaluation order differs')
        delta=np.array([c['psnr']-b['psnr'] for b,c in zip(br,cr)])
        generator=np.random.default_rng(104821)
        boots=np.array([generator.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
        result['datasets'][name]=dict(reference=bs['metrics'][name],candidate=cs['metrics'][name],
            selected_delta_psnr=float(delta.mean()),selected_delta_ssim=cs['metrics'][name]['ssim']-bs['metrics'][name]['ssim'],
            final_delta_psnr=candidate[-1]['metrics'][name]['psnr']-base[-1]['metrics'][name]['psnr'],
            last3_mean_delta_psnr=float(np.mean([c['metrics'][name]['psnr']-b['metrics'][name]['psnr'] for b,c in zip(base[-3:],candidate[-3:])])),
            paired_median=float(np.median(delta)),win_fraction=float((delta>0).mean()),
            image_bootstrap95=list(np.quantile(boots,[.025,.975])),
            per_image=[dict(image=c['image'],delta_psnr=float(d)) for c,d in zip(cr,delta)])
    validation=result['datasets']['DIV2K_valid']
    result['decision']='STOP' if validation['selected_delta_psnr']<=0 or validation['last3_mean_delta_psnr']<=0 else 'REVIEW_BEFORE_LONG_RUN'
    result['limitations']='Single-seed short screen. Image bootstrap does not measure training-seed variation. Positive deltas are not automatic approval for 1M. Inspect resource cost, feature/update logs, tails and all sets together.'
    output.write_text(json.dumps(result,indent=2),encoding='utf8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(compare(a.group,a.run,a.output),indent=2))
