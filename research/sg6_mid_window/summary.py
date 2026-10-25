import argparse
import json
from pathlib import Path


def summarize(a,c):
    if a['direct_reference']!='SG6' or c['direct_reference']!='SG6' or a['fixed_base']!=c['fixed_base']:
        raise ValueError('Reference differs')
    selected={}
    for name,row in a['validation_selected'].items():
        other=c['validation_selected'][name]
        if row['SG6']!=other['SG6']:raise ValueError('SG6 selected metrics differ')
        selected[name]=dict(SG6=row['SG6'],A=row['SG6_A'],C=other['SG6_C'],
                            A_minus_SG6=row['delta_vs_SG6'],C_minus_SG6=other['delta_vs_SG6'],
                            A_minus_C={k:row['SG6_A'][k]-other['SG6_C'][k] for k in ('psnr','ssim')})
    targets=('B100','Urban100','Manga109')
    return dict(status='COMPLETED_SCREEN_NOT_ACCURACY_CERTIFICATION',seed=10,budget_each=50000,
                checkpoints=dict(SG6=a['reference_checkpoint'],A=a['candidate_checkpoint'],C=c['candidate_checkpoint']),
                validation_selected=selected,
                target_all_positive_vs_SG6={kind:all(selected[k][kind+'_minus_SG6']['psnr']>0 for k in targets) for kind in ('A','C')},
                A_positive_vs_C_on_all_targets=all(selected[k]['A_minus_C']['psnr']>0 for k in targets),
                interpretation='No averaging away B100/Urban100/Manga109 losses. Positive signs are screening clues, not significance or seed stability. No test-specific checkpoint selection.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--group-dir',type=Path,required=True);args=p.parse_args()
    result=summarize(*[json.loads((args.group_dir/f'comparison-{k}-50000.json').read_text()) for k in ('A','C')])
    (args.group_dir/'comparison-AC-50000.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
