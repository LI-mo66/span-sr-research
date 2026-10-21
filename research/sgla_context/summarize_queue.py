"""Collect prespecified three arms; never select separate test-set checkpoints."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--group',required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if not a.group or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.group):
        raise ValueError('Invalid group')
    root=Path(__file__).resolve().parents[2]
    records={}
    for mode in ('context','original','local'):
        folder=root/'logs/sgla_context'/f'{a.group}-{mode}'
        value=json.loads((folder/'comparison-50000.json').read_text())
        if value['candidate_mode']!=mode: raise ValueError('Comparison mode mismatch')
        archive=Path((folder/'backup-path.txt').read_text().strip())
        metadata=json.loads((archive/'backup-manifest.json').read_text())
        if not (archive/'BACKUP_COMPLETE').is_file() or metadata['training_exit_code']!=0 or metadata['comparison_exit_code']!=0:
            raise ValueError('Incomplete successful backup: '+mode)
        records[mode]=dict(archive=str(archive),result=value)
    summary=dict(reference='SG6 48K validation-selected weight; existing scratch50K run',arms=records,budget_per_arm=50000,seed=10,rule='Compare selected weights and fixed40K..50K trends; no automatic winner/stability claim')
    summary['context_minus_original_selected']={name:{metric:records['context']['result']['validation_selected'][name]['candidate'][metric]-records['original']['result']['validation_selected'][name]['candidate'][metric] for metric in ('psnr','ssim')} for name in records['context']['result']['validation_selected']}
    summary['context_minus_local_selected']={name:{metric:records['context']['result']['validation_selected'][name]['candidate'][metric]-records['local']['result']['validation_selected'][name]['candidate'][metric] for metric in ('psnr','ssim')} for name in records['context']['result']['validation_selected']}
    a.output.write_text(json.dumps(summary,indent=2)); print('Saved three-arm summary:',a.output)

if __name__=='__main__': main()
