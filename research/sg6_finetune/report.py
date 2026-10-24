"""CPU-only reader for persistent fine-tuning records; no torch import."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--records',type=Path,default=Path('/home/featurize/work/span-x4-research/records'))
    p.add_argument('--group',default='FEAT-SG6-FT-10K-01')
    args=p.parse_args()
    for path in sorted(args.records.glob(args.group+'-job-*-status.json'))[-2:]:
        print('STATUS',path);print(path.read_text())
    backups=sorted(path for path in args.records.glob(args.group+'-*') if path.is_dir() and (path/'results/manifest.json').is_file())
    if not backups:
        print('No persistent result backup yet. Inspect current nohup.log.');return
    backup=backups[-1];folder=backup/'results'
    print('BACKUP',backup,'COMPLETE',(backup/'BACKUP_COMPLETE').is_file())
    inventory=backup/'backup-manifest.json'
    if inventory.is_file():
        d=json.loads(inventory.read_text());print('training_exit_code',d.get('training_exit_code'),'file_count',len(d.get('files',[])))
    manifest=json.loads((folder/'manifest.json').read_text())
    print('PROVENANCE',json.dumps({k:v for k,v in manifest.items() if k!='data'},indent=2))
    for label in ('A-L1','B-MSE'):
        history=folder/label/'history.json'
        print('HISTORY',label)
        if history.is_file():print(history.read_text())
        else:print('No completed evaluation checkpoint')
    comparison=folder/'comparison.json'
    print('COMPARISON')
    if comparison.is_file():print(comparison.read_text())
    else:print('No completed comparison')
    for path in sorted((folder/'curves').glob('*.png')):print('CURVE',path)


if __name__=='__main__':main()
