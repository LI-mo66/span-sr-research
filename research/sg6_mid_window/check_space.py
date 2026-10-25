"""Conservative space gate for both runs plus persistent backup, before 50K."""
import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--group-dir',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args()
    # 50 periodic weights + latest + best, per candidate; includes optimizer states.
    runs=sum(json.loads((args.group_dir/f'real-data-smoke-{kind}.json').read_text())['estimated_checkpoint_bytes']*52 for kind in ('A','C'))
    same=os.stat(args.project).st_dev==os.stat(args.destination).st_dev
    project_needed=(2*runs if same else runs)+1024**3
    backup_needed=project_needed if same else runs+1024**3
    if shutil.disk_usage(args.project).free<project_needed or shutil.disk_usage(args.destination).free<backup_needed:
        raise OSError('Insufficient estimated space for two complete runs and verified backup; instance retained')
    report=dict(status='PASS',estimated_two_run_bytes=runs,project_required_free=project_needed,
                backup_required_free=backup_needed,same_filesystem=same)
    (args.group_dir/'disk-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
