"""Linux one-arm 50K controller, read-only baseline audit, persistent backup.

The finish.sh wrapper implements user-authorized release after this exits.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    temporary.replace(path)


def command(script,*args):
    subprocess.run([sys.executable,'-u',str(HERE/script),*map(str,args)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--id',default='FEAT-SG6-UCAN-IND-V2-50K-01')
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/home/featurize/datasets'))
    p.add_argument('--backup-root',type=Path,default=Path('/home/featurize/work/span-x4-research/records'))
    a=p.parse_args()
    if not a.id.startswith('FEAT-SG6-UCAN-IND-') or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.id):
        raise ValueError('Invalid independent-arm experiment ID')
    a.reference=a.reference.resolve()
    a.backup_root=a.backup_root.resolve()
    if not a.backup_root.is_relative_to(Path('/home/featurize/work')):
        raise ValueError('Persistent backup root must be inside /home/featurize/work')
    if a.backup_root==a.reference or a.backup_root.is_relative_to(a.reference):
        raise ValueError('Never write into the protected SG6 archive')
    job=ROOT/'logs/ucan_qk_pair'/a.id
    run=ROOT/'runs'/a.id
    # launch.sh creates only this unique job and submission log. It does not
    # create status/config/run files. Existing jobs are never silently reused.
    job.mkdir(parents=True,exist_ok=True)
    if (job/'status.json').exists() or run.exists() or (job/'frozen').exists():
        raise FileExistsError('Existing experiment preserved; no automatic restart')
    status=dict(id=a.id,stage='preflight',training_exit_code=None,comparison_exit_code=None,
                plotting_exit_code=None,backup_exit_code=None,
                auto_release=os.environ.get('UCAN_AUTO_RELEASE')=='1',success=False)
    status_path=job/'status.json'
    a.backup_root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock_path=Path('/home/featurize/span-x4-runtime/logs/server-gpu.lock')
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock=lock_path.open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        status.update(error='Shared GPU queue lock held',exit_code=75)
        write(status_path,status)
        raise SystemExit(75)
    frozen=job/'frozen'
    try:
        write(status_path,status)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                        '--format=csv,noheader'],text=True).strip()
        if active:
            raise RuntimeError('GPU compute processes already present: '+active)
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
            raise ValueError('Commit source before submission')
        # No real data training is reached until audit and both synthetic
        # engineering checks succeed on the actual server environment.
        command('prepare.py','--reference-backup',a.reference,'--output',frozen,
                '--data-root',a.data_root)
        for directory in (ROOT/'runs',ROOT/'datasets/image_cache',a.backup_root):
            directory.mkdir(parents=True,exist_ok=True)
            if shutil.disk_usage(directory).free < 8*1024**3:
                raise OSError('Need at least 8GiB free at '+str(directory))
        status['stage']='synthetic-engineering';write(status_path,status)
        command('verify.py','--device','cuda','--output',frozen/'server-engineering.json')
        command('integration.py','--output',frozen/'server-entry-integration.json')
        command('resources.py','--device','cuda','--output',frozen/'server-resources.json')
        status['stage']='training';write(status_path,status)
        try:
            command('run.py','--config',frozen/'config.json','--id',a.id,
                    '--screen','--screen-updates','50000')
            status['training_exit_code']=0
        except subprocess.CalledProcessError as exc:
            status['training_exit_code']=exc.returncode
            raise
        status['stage']='comparison';write(status_path,status)
        try:
            command('compare.py','--audit',frozen/'audit.json','--run',run,
                    '--output',frozen/'comparison-50000.json')
            status['comparison_exit_code']=0
        except subprocess.CalledProcessError as exc:
            status['comparison_exit_code']=exc.returncode
            raise
        status['stage']='plotting';write(status_path,status)
        sys.path.insert(0,str(ROOT/'research/moe_fs'))
        from curves import generate
        try:
            generate({'SG6-UCAN-independent-v2':run},job/'curves')
            status['plotting_exit_code']=0
        except Exception:
            status['plotting_exit_code']=1
            raise
        status['success']=True
    except Exception:
        status['error']=traceback.format_exc()
        print(status['error'],flush=True)
    # Backup also on preflight/training failure. Null exit codes mean not run.
    status['stage']='backup';write(status_path,status)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=a.backup_root/(a.id+'-'+stamp)
    try:
        command('backup.py','--run',run,'--logs',job,'--destination',destination,
                '--training-exit',status['training_exit_code'] if status['training_exit_code'] is not None else -1,
                '--comparison-exit',status['comparison_exit_code'] if status['comparison_exit_code'] is not None else -1)
        status.update(backup_exit_code=0,backup=str(destination))
    except Exception:
        status.update(backup_exit_code=1,backup_error=traceback.format_exc(),success=False)
    status.update(stage='completed' if status['success'] else 'failed',
                  exit_code=0 if status['success'] else 1)
    write(status_path,status)
    if status['backup_exit_code']==0:
        write(destination/'FINAL_STATUS.json',status)
    write(a.backup_root/(a.id+'-'+stamp+'-status.json'),status)
    print(json.dumps(status,indent=2),flush=True)
    print('Controller finished. The wrapper releases only after verified success; failures retain the instance.',flush=True)
    fcntl.flock(lock,fcntl.LOCK_UN)
    lock.close()
    raise SystemExit(status['exit_code'])


if __name__=='__main__':
    main()
