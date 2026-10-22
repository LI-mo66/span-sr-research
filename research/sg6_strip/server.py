"""Linux single-GPU serial queue. Every candidate has an independent ID and backup."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def call(group,label,script,*args):
    command=[sys.executable,'-B','-u',str(HERE/script),*map(str,args)]
    print('START',label,command,flush=True)
    with (group/(label+'.log')).open('a',encoding='utf8') as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    print('END',label,'exit',result.returncode,flush=True)
    if result.returncode:raise RuntimeError(label+' failed; inspect '+str(group/(label+'.log')))

def backup(run,destination,exit_code):
    if not run.exists():return dict(status='NO_RUN_CREATED')
    files=[p for p in run.rglob('*') if p.is_file()]
    size=sum(p.stat().st_size for p in files)
    if shutil.disk_usage(destination.parent).free<size+2*1024**3:raise OSError('Insufficient backup disk space')
    destination.mkdir(exist_ok=False)
    receipt=dict(training_exit_code=exit_code,source=str(run),files={})
    for source in files:
        rel=source.relative_to(run);target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        original=digest(source)
        if digest(target)!=original:raise IOError('Backup hash mismatch: '+str(source))
        receipt['files'][str(rel)]=dict(bytes=source.stat().st_size,sha256=original)
    (destination/'backup-manifest.json').write_text(json.dumps(receipt,indent=2))
    (destination/'BACKUP_COMPLETE').write_text('All copied files verified; training success is a separate field.\n')
    return dict(status='HASH_VERIFIED',path=str(destination),bytes=size)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-backup',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--max-time-ratio',type=float,default=1.10)
    p.add_argument('--resume-group',type=Path)
    p.add_argument('--shutdown-after-queue',action='store_true')
    p.add_argument('--shutdown-bin')
    args=p.parse_args()
    if args.max_time_ratio<1:raise ValueError('Invalid time allowance')
    lockpath=Path('/home/featurize/span-x4-runtime/logs/server-gpu.lock')
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip()
        if occupied:raise RuntimeError('GPU already has compute processes: '+occupied)
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Source checkout must be clean')
        from power import plan,finish
        shutdown_command=plan(args.shutdown_bin) if args.shutdown_after_queue else None
        if args.resume_group:
            group=args.resume_group.resolve()
            audit=json.loads((group/'frozen/reuse-audit.json').read_text())
            current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            if audit['candidate_commit']!=current:raise ValueError('Resume source differs')
            for file,key in [('manifest.json','reference_manifest_sha256'),('epochs.jsonl','reference_history_sha256')]:
                if digest(Path(audit['reference_run'])/file)!=audit[key]:raise ValueError('SG6 reference changed')
            if digest(Path(audit['reference_weight']))!=audit['reference_weight_sha256']:raise ValueError('SG6 reference weight changed')
            if not (group/'preflight-complete.json').exists():raise ValueError('Interrupted preflight: use a new group; old logs are retained')
        else:
            stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            args.output_root.mkdir(parents=True,exist_ok=True)
            group=args.output_root/('SG6-STRIP-'+stamp);group.mkdir(exist_ok=False)
            print('GROUP',group,flush=True)
            call(group,'prepare','prepare.py','--reference-backup',args.reference_backup,'--data-root',args.data_root,'--output',group/'frozen')
            call(group,'integration','integration.py')
            for k in (15,31):call(group,f'verify-K{k}','verify.py','--kernel',k,'--device','cuda','--output',group/f'verify-K{k}.json')
            call(group,'features','features.py','--group',group/'frozen','--output',group/'features.json')
            call(group,'resources','resources.py','--group',group/'frozen','--output',group/'resources.json')
            for k in (15,31):call(group,f'data-preflight-K{k}','run.py','--config',group/f'frozen/K{k}.json','--id',f'PREFLIGHT-STRIP-K{k}','--preflight')
            (group/'preflight-complete.json').write_text(json.dumps(dict(status='PASS',max_time_ratio=args.max_time_ratio)))
        # Freeze the cost threshold at the original preflight; resume cannot weaken it.
        maximum=json.loads((group/'preflight-complete.json').read_text())['max_time_ratio']
        resources=json.loads((group/'resources.json').read_text())
        status={}
        for k in (15,31):
            label=f'K{k}';rid=f'{group.name}-{label}-B3-s10';run=ROOT/'runs'/rid
            ratio=resources['results'][label]['update_time_ratio_vs_SG6']
            if ratio>maximum:
                status[label]=dict(status='HOLD_COST',ratio=ratio,maximum=maximum)
                print(label,'HELD: synthetic update time allowance exceeded',ratio,flush=True);continue
            training_code=1
            try:
                if (run/'latest.pt').exists():
                    import torch
                    checkpoint=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
                    update=checkpoint['update'];del checkpoint
                else:update=0
                if update<50000:
                    extra=['--resume'] if run.exists() else []
                    call(group,'train-'+label,'run.py','--config',group/f'frozen/{label}.json','--id',rid,'--screen','--screen-updates',50000,*extra)
                training_code=0
                call(group,'compare-'+label,'compare.py','--group',group/'frozen','--run',run,'--output',group/(label+'-comparison.json'))
                status[label]=dict(status='COMPLETE',run=str(run))
            except Exception:
                traceback.print_exc();status[label]=dict(status='FAILED',run=str(run))
            finally:
                archive=group/(label+'-backup-'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
                status[label]['training_exit_code']=training_code
                status[label]['backup']=backup(run,archive,training_code)
                (group/'queue-status.json').write_text(json.dumps(status,indent=2))
        (group/'queue-status.json').write_text(json.dumps(status,indent=2))
        from collect import collect
        returned=collect(group)
        print('QUEUE FINISHED',group,json.dumps(status,indent=2),flush=True)
        if shutdown_command:finish(group,status,shutdown_command,returned)
        if any(v['status']!='COMPLETE' for v in status.values()):return 2
    return 0

if __name__=='__main__':sys.exit(main())
