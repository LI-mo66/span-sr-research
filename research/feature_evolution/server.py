"""Linux nohup queue: audited SG6 reuse, two serial scratch screens, backup, shutdown."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from backup_utils import backup,digest
from collect import collect
from power import plan,finish

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LABELS=('add','difference')


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def call(group,label,script,*args,observe=False):
    env=os.environ.copy()
    if observe:env['FE_OBSERVE']='1'
    command=[sys.executable,'-B','-u',str(HERE/script),*map(str,args)]
    print('START',label,command,flush=True)
    with (group/(label+'.log')).open('a',encoding='utf-8') as stream:
        with subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
            for line in child.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
            code=child.wait()
    print('END',label,'exit',code,flush=True)
    if code:raise RuntimeError(label+' failed; see '+str(group/(label+'.log')))


def verify_observations(run):
    path=run/'feature-observations.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    updates={r['update']:r for r in rows if r['kind']=='actual_update'}
    expected=set(range(1,13))|set(range(1000,50001,1000))
    if set(updates)!=expected:raise ValueError('Incomplete real-update observations')
    first=updates[1]['norms']
    if first['output.weight']<=0:raise ValueError('Fusion output did not start')
    for name in first:
        if not any(r['norms'][name]>0 for step,r in updates.items() if step>1):
            raise ValueError('New parameter never updated: '+name)
    return dict(status='PASS',sampled_updates=len(updates),path=str(path),
                note='Sampled real data only; this is not a PSNR or no-gradient-conflict proof')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--group',type=Path,required=True)
    p.add_argument('--reference-backup',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--shutdown',action='store_true')
    args=p.parse_args();group=args.group.resolve()
    lockpath=Path('/home/featurize/span-x4-runtime/logs/server-gpu.lock')
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    status={};active_run=None;active_label=None;exit_code=1
    with lockpath.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                raise RuntimeError('GPU already occupied; no job was interrupted')
            if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
                raise RuntimeError('Clean committed source required')
            shutdown_command=plan() if args.shutdown else None
            write(group/'server-info.json',dict(pid=os.getpid(),python=sys.executable,source=str(ROOT),
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                variants=LABELS,updates_each=50000,automatic_shutdown=args.shutdown,
                shutdown_command=shutdown_command))
            if shutil.disk_usage(group).free<12*1024**3:
                raise OSError('Need at least 12GiB free in persistent output filesystem')
            subprocess.run(['git','archive','--format=tar.gz','--output',str(group/'source.tar.gz'),'HEAD'],cwd=ROOT,check=True)
            call(group,'prepare','prepare.py','--reference-backup',args.reference_backup,
                 '--data-root',args.data_root,'--output',group/'frozen')
            audit=json.loads((group/'frozen/reuse-audit.json').read_text())
            os.environ['FE_REFERENCE_AUDIT']=str(group/'frozen/reuse-audit.json')
            reference=Path(audit['reference_run'])
            refdest=group/'reference';refdest.mkdir()
            for path in reference.iterdir():
                if path.is_file() and path.suffix in ('.json','.jsonl'):
                    shutil.copy2(path,refdest/path.name)
            call(group,'verify-cuda','verify.py','--device','cuda','--output',group/'verify-cuda.json')
            for label in LABELS:
                call(group,'integration-'+label,'integration.py','--variant','SG6-FE-'+label,observe=True)
                call(group,'data-preflight-'+label,'entry.py','--config',group/f'frozen/SG6-FE-{label}.json',
                     '--id','PREFLIGHT-FE-'+label,'--preflight')
            call(group,'profile','inference_profile.py','--output',group/'profile.json')
            write(group/'preflight-complete.json',dict(status='PASS',baseline_reuse='AUDITED',
                shutdown_permission_checked=bool(shutdown_command)))
            for label in LABELS:
                active_label=label
                run_id=group.name+'-'+label+'-s10'
                active_run=ROOT/'runs'/run_id
                if active_run.exists():raise FileExistsError('Existing run preserved: '+str(active_run))
                code=1
                try:
                    call(group,'train-'+label,'entry.py','--config',group/f'frozen/SG6-FE-{label}.json',
                         '--id',run_id,'--screen','--screen-updates','50000',observe=True)
                    code=0
                    call(group,'compare-'+label,'compare.py','--group',group/'frozen','--run',active_run,
                         '--output',group/f'comparison-{label}.json')
                    observation=verify_observations(active_run)
                    status[label]=dict(status='COMPLETE',run=str(active_run),training_exit_code=0,
                                       observations=observation)
                except Exception:
                    status[label]=dict(status='FAILED',run=str(active_run),training_exit_code=code)
                    raise
                finally:
                    saved=backup(active_run,group/(label+'-backup'),code)
                    status.setdefault(label,dict(status='FAILED',run=str(active_run),training_exit_code=code))['backup']=saved
                    write(group/'queue-status.json',status)
                active_run=None
            call(group,'compare-pair','compare_pair.py','--add',status['add']['run'],
                 '--difference',status['difference']['run'],'--output',group/'comparison-difference-vs-add.json')
            sys.path.insert(0,str(ROOT/'research/moe_fs'))
            from curves import generate
            generate(dict(SG6=reference,**{k:Path(v['run']) for k,v in status.items()}),group/'curves/seed-10')
            write(group/'FINAL_STATUS.json',dict(status='COMPLETE',variants=status,automatic_shutdown=args.shutdown))
            output=collect(group,status)
            print('ALL COMPLETE. Full weights:',[v['backup']['path'] for v in status.values()],flush=True)
            print('Download for review:',output,flush=True)
            if shutdown_command:finish(group,status,shutdown_command,output)
            exit_code=0
        except Exception:
            (group/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            write(group/'FINAL_STATUS.json',dict(status='FAILED',variants=status,
                active_label=active_label,automatic_shutdown=False,reason='Inspect failure.txt; server retained'))
            print(traceback.format_exc(),flush=True)
            try:
                if not (group/'SG6-FE-RETURN.zip').exists():collect(group,status)
            except Exception:print('Failure-return packaging also failed:',traceback.format_exc(),flush=True)
            print('QUEUE FAILED; server retained; no automatic resume or shutdown.',flush=True)
    return exit_code


if __name__=='__main__':sys.exit(main())
