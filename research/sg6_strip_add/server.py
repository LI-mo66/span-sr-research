"""One independent seed10/50K screen. No shutdown, release, or dynamic source updates."""
import argparse,datetime,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from backup_utils import backup,digest
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

def call(group,label,script,*args):
    command=[sys.executable,'-B','-u',str(HERE/script),*map(str,args)]
    print('START',label,command,flush=True)
    with (group/(label+'.log')).open('a',encoding='utf8') as log:
        with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
    print('END',label,'exit',code,flush=True)
    if code:raise RuntimeError(label+' failed; inspect '+str(group/(label+'.log')))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-backup',type=Path,required=True);p.add_argument('--k31-reference',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--resume-group',type=Path);p.add_argument('--max-time-ratio',type=float,default=1.10)
    a=p.parse_args()
    if not 1<=a.max_time_ratio<=2:raise ValueError('Invalid cost allowance')
    lock=Path('/home/featurize/span-x4-runtime/logs/server-gpu.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip()
        if occupied:raise RuntimeError('GPU occupied; preserved existing jobs: '+occupied)
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Clean source required')
        if a.resume_group:
            group=a.resume_group.resolve();audit=json.loads((group/'frozen/reuse-audit.json').read_text())
            if not (group/'preflight-complete.json').is_file():raise ValueError('Incomplete preflight: create a new group; old evidence is preserved')
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            if audit['candidate_commit']!=commit:raise ValueError('Resume source mismatch')
            for name,key in [('manifest.json','reference_manifest_sha256'),('epochs.jsonl','reference_history_sha256')]:
                if digest(Path(audit['reference_run'])/name)!=audit[key]:raise ValueError('SG6 archive changed')
        else:
            a.output_root.mkdir(parents=True,exist_ok=True)
            group=a.output_root/('SG6-STRIPADD-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
            group.mkdir(exist_ok=False)
            print('GROUP',group,flush=True)
            (group/'server-info.json').write_text(json.dumps(dict(pid=os.getpid(),source=str(ROOT),automatic_shutdown=False,python=sys.executable),indent=2))
            try:
                call(group,'environment','check_env.py')
                call(group,'prepare','prepare.py','--reference-backup',a.reference_backup,'--k31-reference',a.k31_reference,'--data-root',a.data_root,'--output',group/'frozen')
                call(group,'verify','verify.py','--kernel',31,'--device','cuda','--output',group/'verify.json')
                call(group,'integration','integration.py')
                call(group,'profile','resource_probe.py','--group',group)
                call(group,'data-preflight','run.py','--config',group/'frozen/ADD31.json','--id','PREFLIGHT-STRIPADD','--preflight')
                (group/'preflight-complete.json').write_text(json.dumps(dict(status='PASS',max_time_ratio=a.max_time_ratio)))
            except Exception:
                (group/'preflight-error.txt').write_text(traceback.format_exc())
                print('PREFLIGHT FAILED; no formal training started. Machine retained.',flush=True)
                raise
        audit=json.loads((group/'frozen/reuse-audit.json').read_text())
        for name,r in audit['k31_provenance']['files'].items():
            if digest(Path(audit['k31_reference'])/name)!=r['sha256']:raise ValueError('K31 reference changed')
        maximum=json.loads((group/'preflight-complete.json').read_text())['max_time_ratio']
        profile=json.loads((group/'profile.json').read_text())
        rid=group.name+'-ADD31-B3-s10';run=ROOT/'runs'/rid;code=1
        if profile['gate_ratio']>maximum:
            status=dict(ADD31=dict(status='HOLD_COST',ratio=profile['gate_ratio'],maximum=maximum))
        else:
            try:
                if (run/'latest.pt').exists():
                    import torch
                    ckpt=torch.load(run/'latest.pt',map_location='cpu',weights_only=False);update=ckpt['update'];del ckpt
                else:update=0
                if update<50000:
                    extra=['--resume'] if run.exists() else []
                    call(group,'train','run.py','--config',group/'frozen/ADD31.json','--id',rid,'--screen','--screen-updates',50000,*extra)
                code=0
                call(group,'compare','compare_all.py','--group',group,'--run',run)
                status=dict(ADD31=dict(status='COMPLETE',run=str(run),training_exit_code=0))
            except Exception:
                traceback.print_exc();status=dict(ADD31=dict(status='FAILED',run=str(run),training_exit_code=code))
            finally:
                target=group/('ADD31-backup-'+str(time.time_ns()))
                status['ADD31']['backup']=backup(run,target,code)
        (group/'queue-status.json').write_text(json.dumps(status,indent=2))
        from collect import collect
        collect(group)
        print('QUEUE FINISHED',json.dumps(status,indent=2),'Machine retained; automatic shutdown is disabled.',flush=True)
        return 0 if status['ADD31']['status']=='COMPLETE' else 2

if __name__=='__main__':sys.exit(main())
