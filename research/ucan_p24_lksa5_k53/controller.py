"""Two-phase arm controller; queue owns shared GPU lock for its whole lifetime."""
import argparse,json,os,shutil,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
LOCK='/home/featurize/span-x4-runtime/logs/server-gpu.lock'

def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def command(script,*args):
    subprocess.run([sys.executable,'-u',str(HERE/script),*map(str,args)],cwd=ROOT,check=True)

def context_audit(path,frozen):
    from backup import sha
    path=path.resolve()
    if not (path/'BACKUP_COMPLETE').is_file():raise ValueError('Context archive missing completion marker')
    status=json.loads((path/'FINAL_STATUS.json').read_text())
    if not status.get('success') or status.get('exit_code')!=0:raise ValueError('Context reference failed')
    backup=json.loads((path/'backup-manifest.json').read_text())
    if backup['source_commit']!='d4dbdbcf24a4297410d097bdea800dd103ae22c6':raise ValueError('Wrong context source')
    def check(file):
        key=file.relative_to(path).as_posix()
        if sha(file)!=backup['files'][key]:raise ValueError('Context archive hash mismatch: '+key)
    from prepare import history
    runs=list((path/'runs').glob('*/manifest.json'))
    if len(runs)!=1:raise ValueError('Ambiguous context reference')
    run=runs[0].parent;check(runs[0]);m=json.loads(runs[0].read_text())
    check(run/'epochs.jsonl');rows,best=history(run)
    if best['update']!=48000:raise ValueError('Context selected update mismatch')
    check(run/'epoch00048_u48000.pt')
    cfg=json.loads((frozen/'config.json').read_text());audit=json.loads((frozen/'audit.json').read_text())
    if m['commit']!=backup['source_commit'] or m['environment']!=audit['environment']:raise ValueError('Context provenance mismatch')
    if m['config']['experiment_version']!='SG6-UCAN-QKPair-v2-independent':raise ValueError('Wrong context model')
    if m['config']['ucan_pair']['positions']!=[2,4] or m['config']['ucan_pair']['mode']!='independent':raise ValueError('Wrong context placement')
    for k,v in cfg.items():
        if k not in ('ucan_sweep','paths_file','experiment_version') and m['config'].get(k)!=v:raise ValueError('Context protocol mismatch: '+k)
    base=json.loads((Path(audit['reference_run'])/'manifest.json').read_text())
    for name,old in base['data'].items():
        new=m['data'][name]
        if [(r['hr_sha256'],r['lr_sha256']) for r in old]!=[(r['hr_sha256'],r['lr_sha256']) for r in new]:raise ValueError('Context data/order mismatch')
    audit['context_reference_run']=str(run);audit['context_reference_selected']=best
    write(frozen/'audit.json',audit)

def combo_audit(path,frozen):
    from backup import sha
    from prepare import history
    path=path.resolve()
    if not (path/'BACKUP_COMPLETE').is_file():raise ValueError('Missing prior combo archive')
    status=json.loads((path/'FINAL_STATUS.json').read_text())
    if not status.get('success') or status.get('exit_code')!=0:raise ValueError('Prior combo failed')
    backup=json.loads((path/'backup-manifest.json').read_text())
    if backup['source_commit']!='d42b73fef5b2ec3e088f340ab033a5c4ac13a190':raise ValueError('Wrong prior combo source')
    def check(file):
        if sha(file)!=backup['files'][file.relative_to(path).as_posix()]:raise ValueError('Prior combo hash mismatch: '+str(file))
    files=list((path/'runs').glob('*/manifest.json'))
    if len(files)!=1:raise ValueError('Ambiguous prior combo run')
    run=files[0].parent;check(files[0]);m=json.loads(files[0].read_text())
    check(run/'epochs.jsonl');rows,selected=history(run)
    if selected['update']!=48000 or abs(selected['metrics']['DIV2K_valid']['psnr']-30.28196911823231)>1e-9:raise ValueError('Prior combo selection mismatch')
    check(run/'epoch00048_u48000.pt')
    for row in (selected,rows[-1]):
        for name in row['metrics']:check(run/f'{name}_epoch{row["epoch"]:05d}_u{row["update"]}.json')
    audit=json.loads((frozen/'audit.json').read_text());cfg=audit['candidate_config']
    if m['commit']!=backup['source_commit'] or m['environment']!=audit['environment']:raise ValueError('Prior combo provenance mismatch')
    reference_cfg=json.loads((HERE/'reference_config.json').read_text())
    for k,v in reference_cfg.items():
        if k!='paths_file' and m['config'].get(k)!=v:raise ValueError('Prior combo config mismatch: '+k)
    base=json.loads((Path(audit['reference_run'])/'manifest.json').read_text())
    for name,old in base['data'].items():
        if [(r['hr_sha256'],r['lr_sha256']) for r in old]!=[(r['hr_sha256'],r['lr_sha256']) for r in m['data'][name]]:raise ValueError('Prior combo ordered data mismatch')
    audit['combo_reference_run']=str(run);audit['combo_reference_selected']=selected
    audit['combo_reference_weight_sha256']=sha(run/'epoch00048_u48000.pt')
    write(frozen/'audit.json',audit)

def archive(run,job,status,records):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=records/(status['id']+'-'+stamp)
    try:
        command('backup.py','--run',run,'--logs',job,'--destination',destination,
            '--training-exit',status['training_exit_code'] if status['training_exit_code'] is not None else -1,
            '--comparison-exit',status['comparison_exit_code'] if status['comparison_exit_code'] is not None else -1)
        status.update(backup_exit_code=0,backup=str(destination))
    except Exception:status.update(backup_exit_code=1,backup_error=traceback.format_exc(),success=False)
    status.update(stage='completed' if status['success'] else 'failed',exit_code=0 if status['success'] else 1)
    write(job/'status.json',status)
    if status['backup_exit_code']==0:write(destination/'FINAL_STATUS.json',status)
    print(json.dumps(status,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('preflight','train'),required=True)
    p.add_argument('--id',required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--context-reference',type=Path,required=True);p.add_argument('--records',type=Path,required=True)
    p.add_argument('--combo-reference',type=Path,required=True)
    a=p.parse_args()
    if os.readlink('/proc/self/fd/9')!=LOCK:raise RuntimeError('Run via queue.sh with inherited shared GPU lock')
    import fcntl
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.records=a.records.resolve()
    if not a.records.is_relative_to(Path('/home/featurize/work')):raise ValueError('Persistent records required')
    cfg=json.loads((HERE/'config.json').read_text());arm=cfg['ucan_sweep']['arm']
    if a.id!='FEAT-SG6-UCAN-'+arm.upper()+'-V1-50K-01':raise ValueError('Wrong arm ID')
    job=ROOT/'logs'/HERE.name/a.id;run=ROOT/'runs'/a.id;frozen=job/'frozen'
    a.records.mkdir(parents=True,exist_ok=True)
    if a.phase=='preflight':
        if job.exists() or run.exists():raise FileExistsError('Existing experiment preserved')
        job.mkdir(parents=True)
        status=dict(id=a.id,arm=arm,stage='preflight',success=False,training_exit_code=None,
            comparison_exit_code=None,plotting_exit_code=None,backup_exit_code=None,auto_release='queue-only')
    else:
        status=json.loads((job/'status.json').read_text())
        if status['stage']!='preflight-passed' or run.exists():raise ValueError('Fresh passed preflight required')
    try:
        write(job/'status.json',status)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU busy')
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty source')
        for directory in (ROOT/'runs',ROOT/'datasets/image_cache',a.records):
            directory.mkdir(parents=True,exist_ok=True)
            if shutil.disk_usage(directory).free<12*1024**3:raise OSError('Less than 12GiB free: '+str(directory))
        if a.phase=='preflight':
            command('prepare.py','--reference-backup',a.reference,'--output',frozen)
            context_audit(a.context_reference,frozen)
            combo_audit(a.combo_reference,frozen)
            command('verify.py','--device','cuda','--output',frozen/'server-engineering.json')
            command('integration.py','--output',frozen/'server-integration.json')
            command('resources.py','--device','cuda','--output',frozen/'server-resources.json')
            status['stage']='preflight-passed';write(job/'status.json',status)
            print('PREFLIGHT PASSED',a.id,flush=True);return
        audit=json.loads((frozen/'audit.json').read_text())
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=audit['candidate_source']:raise ValueError('Source changed after preflight')
        if json.loads((frozen/'config.json').read_text())!=audit['candidate_config']:raise ValueError('Frozen config changed')
        status['stage']='training';write(job/'status.json',status)
        try:command('run.py','--config',frozen/'config.json','--id',a.id,'--screen','--screen-updates','50000');status['training_exit_code']=0
        except subprocess.CalledProcessError as ex:status['training_exit_code']=ex.returncode;raise
        status['stage']='comparison';write(job/'status.json',status)
        try:command('compare.py','--audit',frozen/'audit.json','--run',run,'--output',frozen/'comparison.json');status['comparison_exit_code']=0
        except subprocess.CalledProcessError as ex:status['comparison_exit_code']=ex.returncode;raise
        sys.path.insert(0,str(ROOT/'research/moe_fs'))
        from curves import generate
        status['stage']='plotting';write(job/'status.json',status)
        try:generate({arm:run},job/'curves');status['plotting_exit_code']=0
        except Exception:status['plotting_exit_code']=1;raise
        status['success']=True
    except Exception:
        status.update(error=traceback.format_exc(),success=False);print(status['error'],flush=True)
    status['stage']='backup';write(job/'status.json',status)
    archive(run,job,status,a.records)
    raise SystemExit(status['exit_code'])

if __name__=='__main__':main()
