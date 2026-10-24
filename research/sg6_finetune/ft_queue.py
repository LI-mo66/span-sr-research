"""Linux locked queue, figures and verified persistent backup. No shutdown."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from common import ROOT, HERE, sha, write


def plots(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for label in ('A-L1','B-MSE'):
        arm=folder/label
        history=arm/'history.json'
        if not history.is_file():continue
        rows=json.loads(history.read_text())
        steps=[json.loads(s) for s in (arm/'loss_steps.jsonl').read_text().splitlines() if s.strip()]
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        # Shared RGB L1 is comparable between objectives; actual objective also saved separately.
        axes[0].plot([r['update'] for r in steps],[r['train_l1'] for r in steps],lw=.6)
        axes[0].set(ylabel='Training RGB L1 (both objectives)')
        r0=json.loads((folder/'R0.json').read_text())['metrics']['DIV2K_valid']
        for ax,metric in zip(axes[1:],('psnr','ssim')):
            for kind in ('raw','ema'):
                selected=[r for r in rows if r['kind']==kind]
                ax.plot([0]+[r['extra_update'] for r in selected],[r0[metric]]+[r['validation'][metric] for r in selected],marker='o',label=kind)
            selected_path=arm/'selected.json'
            if selected_path.is_file():
                average=json.loads(selected_path.read_text())['average']
                ax.scatter([10000],[average['validation'][metric]],marker='*',s=100,label='6K-10K average')
            ax.axhline(r0[metric],ls='--',color='gray',label='R0')
            ax.set(ylabel='Validation '+metric.upper());ax.legend(fontsize=7)
        for ax in axes:ax.set(xlabel='Additional optimizer updates');ax.grid(alpha=.25)
        fig.suptitle('SG6 '+label+' / independent fine-tuning from R0')
        dest=folder/'curves';dest.mkdir(exist_ok=True)
        for ext in ('png','svg'):fig.savefig(dest/(label+'-overview.'+ext),bbox_inches='tight',dpi=160)
        plt.close(fig)


def backup(output, logs, records, group, code):
    destination=records/(group+'-'+str(time.time_ns()))
    destination.mkdir(parents=True)
    files=[]
    for source,label in ((output,'results'),(logs,'logs')):
        if not source.is_dir():continue
        for path in source.rglob('*'):
            if not path.is_file():continue
            target=destination/label/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            digest=sha(path)
            if sha(target)!=digest:raise IOError(f'Backup SHA mismatch: {path}')
            files.append(dict(path=target.relative_to(destination).as_posix(),sha256=digest,bytes=target.stat().st_size))
    write(destination/'backup-manifest.json',dict(files=files,training_exit_code=code))
    (destination/'BACKUP_COMPLETE').write_text('Copied file hashes verified. Check training_exit_code.\n')
    return destination


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--group',required=True)
    p.add_argument('--backup',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True)
    args=p.parse_args()
    import re
    if not re.fullmatch('[A-Za-z0-9_-]+',args.group):raise ValueError('Invalid group')
    args.records.mkdir(parents=True,exist_ok=True);args.job.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (ROOT/'logs/server-gpu.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another GPU queue holds the lock')
        out=ROOT/'runs'/args.group
        code=0;error=None
        try:
            subprocess.run([sys.executable,'-u',str(HERE/'ft_runner.py'),'--backup',str(args.backup),
                '--data-root',str(args.data_root),'--output',str(out)],cwd=ROOT,check=True)
        except Exception as exc:
            code=1;error=repr(exc)
        try:
            plots(out)
            if code==0 and len(list((out/'curves').glob('*-overview.png')))!=2:
                raise ValueError('Missing required overview figures')
        except Exception as exc:
            code=1;error=(error or '')+'; plotting: '+repr(exc)
        status=dict(exit_code=code,error=error,auto_release=False)
        write(args.job/'status.json',status)
        try:
            dest=backup(out,args.job,args.records,args.group,code)
            status.update(backup_exit_code=0,backup=str(dest))
        except Exception as exc:
            status.update(backup_exit_code=1,backup_error=repr(exc));code=74
        write(args.job/'status.json',status)
        write(args.records/(args.group+'-'+args.job.name+'-status.json'),status)
        print(json.dumps(status,indent=2),flush=True)
        print('Instance retained. No shutdown/release command.',flush=True)
        raise SystemExit(code)


if __name__=='__main__':main()
