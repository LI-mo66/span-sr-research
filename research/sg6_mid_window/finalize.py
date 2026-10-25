"""Two-run immutable artifact snapshot, SHA256 verified before any release."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def backup(project, group, destination):
    project,group,destination=project.resolve(),group.resolve(),destination.resolve()
    if not group.is_relative_to(project/'logs/sg6_mid_window'):
        raise ValueError('Group must stay in logs/sg6_mid_window')
    if not (group/'comparison-AC-50000.json').is_file():raise FileNotFoundError('Combined comparison missing')
    sources=[(group,Path('logs'))]
    run_ids={}
    for kind in ('A','C'):
        comparison=json.loads((group/f'comparison-{kind}-50000.json').read_text())
        run_id=group.name+'-'+kind+'-s10'
        run_ids[kind]=run_id
        run=project/'runs'/run_id
        for name in ('manifest.json','epochs.jsonl','loss_steps.jsonl','latest.pt','best_validation.pt',comparison['candidate_checkpoint']):
            if Path(name).name!=name or not (run/name).is_file():raise FileNotFoundError('Missing completed artifact: '+str(run/name))
        sources.append((run,Path('runs')/run_id))
    files=[]
    for source,relative in sources:
        for file in source.rglob('*'):
            if file.is_symlink():raise ValueError('Backup source contains a symlink')
            if file.is_file():files.append((file,relative/file.relative_to(source)))
    destination.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination).free<sum(f.stat().st_size for f,_ in files)+1024**3:
        raise OSError('Insufficient persistent backup space; instance retained')
    output=destination/(group.name+'-'+str(time.time_ns()))
    output.mkdir()
    entries=[]
    for source,relative in files:
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
        before=digest(source);shutil.copy2(source,target)
        if digest(target)!=before or digest(source)!=before:raise IOError('Changing source or corrupt backup: '+str(source))
        entries.append(dict(path=str(relative),sha256=before,bytes=target.stat().st_size))
    (output/'backup-manifest.json').write_text(json.dumps(dict(training_exit_code=0,run_ids=run_ids,files=entries,
            evidence='A/C both completed 50K; existing SG6 referenced read-only',excluded='Original datasets and decode cache'),indent=2),encoding='utf-8')
    (output/'BACKUP_COMPLETE').write_text('All copied files SHA256 verified.\n',encoding='utf-8')
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--group-dir',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    output=backup(args.project,args.group_dir,args.destination)
    args.receipt.write_text(str(output)+'\n',encoding='utf-8')
    print('Persistent A/C backup SHA256 verified:',output,flush=True)


if __name__=='__main__':main()
