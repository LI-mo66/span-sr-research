"""Copy completed/failed run into a fresh persistent archive, verify every file."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--logs',type=Path,required=True); p.add_argument('--destination',type=Path,required=True); p.add_argument('--training-exit',type=int,required=True); p.add_argument('--comparison-exit',type=int,required=True)
    a=p.parse_args(); root=Path(__file__).resolve().parents[2]
    a.destination.mkdir(parents=True,exist_ok=False)
    files={}
    for source,relative in [(a.run,'runs/'+a.run.name),(a.logs,'logs')]:
        if not source.exists(): continue
        target=a.destination/relative; shutil.copytree(source,target)
        for path in source.rglob('*'):
            if path.is_file():
                copied=target/path.relative_to(source); value=sha(path)
                if sha(copied)!=value: raise ValueError('Backup hash mismatch: '+str(path))
                files[str(copied.relative_to(a.destination))]=value
    subprocess.run(['git','archive','--format=zip','--output='+str((a.destination/'source.zip').resolve()),'HEAD'],cwd=root,check=True)
    files['source.zip']=sha(a.destination/'source.zip')
    metadata=dict(training_exit_code=a.training_exit,comparison_exit_code=a.comparison_exit,created_utc=datetime.now(timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),files=files)
    (a.destination/'backup-manifest.json').write_text(json.dumps(metadata,indent=2))
    (a.destination/'BACKUP_COMPLETE').write_text('Hash-verified copy; inspect exit codes in backup-manifest.json.\n')
    print('Verified persistent archive:',a.destination)

if __name__=='__main__': main()
