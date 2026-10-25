"""System shutdown only after complete, verified queue. No provider release API."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from backup_utils import digest


def plan():
    executable=shutil.which('shutdown')
    if not executable or not Path(executable).is_absolute() or not os.access(executable,os.X_OK):
        raise RuntimeError('Executable shutdown command unavailable')
    command=[executable,'-h','now']
    if os.geteuid()!=0:
        sudo=shutil.which('sudo')
        if not sudo:raise RuntimeError('Need root or noninteractive sudo shutdown permission')
        check=subprocess.run([sudo,'-n','-l',*command],capture_output=True,text=True)
        if check.returncode:raise RuntimeError('Shutdown permission unavailable: '+check.stderr.strip())
        command=[sudo,'-n',*command]
    return command


def finish(group,status,command,returned_zip):
    if set(status)!= {'add','difference'}:raise RuntimeError('Incomplete queue')
    for row in status.values():
        if row.get('status')!='COMPLETE' or row.get('training_exit_code')!=0:
            raise RuntimeError('Candidate incomplete or failed; retaining server')
        saved=row.get('backup',{})
        if saved.get('status')!='HASH_VERIFIED':raise RuntimeError('Unverified backup')
        dest=Path(saved['path'])
        if not (dest/'BACKUP_COMPLETE').is_file() or not (dest/'backup-manifest.json').is_file():
            raise RuntimeError('Missing backup receipt')
        manifest=json.loads((dest/'backup-manifest.json').read_text())
        if manifest.get('training_exit_code')!=0:raise RuntimeError('Backup records failed training')
        for name,expected in manifest['files'].items():
            path=dest/name
            if not path.is_file() or path.stat().st_size!=expected['bytes'] or digest(path)!=expected['sha256']:
                raise RuntimeError('Backup changed before shutdown: '+str(path))
    if not Path(returned_zip).is_file():raise RuntimeError('Missing verified return ZIP')
    receipt_path=Path(group)/'RETURN_VERIFIED.json'
    if not receipt_path.is_file():raise RuntimeError('Missing return ZIP verification receipt')
    verified=json.loads(receipt_path.read_text())
    if verified['sha256']!=digest(Path(returned_zip)):
        raise RuntimeError('Return ZIP changed before shutdown')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU still occupied; shutdown cancelled')
    receipt=dict(status='REQUESTING_SYSTEM_SHUTDOWN',command=command,return_zip=str(returned_zip),
                 instance_release=False)
    path=Path(group)/'shutdown-status.json'
    path.write_text(json.dumps(receipt,indent=2))
    print('Verified queue complete; requesting system shutdown:',command,flush=True)
    getattr(os,'sync',lambda:None)()
    result=subprocess.run(command,capture_output=True,text=True)
    receipt.update(status='COMMAND_ACCEPTED' if result.returncode==0 else 'FAILED',
                   returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    path.write_text(json.dumps(receipt,indent=2))
    getattr(os,'sync',lambda:None)()
    if result.returncode:raise RuntimeError('Shutdown command failed; check shutdown-status.json')
