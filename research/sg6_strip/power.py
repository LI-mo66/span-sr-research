"""Explicit system power-off, after verified queue artifacts. Never releases instances."""
import json
import os
from pathlib import Path
import shutil
import subprocess

def plan(executable=None):
    executable=executable or shutil.which('shutdown')
    if not executable or not Path(executable).is_absolute() or not os.access(executable,os.X_OK):
        raise RuntimeError('Executable system shutdown not found; automatic shutdown unavailable')
    # Preserve argv[0]: shutdown may be a systemctl symlink.
    command=[str(executable),'-h','now']
    if os.geteuid()!=0:
        sudo=shutil.which('sudo')
        if not sudo:raise RuntimeError('Automatic shutdown needs root or noninteractive sudo')
        permission=subprocess.run([sudo,'-n','-l',*command],capture_output=True,text=True)
        if permission.returncode:
            raise RuntimeError('Noninteractive shutdown permission check failed: '+permission.stderr.strip())
        command=[sudo,'-n',*command]
    return command

def finish(group,status,command,returned_zip):
    if set(status)!={'K15','K31'}:raise RuntimeError('Incomplete queue: refuse shutdown')
    for label,row in status.items():
        if row['status']=='HOLD_COST':continue
        if row['status'] not in ('COMPLETE','FAILED'):raise RuntimeError('Unsettled candidate: '+label)
        saved=row.get('backup',{})
        if saved.get('status')=='NO_RUN_CREATED' and row['status']=='FAILED':continue
        if saved.get('status')!='HASH_VERIFIED':raise RuntimeError('Unverified candidate backup: '+label)
        dest=Path(saved['path'])
        if not (dest/'BACKUP_COMPLETE').is_file() or not (dest/'backup-manifest.json').is_file():
            raise RuntimeError('Missing backup receipt: '+label)
    if not Path(returned_zip).is_file():raise RuntimeError('Missing verified return archive')
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip()
    if occupied:raise RuntimeError('Other GPU compute detected after queue; refuse shutdown: '+occupied)
    receipt=dict(status='REQUESTING_SYSTEM_SHUTDOWN',command=command,return_zip=str(returned_zip),
        backups_verified=True,instance_release=False)
    path=group/'shutdown-status.json'
    path.write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print('Queue and backups saved. Invoking system shutdown:',command,flush=True)
    getattr(os,'sync',lambda:None)()
    result=subprocess.run(command,capture_output=True,text=True)
    receipt.update(status='COMMAND_ACCEPTED' if result.returncode==0 else 'FAILED',
        returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    path.write_text(json.dumps(receipt,indent=2),encoding='utf8')
    getattr(os,'sync',lambda:None)()
    if result.returncode:raise RuntimeError('Shutdown command failed; machine retained. Check shutdown-status.json')
