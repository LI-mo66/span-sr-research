import hashlib,json,shutil
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

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

