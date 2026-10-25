"""Produce and verify a lightweight return ZIP; full weights remain in backups."""
import hashlib
import json
from pathlib import Path
import zipfile


def collect(group,status):
    group=Path(group)
    files=[]
    for path in group.rglob('*'):
        if path.is_file() and not any(part.endswith('-backup') for part in path.relative_to(group).parts):
            if path.suffix in ('.json','.jsonl','.log','.txt','.md','.png','.svg','.gz'):
                files.append((path,'group/'+path.relative_to(group).as_posix()))
    for row in status.values():
        if not row.get('run'):continue
        run=Path(row['run'])
        for path in run.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.log'):
                files.append((path,'runs/'+run.name+'/'+path.relative_to(run).as_posix()))
    output=group/'SG6-FE-RETURN.zip'
    hashes={}
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path,name in files:
            data=path.read_bytes();archive.writestr(name,data)
            hashes[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        archive.writestr('return-hashes.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:raise IOError('Return ZIP CRC failed')
        for name,row in hashes.items():
            data=archive.read(name)
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                raise IOError('Return ZIP hash mismatch: '+name)
    (group/'RETURN_VERIFIED.json').write_text(json.dumps(dict(path=str(output),
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),files=len(hashes)),indent=2))
    print('RETURN ZIP VERIFIED',output,flush=True)
    return output
