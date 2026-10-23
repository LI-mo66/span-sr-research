"""Small return archive: diagnostics and full metric/loss logs, no weights or images."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[2]

def collect(group):
    status=json.loads((group/'queue-status.json').read_text())
    sources=[(p,'group/'+p.relative_to(group).as_posix()) for p in group.rglob('*') if p.is_file()
        and not any('-backup-' in part for part in p.relative_to(group).parts)
        and p.suffix in ('.json','.jsonl','.log','.md')]
    for row in status.values():
        if 'run' not in row:continue
        run=Path(row['run'])
        for file in run.rglob('*'):
            if file.is_file() and file.suffix in ('.json','.jsonl','.log'):
                sources.append((file,'runs/'+run.name+'/'+file.relative_to(run).as_posix()))
    output=group/'SG6-STRIPADD-RETURN.zip'
    # Exclusive creation preserves previous returns when resuming a group.
    if output.exists():
        import time
        output=group/f'SG6-STRIPADD-RETURN-{time.time_ns()}.zip'
    receipt={}
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for file,name in sources:
            data=file.read_bytes();archive.writestr(name,data)
            receipt[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        archive.writestr('return-hashes.json',json.dumps(receipt,indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:raise IOError('Return zip CRC verification failed')
    print('RETURN ZIP',output,flush=True)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',type=Path,required=True);args=p.parse_args();collect(args.group)
