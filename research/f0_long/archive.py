"""Stage snapshot with per-file SHA256; never load training weights on GPU."""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUNID='FEAT-SG6-F0-X8-S10-1M-01'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    import torch
    cap=int(sys.argv[1]);run=ROOT/'runs'/RUNID;logs=ROOT/'logs/f0-long'
    state=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
    assert state['update']==cap;del state
    history=[json.loads(s) for s in (run/'epochs.jsonl').read_text().splitlines() if s.strip()]
    assert [r['update'] for r in history]==list(range(1000,cap+1,1000))
    selected={}
    for label in ('DIV2K_valid','DIV2K_valid_x8'):
        best=min(history,key=lambda r:(-r['metrics'][label]['psnr'],r['update']))
        selected[label]=dict(update=best['update'],checkpoint=f'epoch{best["epoch"]:05d}_u{best["update"]}.pt',metrics=best['metrics'])
        assert (run/selected[label]['checkpoint']).is_file()
    (logs/f'selection-{cap}.json').write_text(json.dumps(selected,indent=2))
    files=[]
    for source,relative in ((run,Path('runs')/RUNID),(logs,Path('logs'))):
        for path in source.rglob('*'):
            assert not path.is_symlink()
            if path.is_file():files.append((path,relative/path.relative_to(source)))
    records=Path('/home/featurize/work/span-x4-research/records')
    required=sum(p.stat().st_size for p,_ in files)+1024**3
    assert shutil.disk_usage(records).free>required,'Insufficient space for stage backup; instance retained'
    output=records/(RUNID+f'-U{cap}-'+str(time.time_ns()));output.mkdir()
    entries=[]
    for source,relative in files:
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
        before=sha(source);shutil.copy2(source,target)
        assert before==sha(target)==sha(source),str(source)
        entries.append(dict(path=str(relative),sha256=before))
    (output/'backup-manifest.json').write_text(json.dumps(dict(stage_update=cap,total_budget=1000000,training_exit_code=0,files=entries),indent=2))
    (output/'BACKUP_COMPLETE').write_text('Stage files copied and SHA256 verified.\n')
    os.sync()
    (ROOT/f'logs/f0-long-stage-{cap}/backup-path.txt').write_text(str(output)+'\n')
    print('BACKUP_COMPLETE:',output,flush=True)

if __name__=='__main__':main()
