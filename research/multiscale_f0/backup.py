import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
runid=sys.argv[1];run=ROOT/'runs'/runid;logs=ROOT/'logs/multiscale50k'
history=[json.loads(s) for s in (run/'epochs.jsonl').open() if s.strip()]
assert history[-1]['update']==50000
records=Path('/home/featurize/work/span-x4-research/records');records.mkdir(parents=True,exist_ok=True)
files=[]
for source,rel in ((run,Path('run')),(logs,Path('logs'))):
    for path in source.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():files.append((path,rel/path.relative_to(source)))
assert shutil.disk_usage(records).free>sum(p.stat().st_size for p,_ in files)+1024**3
output=records/(runid+'-'+str(time.time_ns()));output.mkdir()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
entries=[]
for source,relative in files:
    target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
    digest=sha(source);shutil.copy2(source,target)
    assert digest==sha(target)==sha(source)
    entries.append(dict(path=str(relative),sha256=digest))
subprocess.run(['git','archive','--format=tar','--output='+str(output/'source.tar'),'HEAD'],cwd=ROOT,check=True)
entries.append(dict(path='source.tar',sha256=sha(output/'source.tar')))
(output/'backup-manifest.json').write_text(json.dumps(dict(runid=runid,files=entries),indent=2))
(output/'BACKUP_COMPLETE').write_text('Copied and SHA256 verified.\n');os.sync()
(logs/(runid+'-backup-path.txt')).write_text(str(output)+'\n')
print('BACKUP_COMPLETE:',output,flush=True)
