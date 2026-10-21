"""Freeze a candidate against actual successful SG6 archive; never warm-start."""
import argparse
import json
from pathlib import Path, PurePosixPath
import platform
import subprocess
import torch
from run import existing, validate_config

HERE=Path(__file__).resolve().parent
runtime=existing.fs_runtime
BASE='e731483d4ce4e325f3d64dbbc80add342b46f7f1'
WEIGHT_SHA='9c16f95c1c265e628914160f7929ef7b7f0b07b7c6a8af81b6277f6bc252c86b'
COUNTS=dict(train=3450,DIV2K_valid=100,Set5=5,Set14=14,B100=100,Urban100=100,Manga109=109)

def git(*args):
    return subprocess.check_output(['git',*args],cwd=runtime.ROOT,text=True).strip()

def history(folder):
    rows=sorted([json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines() if s.strip()],key=lambda r:r['update'])
    if [r['update'] for r in rows]!=list(range(1000,50001,1000)):
        raise ValueError('Complete evaluation history 1k..50k required')
    return rows,max(rows,key=lambda r:r['metrics']['DIV2K_valid']['psnr'])

def paths_from_manifest(data):
    def specs(rows):
        groups={}
        for row in rows:
            hp,lp=PurePosixPath(row['hr']),PurePosixPath(row['lr'])
            if not lp.stem.startswith(hp.stem): raise ValueError('Invalid LR suffix')
            suffix=lp.stem[len(hp.stem):]
            key=(str(hp.parent),str(lp.parent),suffix)
            groups[key]=dict(hr=key[0],lr=key[1],lr_suffix=suffix)
        return list(groups.values())
    return dict(train=specs(data['train']),validation={k:specs(v) for k,v in data.items() if k!='train'})

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-backup',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('context','original','local'),default='context')
    a=p.parse_args()
    if git('status','--porcelain'): raise ValueError('Commit source before training')
    git('merge-base','--is-ancestor',BASE,'HEAD')
    if any(not v.startswith('research/sgla_context/') for v in git('diff','--name-only',BASE,'HEAD').splitlines()):
        raise ValueError('Protected SG6/shared trainer changed')
    archive=a.reference_backup.resolve()
    if not (archive/'BACKUP_COMPLETE').is_file(): raise ValueError('Missing archive completion marker')
    backup=json.loads((archive/'backup-manifest.json').read_text())
    if backup.get('training_exit_code')!=0: raise ValueError('Reference training did not succeed')
    matches=[]
    for f in (archive/'runs').glob('*/manifest.json'):
        m=json.loads(f.read_text())
        if m.get('config',{}).get('experiment_version')=='LG6-SiLU-GB-SpatialGate6-v1': matches.append((f.parent,m))
    if len(matches)!=1: raise ValueError('Exactly one actual SG6 reference run required')
    folder,manifest=matches[0]
    fixed=json.loads(git('show',BASE+':research/spatial_gate6/configs/LG6-SiLU-SG6.json'))
    for k,v in fixed.items():
        if k!='paths_file' and manifest['config'].get(k)!=v: raise ValueError('Reference protocol mismatch: '+k)
    if manifest['commit']!=BASE or manifest['config'].get('engineering_only'):
        raise ValueError('Incorrect reference source')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.backends.cudnn.allow_tf32=True; torch.backends.cuda.matmul.allow_tf32=False
    env=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),deterministic=True,cudnn_allow_tf32=True,matmul_allow_tf32=False)
    if env!=manifest['environment']: raise ValueError('Reference CUDA environment differs')
    import numpy,cv2,scipy
    if platform.python_version()!='3.11.8' or (numpy.__version__,cv2.__version__,scipy.__version__)!=('1.26.4','4.10.0','1.13.1'):
        raise ValueError('Reference Python/metric/image dependency mismatch')
    paths=paths_from_manifest(manifest['data'])
    for name,spec in dict(train=paths['train'],**paths['validation']).items():
        actual=runtime.trainer.pairs(spec); old=manifest['data'][name]
        if len(actual)!=COUNTS[name] or len(old)!=COUNTS[name]: raise ValueError('Data count mismatch: '+name)
        print('Audit ordered image hashes:',name,len(actual),flush=True)
        for (hp,lp),row in zip(actual,old):
            if runtime.trainer.digest(hp)!=row['hr_sha256'] or runtime.trainer.digest(lp)!=row['lr_sha256']:
                raise ValueError('Data hash mismatch: '+str(hp))
    rows,best=history(folder)
    if best['update']!=48000 or abs(best['metrics']['DIV2K_valid']['psnr']-30.250214584429138)>1e-9:
        raise ValueError('SG6 selected checkpoint/metric mismatch')
    weight=folder/'epoch00048_u48000.pt'
    if runtime.trainer.digest(weight)!=WEIGHT_SHA: raise ValueError('Reference checkpoint hash mismatch')
    config=json.loads((HERE/'configs'/('SG6-'+a.mode+'.json')).read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    pathfile=a.output/'paths.json'; pathfile.write_text(json.dumps(paths,indent=2))
    config['paths_file']=str(pathfile.resolve()); validate_config(config)
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    audit=dict(status='PASS',fixed_base=BASE,candidate_source=git('rev-parse','HEAD'),candidate_config=config,reference_selected=best,reference_final=rows[-1],reference_run=str(folder),reference_weight_sha256=WEIGHT_SHA,environment=env,scope='scratch seed10; no reference weights loaded')
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2))
    print('PASS frozen',a.mode,'against SG6; no training started')

if __name__=='__main__': main()
