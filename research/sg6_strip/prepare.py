"""Read-only SG6 provenance/data audit; freeze two independent scratch configs."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import torch
from bootstrap import ROOT,trainer
from strip_model import BASE
from run import validate_config

HERE=Path(__file__).resolve().parent
COUNTS=dict(train=3450,DIV2K_valid=100,Set5=5,Set14=14,B100=100,Urban100=100,Manga109=109)
REFERENCE_METRICS=dict(DIV2K_valid=30.25021,Set5=31.94137,Set14=28.44490,B100=27.47339,Urban100=25.74270,Manga109=29.97383)

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def history(run):
    rows=[json.loads(s) for s in (run/'epochs.jsonl').read_text().splitlines() if s.strip()]
    if [r['update'] for r in rows]!=list(range(1000,50001,1000)):
        raise ValueError('Complete ordered 1K..50K reference required')
    return rows,max(rows,key=lambda r:r['metrics']['DIV2K_valid']['psnr'])

def relocate(old,data_root):
    p=Path(old)
    if p.is_file():return p
    parts=list(p.parts)
    for token in ('DIV2K','Flickr2K','benchmark'):
        if token in parts:
            tail=Path(*parts[parts.index(token):])
            choices=[data_root/tail]
            if token=='Flickr2K':choices.append(data_root/'downloads'/tail)
            if token=='benchmark' and 'Manga109' in tail.parts:
                choices.append(data_root/str(tail).replace('Manga109','manga109'))
            for q in choices:
                if q.is_file():return q
    raise FileNotFoundError('Original image unavailable; data-root may be wrong: '+old)

def dataset_paths(data,data_root):
    result={}
    for name,records in data.items():
        groups={}
        for row in records:
            hp,lp=relocate(row['hr'],data_root),relocate(row['lr'],data_root)
            if not lp.stem.startswith(hp.stem):raise ValueError('Unknown LR naming')
            key=(str(hp.parent),str(lp.parent),lp.stem[len(hp.stem):])
            groups[key]=dict(hr=key[0],lr=key[1],lr_suffix=key[2])
        result[name]=list(groups.values())
    return dict(train=result.pop('train'),validation=result)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-backup',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kernels',nargs='+',type=int,choices=[15,31],default=[15,31])
    args=p.parse_args()
    if git('status','--porcelain'):raise ValueError('Commit source changes before use')
    git('merge-base','--is-ancestor',BASE,'HEAD')
    if any(not n.startswith('research/sg6_strip/') for n in git('diff','--name-only',BASE,'HEAD').splitlines()):
        raise ValueError('Fixed SG6/common source changed')
    archive=args.reference_backup.resolve()
    if not (archive/'BACKUP_COMPLETE').is_file():raise ValueError('Reference archive incomplete')
    receipt=json.loads((archive/'backup-manifest.json').read_text())
    if receipt.get('training_exit_code')!=0:raise ValueError('Reference training not successful')
    found=[]
    for file in (archive/'runs').glob('*/manifest.json'):
        info=json.loads(file.read_text())
        if info.get('config',{}).get('experiment_version')=='LG6-SiLU-GB-SpatialGate6-v1':found.append((file.parent,info))
    if len(found)!=1:raise ValueError('Expected exactly one SG6 archive run')
    reference,info=found[0]
    if info['commit']!=BASE:raise ValueError('SG6 source commit differs')
    frozen=json.loads(git('show',BASE+':research/spatial_gate6/configs/LG6-SiLU-SG6.json'))
    for key,value in frozen.items():
        if key!='paths_file' and info['config'].get(key)!=value:raise ValueError('Reference config differs: '+key)
    torch.backends.cudnn.allow_tf32=info['environment']['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32=info['environment']['matmul_allow_tf32']
    env=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        deterministic=True,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    if env!=info['environment']:raise ValueError('Reference torch/CUDA/GPU/precision environment differs')
    import numpy,cv2,scipy
    if (platform.python_version(),numpy.__version__,cv2.__version__,scipy.__version__)!=('3.11.8','1.26.4','4.10.0','1.13.1'):
        raise ValueError('Reference Python/image/metric dependencies differ')
    paths=dataset_paths(info['data'],args.data_root.resolve())
    specs=dict(train=paths['train'],**paths['validation'])
    for name,count in COUNTS.items():
        pairs=trainer.pairs(specs[name]);old=info['data'][name]
        if len(pairs)!=len(old) or len(pairs)!=count:raise ValueError('Dataset count differs: '+name)
        for j,((hp,lp),row) in enumerate(zip(pairs,old),1):
            if hp.name!=Path(row['hr']).name or lp.name!=Path(row['lr']).name or trainer.digest(hp)!=row['hr_sha256'] or trainer.digest(lp)!=row['lr_sha256']:
                raise ValueError('Ordered image names/hashes differ: '+str(hp))
            if j%250==0:print('HASH AUDIT',name,j,count,flush=True)
    rows,best=history(reference)
    if best['update']!=48000:raise ValueError('Reference validation selection differs')
    for name,psnr in REFERENCE_METRICS.items():
        if abs(best['metrics'][name]['psnr']-psnr)>5.1e-6:raise ValueError('Reference reported metric differs: '+name)
    weight=reference/f'epoch{best["epoch"]:05d}_u{best["update"]}.pt'
    weight_hash=trainer.digest(weight)
    if weight_hash!='9c16f95c1c265e628914160f7929ef7b7f0b07b7c6a8af81b6277f6bc252c86b':
        raise ValueError('Selected SG6 weight differs from the returned read-only audit')
    args.output.mkdir(parents=True,exist_ok=False)
    pathfile=args.output/'paths.json'
    pathfile.write_text(json.dumps(paths,indent=2))
    for k in args.kernels:
        candidate=dict(info['config'])
        template=json.loads((HERE/f'configs/K{k}.json').read_text())
        candidate.update(experiment_version=template['experiment_version'],partial_strip=template['partial_strip'],paths_file=str(pathfile.resolve()))
        validate_config(candidate)
        (args.output/f'K{k}.json').write_text(json.dumps(candidate,indent=2))
    report=dict(status='PASS',fixed_base=BASE,candidate_commit=git('rev-parse','HEAD'),reference_run=str(reference),
        reference_weight=str(weight),reference_weight_sha256=weight_hash,reference_manifest_sha256=trainer.digest(reference/'manifest.json'),
        reference_history_sha256=trainer.digest(reference/'epochs.jsonl'),reference_selected=best,reference_final=rows[-1],
        ordered_data_hashes_verified=True,environment=env,kernels=args.kernels,
        scratch_training=True,note='Reference weights are read-only; frozen feature probe may load them, formal candidate trainer never does.')
    (args.output/'reuse-audit.json').write_text(json.dumps(report,indent=2))
    print('SG6 REUSE AUDIT PASS',json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
