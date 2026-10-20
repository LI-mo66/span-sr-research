"""Audit the fixed SG6 archive and freeze exactly one scratch ATD candidate."""
import argparse
import json
from pathlib import Path, PurePosixPath
import platform
import subprocess
from run import atd_runtime as runtime, validate_config
import torch

BASE='e731483d4ce4e325f3d64dbbc80add342b46f7f1'
VERSION='LG6-SiLU-GB-SpatialGate6-v1'
EXPECTED_COUNTS=dict(train=3450,DIV2K_valid=100,Set5=5,Set14=14,B100=100,Urban100=100,Manga109=109)
ROUNDED=dict(DIV2K_valid=30.25021,Set5=31.94137,Set14=28.44490,B100=27.47339,Urban100=25.74270,Manga109=29.97383)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=runtime.ROOT,text=True).strip()


def paths_from_manifest(data):
    def specs(rows):
        groups={}
        for row in rows:
            hp,lp=PurePosixPath(row['hr']),PurePosixPath(row['lr'])
            if not lp.stem.startswith(hp.stem):raise ValueError('Invalid LR suffix')
            suffix=lp.stem[len(hp.stem):]
            groups[(str(hp.parent),str(lp.parent),suffix)]=dict(hr=str(hp.parent),lr=str(lp.parent),lr_suffix=suffix)
        return list(groups.values())
    return dict(train=specs(data['train']),validation={k:specs(v) for k,v in data.items() if k!='train'})


def history(folder):
    rows=sorted([json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines() if s.strip()],key=lambda r:r['update'])
    if [r['update'] for r in rows]!=list(range(1000,50001,1000)):
        raise ValueError('Complete 1k..50k history required')
    best=max(rows,key=lambda r:r['metrics']['DIV2K_valid']['psnr'])
    return rows,best


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-backup',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if git('status','--porcelain'):raise ValueError('Source must be committed and clean')
    git('merge-base','--is-ancestor',BASE,'HEAD')
    changed=git('diff','--name-only',BASE,'HEAD').splitlines()
    if any(not p.startswith('research/sg6_atd/') for p in changed):
        raise ValueError('Only the new sg6_atd directory may differ from fixed SG6')
    archive=args.reference_backup.resolve()
    if not (archive/'BACKUP_COMPLETE').is_file():raise ValueError('SG6 archive marker missing')
    backup_manifest=json.loads((archive/'backup-manifest.json').read_text())
    if backup_manifest.get('training_exit_code')!=0:raise ValueError('SG6 archive did not record successful training')
    matches=[]
    for file in (archive/'runs').glob('*/manifest.json'):
        value=json.loads(file.read_text())
        if value.get('config',{}).get('experiment_version')==VERSION:matches.append((file.parent,value))
    if len(matches)!=1:raise ValueError('Exactly one SG6 reference run required')
    reference,manifest=matches[0]
    cfg=manifest['config']
    template=json.loads((Path(__file__).parent/'configs/A1-sg6-atd-lr2e4.json').read_text())
    fixed=json.loads(git('show',BASE+':research/spatial_gate6/configs/LG6-SiLU-SG6.json'))
    for key,value in fixed.items():
        if key!='paths_file' and cfg.get(key)!=value:raise ValueError('SG6 config differs: '+key)
    if cfg.get('engineering_only') or cfg.get('atd_attention') or cfg.get('variants'):
        raise ValueError('Reference must be unmodified real SG6')
    protected=['basicsr','research/moe_fs/trainer.py','research/moe_fs/run.py','research/moe_fs/runtime.py',
               'research/runtime.py','research/image_io.py','research/parallel_loader.py','research/deploy.py',
               'research/spatial_gate6/sg_model.py','research/spatial_gate6/entry.py',
               'research/gated_bypass/gb_model.py','research/linear_gate/lg_model.py']
    if git('diff','--name-only',manifest['commit'],BASE,'--',*protected):
        raise ValueError('SG6 training source differs from locked implementation')
    if manifest.get('extra_deployed_parameters')!=37564:
        raise ValueError('SG6 archive parameter metadata differs')
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    env=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
             deterministic=True,cudnn_allow_tf32=True,matmul_allow_tf32=False)
    if env!=manifest['environment']:raise ValueError('SG6 CUDA environment differs')
    import numpy,cv2,scipy
    if platform.python_version()!='3.11.8' or (numpy.__version__,cv2.__version__,scipy.__version__)!=('1.26.4','4.10.0','1.13.1'):
        raise ValueError('SG6 Python/image/metric dependency versions differ')
    paths=paths_from_manifest(manifest['data'])
    specs=dict(train=paths['train'],**paths['validation'])
    if set(specs)!=set(EXPECTED_COUNTS):raise ValueError('Dataset set differs')
    for name,spec in specs.items():
        actual=runtime.trainer.pairs(spec); old=manifest['data'][name]
        if len(actual)!=len(old) or len(old)!=EXPECTED_COUNTS[name]:raise ValueError('Dataset count differs: '+name)
        print('Auditing SG6 ordered data hashes:',name,len(actual),'pairs',flush=True)
        for (hp,lp),row in zip(actual,old):
            if runtime.trainer.digest(hp)!=row['hr_sha256'] or runtime.trainer.digest(lp)!=row['lr_sha256']:
                raise ValueError('SG6 ordered data mismatch: '+str(hp))
    rows,best=history(reference)
    if best['update']!=48000:raise ValueError('SG6 selected reference is not 48k')
    for name,rounded in ROUNDED.items():
        if abs(best['metrics'][name]['psnr']-rounded)>5.1e-6:
            raise ValueError('SG6 reported reference differs: '+name)
    weight=reference/'epoch00048_u48000.pt'
    weight_hash=runtime.trainer.digest(weight)
    args.output.mkdir(parents=True,exist_ok=False)
    path_file=args.output/'paths.json';path_file.write_text(json.dumps(paths,indent=2))
    candidate=dict(cfg)
    for key in ('atd_attention','atd_settings','experiment_version','deployed_reference_parameters',
                'max_extra_deployed_parameters','max_new_deployed_parameters'):
        candidate[key]=template[key]
    candidate['paths_file']=str(path_file.resolve())
    validate_config(candidate)
    (args.output/'A1-config.json').write_text(json.dumps(candidate,indent=2))
    audit=dict(status='PASS',direct_reference='SG6',fixed_base=BASE,
               reference_run=str(reference),reference_training_source=manifest['commit'],
               candidate_source=git('rev-parse','HEAD'),reference_selected=best,reference_final=rows[-1],
               reference_weight=weight.name,reference_weight_sha256=weight_hash,environment=env,
               reference_parameters=535772,new_parameters=21813,deployed_parameters=557585,
               initialization='From scratch; SG6 weights only hashed, never loaded',
               candidate_config=candidate)
    (args.output/'reuse-audit.json').write_text(json.dumps(audit,indent=2))
    print('PASS: SG6 direct reference; SG6+ATD only, scratch seed10, cap50k',flush=True)


if __name__=='__main__':main()
