"""Read-only fixed SG6 audit and two frozen scratch candidate configurations."""
import argparse
import json
import math
import platform
import subprocess
from pathlib import Path, PurePosixPath
import torch
from run import mid_runtime as runtime, validate_config

BASE = 'e731483d4ce4e325f3d64dbbc80add342b46f7f1'
VERSION = 'LG6-SiLU-GB-SpatialGate6-v1'
COUNTS = dict(train=3450, DIV2K_valid=100, Set5=5, Set14=14, B100=100, Urban100=100, Manga109=109)
REFERENCE_PSNR = dict(DIV2K_valid=30.250214584429138, Set5=31.94137121525808,
                      Set14=28.444896277229862, B100=27.473392796193416,
                      Urban100=25.7427024714934, Manga109=29.97382994931983)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=runtime.ROOT, text=True).strip()


def history(folder):
    rows = sorted([json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines() if s.strip()], key=lambda r:r['update'])
    if [r['update'] for r in rows] != list(range(1000,50001,1000)):
        raise ValueError('Complete unique 1K..50K history required')
    if any(not math.isfinite(v) for r in rows for scores in r['metrics'].values() for v in scores.values()):
        raise ValueError('Nonfinite metrics')
    return rows, max(rows, key=lambda r:r['metrics']['DIV2K_valid']['psnr'])


def paths_from_manifest(data):
    def specs(rows):
        groups = {}
        for row in rows:
            hr, lr = PurePosixPath(row['hr']), PurePosixPath(row['lr'])
            if not lr.stem.startswith(hr.stem):
                raise ValueError('Invalid LR filename suffix')
            suffix = lr.stem[len(hr.stem):]
            groups[(str(hr.parent),str(lr.parent),suffix)] = dict(hr=str(hr.parent),lr=str(lr.parent),lr_suffix=suffix)
        return list(groups.values())
    return dict(train=specs(data['train']),validation={k:specs(v) for k,v in data.items() if k!='train'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference-backup', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if git('status','--porcelain'):
        raise ValueError('Source must be committed and clean')
    git('merge-base','--is-ancestor',BASE,'HEAD')
    if any(not path.startswith('research/sg6_mid_window/') for path in git('diff','--name-only',BASE,'HEAD').splitlines()):
        raise ValueError('Protected SG6 or shared training code changed')
    archive = args.reference_backup.resolve()
    if not (archive/'BACKUP_COMPLETE').is_file():
        raise ValueError('SG6 persistent backup marker missing')
    if json.loads((archive/'backup-manifest.json').read_text()).get('training_exit_code') != 0:
        raise ValueError('SG6 archive did not record successful training')
    matches = [(file.parent,json.loads(file.read_text())) for file in (archive/'runs').glob('*/manifest.json')]
    matches = [(folder,m) for folder,m in matches if m.get('config',{}).get('experiment_version')==VERSION]
    if len(matches)!=1:
        raise ValueError('Exactly one fixed SG6 run required')
    reference, manifest = matches[0]
    fixed = json.loads(git('show',BASE+':research/spatial_gate6/configs/LG6-SiLU-SG6.json'))
    cfg = manifest['config']
    if any(cfg.get(k)!=v for k,v in fixed.items() if k!='paths_file'):
        raise ValueError('Archived SG6 configuration differs from fixed base')
    if any(cfg.get(k) for k in ('engineering_only','assm','atd_attention','mid_interaction','variants')):
        raise ValueError('Reference is not plain SG6')
    protected = ['basicsr','research/moe_fs/trainer.py','research/moe_fs/run.py','research/moe_fs/runtime.py',
                 'research/runtime.py','research/image_io.py','research/parallel_loader.py','research/deploy.py',
                 'research/spatial_gate6/sg_model.py','research/spatial_gate6/entry.py',
                 'research/gated_bypass/gb_model.py','research/linear_gate/lg_model.py']
    if git('diff','--name-only',manifest['commit'],BASE,'--',*protected):
        raise ValueError('Archived SG6 shared training implementation differs')
    if manifest.get('extra_deployed_parameters') != 37564:
        raise ValueError('SG6 parameter metadata differs')
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    env = dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
               deterministic=True,cudnn_allow_tf32=True,matmul_allow_tf32=False)
    if env!=manifest['environment']:
        raise ValueError('GPU/PyTorch/CUDA environment differs from SG6: '+str(env))
    import numpy, cv2, scipy
    if platform.python_version()!='3.11.8' or (numpy.__version__,cv2.__version__,scipy.__version__)!=('1.26.4','4.10.0','1.13.1'):
        raise ValueError('Python/image/metric dependency versions differ from SG6')
    paths = paths_from_manifest(manifest['data'])
    specs = dict(train=paths['train'], **paths['validation'])
    if set(specs)!=set(COUNTS):
        raise ValueError('Dataset names differ')
    for name,spec in specs.items():
        actual, old = runtime.trainer.pairs(spec), manifest['data'][name]
        if len(actual)!=len(old) or len(old)!=COUNTS[name]:
            raise ValueError('Dataset count differs: '+name)
        print('Auditing SG6 ordered SHA256:',name,len(actual),'pairs',flush=True)
        for (hp,lp),row in zip(actual,old):
            if runtime.trainer.digest(hp)!=row['hr_sha256'] or runtime.trainer.digest(lp)!=row['lr_sha256']:
                raise ValueError('Ordered dataset content differs: '+str(hp))
    rows,best = history(reference)
    if best['update']!=48000 or any(abs(best['metrics'][k]['psnr']-v)>1e-9 for k,v in REFERENCE_PSNR.items()):
        raise ValueError('Fixed SG6 selected metrics differ')
    weight = reference/'epoch00048_u48000.pt'
    weight_hash = runtime.trainer.digest(weight)
    if weight_hash!='9c16f95c1c265e628914160f7929ef7b7f0b07b7c6a8af81b6277f6bc252c86b':
        raise ValueError('Fixed SG6 weight SHA256 mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    path_file = args.output/'paths.json'
    path_file.write_text(json.dumps(paths,indent=2),encoding='utf-8')
    candidates = {}
    for kind in ('A','C'):
        template = json.loads((Path(__file__).parent/'configs'/f'{kind}.json').read_text())
        candidate = dict(cfg)
        for key in ('mid_interaction','mid_settings','experiment_version','deployed_reference_parameters',
                    'max_extra_deployed_parameters','max_new_deployed_parameters'):
            candidate[key]=template[key]
        candidate['paths_file']=str(path_file.resolve())
        validate_config(candidate)
        (args.output/f'{kind}-config.json').write_text(json.dumps(candidate,indent=2),encoding='utf-8')
        candidates[kind]=candidate
    audit = dict(status='PASS',direct_reference='SG6',fixed_base=BASE,
                 reference_run=str(reference),reference_training_source=manifest['commit'],
                 reference_weight=weight.name,reference_weight_sha256=weight_hash,
                 reference_selected=best,reference_final=rows[-1],candidate_source=git('rev-parse','HEAD'),
                 candidate_configs=candidates,environment=env,
                 initialization='Both candidates from scratch; reference weight only hashed, never loaded')
    (args.output/'reuse-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('PASS: SG6 reference locked; A and C scratch seed10; 50K each',flush=True)


if __name__=='__main__':main()
