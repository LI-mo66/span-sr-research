"""Candidate engineering checks on the B0 environment; no dataset training/release."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from run import atd_runtime as runtime
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],cwd=runtime.ROOT,text=True).strip():
        raise ValueError('Diagnostic source must be committed and clean')
    manifest=json.loads((args.baseline_run/'manifest.json').read_text())
    if manifest['commit']!='d280dc5ebcffc46c50983dfe27e81c6096d90dcf':
        raise ValueError('Unexpected B0 source')
    if subprocess.check_output(['git','diff',manifest['commit'],'--','basicsr','research/moe_fs/trainer.py',
                                'research/runtime.py','research/image_io.py','research/parallel_loader.py'],
                               cwd=runtime.ROOT,text=True).strip():
        raise ValueError('Common baseline/evaluation source changed')
    actual=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
                deterministic=True,cudnn_allow_tf32=True,matmul_allow_tf32=False)
    if actual!=manifest['environment']:
        raise ValueError(f'B0 CUDA environment mismatch: {actual}')
    cfg=dict(manifest['config']);cfg.pop('moe_fs',None)
    cfg.update(atd_attention=True,atd_settings=dict(position='tail',query_chunk=65536,group_chunk=256),
               experiment_version='ATD-Attention-tail-v1')
    # Only held-out images are needed for whole-image memory checking.
    data={name:rows for name,rows in manifest['data'].items() if name!='train'}
    paths=dict(validation={})
    for name,rows in data.items():
        groups={}
        for row in rows:
            hp,lp=Path(row['hr']),Path(row['lr'])
            if not lp.stem.startswith(hp.stem):raise ValueError('Invalid paired filename')
            suffix=lp.stem[len(hp.stem):]
            groups[(str(hp.parent),str(lp.parent),suffix)]=dict(hr=str(hp.parent),lr=str(lp.parent),lr_suffix=suffix)
        paths['validation'][name]=list(groups.values())
        pairs=runtime.trainer.pairs(paths['validation'][name])
        if len(pairs)!=len(rows):raise ValueError('Evaluation image count differs')
        for (hp,lp),row in zip(pairs,rows):
            if runtime.trainer.digest(hp)!=row['hr_sha256'] or runtime.trainer.digest(lp)!=row['lr_sha256']:
                raise ValueError(f'Image/order hash mismatch: {name}/{hp.name}')
    args.output.mkdir(parents=True,exist_ok=False)
    path_file=args.output/'eval-paths.json';path_file.write_text(json.dumps(paths,indent=2))
    cfg['paths_file']=str(path_file.resolve())
    config=args.output/'config.json';config.write_text(json.dumps(cfg,indent=2))
    here=Path(__file__).resolve().parent
    commands=[['check_reference.py','--device','cuda','--output',str(args.output/'reference-cuda.json')],
              ['verify.py','--device','cuda','--config',str(config),'--output',str(args.output/'engineering-cuda.json')],
              ['server_smoke.py','--config',str(config),'--output',str(args.output/'server-smoke.json')],
              ['resources.py','--config',str(config),'--output',str(args.output/'resources.json')]]
    for script,*options in commands:
        print('START',script,flush=True)
        subprocess.run([sys.executable,'-u',str(here/script),*options],cwd=runtime.ROOT,check=True)
        print('PASS',script,flush=True)
    (args.output/'COMPLETE').write_text('Candidate engineering checks only; no real training or automatic release.\n')
    print('COMPLETE: target-server engineering checks; no PSNR conclusion',flush=True)


if __name__=='__main__':main()
