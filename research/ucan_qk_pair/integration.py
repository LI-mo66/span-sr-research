"""Real shared entry point, synthetic CUDA images, accumulation/resume audit.

Requires a clean committed checkout. Retains unique tmp/ and runs/ artifacts.
No SG6 weight or real dataset is loaded, and no 50K training is launched.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import cv2
import numpy as np
import torch
from run import HERE, existing
from verify import recursive_equal


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=existing.fs_runtime.ROOT
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('Commit source before the shared-trainer integration check')
    token=uuid.uuid4().hex[:10]
    fixture=root/'tmp'/('ucan-pair-integration-'+token)
    fixture.mkdir(parents=True)
    rng=np.random.default_rng(20261023)
    paths={'validation':{}}
    for name in ('train','DIV2K_valid','Set5','Set14','B100','Urban100','Manga109'):
        hr,lr=fixture/name/'HR',fixture/name/'LR'
        hr.mkdir(parents=True);lr.mkdir(parents=True)
        assert cv2.imwrite(str(hr/'fixture.png'),rng.integers(0,256,(64,80,3),dtype=np.uint8))
        assert cv2.imwrite(str(lr/'fixture.png'),rng.integers(0,256,(16,20,3),dtype=np.uint8))
        pair=dict(hr=str(hr),lr=str(lr))
        if name=='train':
            paths['train']=pair
        else:
            paths['validation'][name]=pair
    paths_file=fixture/'paths.json'
    paths_file.write_text(json.dumps(paths),encoding='utf8')
    cfg=json.loads((HERE/'configs/SG6-UCAN-independent.json').read_text())
    cfg.update(engineering_only=True,paths_file=str(paths_file),effective_batch=2,
               micro_batch=1,hr_patch=32,total_updates=4,record_updates=1,screen_epochs=1)
    config=fixture/'config.json'
    config.write_text(json.dumps(cfg),encoding='utf8')
    env=dict(os.environ,UCAN_PAIR_ENGINEERING='1')
    run_id='ENGINEERING-UCAN-IND-'+token
    command=[sys.executable,'-u',str(HERE/'run.py'),'--config',str(config),'--id',run_id,'--screen']
    def execute(command):
        subprocess.run(command,cwd=root,env=env,check=True)
    execute(command)
    out=root/'runs'/run_id
    with (out/'epochs.jsonl').open('a',encoding='utf8') as f:
        f.write('{"interrupted":')
    with (out/'loss_steps.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(dict(update=999,train_l1=999))+'\n'+ '{"interrupted":')
    execute(command+['--screen-updates','50000','--resume'])
    direct_id=run_id+'-continuous'
    direct_cfg=dict(cfg,record_updates=4)
    direct_file=fixture/'continuous-config.json'
    direct_file.write_text(json.dumps(direct_cfg),encoding='utf8')
    execute([sys.executable,'-u',str(HERE/'run.py'),'--config',str(direct_file),'--id',direct_id,
             '--screen','--screen-updates','50000'])
    resumed=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    direct=torch.load(root/'runs'/direct_id/'latest.pt',map_location='cpu',weights_only=False)
    assert resumed['update']==direct['update']==4
    for key in ('model','optimizer','scheduler','python_rng','torch_rng','cuda_rng'):
        recursive_equal(resumed[key],direct[key])
    assert resumed['numpy_rng'][0]==direct['numpy_rng'][0]
    assert np.array_equal(resumed['numpy_rng'][1],direct['numpy_rng'][1])
    assert resumed['numpy_rng'][2:]==direct['numpy_rng'][2:]
    def rows(path):
        return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
    assert rows(out/'loss_steps.jsonl')==rows(root/'runs'/direct_id/'loss_steps.jsonl')
    assert [v['update'] for v in rows(out/'epochs.jsonl')]==[1,2,3,4]
    assert (out/'best_validation.pt').is_file()
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['extra_deployed_parameters']==37564+18356
    report=dict(status='PASS',scope='four synthetic updates; not real-data performance',
                source_commit=manifest['commit'],run_id=run_id,continuous_id=direct_id,
                exact_model_optimizer_scheduler_rng_resume=True,
                exact_losses_with_different_evaluation_cadence=True,
                partial_log_recovery=True,best_validation_checkpoint=True,
                accumulation=2,micro_batch=1,npy_mmap=True,environment=manifest['environment'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('PASS',run_id)


if __name__=='__main__':
    main()
