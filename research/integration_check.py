"""One-update synthetic fixture checks logging/evaluation/checkpoint/resume wiring."""
import json
import argparse
import subprocess
import sys
import uuid
import cv2
import numpy as np
import torch
from runtime import ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate-cache',action='store_true');args=parser.parse_args()
    fixture=ROOT/'tmp'/('integration-'+uuid.uuid4().hex[:8]); fixture.mkdir(parents=True)
    rng=np.random.default_rng(2026); paths={'validation':{}}
    for name in ('train','DIV2K_valid','Set5','Set14','B100','Urban100','Manga109'):
        hr=fixture/name/'HR'; lr=fixture/name/'LR'; hr.mkdir(parents=True); lr.mkdir(parents=True)
        cv2.imwrite(str(hr/'a.png'),rng.integers(0,256,(64,80,3),dtype=np.uint8)); cv2.imwrite(str(lr/'a.png'),rng.integers(0,256,(16,20,3),dtype=np.uint8))
        spec=dict(hr=str(hr),lr=str(lr))
        if name=='train': paths['train']=spec
        else: paths['validation'][name]=spec
    paths_file=fixture/'paths.json'; paths_file.write_text(json.dumps(paths))
    cfg=json.loads((ROOT/'research/configs/baseline_x4.json').read_text()); cfg.update(paths_file=str(paths_file),effective_batch=1,micro_batch=1,hr_patch=32,total_updates=2,record_updates=1,screen_epochs=1)
    if args.candidate_cache:cfg.update(variants=[dict(name='fusion',position='fusion')],image_backend='npy_mmap')
    config=fixture/'config.json'; config.write_text(json.dumps(cfg))
    run_id='ENGINEERING-ONLY-'+uuid.uuid4().hex[:10]
    command=[sys.executable,str(ROOT/'research/train.py'),'--config',str(config),'--id',run_id,'--screen']
    subprocess.run(command,cwd=ROOT,check=True)
    # Simulate an interrupted/stale metric append beyond the committed checkpoint.
    with open(ROOT/'runs'/run_id/'epochs.jsonl','a') as f:f.write(json.dumps(dict(update=999,epoch=999))+'\n'+'{"interrupted":')
    subprocess.run(command+['--resume','--screen-updates','5000'],cwd=ROOT,check=True)
    subprocess.run(command[:-1]+['--resume'],cwd=ROOT,check=True)
    out=ROOT/'runs'/run_id
    assert len((out/'epochs.jsonl').read_text().splitlines())==2
    assert (out/'latest.pt').exists() and (out/'best_validation.pt').exists()
    assert (out/'epoch00001_u1.pt').exists() and (out/'epoch00002_u2.pt').exists()
    assert list(out.glob('epochs_before_resume_*.jsonl'))
    continuous=run_id+'-continuous'
    subprocess.run([sys.executable,str(ROOT/'research/train.py'),'--config',str(config),'--id',continuous,'--screen','--screen-updates','5000'],cwd=ROOT,check=True)
    resumed=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    direct=torch.load(ROOT/'runs'/continuous/'latest.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(value,direct['model'][key]) for key,value in resumed['model'].items())
    print(f'PASS resumed screening-cap extension equals continuous training exactly: {run_id}; no real accuracy evidence')
if __name__=='__main__': main()
