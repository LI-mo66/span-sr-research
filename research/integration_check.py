"""One-update synthetic fixture checks logging/evaluation/checkpoint/resume wiring."""
import json
import subprocess
import sys
import uuid
import cv2
import numpy as np
from runtime import ROOT

def main():
    fixture=ROOT/'tmp'/('integration-'+uuid.uuid4().hex[:8]); fixture.mkdir(parents=True)
    rng=np.random.default_rng(2026); paths={'validation':{}}
    for name in ('train','DIV2K_valid','Set5','Set14','B100','Urban100','Manga109'):
        hr=fixture/name/'HR'; lr=fixture/name/'LR'; hr.mkdir(parents=True); lr.mkdir(parents=True)
        cv2.imwrite(str(hr/'a.png'),rng.integers(0,256,(64,80,3),dtype=np.uint8)); cv2.imwrite(str(lr/'a.png'),rng.integers(0,256,(16,20,3),dtype=np.uint8))
        spec=dict(hr=str(hr),lr=str(lr))
        if name=='train': paths['train']=spec
        else: paths['validation'][name]=spec
    paths_file=fixture/'paths.json'; paths_file.write_text(json.dumps(paths))
    cfg=json.loads((ROOT/'research/configs/baseline_x4.json').read_text()); cfg.update(paths_file=str(paths_file),effective_batch=1,micro_batch=1,hr_patch=32,total_updates=1)
    config=fixture/'config.json'; config.write_text(json.dumps(cfg))
    run_id='ENGINEERING-ONLY-'+uuid.uuid4().hex[:10]
    command=[sys.executable,str(ROOT/'research/train.py'),'--config',str(config),'--id',run_id]
    subprocess.run(command,cwd=ROOT,check=True)
    subprocess.run(command+['--resume'],cwd=ROOT,check=True)
    out=ROOT/'runs'/run_id
    assert len((out/'epochs.jsonl').read_text().splitlines())==1
    assert (out/'latest.pt').exists() and (out/'best_validation.pt').exists()
    print(f'PASS synthetic training/evaluation/save/resume: {run_id}; no real accuracy evidence')
if __name__=='__main__': main()
