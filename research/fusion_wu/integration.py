"""Exercise actual shared training entry and exact checkpoint resume on synthetic data."""
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
import cv2
import numpy as np
import torch
from fusion import ROOT,HERE

def exact(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b

def main():
    token=uuid.uuid4().hex[:10];fixture=ROOT/'tmp'/('fusion-'+token);fixture.mkdir(parents=True)
    rng=np.random.default_rng(10);paths={'validation':{}}
    for name in ('train','DIV2K_valid','Set5','Set14','B100','Urban100','Manga109'):
        hr=fixture/name/'HR';lr=fixture/name/'LR';hr.mkdir(parents=True);lr.mkdir()
        assert cv2.imwrite(str(hr/'a.png'),rng.integers(0,256,(64,80,3),dtype=np.uint8))
        assert cv2.imwrite(str(lr/'a.png'),rng.integers(0,256,(16,20,3),dtype=np.uint8))
        spec=dict(hr=str(hr),lr=str(lr))
        if name=='train':paths['train']=spec
        else:paths['validation'][name]=spec
    (fixture/'paths.json').write_text(json.dumps(paths))
    cfg=json.loads((ROOT/'logs/fusion-WU/frozen/config.json').read_text())
    cfg.update(engineering_only=True,paths_file=str(fixture/'paths.json'),hr_patch=32,
        effective_batch=2,micro_batch=1,total_updates=2,record_updates=1,screen_epochs=1)
    (fixture/'config.json').write_text(json.dumps(cfg))
    identity='ENGINEERING-FUSION-'+token
    command=[sys.executable,'-u',str(HERE/'run.py'),'--config',str(fixture/'config.json'),'--id',identity,'--screen']
    env=dict(os.environ,FUSION_ENGINEERING='1')
    subprocess.run(command,cwd=ROOT,env=env,check=True)
    subprocess.run(command+['--screen-updates','50000','--resume'],cwd=ROOT,env=env,check=True)
    direct=dict(cfg,record_updates=2);(fixture/'direct.json').write_text(json.dumps(direct))
    subprocess.run([sys.executable,'-u',str(HERE/'run.py'),'--config',str(fixture/'direct.json'),
        '--id',identity+'-direct','--screen','--screen-updates','50000'],cwd=ROOT,env=env,check=True)
    a=torch.load(ROOT/'runs'/identity/'latest.pt',map_location='cpu',weights_only=False)
    b=torch.load(ROOT/'runs'/(identity+'-direct')/'latest.pt',map_location='cpu',weights_only=False)
    assert a['update']==b['update']==2
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    for key in ('optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng'):
        exact(a[key],b[key])
    result=dict(status='PASS',synthetic_updates=2,exact_resumed_weights_optimizer_scheduler_rng=True,
        training_entry=str(HERE/'run.py'),scope='Synthetic integration only; no real-data PSNR claim')
    (ROOT/'logs/fusion-WU/integration.json').write_text(json.dumps(result,indent=2))
    print('PASS: shared trainer checkpoint resume',flush=True)

if __name__=='__main__':main()
