"""Two synthetic updates through the unchanged real trainer, exact resume check."""
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import cv2
import numpy as np
import torch
from bootstrap import ROOT

HERE=Path(__file__).resolve().parent

def main():
    key=uuid.uuid4().hex[:10]
    out=ROOT/'tmp'/('strip-integration-'+key);out.mkdir(parents=True)
    paths={'validation':{}}
    rng=np.random.default_rng(10)
    for name in ('train','DIV2K_valid','Set5','Set14','B100','Urban100','Manga109'):
        folder=out/name;folder.mkdir()
        hr,lr=folder/'hr',folder/'lr';hr.mkdir();lr.mkdir()
        cv2.imwrite(str(hr/'a.png'),rng.integers(0,256,(64,80,3),dtype=np.uint8))
        cv2.imwrite(str(lr/'a.png'),rng.integers(0,256,(16,20,3),dtype=np.uint8))
        spec=dict(hr=str(hr),lr=str(lr))
        if name=='train':paths['train']=spec
        else:paths['validation'][name]=spec
    pathfile=out/'paths.json';pathfile.write_text(json.dumps(paths))
    env=dict(os.environ,SG6_ADD_ENGINEERING='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    results={}
    for k in (31,):
        cfg=json.loads((HERE/f'configs/ADD{k}.json').read_text())
        cfg.update(engineering_only=True,total_updates=4,record_updates=1,screen_epochs=2,effective_batch=2,micro_batch=1,
                   hr_patch=32,image_backend='png',paths_file=str(pathfile))
        config=out/f'ADD{k}.json';config.write_text(json.dumps(cfg))
        rid=f'ENGINEERING-STRIPADD-{key}-K{k}'
        command=[sys.executable,'-B','-u',str(HERE/'run.py'),'--config',str(config),'--id',rid,'--screen']
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        subprocess.run(command+['--screen-updates','50000','--resume'],cwd=ROOT,env=env,check=True)
        direct=rid+'-direct'
        subprocess.run([sys.executable,'-B','-u',str(HERE/'run.py'),'--config',str(config),'--id',direct,'--screen','--screen-updates','50000'],cwd=ROOT,env=env,check=True)
        a=torch.load(ROOT/'runs'/rid/'latest.pt',map_location='cpu',weights_only=False)
        b=torch.load(ROOT/'runs'/direct/'latest.pt',map_location='cpu',weights_only=False)
        assert a['update']==b['update']==4
        assert all(torch.equal(v,b['model'][n]) for n,v in a['model'].items())
        assert a['scheduler']==b['scheduler']
        la=[json.loads(s) for s in (ROOT/'runs'/rid/'loss_steps.jsonl').read_text().splitlines()]
        lb=[json.loads(s) for s in (ROOT/'runs'/direct/'loss_steps.jsonl').read_text().splitlines()]
        assert la==lb
        results[str(k)]=dict(status='PASS',resumed=rid,continuous=direct,updates=4)
    (out/'result.json').write_text(json.dumps(results,indent=2))
    print('SHARED TRAINER INTEGRATION PASS',json.dumps(results))

if __name__=='__main__':main()
