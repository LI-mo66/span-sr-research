"""D4 inverse/independent oracle + real-entry stages, LR, RNG and old-training equivalence."""
import copy
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import torch
import numpy as np
import cv2
from ensemble import TRANSFORMS,transform,predict

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
    torch.set_num_threads(2)
    x=torch.arange(45,dtype=torch.float32).reshape(1,3,3,5)/17
    shapes=set()
    for code in TRANSFORMS:
        y=transform(x,code)
        shapes.add((tuple(y.shape),y.numpy().tobytes()))
        assert torch.equal(transform(y,code,inverse=True),x)
    assert len(shapes)==8
    class Dummy(torch.nn.Module):
        def forward(self,x):
            y=x.repeat_interleave(4,-2).repeat_interleave(4,-1)
            return y+torch.linspace(-.7,.8,y.shape[-1])[None,None,None,:]
    model=Dummy();plain,average=predict(model,x)
    assert torch.equal(plain,model(x))
    oracle=[]
    for flip in (False,True):
        for k in range(4):
            z=x.flip(-1) if flip else x
            z=model(torch.rot90(z,k,(-2,-1)))
            z=torch.rot90(z,-k,(-2,-1))
            oracle.append(z.flip(-1) if flip else z)
    torch.testing.assert_close(average,torch.stack(oracle).mean(0),atol=1e-6,rtol=1e-6)
    assert not torch.equal(average,average.clamp(0,1)), 'Test must include out-of-range values'
    print('PASS: all D4 inverses, unique transforms, independent rot90 oracle, unclipped average',flush=True)

    token=uuid.uuid4().hex[:10];fixture=ROOT/'tmp'/('long-'+token);fixture.mkdir(parents=True)
    rng=np.random.default_rng(12);paths={'validation':{}}
    for name in ('train','DIV2K_valid','Set5','Set14','B100','Urban100','Manga109'):
        hr=fixture/name/'HR';lr=fixture/name/'LR';hr.mkdir(parents=True);lr.mkdir()
        assert cv2.imwrite(str(hr/'a.png'),rng.integers(0,256,(64,80,3),dtype=np.uint8))
        assert cv2.imwrite(str(lr/'a.png'),rng.integers(0,256,(16,20,3),dtype=np.uint8))
        spec=dict(hr=str(hr),lr=str(lr))
        if name=='train':paths['train']=spec
        else:paths['validation'][name]=spec
    (fixture/'paths.json').write_text(json.dumps(paths))
    cfg=json.loads((ROOT/'logs/f0-long/frozen/config.json').read_text())
    cfg.update(engineering_only=True,paths_file=str(fixture/'paths.json'),hr_patch=32,
        effective_batch=2,micro_batch=1,total_updates=4,record_updates=2,milestones=[2,3],screen_epochs=1)
    path=fixture/'config.json';path.write_text(json.dumps(cfg))
    identity='ENGINEERING-F0-LONG-'+token
    env=dict(os.environ,F0_LONG_ENGINEERING='1',FUSION_ENGINEERING='1')
    base=[sys.executable,'-u',str(HERE/'run.py'),'--config',str(path)]
    def run(command):subprocess.run(command,cwd=ROOT,env=env,check=True)
    run(base+['--id',identity,'--until-update','2'])
    # Verify recovery does not retain uncommitted future/partial rows.
    folder=ROOT/'runs'/identity
    with (folder/'loss_steps.jsonl').open('a') as f:f.write('{"update":999,"train_l1":999}\n{"partial":')
    with (folder/'epochs.jsonl').open('a') as f:f.write('{"partial":')
    run(base+['--id',identity,'--until-update','4','--resume'])
    run(base+['--id',identity+'-direct','--until-update','4'])
    def checkpoint(suffix=''):
        return torch.load(ROOT/'runs'/(identity+suffix)/'latest.pt',map_location='cpu',weights_only=False)
    staged,direct=checkpoint(),checkpoint('-direct')
    for key in ('model','optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng'):
        exact(staged[key],direct[key])
    for name in ('loss_steps.jsonl','epochs.jsonl'):
        values=[json.loads(s) for s in (folder/name).read_text().splitlines()]
        assert all(v['update']<=4 for v in values)
    assert staged['update']==4 and staged['scheduler']['last_epoch']==4
    assert staged['optimizer']['param_groups'][0]['lr']==5e-5
    old=copy.deepcopy(cfg);old.update(self_ensemble=False,selection_dataset='DIV2K_valid')
    oldpath=fixture/'old.json';oldpath.write_text(json.dumps(old))
    oldfrozen=ROOT/'logs/fusion-WU/frozen/config.json';oldfrozen.parent.mkdir(parents=True,exist_ok=True)
    if not oldfrozen.exists():oldfrozen.write_text(json.dumps(old))
    oldid='ENGINEERING-F0-OLD-'+token
    run([sys.executable,'-u',str(ROOT/'research/fusion_wu/run.py'),'--config',str(oldpath),
         '--id',oldid,'--screen','--screen-updates','50000'])
    reference=torch.load(ROOT/'runs'/oldid/'latest.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng'):
        exact(reference[key],direct[key])
    a=[json.loads(s)['train_l1'] for s in (ROOT/'runs'/oldid/'loss_steps.jsonl').read_text().splitlines()]
    b=[json.loads(s)['train_l1'] for s in (ROOT/'runs'/(identity+'-direct')/'loss_steps.jsonl').read_text().splitlines()]
    assert a==b
    assert 'DIV2K_valid_x8' in direct['epoch_record']['metrics']
    assert (ROOT/'runs'/(identity+'-direct')/'best_plain.pt').is_file()
    report=dict(status='PASS',torch=torch.__version__,device=torch.cuda.get_device_name(),
        transforms_unique_and_inverse_exact=True,independent_x8_oracle=True,
        old_vs_new_training_weights_optimizer_scheduler_rng_losses_exact=True,
        staged_vs_continuous_exact=True,partial_record_recovery=True,lr_milestone_boundary=True,
        both_metric_modes_recorded=True,scope='Four synthetic updates; not real-data convergence or stability')
    out=ROOT/'logs/f0-long/test-protocol.json';out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
