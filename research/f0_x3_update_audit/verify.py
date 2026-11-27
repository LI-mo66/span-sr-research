"""Synthetic equivalence: instrumentation may stop replay, never alter updates."""
import argparse
import copy
import json
from pathlib import Path
import random
import tempfile

import cv2
import numpy as np
import torch
import audit
import follow_on


def equal(a,b):
    if torch.is_tensor(a): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    else: assert a==b


def main(device):
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.manual_seed(10)
    with tempfile.TemporaryDirectory() as tmp:
        low=np.random.default_rng(10).integers(0,256,(70,75,3),dtype=np.uint8)
        high=np.repeat(np.repeat(low,3,0),3,1)
        lp,hp=Path(tmp)/'lr.png',Path(tmp)/'hr.png'
        cv2.imwrite(str(lp),low);cv2.imwrite(str(hp),high)
        audit.trainer.SCALE=3
        a,b=random.Random(10),random.Random(10)
        left=audit.trainer.sample((hp,lp),192,a)
        trace=audit.RandomTrace(b);right=audit.trainer.sample((hp,lp),192,trace)
        equal(left,right);assert a.getstate()==b.getstate() and len(trace.draws)==5
    cfg=json.loads((audit.ROOT/'research/multiscale_f0/template.json').read_text())
    cfg['arm']='F0';cfg['model']['upscale']=3
    arch,_=audit.fusion.components()
    reference=audit.fusion.build_model(arch,cfg).to(device).train()
    model=copy.deepcopy(reference)
    opt1=torch.optim.Adam((p for p in reference.parameters() if p.requires_grad),lr=1e-4)
    opt2=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    sc1=torch.optim.lr_scheduler.MultiStepLR(opt1,[200000,400000],gamma=.5)
    sc2=torch.optim.lr_scheduler.MultiStepLR(opt2,[200000,400000],gamma=.5)
    hooks=audit.FeatureTrace(model);hooks.enabled=True
    batches=[]
    for _ in range(4):
        x=torch.rand(2,3,8,8,device=device)
        with torch.no_grad():
            y=reference(x).detach()+torch.rand(2,3,24,24,device=device)*.02
        batches.append((x,y))
    for _ in range(2):
        expected_rng=audit.rng()
        opt1.zero_grad(set_to_none=True)
        for x,y in batches:
            (torch.nn.functional.l1_loss(reference(x),y)/4).backward()
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad],.1,error_if_nonfinite=True))
        opt1.step();sc1.step()
        actual=audit.step(model,opt2,sc2,batches)
        assert actual['gradient_norm_before_clip']==norm
        assert actual['gradient_norm_after_clip']<=.100005
        audit.probe(model,batches[0])
        follow_on.batch_score(model,batches,device)
        equal(reference.state_dict(),model.state_dict())
        equal(opt1.state_dict(),opt2.state_dict());equal(sc1.state_dict(),sc2.state_dict())
        equal(expected_rng,audit.rng())
    assert hooks.rows and any('raw_product' in r for r in hooks.rows)
    frozen=audit.snapshot(model,opt2,sc2,2,audit.rng())
    audit.step(model,opt2,sc2,batches)
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen['model'].items())
    expected=copy.deepcopy((model.state_dict(),opt2.state_dict(),sc2.state_dict()))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'before.pt';torch.save(frozen,path)
        saved=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True);opt2.load_state_dict(saved['optimizer'])
    sc2.load_state_dict(saved['scheduler']);audit.restore(saved['rng'])
    audit.step(model,opt2,sc2,batches)
    equal(expected,(model.state_dict(),opt2.state_dict(),sc2.state_dict()))
    toy=torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.SiLU(),torch.nn.Conv2d(4,3,1)).to(device).train()
    x=torch.rand(2,3,5,7,device=device)
    with torch.no_grad(): y=toy(x).clone()
    left=copy.deepcopy(toy.state_dict())
    with torch.no_grad(): toy[2].bias.add_(.2)
    right=copy.deepcopy(toy.state_dict())
    output=follow_on.rollback_sweep(toy,left,right,[(x,y)],['0','2'],device)
    assert output['before_l1']==0 and output['after_l1']>.19
    assert output['rows'][1]['l1']==0
    equal(toy.state_dict(),right)
    assert not torch.equal(left['2.bias'],right['2.bias'])
    print('PASS traced sampler, exact native F0 two-step model/Adam/scheduler/RNG equivalence, real postclip norm, immutable snapshot',device,flush=True)
    print('PASS same-batch probes preserve updates; frozen rollback recovers toy effect and leaves source states unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    main(parser.parse_args().device)
