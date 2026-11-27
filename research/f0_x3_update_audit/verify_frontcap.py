"""Active-cap actual F0 x3 save/restore exactness. Not an accuracy test."""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import audit as a
from cap_screen import apply_cap
from follow_on import equal
from frontcap_train import checkpoint,committed_rows,expected_lr


def main(device):
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.manual_seed(10)
    assert expected_lr(295000)==expected_lr(399999)==1e-4
    assert expected_lr(400000)==expected_lr(599999)==5e-5
    assert expected_lr(600000)==2.5e-5 and expected_lr(800000)==expected_lr(1000000)==1.25e-5
    cfg=json.loads((a.ROOT/'research/multiscale_f0/template.json').read_text())
    cfg['arm']='F0';cfg['model']['upscale']=3
    arch,_=a.fusion.components()
    model=a.fusion.build_model(arch,cfg).to(device).train()
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[200000,400000],gamma=.5)
    batches=[]
    for _ in range(4):
        x=torch.rand(2,3,8,8,device=device)
        with torch.no_grad():y=model(x).detach()+torch.rand(2,3,24,24,device=device)*.02
        batches.append((x,y))
    active=0
    def advance():
        nonlocal active
        old=copy.deepcopy(model.state_dict())
        result=a.step(model,opt,scheduler,batches)
        assert not result.get('original_loss_guard')
        caps=apply_cap(model,old,.0005)
        active+=sum(r['cap_active'] for r in caps)
    advance();advance()
    saved=checkpoint(model,opt,scheduler,2,'fixture',dict(update=2))
    advance();advance()
    expected=copy.deepcopy((model.state_dict(),opt.state_dict(),scheduler.state_dict(),a.rng()))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'latest.pt';torch.save(saved,path)
        loaded=torch.load(path,map_location='cpu',weights_only=False)
        log=Path(tmp)/'steps.jsonl'
        log.write_text('{"update":295001,"mean_l1":0.02}\n{"update":295002,"mean_l1":0.03}\n{"update":295003')
        rows=committed_rows(log,295001,1)
        assert [r['update'] for r in rows]==[295001]
        assert list(Path(tmp).glob('steps.jsonl.before-resume-*.txt'))
        log.write_text('{"update":295002,"mean_l1":0.03}\n')
        try:committed_rows(log,295002,1)
        except AssertionError:pass
        else:raise AssertionError('Committed log gap accepted')
    assert loaded['format']=='F0-X3-frontcap-candidate-v1' and loaded['fingerprint']=='fixture'
    model.load_state_dict(loaded['model'],strict=True);opt.load_state_dict(loaded['optimizer'])
    scheduler.load_state_dict(loaded['scheduler']);a.restore(loaded['rng'])
    advance();advance()
    equal(expected[:3],(model.state_dict(),opt.state_dict(),scheduler.state_dict()))
    observed_rng=a.rng()
    assert expected[3]['python']==observed_rng['python']
    assert expected[3]['numpy'][0]==observed_rng['numpy'][0]
    assert np.array_equal(expected[3]['numpy'][1],observed_rng['numpy'][1])
    assert expected[3]['numpy'][2:]==observed_rng['numpy'][2:]
    assert torch.equal(expected[3]['torch'],observed_rng['torch'])
    assert len(expected[3]['cuda'])==len(observed_rng['cuda'])
    assert all(torch.equal(x,y) for x,y in zip(expected[3]['cuda'],observed_rng['cuda']))
    assert active>0 and [g['lr'] for g in opt.param_groups]==[1e-4]
    print('PASS native F0 x3 active front-cap2+2 resume equals direct4: model, Adam, scheduler, RNG',device,flush=True)
    print('PASS resume log keeps committed rows, backs up uncommitted/partial tail, rejects committed gaps',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    main(p.parse_args().device)
