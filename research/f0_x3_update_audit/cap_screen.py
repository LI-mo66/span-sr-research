"""Fixed two-update counterfactuals, not resumed long training or PSNR screening."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import torch
import audit as a
from follow_on import equal, batch_score


def apply_cap(model, before, rho, groups=('conv_1','block_1')):
    """Project the proposed Adam delta of each selected group, keeping Adam moments."""
    result=[]
    with torch.no_grad():
        named={n:p for n,p in model.named_parameters() if p.requires_grad}
        for group in groups:
            keys=[n for n in named if n==group or n.startswith(group+'.')]
            assert keys,group
            old_norm=torch.stack([before[k].double().square().sum() for k in keys]).sum().sqrt()
            delta_norm=torch.stack([(named[k].detach().double()-before[k].double()).square().sum()
                                    for k in keys]).sum().sqrt()
            limit=float(old_norm)*rho
            scale=min(1.,limit/max(float(delta_norm),1e-30))
            if scale<1.:
                for key in keys:
                    named[key].copy_(before[key]+(named[key]-before[key])*scale)
            applied_norm=float(torch.stack([(named[k].detach().double()-before[k].double()).square().sum()
                                            for k in keys]).sum().sqrt())
            assert applied_norm<=limit*(1+1e-3)+1e-6, (group,applied_norm,limit)
            result.append(dict(group=group,rho=rho,parameter_norm=float(old_norm),
                proposed_delta_norm=float(delta_norm),applied_delta_norm=applied_norm,
                scale=scale,cap_active=scale<1.))
    return result


def load_state(model,optimizer,scheduler,state):
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    scheduler.load_state_dict(copy.deepcopy(state['scheduler']))
    a.restore(state['rng'])


def scores(model,first,second,fixed,device):
    return dict(batch295382=batch_score(model,first,device),
                batch295383=batch_score(model,second,device),fixed_probe=a.probe(model,fixed))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    capture,out=args.capture.resolve(),args.output.resolve()
    assert out.is_relative_to(Path('/home/featurize/work').resolve()) and not (out/'protocol.json').exists()
    assert torch.__version__=='2.2.2+cu121' and torch.cuda.is_available()
    assert '4090' in torch.cuda.get_device_name()
    report=json.loads((capture/'report.json').read_text())
    assert report['status']=='QUALITY_WARNING_ON_MATCHED_TRAJECTORY'
    assert report['last_update']==295383 and report['matched_follow_on_updates']==7
    assert report['maximum_loss_difference']==0 and report['input_files_unchanged'] is True
    capture_protocol=json.loads((capture/'protocol.json').read_text())
    assert capture_protocol['source_sha256']==a.SOURCE_SHA
    input_names=('previous-before.pt','previous-after.pt','before.pt','after.pt',
                 'previous-batches.pt','batches.pt','report.json','protocol.json','rollback.json')
    hashes={str(capture/n):a.trainer.digest(capture/n) for n in input_names}
    seed=torch.load(capture/'previous-before.pt',map_location='cpu',weights_only=False)
    middle=torch.load(capture/'before.pt',map_location='cpu',weights_only=False)
    expected=torch.load(capture/'after.pt',map_location='cpu',weights_only=False)
    previous_after=torch.load(capture/'previous-after.pt',map_location='cpu',weights_only=False)
    assert all(s['format']=='X3-native-update-diagnostic-v1' for s in (seed,middle,expected,previous_after))
    equal(middle['model'],previous_after['model'])
    equal(middle['optimizer'],previous_after['optimizer'])
    assert [s['update'] for s in (seed,middle,expected)]==[295381,295382,295383]
    assert [g['lr'] for g in seed['optimizer']['param_groups']]==[1e-4]
    assert seed['scheduler']['last_epoch']==295381 and seed['scheduler']['base_lrs']==[2e-4]
    first=torch.load(capture/'previous-batches.pt',map_location='cpu',weights_only=False)['batches']
    data=torch.load(capture/'batches.pt',map_location='cpu',weights_only=False)
    second,cfg=data['batches'],data['cfg']
    assert len(first)==len(second)==4
    assert cfg['model']['upscale']==3 and cfg['hr_patch']==192 and cfg['micro_batch']==16
    for x,y in first+second:
        assert x.shape==(16,3,64,64) and y.shape==(16,3,192,192)
        assert x.dtype==y.dtype==torch.float32 and torch.isfinite(x).all() and torch.isfinite(y).all()
    code_hashes={}
    for name in ('fusion.py','sg_model.py','bound.py','gb_model.py','lg_model.py',
                 'attention.py','lksa.py','window_adapter.py','ensemble.py'):
        rel='research/multiscale_f0/'+name
        original=subprocess.check_output(['git','show','0748bd60c1eaddda4e0db7f01ba768a0f2a22be3:'+rel],cwd=a.ROOT)
        assert original==(a.ROOT/rel).read_bytes(),rel
        code_hashes[rel]=a.trainer.digest(a.ROOT/rel)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    arch,_=a.fusion.components()
    model=a.fusion.build_model(arch,cfg).cuda().train()
    assert not any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout,
        torch.nn.Dropout2d,torch.nn.Dropout3d)) for m in model.modules())
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=cfg['lr'],
        betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'])
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg['milestones'],gamma=cfg['gamma'])
    fixed=tuple(t.cuda() for t in data['fixed_probe'])
    plans={'original':None,'front_rho_0p002':.002,'front_rho_0p001':.001,'front_rho_0p0005':.0005}
    protocol=dict(input_hashes=hashes,model_code_sha256=code_hashes,start=295381,stop=295383,
        fixed_batches=True,lr=1e-4,gradient_clip=.1,plans=plans,groups=['conv_1','block_1'],
        method='Per-group ||applied Adam delta|| <= rho*||pre-step parameters||; moments unchanged',
        scope='Two saved training updates only. New optimization rule, unchanged architecture and nominal LR. No PSNR or long stability claim.')
    a.write_json(out/'protocol.json',protocol)
    results={}
    for label,rho in plans.items():
        print('CASE',label,flush=True)
        load_state(model,optimizer,scheduler,seed)
        rows=[]
        for index,batches in enumerate((first,second)):
            before=copy.deepcopy(model.state_dict())
            measured_before=scores(model,first,second,fixed,'cuda')
            value=a.step(model,optimizer,scheduler,[tuple(t.cuda() for t in b) for b in batches])
            if value.get('original_loss_guard'):
                rows.append(dict(update=295382+index,status='LOSS_GUARD',values=value));break
            caps=apply_cap(model,before,rho) if rho is not None else []
            measured_after=scores(model,first,second,fixed,'cuda')
            if rho is None:
                target=middle if index==0 else expected
                equal(model.state_dict(),target['model'])
                equal(optimizer.state_dict(),target['optimizer'])
                equal(scheduler.state_dict(),target['scheduler'])
                print('PASS exact historical model/Adam/scheduler',target['update'],flush=True)
            row=dict(update=295382+index,train_l1=value['mean_l1'],
                gradient_norm_before_clip=value['gradient_norm_before_clip'],
                gradient_norm_after_clip=value['gradient_norm_after_clip'],
                before=measured_before,after=measured_after,caps=caps,
                lr=[g['lr'] for g in optimizer.param_groups],scheduler_last_epoch=scheduler.last_epoch)
            rows.append(row);print(json.dumps(row),flush=True)
        results[label]=dict(rows=rows,completed_two_updates=len(rows)==2 and all('status' not in r for r in rows))
        if results[label]['completed_two_updates']:
            results[label]['final']=rows[-1]['after']
            results[label]['relative_to_start']={k:rows[-1]['after'][k]/rows[0]['before'][k]
                                                for k in rows[-1]['after']}
        a.write_json(out/'comparison.json',results)
    unchanged=all(a.trainer.digest(Path(p))==digest for p,digest in hashes.items())
    assert unchanged
    summary=dict(status='FIXED_TWO_UPDATE_SCREEN_COMPLETE',source_unchanged=unchanged,
        original_transition_exact=True,results=results,scope=protocol['scope'])
    a.write_json(out/'report.json',summary)
    print('SCREEN_COMPLETE',json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
