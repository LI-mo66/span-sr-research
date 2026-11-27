"""One registered front-cap candidate. Stage cap is not a scheduler horizon."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time

import torch
import audit as a
from cap_screen import apply_cap


def checkpoint(model,optimizer,scheduler,update,fingerprint,record):
    state=a.snapshot(model,optimizer,scheduler,update,a.rng())
    state.update(format='F0-X3-frontcap-candidate-v1',fingerprint=fingerprint,
                 epoch=update//1000,epoch_record=record)
    return state


def committed_rows(path,update,stride):
    raw=path.read_bytes()
    lines=raw.decode('utf-8').splitlines()
    rows=[]
    for index,line in enumerate(lines):
        if not line.strip():continue
        try:row=json.loads(line)
        except json.JSONDecodeError:
            if index==len(lines)-1 and not raw.endswith(b'\n'):break
            raise
        if row['update']<=update:rows.append(row)
    assert [r['update'] for r in rows]==list(range(295000+stride,update+1,stride))
    text=''.join(json.dumps(r)+'\n' for r in rows)
    if text.encode()!=raw:
        path.with_name(path.name+f'.before-resume-{time.time_ns()}.txt').write_bytes(raw)
        tmp=path.with_suffix('.tmp');tmp.write_text(text);tmp.replace(path)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-project',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--until-update',type=int,choices=(296000,300000),required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    out=args.output.resolve();source=args.source_project.resolve()
    assert out.is_relative_to(Path('/home/featurize/work').resolve())
    assert torch.__version__=='2.2.2+cu121' and torch.cuda.is_available()
    assert '4090' in torch.cuda.get_device_name()
    out.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(out.parent).free>2*1024**3
    original=source/'runs/F0-X3-CLIP01-U300K-14/epoch00295_u295000.pt'
    assert a.trainer.digest(original)==a.SOURCE_SHA
    metadata=json.loads((original.parent/'manifest.json').read_text())
    cfg=metadata['config']
    assert cfg['arm']=='F0' and cfg['model']['upscale']==3
    assert cfg['hr_patch']==192 and cfg['effective_batch']==64 and cfg['micro_batch']==16
    assert cfg['loader_workers']==0 and cfg['self_ensemble'] is True and cfg['crop_border']==3
    code_hashes={}
    for name in ('fusion.py','sg_model.py','bound.py','gb_model.py','lg_model.py',
                 'attention.py','lksa.py','window_adapter.py','ensemble.py'):
        rel='research/multiscale_f0/'+name
        raw=subprocess.check_output(['git','show','0748bd60c1eaddda4e0db7f01ba768a0f2a22be3:'+rel],cwd=a.ROOT)
        assert raw==(a.ROOT/rel).read_bytes(),rel
        code_hashes[rel]=a.trainer.digest(a.ROOT/rel)
    for name in ('audit.py','cap_screen.py','frontcap_train.py'):
        code_hashes['research/f0_x3_update_audit/'+name]=a.trainer.digest(Path(__file__).parent/name)
    paths=json.loads(Path(cfg['paths_file']).read_text(encoding='utf-8-sig'))
    a.trainer.SCALE=3
    data={'train':a.trainer.pairs(paths['train'])}
    data.update({name:a.trainer.pairs(spec) for name,spec in paths['validation'].items()})
    counts=dict(train=3450,DIV2K_valid=100,Set5=5,Set14=14,B100=100,Urban100=100,Manga109=109)
    assert data.keys()==counts.keys()==metadata['data'].keys()
    for name,pairs in data.items():
        assert len(pairs)==len(metadata['data'][name])==counts[name]
        for (hr,lr),row in zip(pairs,metadata['data'][name]):
            assert str(hr)==row['hr'] and str(lr)==row['lr']
            assert a.trainer.digest(hr)==row['hr_sha256'] and a.trainer.digest(lr)==row['lr_sha256']
            high,low=a.trainer.aligned(hr,lr)
            if name=='train':assert min(low.shape[:2])>=64
        print('PASS ordered data SHA256 and geometry',name,len(pairs),flush=True)
    train_hashes={r['hr_sha256'] for r in metadata['data']['train']}
    for name in data:
        if name!='train':assert not train_hashes & {r['hr_sha256'] for r in metadata['data'][name]}
    a.enable_cache(metadata['data'],args.cache)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    protocol=dict(source=str(original),source_sha256=a.SOURCE_SHA,original_cfg=cfg,
        optimization='native Adam proposal, then per-group parameter-delta projection',
        groups=['conv_1','block_1'],rho=.0005,gradient_clip=.1,loss='RGB L1',
        effective_batch=64,micro_batch=16,accum=4,hr_patch=192,lr_patch=64,seed=10,
        optimizer_reset=False,source_start=295000,code_hashes=code_hashes,
        data=metadata['data'],environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
                                              device=torch.cuda.get_device_name()),
        diagnostic_guard='Abort only on original raw micro L1>1, whole-step L1>max(.1,4*recentMedian), fixed probe doubles with rise>.001; no rescue',
        scope='New optimization rule. Nominal original MultiStep preserved; no claim of unchanged optimizer trajectory.')
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if args.resume:
        assert out.is_dir()
        assert json.loads((out/'protocol.json').read_text())==protocol
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert state['format']=='F0-X3-frontcap-candidate-v1' and state['fingerprint']==fingerprint
    else:
        assert not out.exists(),'Candidate output already exists'
        out.mkdir(parents=True)
        a.write_json(out/'protocol.json',protocol)
        state=torch.load(original,map_location='cpu',weights_only=False)
        assert state['update']==295000 and state['fingerprint']==metadata['fingerprint']
    start=update=state['update']
    assert 295000<=update<args.until_update<=300000
    assert state['scheduler']['last_epoch']==update and state['scheduler']['base_lrs']==[2e-4]
    assert state['scheduler']['_last_lr']==[1e-4]
    assert dict(state['scheduler']['milestones'])=={200000:1,400000:1,600000:1,800000:1}
    assert [g['lr'] for g in state['optimizer']['param_groups']]==[1e-4]
    arch,metrics=a.fusion.components()
    model=a.fusion.build_model(arch,cfg).cuda().train()
    model.load_state_dict(state['model'],strict=True)
    assert not any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout,
                                torch.nn.Dropout2d,torch.nn.Dropout3d)) for m in model.modules())
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=cfg['lr'],
        betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'])
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg['milestones'],gamma=cfg['gamma'])
    optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    assert len(optimizer.state)==272
    private=random.Random(12345)
    samples=[a.trainer.sample(private.choice(data['train']),192,private) for _ in range(4)]
    fixed=tuple(torch.stack([s[k] for s in samples]).cuda() for k in (0,1))
    source_rng=state['rng'] if args.resume else dict(python=state['python_rng'],numpy=state['numpy_rng'],
        torch=state['torch_rng'],cuda=state['cuda_rng'])
    if not args.resume:
        folder=out/'reference295k';folder.mkdir()
        observed=a.trainer.evaluate(model,{k:v for k,v in data.items() if k!='train'},metrics,cfg,folder,295,295000)
        expected=state['epoch_record']['metrics']
        for name,score in observed.items():
            for key,value in score.items():assert abs(value-expected[name][key])<2e-6,(name,key,value,expected[name][key])
        a.write_json(out/'reference295k.json',observed)
        (out/'steps.jsonl').touch();(out/'epochs.jsonl').touch()
        print('PASS source295K six-set plain/x8 metrics reproduction',flush=True)
    a.restore(source_rng)
    del state
    old=[]
    with (original.parent/'loss_steps.jsonl').open() as stream:
        for line in stream:
            if line.strip():
                r=json.loads(line)
                if 294001<=r['update']<=295000:old.append(r)
    assert [r['update'] for r in old]==list(range(294001,295001))
    history=[r['train_l1'] for r in old]
    if args.resume:
        committed_rows(out/'epochs.jsonl',update,1000)
        rows=committed_rows(out/'steps.jsonl',update,1)
        history=(history+[r['mean_l1'] for r in rows])[-1000:]
    else:
        initial=checkpoint(model,optimizer,scheduler,update,fingerprint,None)
        a.trainer.atomic_save(initial,out/'latest.pt');del initial
    baseline_probe=a.probe(model,fixed)
    with (out/'steps.jsonl').open('a',buffering=1) as log:
        while update<args.until_update:
            before=a.snapshot(model,optimizer,scheduler,update,a.rng())
            probe_before=a.probe(model,fixed)
            batches=[]
            def draw():
                for _ in range(4):
                    samples=[a.trainer.sample(random.choice(data['train']),192) for _ in range(16)]
                    b=tuple(torch.stack([s[k] for s in samples]) for k in (0,1))
                    batches.append(b);yield tuple(t.cuda() for t in b)
            lr_used=[g['lr'] for g in optimizer.param_groups]
            try:
                value=a.step(model,optimizer,scheduler,draw())
                assert not value.get('original_loss_guard'),value
                caps=apply_cap(model,before['model'],.0005)
                probe_after=a.probe(model,fixed)
                loss=value['mean_l1']
                assert loss<=max(.1,4*statistics.median(history)),('whole-step loss warning',loss)
                assert not (probe_after>2*probe_before and probe_after-probe_before>.001),('probe jump',probe_before,probe_after)
                assert [g['lr'] for g in optimizer.param_groups]==[1e-4]
                assert all(torch.isfinite(p).all() for p in model.parameters())
            except Exception as error:
                a.trainer.atomic_save(before,out/'failure-before.pt')
                failed=a.snapshot(model,optimizer,scheduler,scheduler.last_epoch,a.rng())
                failed.update(diagnostic_only=True,attempted_update=update+1)
                a.trainer.atomic_save(failed,out/'failure-after.pt')
                torch.save(dict(batches=batches,fixed_probe=fixed,cfg=cfg),out/'failure-batches.pt')
                a.write_json(out/'ANOMALY.json',dict(attempted_update=update+1,error=repr(error),action='Stopped; no automatic rescue; latest complete checkpoint unchanged'))
                raise
            update+=1;history.append(loss);history=history[-1000:]
            record=dict(update=update,mean_l1=loss,lr_used=lr_used,lr_after=[g['lr'] for g in optimizer.param_groups],
                gradient_norm_before_clip=value['gradient_norm_before_clip'],gradient_norm_after_clip=value['gradient_norm_after_clip'],
                fixed_probe_before=probe_before,fixed_probe_after=probe_after,caps=caps)
            log.write(json.dumps(record,allow_nan=False)+'\n')
            if update==start+1 or update%100==0:print('FRONTCAP',update,'/',args.until_update,json.dumps(record),flush=True)
            del before,value,batches
            if update%1000==0:
                scores=a.trainer.evaluate(model,{k:v for k,v in data.items() if k!='train'},metrics,cfg,out,update//1000,update)
                assert all(math.isfinite(v) for s in scores.values() for v in s.values())
                epoch_record=dict(update=update,metrics=scores)
                saved=checkpoint(model,optimizer,scheduler,update,fingerprint,epoch_record)
                a.trainer.atomic_save(saved,out/f'epoch{update//1000:05d}_u{update}.pt')
                with (out/'epochs.jsonl').open('a') as stream:stream.write(json.dumps(epoch_record)+'\n')
                a.trainer.atomic_save(saved,out/'latest.pt');del saved
                print('EVALUATED',update,json.dumps(scores),flush=True)
    reference=json.loads((out/'reference295k.json').read_text())
    comparison={name:dict(psnr=value['psnr'],ssim=value['ssim'],
        delta_psnr_vs295k=value['psnr']-reference[name]['psnr'],
        delta_ssim_vs295k=value['ssim']-reference[name]['ssim']) for name,value in scores.items()}
    assert a.trainer.digest(original)==a.SOURCE_SHA
    receipt=dict(status='COMPLETED_REVIEW_REQUIRED',update=update,start=start,
        source_unchanged=True,lr=[g['lr'] for g in optimizer.param_groups],
        baseline_probe_l1=baseline_probe,comparison=comparison,
        scope='One candidate, staged continuation, new front-group update cap. Nominal LR schedule preserved; no long-term stability claim.')
    a.write_json(out/f'comparison-{update}.json',receipt)
    hashes={}
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.suffix in ('.pt','.json','.jsonl'):
            for attempt in range(3):
                try:
                    with path.open('rb') as stream:os.fsync(stream.fileno())
                    hashes[str(path.relative_to(out))]=a.trainer.digest(path)
                    break
                except OSError:
                    if attempt==2:raise
                    time.sleep(2)
    a.write_json(out/f'verification-{update}.json',dict(status='PASS',as_of_update=update,
        mutable_on_future_resume=['latest.pt','steps.jsonl','epochs.jsonl'],file_sha256=hashes))
    with (out/f'verification-{update}.json').open('rb') as stream:os.fsync(stream.fileno())
    a.write_json(out/f'COMPLETE-{update}.json',dict(status='PASS_FILES_WRITTEN_NOT_STABILITY_PROOF',update=update))
    with (out/f'COMPLETE-{update}.json').open('rb') as stream:os.fsync(stream.fileno())
    print('STAGE_FINISHED',json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
