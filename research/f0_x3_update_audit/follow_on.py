"""Resume the exactly matched warning state; capture quality degradation and rollback clues."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics

import torch
import audit as a


def equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left.detach().cpu(), right.detach().cpu())
    elif isinstance(left, dict):
        assert left.keys()==right.keys()
        for k in left: equal(left[k], right[k])
    elif isinstance(left, (list, tuple)):
        assert len(left)==len(right)
        for x,y in zip(left,right): equal(x,y)
    else:
        assert left==right


def batch_score(model, batches, device):
    return sum(a.probe(model, tuple(t.to(device) for t in b)) for b in batches)/len(batches)


def rollback_sweep(model, before, after, batches, groups, device):
    """Frozen-state dependence only: do not interpret as a training ablation."""
    parameter_names={n for n,p in model.named_parameters() if p.requires_grad}
    try:
        model.load_state_dict(before, strict=True)
        left=batch_score(model,batches,device)
        model.load_state_dict(after, strict=True)
        right=batch_score(model,batches,device)
        rows=[]
        for group in groups:
            state=dict(after)
            changed=[k for k in parameter_names if k==group or k.startswith(group+'.')]
            assert changed,group
            for key in changed: state[key]=before[key]
            model.load_state_dict(state,strict=True)
            loss=batch_score(model,batches,device)
            rows.append(dict(group=group,l1=loss,delta_vs_after=loss-right,
                             parameter_tensors=len(changed)))
        return dict(before_l1=left,after_l1=right,rows=rows,
            scope='One update, frozen parameters, fixed batch only; not proof of training benefit or a unique cause.')
    finally:
        model.load_state_dict(after,strict=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-project',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    args=p.parse_args()
    source,previous,out=args.source_project.resolve(),args.previous.resolve(),args.output.resolve()
    assert out.is_relative_to(Path('/home/featurize/work').resolve()) and not (out/'protocol.json').exists()
    assert torch.__version__=='2.2.2+cu121' and torch.cuda.is_available()
    assert '4090' in torch.cuda.get_device_name()
    old_report=json.loads((previous/'report.json').read_text())
    assert old_report['status']=='WARNING_ON_MATCHED_TRAJECTORY'
    assert old_report['matched_updates']==376 and old_report['maximum_loss_difference']==0
    assert old_report['first_capture_update']==295376 and old_report['source_unchanged'] is True
    old_protocol=json.loads((previous/'protocol.json').read_text())
    assert old_protocol['source_sha256']==a.SOURCE_SHA
    for path,digest in old_protocol['model_code_sha256'].items():
        assert a.trainer.digest(a.ROOT/path)==digest,path
    input_paths=[previous/n for n in ('before.pt','after.pt','batches.pt','report.json','protocol.json')]
    hashes={str(f):a.trainer.digest(f) for f in input_paths}
    before_saved=torch.load(previous/'before.pt',map_location='cpu',weights_only=False)
    after_saved=torch.load(previous/'after.pt',map_location='cpu',weights_only=False)
    frozen=torch.load(previous/'batches.pt',map_location='cpu',weights_only=False)
    assert before_saved['update']==295375 and after_saved['update']==295376
    assert after_saved['scheduler']['last_epoch']==295376
    assert [g['lr'] for g in after_saved['optimizer']['param_groups']]==[1e-4]
    cfg=frozen['cfg']
    assert cfg['hr_patch']==192 and cfg['micro_batch']==16 and cfg['effective_batch']==64
    assert cfg['model']['upscale']==3 and cfg['loader_workers']==0
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    arch,_=a.fusion.components()
    model=a.fusion.build_model(arch,cfg).cuda().train()
    assert not any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout,
                                torch.nn.Dropout2d,torch.nn.Dropout3d)) for m in model.modules())
    optimizer=torch.optim.Adam((v for v in model.parameters() if v.requires_grad),lr=cfg['lr'],
        betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'])
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg['milestones'],gamma=cfg['gamma'])
    model.load_state_dict(before_saved['model'],strict=True)
    optimizer.load_state_dict(before_saved['optimizer']);scheduler.load_state_dict(before_saved['scheduler'])
    replay=a.step(model,optimizer,scheduler,[tuple(t.cuda() for t in b) for b in frozen['batches']])
    assert not replay.get('original_loss_guard')
    equal(model.state_dict(),after_saved['model'])
    equal(optimizer.state_dict(),after_saved['optimizer'])
    equal(scheduler.state_dict(),after_saved['scheduler'])
    print('PASS saved295376 update exactly reproduced: model, Adam, scheduler',flush=True)
    fixed=tuple(t.cuda() for t in frozen['fixed_probe'])
    checkpoint=source/'runs/F0-X3-CLIP01-U300K-14/epoch00295_u295000.pt'
    assert a.trainer.digest(checkpoint)==a.SOURCE_SHA
    metadata=json.loads((checkpoint.parent/'manifest.json').read_text())
    assert cfg==metadata['config']
    paths=json.loads(Path(cfg['paths_file']).read_text(encoding='utf-8-sig'))
    a.trainer.SCALE=3
    train=a.trainer.pairs(paths['train'])
    assert len(train)==len(metadata['data']['train'])==3450
    for i,((hr,lr),row) in enumerate(zip(train,metadata['data']['train']),1):
        assert str(hr)==row['hr'] and str(lr)==row['lr']
        assert a.trainer.digest(hr)==row['hr_sha256'] and a.trainer.digest(lr)==row['lr_sha256']
        if i%500==0:print('Auditing train',i,'/3450',flush=True)
    a.enable_cache({'train':metadata['data']['train']},args.cache)
    references={}
    for line in (checkpoint.parent/'loss_steps.jsonl').open():
        if line.strip():
            row=json.loads(line)
            if 294001<=row['update']<=295500:references[row['update']]=row['train_l1']
    assert all(u in references for u in range(294001,295501))
    history=[references[u] for u in range(294377,295377)]
    hooks=a.FeatureTrace(model)
    a.restore(after_saved['rng'])
    warning_before,warning_after=before_saved,after_saved
    previous_before=before_saved
    previous_batches=[tuple(t.cpu() for t in b) for b in frozen['batches']]
    previous_samples=frozen['samples']
    del after_saved,frozen,replay
    protocol=dict(start_update=295376,maximum_stop=295500,lr=1e-4,clip=.1,
        source_sha256=a.SOURCE_SHA,input_hashes=hashes,exact_saved_transition=True,
        stops='Original loss guard; historical mismatch; trainingL1>max(.1,4*median); fixed probe doubles; same training batch after Adam doubles and rises>.02. Gradient size alone does not stop.',
        scope='Continuation diagnostic and frozen rollback only. No stability fix or formal long training.')
    a.write_json(out/'protocol.json',protocol)
    matched=0;maximum_difference=0.;status='MAXIMUM_REPLAY_REACHED_NO_QUALITY_WARNING'
    with (out/'observations.jsonl').open('w',buffering=1) as stream:
        for update in range(295377,295501):
            before=a.snapshot(model,optimizer,scheduler,update-1,a.rng())
            batches=[];samples_log=[]
            def draw():
                for micro in range(4):
                    samples=[]
                    for _ in range(16):
                        pair=random.choice(train);trace=a.RandomTrace(random)
                        samples.append(a.trainer.sample(pair,192,trace))
                        samples_log.append(dict(micro=micro,hr=str(pair[0]),lr=str(pair[1]),draws=trace.draws))
                    cpu=tuple(torch.stack([s[k] for s in samples]) for k in (0,1))
                    batches.append(cpu);yield tuple(t.cuda() for t in cpu)
            probe_before=a.probe(model,fixed)
            hooks.rows=[];hooks.enabled=True
            try:
                value=a.step(model,optimizer,scheduler,draw())
            except Exception as error:
                torch.save(before,out/'before.pt')
                torch.save(dict(batches=batches,samples=samples_log,cfg=cfg,fixed_probe=fixed),out/'batches.pt')
                a.write_json(out/'report.json',dict(status='STEP_FAILED',update=update,error=repr(error)))
                raise
            hooks.enabled=False
            activation_before=list(hooks.rows)
            reasons=[]
            if value.get('original_loss_guard'):
                status='ORIGINAL_LOSS_GUARD_REPRODUCED';reasons=['original_loss_guard']
                after_batch=None;probe_after=None
            else:
                difference=abs(value['mean_l1']-references[update])
                maximum_difference=max(maximum_difference,difference)
                after_batch=batch_score(model,batches,'cuda')
                probe_after=a.probe(model,fixed)
                if difference>1e-7:
                    reasons=['historical_loss_mismatch'];status='HISTORY_DIVERGENCE'
                else:
                    matched+=1
                    if value['mean_l1']>max(.1,4*statistics.median(history)):reasons.append('training_loss_warning')
                    if probe_after>2*probe_before and probe_after-probe_before>.001:reasons.append('fixed_probe_jump')
                    if after_batch>2*value['mean_l1'] and after_batch-value['mean_l1']>.02:reasons.append('same_batch_worse_after_adam')
                    if reasons:status='QUALITY_WARNING_ON_MATCHED_TRAJECTORY'
                history.append(value['mean_l1']);history=history[-1000:]
            gradients=value.pop('gradients_after_clip',None);changes=value.pop('changes',[])
            row=dict(update=update,reference_l1=references[update],fixed_probe_before=probe_before,
                fixed_probe_after=probe_after,same_batch_after=after_batch,reasons=reasons,**value,
                top_updates=sorted(changes,key=lambda r:-r['update_norm'])[:12])
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            print('FOLLOW',update,'L1',value.get('mean_l1'),'same_batch_after',after_batch,
                  'norms',value.get('gradient_norm_before_clip'),value.get('gradient_norm_after_clip'),reasons,flush=True)
            if reasons or update==295500:
                after=a.snapshot(model,optimizer,scheduler,update if probe_after is not None else update-1,a.rng())
                for name,data in (('before.pt',before),('after.pt',after),('previous-before.pt',previous_before),
                    ('previous-after.pt',before),('previous-batches.pt',dict(batches=previous_batches,samples=previous_samples)),
                    ('batches.pt',dict(batches=batches,samples=samples_log,cfg=cfg,fixed_probe=fixed,gradients_after_clip=gradients))):
                    torch.save(data,out/name)
                a.write_json(out/'activation-before.json',activation_before)
                hooks.rows=[];hooks.enabled=True
                batch_score(model,batches,'cuda');hooks.enabled=False
                a.write_json(out/'activation-after.json',hooks.rows)
                a.write_json(out/'transition.json',dict(row=row,all_parameter_updates=changes))
                if status in ('QUALITY_WARNING_ON_MATCHED_TRAJECTORY','ORIGINAL_LOSS_GUARD_REPRODUCED'):
                    groups=['conv_1']+[f'block_{i}' for i in range(1,7)]+['conv_2','conv_cat','upsampler','image_residual','image_gate']
                    sweeps={}
                    # Test both the current and preceding transition on the captured challenging batch.
                    for label,left,right in (('current_update',before['model'],after['model']),
                                            ('previous_update',previous_before['model'],before['model']),
                                            ('initial_gradient_warning',warning_before['model'],warning_after['model'])):
                        sweeps[label]=rollback_sweep(model,left,right,batches,groups,'cuda')
                    model.load_state_dict(after['model'],strict=True)
                    a.write_json(out/'rollback.json',sweeps)
                break
            previous_before=before;previous_batches=batches;previous_samples=samples_log
    unchanged=all(a.trainer.digest(Path(path))==digest for path,digest in hashes.items())
    assert unchanged and a.trainer.digest(checkpoint)==a.SOURCE_SHA
    report=dict(status=status,last_update=update,matched_follow_on_updates=matched,
        maximum_loss_difference=maximum_difference,input_files_unchanged=unchanged,
        scope=protocol['scope'])
    a.write_json(out/'report.json',report)
    print('FOLLOW_ON_FINISHED',json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
