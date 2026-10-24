"""Two independent 10K fine-tuning arms; validation-only weight selection."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from common import *

BASE_COMMIT = 'e731483d4ce4e325f3d64dbbc80add342b46f7f1'
PROTOCOL = dict(version='SG6-FT-L1-MSE-v1', seed=10, updates=10000,
                evaluate_every=1000, lr=1e-5, eta_min=1e-6, ema_decay=.999,
                average_updates=[6000,7000,8000,9000,10000], optimizer_reset=True,
                augmentation_stream='reset_seed10_each_arm', large_patch=False)


def relocated(root, old):
    parts = Path(old).parts
    for anchor in ('DIV2K','Flickr2K','benchmark'):
        if anchor in parts:
            return root.joinpath(*parts[parts.index(anchor):])
    raise ValueError(f'Unknown dataset path: {old}')


def audit(backup, root):
    if not (backup/'BACKUP_COMPLETE').is_file():
        raise ValueError('SG6 backup incomplete')
    runs = list((backup/'runs').glob('*-LG6-SiLU-SG6-s10'))
    if len(runs) != 1:
        raise ValueError('Expected one SG6 run')
    run = runs[0]
    info = json.loads((run/'manifest.json').read_text())
    canonical = json.loads((ROOT/'research/spatial_gate6/configs/LG6-SiLU-SG6.json').read_text())
    for key in ('model','seed','lr','adam_betas','adam_eps','milestones','gamma','hr_patch',
                'effective_batch','micro_batch','crop_border','test_y_channel','self_ensemble',
                'local_gate','image_bypass','spatial_gate','experiment_version','image_backend'):
        if info['config'].get(key) != canonical.get(key):
            raise ValueError(f'Fixed SG6 protocol mismatch: {key}')
    if info['commit'] != BASE_COMMIT:
        raise ValueError('Not the fixed SG6 source commit')
    history = sorted([json.loads(s) for s in (run/'epochs.jsonl').read_text().splitlines() if s.strip()],key=lambda r:r['update'])
    if [r['update'] for r in history] != list(range(1000,50001,1000)):
        raise ValueError('Expected complete 50K reference history')
    selected = max(history,key=lambda r:r['metrics']['DIV2K_valid']['psnr'])
    if selected['update'] != 48000:
        raise ValueError('Fixed reference selected update differs')
    checkpoint = run/f'epoch{selected["epoch"]:05d}_u{selected["update"]}.pt'
    inventory = json.loads((backup/'backup-manifest.json').read_text())
    if inventory['training_exit_code'] != 0:
        raise ValueError('Reference training failed')
    expected = next(r['sha256'] for r in inventory['files'] if r['path']==checkpoint.relative_to(backup).as_posix())
    if sha(checkpoint) != expected:
        raise ValueError('Reference checkpoint SHA mismatch')
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['fingerprint'] != info['fingerprint'] or state['update'] != 48000:
        raise ValueError('Reference checkpoint identity mismatch')
    env = info['environment']
    if not torch.cuda.is_available() or str(torch.__version__) != env['torch'] or torch.version.cuda != env['cuda'] or torch.cuda.get_device_name() != env['device']:
        raise ValueError('Reference CUDA/PyTorch/GPU differs')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=env['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32=env['matmul_allow_tf32']
    torch.use_deterministic_algorithms(True)
    pairs = {}; relocated_manifest = {}
    for name, records in info['data'].items():
        pairs[name] = []; relocated_manifest[name] = []
        for i, rec in enumerate(records,1):
            hp,lp = [relocated(root,rec[side]) for side in ('hr','lr')]
            if sha(hp) != rec['hr_sha256'] or sha(lp) != rec['lr_sha256']:
                raise ValueError(f'Dataset hash differs: {hp}')
            pairs[name].append((hp,lp))
            relocated_manifest[name].append(dict(hr=str(hp),lr=str(lp),hr_sha256=rec['hr_sha256'],lr_sha256=rec['lr_sha256']))
            if i%250==0:print(f'Hash audit {name}: {i}/{len(records)}',flush=True)
    if len(pairs['train']) != 3450 or len(pairs['DIV2K_valid']) != 100:
        raise ValueError('Original DF2K/validation counts differ')
    train_hashes={r['hr_sha256'] for r in info['data']['train']}
    if any(train_hashes & {r['hr_sha256'] for r in records} for name,records in info['data'].items() if name!='train'):
        raise ValueError('Train/evaluation overlap')
    trainer.enable_cache(relocated_manifest,root/'image_cache')
    return info, state, selected, checkpoint, expected, pairs


def branch(label, initial, cfg, data, arch, metrics, root, fingerprint, engineering_settings=None):
    settings=PROTOCOL
    if engineering_settings is not None:
        if os.environ.get('SG6_FINETUNE_ENGINEERING')!='1':
            raise ValueError('Engineering override requires explicit environment flag')
        settings=dict(PROTOCOL,**engineering_settings)
    total=settings['updates'];period=settings['evaluate_every']
    out=root/label;out.mkdir(exist_ok=True)
    seed(10)
    model=build_model(arch,cfg).cuda();model.load_state_dict(initial,strict=True)
    ema_model=build_model(arch,cfg).cuda();ema_model.load_state_dict(initial,strict=True)
    if list(model.named_buffers()):
        raise ValueError('Unexpected buffers: explicit averaging policy required')
    shadow={k:v.detach().clone() for k,v in trainable(model).items()}
    opt=torch.optim.Adam(trainable(model).values(),lr=PROTOCOL['lr'],betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total,eta_min=1e-6)
    latest=out/'latest.pt';update=0;history=[];steps=[];elapsed=0.
    if latest.is_file():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        if saved['fingerprint']!=fingerprint or saved['label']!=label:
            raise ValueError('Fine-tuning resume fingerprint differs')
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        shadow={k:v.cuda() for k,v in saved['ema'].items()}
        install(ema_model,shadow,arch)
        update=saved['update'];history=saved['history'];steps=saved['steps'];elapsed=saved['elapsed_seconds']
        restore_rng(saved['rng'])
        if not 0<=update<=total:raise ValueError('Invalid saved update')
    else:
        # Same draw stream for both objectives, independent of model construction.
        seed(10)
    start=time.perf_counter()
    with (out/'loss_steps.jsonl').open('w',encoding='utf8') as f:
        for r in steps:f.write(json.dumps(r)+'\n')
    log=(out/'loss_steps.jsonl').open('a',encoding='utf8',buffering=1)
    try:
        while update<total:
            row=step(model,opt,scheduler,shadow,data['train'],cfg,'L1' if label=='A-L1' else 'MSE','cuda')
            update+=1;row['update']=update;steps.append(row);log.write(json.dumps(row)+'\n')
            if update==1 or update%100==0:
                print(f'{label} extra_update={update}/{total} objective={row["objective"]:.9f} RGB_L1={row["train_l1"]:.9f} lr_next={row["lr_next"]:.8g}',flush=True)
            if update%period:continue
            install(ema_model,shadow,arch)
            for kind,net in (('raw',model),('ema',ema_model)):
                name=f'{kind}_u{update:05d}.pt'
                score=evaluate(net,data['DIV2K_valid'],metrics,cfg,out/f'{kind}_u{update:05d}_DIV2K_valid.json','cuda')
                history.append(dict(kind=kind,extra_update=update,checkpoint=name,validation=score))
                save(out/name,dict(model=state_copy(net,arch),fingerprint=fingerprint,label=label,kind=kind,extra_update=update,config=cfg))
                print(label,kind,update,json.dumps(score),flush=True)
            elapsed_now=elapsed+time.perf_counter()-start
            save(latest,dict(fingerprint=fingerprint,label=label,update=update,model=state_copy(model,arch),
                optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),ema={k:v.cpu() for k,v in shadow.items()},
                rng=rng_state(),history=history,steps=steps,elapsed_seconds=elapsed_now))
            write(out/'history.json',history)
    finally:
        log.close()
    # Strictly same arm, same structure, ordinary training parameters only.
    states=[torch.load(out/f'raw_u{u:05d}.pt',map_location='cpu',weights_only=False) for u in settings['average_updates']]
    for s in states:
        if s['fingerprint']!=fingerprint or s['label']!=label or s['kind']!='raw':
            raise ValueError('Average input provenance differs')
    install(ema_model,average([s['model'] for s in states],trainable(ema_model)),arch)
    score=evaluate(ema_model,data['DIV2K_valid'],metrics,cfg,out/'average_DIV2K_valid.json','cuda')
    save(out/'average_6k_10k.pt',dict(model=state_copy(ema_model,arch),fingerprint=fingerprint,label=label,kind='checkpoint_average',config=cfg,sources=settings['average_updates']))
    selected={}
    for kind in ('raw','ema'):
        best=max([r for r in history if r['kind']==kind],key=lambda r:r['validation']['psnr'])
        selected[kind]=dict(best)
    selected['average']=dict(checkpoint='average_6k_10k.pt',validation=score,extra_update=total,kind='checkpoint_average')
    for kind,result in selected.items():
        saved=torch.load(out/result['checkpoint'],map_location='cpu',weights_only=False)
        ema_model.load_state_dict(saved['model'],strict=True)
        result['deployment']=deployment_check(ema_model,arch,'cuda')
        result['metrics']={'DIV2K_valid':result['validation']}
        for name,pairs in data.items():
            if name in ('train','DIV2K_valid'):continue
            result['metrics'][name]=evaluate(ema_model,pairs,metrics,cfg,out/f'FINAL_{kind}_{name}.json','cuda')
        result['checkpoint_sha256']=sha(out/result['checkpoint'])
    write(out/'selected.json',selected)
    return dict(selected=selected,elapsed_seconds=elapsed+time.perf_counter()-start,loss=label,updates=total)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backup',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Source must be committed and clean')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    info,state,best,checkpoint,digest,data=audit(args.backup,args.data_root)
    cfg=info['config'];arch,metrics=sg_entry.existing.fs_runtime.components()
    metadata=dict(protocol=PROTOCOL,source_checkpoint=str(checkpoint),source_sha256=digest,
                  source_commit=info['commit'],runner_commit=commit,config=cfg,environment=info['environment'],data=info['data'])
    fingerprint=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=metadata:
        raise ValueError('Output directory belongs to a different protocol/source')
    write(manifest,metadata)
    model=build_model(arch,cfg).cuda();model.load_state_dict(state['model'],strict=True)
    initial={k:v.cpu().clone() for k,v in state['model'].items()}
    del state
    r0={name:evaluate(model,pairs,metrics,cfg,args.output/f'R0_{name}.json','cuda') for name,pairs in data.items() if name!='train'}
    for name,values in r0.items():
        for metric,value in values.items():
            if abs(value-best['metrics'][name][metric])>1e-6:
                raise ValueError(f'R0 archive mismatch: {name}/{metric}')
    write(args.output/'R0.json',dict(metrics=r0,checkpoint=str(checkpoint),sha256=digest,deployment=deployment_check(model,arch,'cuda')))
    del model
    result=dict(R0=r0,source_update=48000,protocol=PROTOCOL,arms={})
    for label in ('A-L1','B-MSE'):
        result['arms'][label]=branch(label,initial,cfg,data,arch,metrics,args.output,fingerprint)
        for candidate in result['arms'][label]['selected'].values():
            candidate['delta_vs_R0']={name:{m:values[m]-r0[name][m] for m in ('psnr','ssim')} for name,values in candidate['metrics'].items()}
        write(args.output/'comparison.json',result)
    options=[dict(label='R0',checkpoint=str(checkpoint),validation=r0['DIV2K_valid'])]
    for label,arm in result['arms'].items():
        options.extend(dict(label=label+'/'+kind,checkpoint=str(args.output/label/r['checkpoint']),validation=r['validation']) for kind,r in arm['selected'].items())
    result['overall_validation_winner']=max(options,key=lambda r:r['validation']['psnr'])
    result['note']='48K checkpoint followed by independent 10K optimization; no prediction of 1M training. Selection uses validation only; R0 eligible if no improvement.'
    write(args.output/'comparison.json',result)
    print('FINE-TUNING COMPLETE\n'+json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
