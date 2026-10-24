"""Read-only trained-weight interventions. No optimizer, step, or checkpoint save.

Uses the exact archived training source, not this checkout's current model code.
Fixed-weight dependency tests cannot substitute for independently trained ablations.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import statistics
import sys
import traceback
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True

import numpy as np
import torch
from torch.nn import functional as F

TRAIN_COMMIT='d4dbdbcf24a4297410d097bdea800dd103ae22c6'
BASE='e731483d4ce4e325f3d64dbbc80add342b46f7f1'
RUN='FEAT-SG6-UCAN-IND-V2-50K-03'
UNIFORM_INDEX=(0,14,28,42,57,71,85,99)
MODES={'full':{},'off_2':{2:'off'},'off_4':{4:'off'},'off_both':{2:'off',4:'off'},
       'spatial_off_2':{2:'spatial'},'spatial_off_4':{4:'spatial'},
       'channel_off_2':{2:'channel'},'channel_off_4':{4:'channel'},
       'spatial_off_both':{2:'spatial',4:'spatial'},
       'channel_off_both':{2:'channel',4:'channel'}}


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def read(path):return json.loads(path.read_text(encoding='utf8'))


def backup_hashes(folder):
    rows=read(folder/'backup-manifest.json')['files']
    return rows if isinstance(rows,dict) else {r['path']:r['sha256'] for r in rows}


def verify_file(path,expected):
    if not path.is_file() or sha(path)!=expected:raise ValueError('Missing or hash-mismatched file: '+str(path))


def verify_payload(folder,path):
    relative=path.relative_to(folder).as_posix()
    verify_file(path,backup_hashes(folder)[relative])


def extract_source(archive,target):
    # The source ZIP is separately checked against the run's backup manifest.
    target.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name=PurePosixPath(info.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in info.filename:
                raise ValueError('Unsafe source ZIP path')
            if ((info.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('Unexpected symlink in source ZIP')
            path=target.joinpath(*name.parts)
            if info.is_dir():path.mkdir(parents=True,exist_ok=True)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(z.read(info))


@contextmanager
def intervene(model,settings):
    handles=[]
    try:
        for position,mode in settings.items():
            adapter=getattr(model,f'context_{position}')
            if mode=='off':
                def disable(module,args,output):return args[0],output[1]
                handles.append(adapter.register_forward_hook(disable))
            else:
                def mask(module,args,mode=mode):
                    value=args[0].clone()
                    if mode=='spatial':value[...,:26]=0
                    elif mode=='channel':value[...,26:]=0
                    else:raise ValueError('Unknown intervention')
                    return (value,)
                handles.append(adapter.core.proj.register_forward_pre_hook(mask))
        yield
    finally:
        for handle in handles:handle.remove()


def stats(t):
    x=t.detach().double()
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite feature')
    return dict(mean=float(x.mean()),rms=float(x.square().mean().sqrt()),absmax=float(x.abs().max()))


def cosine(a,b):
    a,b=a.detach().double().flatten(),b.detach().double().flatten()
    den=a.norm()*b.norm()
    return None if den==0 else float((a@b)/den)


def parameter_digest(model):
    h=hashlib.sha256()
    # Upstream eval_conv is a nontrainable forward cache, intentionally refreshed
    # by eval. Check every trainable weight and persistent frequency buffer.
    for name,p in list(model.named_parameters())+list(model.named_buffers()):
        if '.eval_conv.' in name:continue
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def record_features(model,x):
    rows={};handles=[]
    for pos in (2,4):
        if not hasattr(model,f'context_{pos}'):continue
        unit=getattr(model,f'context_{pos}');row=rows[str(pos)]={}
        def norm_hook(module,args,out,row=row):
            row['pre_ln']=stats(args[0]);row['post_ln']=stats(out)
        def adapter_hook(module,args,out,row=row):
            row['input']=stats(args[0]);row['correction']=stats(out[0]-args[0])
            row['output']=stats(out[0])
            row['correction_input_cosine']=cosine(out[0]-args[0],args[0])
            row['correction_input_rms_ratio']=row['correction']['rms']/max(row['input']['rms'],1e-30)
        def proj_hook(module,args,row=row,unit=unit):
            v=args[0];scale=1.0
            # Projection halves are diagnostic contributions before RMS scaling;
            # the final common bias is reported separately, not double counted.
            spatial=F.linear(v[...,:26],module.weight[:,:26],None)
            channel=F.linear(v[...,26:],module.weight[:,26:],None)
            row['projected_spatial']=stats(spatial);row['projected_channel']=stats(channel)
            row['projected_branch_cosine']=cosine(spatial,channel)
            row['projection_bias']=stats(module.bias)
        def qhook(module,args,out,row=row):row['_q']=out
        def khook(module,args,out,row=row):
            q=row.pop('_q')
            denominator=q@out.mean(-2,keepdim=True).transpose(-2,-1)+1e-6
            row['spatial_denominator_min']=float(denominator.min())
            row['spatial_inverse_max']=float(denominator.reciprocal().max())
        handles.extend([unit.norm.register_forward_hook(norm_hook),unit.register_forward_hook(adapter_hook),
                        unit.core.proj.register_forward_pre_hook(proj_hook),
                        unit.core.ln_q.register_forward_hook(qhook),unit.core.ln_k.register_forward_hook(khook)])
        row['fourier']={}
        for name in ('sin_coef','cos_coef'):
            coef=getattr(unit.core.pos_emb,name).detach()
            sums=coef.abs().sum(-2)
            row['fourier'][name]=dict(min_abs_sum=float(sums.min()),
                floored_columns=int((sums<unit.core.pos_emb.floor).sum()),
                normalized_column_abs_sum_max=float(unit.core.pos_emb.normalized(coef).abs().sum(-2).max()))
        row['temperature']=stats(unit.core.temperature)
    rows['sg6_blocks']={}
    for pos in range(1,7):
        if not hasattr(model,f'block_{pos}'):continue
        block=getattr(model,f'block_{pos}');block_row=rows['sg6_blocks'][str(pos)]={}
        def block_hook(module,args,out,row=block_row):
            row['input']=stats(args[0]);row['output']=stats(out[0]);row['combined_gate']=stats(out[2])
        handles.append(block.register_forward_hook(block_hook))
        for key in ('local_gate','spatial_gate'):
            def gate_hook(module,args,out,row=block_row,key=key):row[key]=stats(out)
            handles.append(getattr(block,key).register_forward_hook(gate_hook))
    try:
        with torch.inference_mode():model(x)
    finally:
        for handle in handles:handle.remove()
    return rows


def quantize(y):
    if not torch.isfinite(y).all():raise FloatingPointError('Nonfinite model prediction')
    return np.rint(y[0].detach().clamp(0,1).cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8)


def score(metrics,sr,hr):
    kw=dict(crop_border=4,test_y_channel=True)
    return {m:getattr(metrics,'calculate_'+m)(sr,hr,**kw) for m in ('psnr','ssim')}


def infer(model,x):
    with torch.inference_mode():return quantize(model(x))


def patch_probe(model,x,hr,trainer):
    import copy
    probe=copy.deepcopy(model).double()
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in probe.modules()):
        raise ValueError('Unexpected running-state normalization; gradient probe blocked')
    h,w=x.shape[-2:];ph,pw=min(64,h),min(64,w);top,left=(h-ph)//2,(w-pw)//2
    patch=x[:,:,top:top+ph,left:left+pw].double()
    target=trainer.tensor(hr[top*4:(top+ph)*4,left*4:(left+pw)*4]).unsqueeze(0).to(patch).double()
    with torch.no_grad():
        probe.eval();eval_y=probe(patch)
        probe.train();train_y=probe(patch)
        error=float((eval_y-train_y).abs().max())
    result=dict(shape=list(patch.shape),train_eval_fp64_absmax=error,loss='RGB L1 on fixed center validation crop',optimizer_step=False)
    if error>1e-8:
        result.update(status='GRADIENT_SKIPPED_NUMERICAL_GATE',reason='FP64 train/eval mismatch; not a performance failure')
        return result
    captured={};handles=[]
    for pos in (2,4):
        def capture(module,args,out,pos=pos):captured[pos]=(args[0],out[0])
        handles.append(getattr(probe,f'context_{pos}').register_forward_hook(capture))
    try:
        pred=probe(patch);loss=F.l1_loss(pred,target)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite gradient-probe loss')
        result['loss_value']=float(loss.detach());result['path_gradients']={}
        for pos,(u,v) in captured.items():
            cotangent=torch.autograd.grad(loss,v,retain_graph=True)[0]
            correction_gradient=torch.autograd.grad(v-u,u,grad_outputs=cotangent,retain_graph=True)[0]
            result['path_gradients'][str(pos)]=dict(identity_cotangent=stats(cotangent),
                correction_path=stats(correction_gradient),
                same_input_path_cosine=cosine(cotangent,correction_gradient),
                meaning='local path alignment only; not proof of optimizer conflict or benefit')
        params=[(n,p) for n,p in probe.named_parameters() if n.startswith('context_')]
        grads=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
        result['parameters']={n:dict(weight=stats(p),gradient=None if g is None else stats(g)) for (n,p),g in zip(params,grads)}
        result['status']='PASS'
    finally:
        for handle in handles:handle.remove()
        del probe
    return result


def execute(args,status):
    archive=args.candidate.resolve();reference=args.reference.resolve();out=args.output
    hashes=backup_hashes(archive)
    for marker in ('BACKUP_COMPLETE','FINAL_STATUS.json'):
        if not (archive/marker).is_file():raise ValueError('Incomplete candidate backup')
    final=read(archive/'FINAL_STATUS.json')
    if not final.get('success') or final['id']!=RUN:raise ValueError('Wrong or failed candidate run')
    if read(archive/'backup-manifest.json')['source_commit']!=TRAIN_COMMIT:
        raise ValueError('Unexpected trained source revision')
    verify_file(archive/'source.zip',hashes['source.zip'])
    source=args.work/'archived-source';extract_source(archive/'source.zip',source)
    sys.path.insert(0,str(source/'research/ucan_qk_pair'))
    from run import existing
    from pair_model import build_model,build_sg6
    trainer=existing.fs_runtime.trainer
    arch,metrics=existing.fs_runtime.components()
    run=archive/'runs'/RUN;manifest=read(run/'manifest.json');audit=read(archive/'logs/frozen/audit.json')
    for file in (run/'manifest.json',archive/'logs/frozen/audit.json'):
        verify_payload(archive,file)
    if manifest['config']!=audit['candidate_config'] or manifest['commit']!=TRAIN_COMMIT:
        raise ValueError('Candidate source/config audit mismatch')
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
    env=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
             deterministic=True,cudnn_allow_tf32=True,matmul_allow_tf32=False)
    if env!=manifest['environment']:raise ValueError('Server precision/environment differs from archived training')
    status['environment']=env
    candidates=list((reference/'runs').glob('*/manifest.json'))
    baselines=[p.parent for p in candidates if read(p).get('commit')==BASE and read(p).get('config',{}).get('experiment_version')=='LG6-SiLU-GB-SpatialGate6-v1']
    if len(baselines)!=1:raise ValueError('Unique SG6 reference run required')
    base_run=baselines[0];base_manifest=read(base_run/'manifest.json')
    if base_manifest['environment']!=env:raise ValueError('Reference environment mismatch')
    base_weight=base_run/'epoch00048_u48000.pt';verify_file(base_weight,audit['reference_weight_sha256'])
    base_state=torch.load(base_weight,map_location='cpu',weights_only=False)
    baseline=build_sg6(arch,base_manifest['config']);baseline.load_state_dict(base_state['model'],strict=True)
    del base_state;baseline=baseline.cuda().eval()
    weights=[run/'epoch00048_u48000.pt',run/'epoch00050_u50000.pt']
    for p in weights:verify_file(p,hashes[p.relative_to(archive).as_posix()])
    frozen_files={str(p):sha(p) for p in [base_weight,*weights]}
    data=manifest['data']['DIV2K_valid']
    base_data=base_manifest['data']['DIV2K_valid']
    if len(data)!=100 or len(base_data)!=100:raise ValueError('Expected fixed 100 validation images')
    old={r['image']:r for r in read(base_run/'DIV2K_valid_epoch00048_u48000.json')}
    current={r['image']:r for r in read(run/'DIV2K_valid_epoch00048_u48000.json')}
    verify_payload(reference,base_run/'DIV2K_valid_epoch00048_u48000.json')
    verify_payload(archive,run/'DIV2K_valid_epoch00048_u48000.json')
    if set(old)!=set(current) or set(current)!={Path(r['hr']).name for r in data}:
        raise ValueError('Validation image identity mismatch')
    uniform=[Path(data[i]['hr']).name for i in UNIFORM_INDEX]
    worst=sorted(current,key=lambda n:current[n]['psnr']-old[n]['psnr'])[:4]
    chosen=list(dict.fromkeys(uniform+worst))
    groups={name:('uniform' if name in uniform else 'exploratory_worst') for name in chosen}
    write(out/'selection.json',dict(uniform=uniform,exploratory_worst=worst,
        rule='8 fixed evenly spaced validation images; worst 4 are post-hoc exploratory and never merged into uniform averages',
        interventions=MODES,gradient_images=uniform[::2],checkpoint_updates=[48000,50000]))
    records=[];gradients=[];features=[];learned={}
    for update,weight in zip((48000,50000),weights):
        state=torch.load(weight,map_location='cpu',weights_only=False)
        if state['update']!=update:raise ValueError('Checkpoint update mismatch')
        model=build_model(arch,manifest['config'])
        initial={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('context_')}
        model.load_state_dict(state['model'],strict=True)
        learned[str(update)]={n:dict(initial_rms=stats(initial[n])['rms'],trained_rms=stats(p)['rms'],
            change_rms=stats(p.detach()-initial[n])['rms']) for n,p in model.named_parameters() if n in initial}
        write(out/'initialization_deviations.json',learned)
        del initial
        del state;model=model.cuda().eval();before=parameter_digest(model)
        archived={r['image']:r for r in read(run/f'DIV2K_valid_epoch{update//1000:05d}_u{update}.json')}
        verify_payload(archive,run/f'DIV2K_valid_epoch{update//1000:05d}_u{update}.json')
        image_names=chosen if update==48000 else uniform
        for index,name in enumerate(image_names,1):
            row=next(r for r in data if Path(r['hr']).name==name)
            br=next(r for r in base_data if Path(r['hr']).name==name)
            for key in ('hr_sha256','lr_sha256'):
                if row[key]!=br[key]:raise ValueError('Baseline/candidate image mismatch')
            hp,lp=Path(row['hr']),Path(row['lr'])
            verify_file(hp,row['hr_sha256']);verify_file(lp,row['lr_sha256'])
            hr,lr=trainer.aligned(hp,lp);x=trainer.tensor(lr).unsqueeze(0).cuda()
            original=infer(model,x);full=score(metrics,original,hr)
            if abs(full['psnr']-archived[name]['psnr'])>0.0002 or abs(full['ssim']-archived[name]['ssim'])>2e-6:
                raise ValueError('Unmodified candidate replay mismatch: '+name)
            if update==48000:
                bsr=infer(baseline,x);bscore=score(metrics,bsr,hr)
                if abs(bscore['psnr']-old[name]['psnr'])>0.0002:raise ValueError('SG6 replay mismatch')
                features.append(dict(update=update,image=name,group=groups[name],
                    units=record_features(model,x),sg6_reference_units=record_features(baseline,x)))
            modes=MODES if update==48000 else {k:MODES[k] for k in ('full','off_2','off_4','off_both')}
            for mode,settings in modes.items():
                if mode=='full':values=full
                else:
                    with intervene(model,settings):values=score(metrics,infer(model,x),hr)
                records.append(dict(update=update,image=name,group=groups[name],mode=mode,**values,
                    delta_from_full_psnr=values['psnr']-full['psnr'],
                    full_delta_vs_sg6=full['psnr']-old[name]['psnr'],reference_update=48000))
            if update==48000 and name in uniform[::2]:
                gradients.append(dict(image=name,**patch_probe(model,x,hr,trainer)))
            print(f'CHECKPOINT {update} image {index}/{len(image_names)} {name}: full PSNR={full["psnr"]:.6f}',flush=True)
            write(out/'interventions.json',records);write(out/'features.json',features);write(out/'gradients.json',gradients)
            del x,original
        if before!=parameter_digest(model):raise ValueError('Trained parameters changed during diagnostic')
        del model;torch.cuda.empty_cache()
    for path,digest in frozen_files.items():verify_file(Path(path),digest)
    summary=dict(scope='fixed-weight dependencies; not independent training ablations, no optimizer step',
                 uniform_images=uniform,exploratory_worst=worst,weights_verified=True,
                 trained_parameters_unchanged=True,original_checkpoint_files_unchanged=True,
                 gradient_status={r['image']:r['status'] for r in gradients},effects={})
    for update in (48000,50000):
        summary['effects'][str(update)]={}
        for mode in MODES:
            subset=[r for r in records if r['image'] in uniform and r['update']==update and r['mode']==mode]
            if subset:summary['effects'][str(update)][mode]=dict(images=len(subset),
                mean_delta_from_full_psnr=statistics.mean(r['delta_from_full_psnr'] for r in subset),
                median_delta_from_full_psnr=statistics.median(r['delta_from_full_psnr'] for r in subset))
    write(out/'summary.json',summary)
    (out/'PASTE_SUMMARY.txt').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    status.update(success=True,stage='completed',scope=summary['scope'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    args.work.mkdir(parents=True,exist_ok=False)
    status=dict(success=False,stage='initializing',training_started=False,optimizer_step=False)
    try:execute(args,status)
    except Exception:
        status.update(stage='failed',error=traceback.format_exc());print(status['error'],flush=True)
    finally:
        write(args.output/'STATUS.json',status)
        packet=args.output/'diagnostic-results.zip'
        with zipfile.ZipFile(packet,'x',zipfile.ZIP_DEFLATED) as z:
            for file in args.output.rglob('*'):
                if file.is_file() and file!=packet:z.write(file,file.relative_to(args.output))
        (args.output/'diagnostic-results.sha256').write_text(sha(packet)+'\n',encoding='ascii')
        print('STATUS:',json.dumps(status,ensure_ascii=False),flush=True)
        print('RESULT_ZIP:',packet,flush=True)
        if (args.output/'PASTE_SUMMARY.txt').is_file():print((args.output/'PASTE_SUMMARY.txt').read_text(),flush=True)
    raise SystemExit(0 if status['success'] else 1)


if __name__=='__main__':main()
