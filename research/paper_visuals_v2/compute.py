"""Independent inference actions. No training, optimizer, or test-time fitting."""
import argparse
import ast
import gc
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

from common import *


def ddistill(device):
    folder=OUT/'external/DDistill-SR'
    entries=[]
    for name,blob in [('DDistill_SR_Inference_arch.py','b099e6c356795a6895c67b538fdb3625d97eabf1'),
                      ('model-backup/DDistill-SRx4_inference.pth','2a73d71e323b1ed0af10513ab3f728f9001a0f54'),
                      ('README.md','f6884d62ead85b4cca07373f21f7b3bd0d3b74ae')]:
        entries.append(fetch(f'https://raw.githubusercontent.com/icandle/DDistill-SR/{DD_COMMIT}/{name}',folder/name,blob=blob))
    module=load('paper_ddistill',folder/'DDistill_SR_Inference_arch.py')
    model=module.DDistill_SR_Inference(in_nc=3,out_nc=3,nf=56,num_modules=4,scale=4,deploy=True,dynamic=True,L=16,style='DBB',res=True)
    checkpoint=folder/'model-backup/DDistill-SRx4_inference.pth'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state,strict=True);model=model.to(device).eval()
    with torch.no_grad():
        for h,w in [(32,40),(31,37)]:
            y=model(torch.zeros(1,3,h,w,device=device))
            if y.shape!=(1,3,h*4,w*4) or not torch.isfinite(y).all(): raise ValueError('DDistill shape/finite failure')
    rows=[];sources=[]
    with torch.no_grad():
        for ds,count in [('Set14',14),('Urban100',100),('Manga109',109)]:
            images=sorted((dataset(ds)/'HR').glob('*.png'))
            if len(images)!=count: raise ValueError(f'Unexpected {ds} count')
            started=time.monotonic()
            for i,hp in enumerate(images,1):
                gt,lr,provenance=pair(ds,hp.name)
                # Check against the already verified GT stored in the paper output.
                if not np.array_equal(gt,get_image(ds,'GT',hp.name)): raise ValueError('Existing GT differs from aligned dataset')
                sr=quantize(model(tensor(lr,device)))
                if sr.shape!=gt.shape:raise ValueError('DDistill whole-image shape mismatch')
                dest=OUT/'sr_outputs'/ds/'ddistill_sr_single'/hp.name;save_png(dest,sr)
                score=metrics(np.array(Image.open(dest).convert('RGB')),gt)
                rows.append(dict(dataset=ds,image_name=hp.name,method='ddistill_sr_single',psnr_y=score['psnr'],ssim_y=score['ssim'],width=gt.shape[1],height=gt.shape[0],png_sha256=sha(dest)))
                sources.append(provenance)
                if i%10==0 or i==count: print(f'DDISTILL_PNG_PASS {ds} {i}/{count} elapsed_s={time.monotonic()-started:.1f}',flush=True)
    write_json(OUT/'ddistill-results.json',dict(repository='https://github.com/icandle/DDistill-SR',commit=DD_COMMIT,official_files=entries,strict_load=True,parameters=sum(p.numel() for p in model.parameters()),mode='single',rows=rows,inputs=sources))


def run_ablation(name,device):
    model,run,provenance=ablation_model(name,device)
    rows=[]
    with torch.no_grad():
        for image in ['img004.png','img011.png']:
            gt,lr,inp=pair('Urban100',image)
            sr=quantize(model(tensor(lr,device)))
            if sr.shape!=gt.shape:raise ValueError('Ablation output shape mismatch')
            score=metrics(sr,gt)
            historical=read_json(run/'Urban100_epoch00048_u48000.json')
            match=[r for r in historical if r['image']==image]
            if len(match)!=1:raise ValueError('Historical per-image result missing')
            error={k:score[k]-float(match[0][k]) for k in ['psnr','ssim']}
            if abs(error['psnr'])>5e-5 or abs(error['ssim'])>1e-6:
                raise ValueError(f'Historical reproduction failure {name} {image}: {error}')
            path=OUT/'sr_outputs/Urban100'/f'{name}_50k_single'/image;save_png(path,sr)
            rows.append(dict(image=image,score=score,historical_delta=error,png=str(path),png_sha256=sha(path),input=inp))
            print('ABLATION_REPRODUCTION_PASS',name,image,score,flush=True)
    write_json(OUT/f'ablation-{name}.json',dict(model=name,checkpoint=provenance,training_budget=50000,selected_update=48000,mode='single',rows=rows,scope='historical independent training; no eval-time module disabling'))


def imdn(device):
    import types
    ref='8f158e6a5ac9db6e5857d9159fd4a6c4214da574'
    folder=OUT/'external/IMDN';entries=[]
    for name,blob in [('model/architecture.py','b297f9c39bc4cd8398b18d9da705a693b3519c9c'),
                      ('model/block.py','86feb771614dacb83966d95989f9cc4c239cf60d'),
                      ('checkpoints/IMDN_x4.pth','3b9bf90db6985f7f97015a6629d001c46f3e2c33')]:
        entries.append(fetch(f'https://raw.githubusercontent.com/Zheng222/IMDN/{ref}/{name}',folder/name,blob=blob))
    package=types.ModuleType('paper_imdn_pkg');package.__path__=[str(folder/'model')];sys.modules[package.__name__]=package
    module=load('paper_imdn_pkg.architecture',folder/'model/architecture.py')
    model=module.IMDN(upscale=4)
    state=torch.load(folder/'checkpoints/IMDN_x4.pth',map_location='cpu',weights_only=False)
    strip_prefix=all(k.startswith('module.') for k in state)
    if strip_prefix:state={k[len('module.'):]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True);model=model.to(device).eval()
    with torch.no_grad():
        for h,w in [(32,40),(31,37)]:
            pred=model(torch.zeros(1,3,h,w,device=device))
            if pred.shape!=(1,3,h*4,w*4) or not torch.isfinite(pred).all():raise ValueError('IMDN forward failure')
    rows=[];sources=[]
    with torch.no_grad():
        for ds,count in [('Set14',14),('Urban100',100),('Manga109',109)]:
            images=sorted((dataset(ds)/'HR').glob('*.png'))
            if len(images)!=count:raise ValueError('Unexpected dataset size')
            started=time.monotonic()
            for i,hp in enumerate(images,1):
                gt,lr,inp=pair(ds,hp.name)
                if not np.array_equal(gt,get_image(ds,'GT',hp.name)):raise ValueError('Existing GT differs')
                sr=quantize(model(tensor(lr,device)))
                if sr.shape!=gt.shape:raise ValueError('IMDN whole-image output mismatch')
                dest=OUT/'sr_outputs'/ds/'imdn_single'/hp.name;save_png(dest,sr)
                score=metrics(np.array(Image.open(dest).convert('RGB')),gt)
                rows.append(dict(dataset=ds,image_name=hp.name,method='imdn_single',psnr_y=score['psnr'],ssim_y=score['ssim'],width=gt.shape[1],height=gt.shape[0],png_sha256=sha(dest)))
                sources.append(inp)
                if i%10==0 or i==count:print(f'IMDN_PNG_PASS {ds} {i}/{count} elapsed_s={time.monotonic()-started:.1f}',flush=True)
    write_json(OUT/'imdn-results.json',dict(repository='https://github.com/Zheng222/IMDN',commit=ref,official_files=entries,strict_load=True,stripped_uniform_DataParallel_prefix=strip_prefix,parameters=sum(p.numel() for p in model.parameters()),mode='single',rows=rows,inputs=sources))


def official_functions():
    folder=OUT/'external/LAM_Demo'
    files=[('SaliencyModel/BackProp.py','1401bc836cf88b2eb255565e9c00e66b6ada4ce2'),
           ('SaliencyModel/utils.py','dc9d8f35c214eaa51b19b93bb921e47592b2c587'),
           ('SaliencyModel/attributes.py','abda92e9eeb6bc0a7684f5dee116aacd883e6c50')]
    keep={'GaussianBlurPath','isotropic_gaussian_kernel','grad_abs_norm','gini','attr_grad','reduce_func'}
    body=[];sources=[]
    for name,blob in files:
        path=folder/name
        sources.append(fetch(f'https://raw.githubusercontent.com/X-Lowlevel-Vision/LAM_Demo/{LAM_COMMIT}/{name}',path,blob=blob))
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        body.extend(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in keep)
    namespace=dict(np=np,torch=torch,cv2=cv2)
    exec(compile(ast.Module(body=body,type_ignores=[]),'official_LAM_selected_functions','exec'),namespace)
    if not keep.issubset(namespace):raise ValueError('Missing pinned official LAM functions')
    return namespace,sources


def stable_objective(prediction,y,x,window=16):
    """Official RGB gradient-magnitude sum; choose zero subgradient at zero norm.

    Cropping the two finite differences before the norm avoids computing
    unrelated sqrt(0) derivatives outside the objective's target region.
    torch.linalg.vector_norm defines a finite zero subgradient at exact zero.
    """
    a=prediction[:,:,y:y+window,x:x+window]
    dy=a-prediction[:,:,y+1:y+window+1,x:x+window]
    dx=a-prediction[:,:,y:y+window,x+1:x+window+1]
    return torch.linalg.vector_norm(torch.stack([dy,dx]),dim=0).sum()


def lam(name,device):
    fn,sources=official_functions()
    model,weight=final_model(device) if name=='Ours' else public_model(name,device)
    model.requires_grad_(False)
    records=[]
    for image,target in [('img004.png',[904,472,16,16]),('img011.png',[296,920,16,16])]:
        gt,lr,inputs=pair('Urban100',image);x,y,w,h=target
        if x<0 or y<0 or x+w>=gt.shape[1] or y+h>=gt.shape[0]:raise ValueError('LAM target outside output-gradient domain')
        method={'SPAN52':'span52_official_single','SwinIR-light':'swinir_light_single','Ours':'ours_single'}[name]
        with torch.no_grad():observed=quantize(model(tensor(lr,device)))
        if not np.array_equal(observed,get_image('Urban100',method,image)):
            raise ValueError(f'LAM model forward differs from verified single-pass PNG: {name} {image}')
        print('LAM_MODEL_REPRODUCTION_PASS',name,image,flush=True)
        raw=lr.astype(np.float32)/255.
        path,velocity=fn['GaussianBlurPath'](1.2,50,l=5)(raw)
        attribution=np.zeros_like(path[0]);objectives=[]
        started=time.monotonic()
        for i in range(50):
            value=torch.from_numpy(path[i:i+1]).to(device).requires_grad_(True)
            prediction=model(value)
            objective=stable_objective(prediction,y,x,window=16)
            gradient,=torch.autograd.grad(objective,value)
            if not torch.isfinite(gradient).all():
                # The demo masks NaNs. We instead stop and report the actual defect.
                raise FloatingPointError(f'LAM nonfinite input gradient: {name} {image} fold{i+1}')
            attribution+=gradient[0].detach().cpu().numpy()*velocity[i]/50.
            objectives.append(float(objective.detach()))
            del gradient,objective,prediction,value
            if (i+1)%10==0:print('LAM_PROGRESS',name,image,i+1,'/50',f'elapsed_s={time.monotonic()-started:.1f}',flush=True)
        norm=fn['grad_abs_norm'](attribution)
        if not np.isfinite(norm).all() or norm.max()<=0:raise ValueError('Empty or invalid LAM map')
        di=float((1.-fn['gini'](norm.copy()))*100.)
        if not(0<=di<=100):raise ValueError('Invalid diffusion index')
        folder=OUT/'lam'/name;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/(Path(image).stem+'.npz'),attribution=attribution,normalized=norm,lr_rgb=lr,objectives=np.asarray(objectives))
        records.append(dict(image=image,target_hr_xywh=target,DI=di,input=inputs,lr_shape=list(lr.shape),seconds=time.monotonic()-started))
        print('LAM_PASS',name,image,'DI',di,flush=True)
    write_json(OUT/f'lam-{name}.json',dict(model=name,weight=weight,official_commit=LAM_COMMIT,sources=sources,settings=dict(sigma_LR=1.2,fold=50,kernel=5,target_window_HR=16,objective='official attr_grad RGB spatial-gradient magnitude sum; stable L2 zero subgradient',channel_reduce='abs(sum signed attribution across RGB); official grad_abs_norm',DI='100*(1-official_gini(normalized_attribution))'),implementation='Exact official blur path, normalization and Gini. Equivalent gradient-magnitude objective crops differences before L2 and uses vector_norm (zero subgradient at exact zero). Streaming autograd.grad(input) replaces demo graph retention. Nonfinite gradients stop instead of post-hoc masking. Equivalence on nonzero toy gradients and finite flat-region gradients tested locally.',mode='single',rows=records,scope='DI conditional on these images, output targets and numerical settings; no causality or universal performance claim'))


def features(device):
    model,weight=final_model(device);captured={};handles=[];rows=[]
    def stage_hook(label):
        def callback(module,args,output):
            a=output[0]
            captured[label]=a.detach().float().cpu().clone().numpy()[0]
        return callback
    for i in [2,3,4,5]:handles.append(getattr(model,f'block_{i}').register_forward_hook(stage_hook(f'F{i}')))
    def gate_hook(module,args,output):
        captured['G']=output[2].detach().float().cpu().clone().numpy()[0]
    handles.append(model.block_3.original.register_forward_hook(gate_hook))
    try:
        for image in ['img004.png','img011.png']:
            captured.clear();gt,lr,inputs=pair('Urban100',image)
            with torch.no_grad():sr=quantize(model(tensor(lr,device)))
            if not np.array_equal(sr,get_image('Urban100','ours_single',image)):
                raise ValueError('Feature-hook inference differs from verified final Ours PNG')
            if set(captured)!={'G','F2','F3','F4','F5'}:raise ValueError('Missing actual feature tensors')
            if any(a.shape!=(52,*lr.shape[:2]) or not np.isfinite(a).all() for a in captured.values()):
                raise ValueError('Feature/gate shape or numerical mismatch')
            maps={k:np.sqrt(np.mean(a.astype(np.float64)**2,axis=0)) for k,a in captured.items() if k!='G'}
            maps['G_mean_signed']=captured['G'].mean(0)
            path=OUT/'features';path.mkdir(exist_ok=True)
            np.savez_compressed(path/(Path(image).stem+'.npz'),**captured,**{k+'_rms':v for k,v in maps.items() if k!='G_mean_signed'},G_mean_signed=maps['G_mean_signed'])
            stats={k:dict(shape=list(a.shape),mean=float(a.mean()),rms=float(np.sqrt(np.mean(a.astype(np.float64)**2))),min=float(a.min()),max=float(a.max())) for k,a in captured.items()}
            rows.append(dict(image=image,inputs=inputs,statistics=stats))
            print('FEATURES_PASS',image,flush=True)
    finally:
        for handle in handles:handle.remove()
    write_json(OUT/'features.json',dict(weight=weight,gate='True SG6 block3 output[2], 52-channel signed gate; shown as channel mean, no extra sigmoid',stages='F2/F3/F4/F5 taken after respective post-block UCAN/W/UCAN/LKSA wrappers and installed Bound8 hooks',rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['ddistill','imdn','ablation','lam','features']);p.add_argument('--name');p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    if a.action=='ddistill':ddistill(a.device)
    elif a.action=='imdn':imdn(a.device)
    elif a.action=='ablation':run_ablation(a.name,a.device)
    elif a.action=='lam':lam(a.name,a.device)
    else:features(a.device)
