"""Shared I/O, pinned model factories and invariants for paper-only inference."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import torch

BASE = Path(os.environ.get('PAPER_BASE', '/home/featurize/work/span-x4-research'))
PROJECT = Path(os.environ.get('PAPER_PROJECT', str(BASE/'code/span-sg6-f0-long-x8-1m')))
ROOT = Path(os.environ.get('PAPER_ROOT', str(BASE/'records/paper_visuals_20261126')))
OUT = Path(os.environ.get('PAPER_OUT', str(ROOT/'step10-paper-suite-v2-01')))
DATA = Path(os.environ.get('PAPER_DATA', '/home/featurize/datasets/benchmark'))
DD_COMMIT = 'b34bb97ba68f91a25afee23d737d97ca0a270bde'
LAM_COMMIT = '6e75cc026ee8570f3966a6f017e9152556400876'
FINAL_SHA = '1f691364ec3958db9eb67800c583e032c1813d8ddd8d616b466f4717600d7448'
BOUND_SHA = '19bef1a8a112c5cb6136478c0130fab9466abdd80987c4f336a3ffbd133f16fb'
RUNS = {
    'B0': ('d280dc5ebcffc46c50983dfe27e81c6096d90dcf',
           'FEAT-MOEFS-G4-LR2E4-20K-01-1791983354063378494/runs/FEAT-MOEFS-G4-LR2E4-20K-01-B0-s10'),
    'SG6': ('e731483d4ce4e325f3d64dbbc80add342b46f7f1',
            'FEAT-LG6-SG6-50K-01-1792264356078917269/runs/FEAT-LG6-SG6-50K-01-LG6-SiLU-SG6-s10'),
    'W': ('78cc1dda90a5bafcefa6c62225197f6eb95c92e2',
          'FEAT-SG6-MID-W16-PC26-AC-LR2E4-50K-01-INTERRUPTED-20261026T033650Z-20587/runs/FEAT-SG6-MID-W16-PC26-AC-LR2E4-50K-01-A-s10'),
    'U': ('d42b73fef5b2ec3e088f340ab033a5c4ac13a190',
          'FEAT-SG6-UCAN-P24-LKSA5-V1-50K-01-20261025T125002Z/runs/FEAT-SG6-UCAN-P24-LKSA5-V1-50K-01'),
    'F0': ('2416cd660f25c3a8351330a8ad56d7fc8aa233e0',
           'FEAT-SG6-W-UCAN-P24-LKSA5-FUSION-V1-50K-01-1793142163941762977/runs/FEAT-SG6-W-UCAN-P24-LKSA5-FUSION-V1-50K-01'),
}

# Actual files in the archived commits, checked against the GitHub trees.
ABLATION_ENTRIES = {
    'W': ('research/sg6_mid_window/model.py','b5aa9d8d24aa2dd4143ff70040bd70587c64d063'),
    'U': ('research/ucan_p24_lksa5/candidate.py','7aa5cf8c6e3f4f86d4bfc458c3aba9130835a6de'),
    'F0': ('research/fusion_wu/fusion.py','ab57d5a5e3a87bb7419a52497e29aa2733a6de0b'),
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8')==data: return
        raise FileExistsError(f'Refusing to overwrite existing evidence: {path}')
    temporary=path.with_name(path.name+'.part')
    with temporary.open('w',encoding='utf-8') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def fetch(url,path,blob=None,expected_sha=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'SPAN-paper-reproduction'})
                with urllib.request.urlopen(req,timeout=45) as response:
                    raw=response.read(100_000_001)
                if len(raw)>100_000_000: raise ValueError('Unexpected oversized download')
                if raw.startswith(b'<!DOCTYPE html') or raw.startswith(b'<html'): raise ValueError('HTML instead of asset')
                part=path.with_name(path.name+'.part');part.write_bytes(raw);part.replace(path)
                break
            except Exception:
                if attempt==2: raise
                time.sleep(2)
    raw=path.read_bytes()
    if blob and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=blob:
        raise ValueError(f'Official git blob mismatch: {path}')
    if expected_sha and sha(path)!=expected_sha: raise ValueError(f'SHA256 mismatch: {path}')
    return dict(path=str(path),bytes=len(raw),sha256=sha(path),url=url,git_blob=blob)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def runtime(project=PROJECT):
    return load('paper_runtime',Path(project)/'research/runtime.py').components()


def metrics(sr,gt,project=PROJECT):
    global _metric_module
    if '_metric_module' not in globals(): _,_metric_module=runtime(project)
    a,b=sr[:,:,::-1].copy(),gt[:,:,::-1].copy()
    return dict(psnr=float(_metric_module.calculate_psnr(a,b,crop_border=4,test_y_channel=True)),
                ssim=float(_metric_module.calculate_ssim(a,b,crop_border=4,test_y_channel=True)))


def dataset(name):
    folders=[p for p in DATA.iterdir() if p.is_dir() and p.name.casefold()==name.casefold()]
    if len(folders)!=1: raise ValueError(f'Expected one dataset directory: {name}')
    return folders[0]


def pair(name,image):
    root=dataset(name)
    hp=root/'HR'/image; lp=root/'LR_bicubic/X4'/(Path(image).stem+'x4.png')
    hr=np.array(Image.open(hp).convert('RGB'));lr=np.array(Image.open(lp).convert('RGB'))
    h,w=lr.shape[0]*4,lr.shape[1]*4
    if not(0<=hr.shape[0]-h<4 and 0<=hr.shape[1]-w<4): raise ValueError('HR/LR alignment mismatch')
    return hr[:h,:w].copy(),lr,dict(hr=str(hp),lr=str(lp),hr_sha256=sha(hp),lr_sha256=sha(lp))


def tensor(rgb,device):
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float().unsqueeze(0).to(device)/255.


def quantize(output):
    if output.ndim!=4 or output.shape[:2]!=(1,3) or not torch.isfinite(output).all():
        raise ValueError('Invalid RGB prediction')
    return np.rint(output[0].clamp(0,1).detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)


def save_png(path,rgb):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not np.array_equal(np.array(Image.open(path).convert('RGB')),rgb): raise ValueError(f'Existing PNG differs: {path}')
    else:
        temporary=path.with_name(path.stem+'.part.png')
        Image.fromarray(rgb).save(temporary,format='PNG')
        if not np.array_equal(np.array(Image.open(temporary).convert('RGB')),rgb):raise ValueError('Temporary PNG readback mismatch')
        temporary.replace(path)
    if not np.array_equal(np.array(Image.open(path).convert('RGB')),rgb): raise ValueError('PNG readback mismatch')


def image_path(ds,method,image):
    candidates=[OUT/'sr_outputs'/ds/method/image,ROOT/'sr_outputs'/ds/method/image]
    present=[p for p in candidates if p.is_file()]
    if not present: raise FileNotFoundError(str(candidates))
    if len(present)>1 and sha(present[0])!=sha(present[1]): raise ValueError('Conflicting method PNGs')
    return present[0]


def get_image(ds,method,image):
    return np.array(Image.open(image_path(ds,method,image)).convert('RGB'))


def final_model(device):
    cfg=read_json(PROJECT/'logs/f0-finetune-five-01/FT2/config.json')
    checkpoint=BASE/'records/F0-X4-FT2-20K-FINAL-ARCHIVE-01/assets/F0-x4-FT2-20K-full.pt'
    boundpath=BASE/'records/F0-FEATURE-BOUND8-1M-FINAL-01/candidate/feature_bound.py'
    if sha(checkpoint)!=FINAL_SHA or sha(boundpath)!=BOUND_SHA: raise ValueError('Final weight/bound hash changed')
    fusion=load('paper_fusion_final',PROJECT/'research/fusion_wu/fusion.py')
    arch,_=runtime(PROJECT);model=fusion.build_model(arch,cfg)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['update']!=20000: raise ValueError('Wrong finetune update')
    model.load_state_dict(state['model'],strict=True)
    load('paper_bound8',boundpath).install(model,8.)
    return model.to(device).eval(),dict(path=str(checkpoint),sha256=sha(checkpoint),weight_key='model',bound8=str(boundpath),bound_sha256=sha(boundpath))


def check_ablation_entry(name,src):
    relative,expected=ABLATION_ENTRIES[name]
    path=Path(src)/relative
    raw=path.read_bytes()
    observed=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if observed!=expected:raise ValueError('Archived factory blob mismatch: '+str(path))
    return path,observed


def build_ablation(name,src,cfg):
    """Use an isolated process per model, with its exact historical factory."""
    src=Path(src)
    arch,_=runtime(src)
    for folder in ['linear_gate','gated_bypass','spatial_gate6']:
        sys.path.insert(0,str(src/'research'/folder))
    if name=='B0': model=arch.SPAN(**cfg['model'])
    elif name=='SG6': model=load('paper_sg6',src/'research/spatial_gate6/sg_model.py').build_model(arch,cfg)
    else:
        entry,_=check_ablation_entry(name,src)
        sys.path.insert(0,str(entry.parent))
        module=load('paper_candidate',entry)
        model=module.build_model(arch,cfg)
    return model


def ablation_model(name,device):
    _,run_relative=RUNS[name];run=BASE/'records'/run_relative
    manifest=read_json(run/'manifest.json');cfg=manifest['config']
    src=OUT/'source50'/name
    model=build_ablation(name,src,cfg)
    checkpoint=run/'epoch00048_u48000.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['update']!=48000: raise ValueError('Wrong ablation update')
    model.load_state_dict(state['model'],strict=True)
    provenance=dict(path=str(checkpoint),sha256=sha(checkpoint),weight_key='model',source_commit=RUNS[name][0],bound8=False)
    if name in ABLATION_ENTRIES:
        entry,blob=check_ablation_entry(name,src)
        provenance.update(factory_path=str(entry),factory_git_blob=blob)
    return model.to(device).eval(),run,provenance


def public_model(name,device):
    arch,_=runtime()
    if name=='SPAN52':
        # Same released architecture and strict state loading as successful step7.
        model=arch.SPAN(num_in_ch=3,num_out_ch=3,feature_channels=52,upscale=4,img_range=255.)
        path=ROOT/'weights/4x-spanx4_ch52.pth'
        if not path.exists():
            choices=list((ROOT/'weights').glob('*span*.pth'))
            choices=[p for p in choices if sha(p)=='db3aab7cbd91d47d747dd2320c8f915e30580ac87c59826869c909cd094a397f']
            if len(choices)!=1: raise FileNotFoundError('Cannot locate exact released SPAN52 weight')
            path=choices[0]
        if sha(path)!='db3aab7cbd91d47d747dd2320c8f915e30580ac87c59826869c909cd094a397f': raise ValueError('Wrong SPAN weight')
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False),strict=True)
    elif name=='SwinIR-light':
        sys.path.insert(0,str(ROOT/'python_deps'))
        mod=load('paper_swinir',ROOT/'external/SwinIR/models/network_swinir.py')
        core=mod.SwinIR(upscale=4,in_chans=3,img_size=64,window_size=8,img_range=1.,depths=[6]*4,embed_dim=60,num_heads=[6]*4,mlp_ratio=2,upsampler='pixelshuffledirect',resi_connection='1conv')
        path=ROOT/'weights/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth'
        if sha(path)!='09fad24e32ae62722e1a055efde9921328f4137981bab0a42a4a3a806306c58e': raise ValueError('Wrong SwinIR weight')
        core.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['params'],strict=True)
        class Padded(torch.nn.Module):
            def __init__(self):super().__init__();self.core=core
            def forward(self,x):
                h,w=x.shape[-2:]; ph=(h//8+1)*8-h;pw=(w//8+1)*8-w
                padded=torch.cat([x,x.flip([2])],2)[:,:,:h+ph,:]
                padded=torch.cat([padded,padded.flip([3])],3)[:,:,:,:w+pw]
                return self.core(padded)[:,:,:h*4,:w*4]
        model=Padded()
    else: raise ValueError(name)
    return model.to(device).eval(),dict(path=str(path),sha256=sha(path),mode='single')
