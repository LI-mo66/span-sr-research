"""Frozen real SG6 feature-context probe. Not candidate PSNR or training initialization."""
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import torch
from bootstrap import components,trainer
from sg_model import build_model as sg6
from strip_model import PartialStripDepth

def stat(x):
    x=x.detach().float()
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite feature probe')
    return dict(rms=float(x.square().mean().sqrt()),mean=float(x.mean()),std=float(x.std(unbiased=False)),
        absmax=float(x.abs().max()),negative_fraction=float((x<0).float().mean()))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--group',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    audit=json.loads((args.group/'reuse-audit.json').read_text())
    cfg=json.loads((args.group/f'K{audit["kernels"][0]}.json').read_text())
    reference=Path(audit['reference_run'])
    info=json.loads((reference/'manifest.json').read_text())
    torch.backends.cudnn.allow_tf32=info['environment']['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32=info['environment']['matmul_allow_tf32']
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    arch,_=components();model=sg6(arch,cfg).cuda()
    state=torch.load(audit['reference_weight'],map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True);model.eval()
    layers={}
    for k in audit['kernels']:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(70031)
            # Diagnostic-only energy calibration to trained reference depth, not formal init.
            import copy
            layers[k]=PartialStripDepth(copy.deepcopy(model.block_3.spatial_gate.depth).cpu(),k).cuda()
    paths=json.loads(Path(cfg['paths_file']).read_text())
    pairs=trainer.pairs(paths['validation']['DIV2K_valid'])
    indices=[0,len(pairs)//3,2*len(pairs)//3,len(pairs)-1]
    rows=[]
    captured={}
    h=model.block_3.c3_r.register_forward_hook(lambda m,inp,out:captured.update(F=out.detach().clone()))
    try:
        with torch.inference_mode():
            for index in indices:
                hp,lp=pairs[index];hr,lr=trainer.aligned(hp,lp)
                size=min(64,*lr.shape[:2]);top=(lr.shape[0]-size)//2;left=(lr.shape[1]-size)//2
                model(trainer.tensor(lr[top:top+size,left:left+size]).unsqueeze(0).cuda())
                f=captured['F'];branch=model.block_3.spatial_gate
                old=branch.depth(f);original=branch.project(branch.act(old))
                row=dict(image=hp.name,crop_lr=[top,left,size,size],F=stat(f),original_depth=stat(old),original_spatial=stat(original),variants={})
                for k,layer in layers.items():
                    new=layer(f);spatial=branch.project(branch.act(new))
                    row['variants'][str(k)]=dict(depth=stat(new),spatial=stat(spatial),
                        selected_depth_rms_ratio=stat(new[:,:13])['rms']/max(stat(old[:,:13])['rms'],1e-30),
                        full_depth_rms_ratio=stat(new)['rms']/max(stat(old)['rms'],1e-30),
                        spatial_rms_ratio=stat(spatial)['rms']/max(stat(original)['rms'],1e-30),
                        changed_spatial_rms=stat(spatial-original)['rms'])
                rows.append(row)
    finally:h.remove()
    if trainer.digest(Path(audit['reference_weight']))!=audit['reference_weight_sha256']:
        raise ValueError('Reference checkpoint changed')
    result=dict(status='PASS',scope='FROZEN_CONTEXT_ONLY',images=rows,normalization_added=False,
        limitations='Random strip core energy-matched to trained SG6 depth only for scale stress probing. This is not candidate training initialization, learned feature matching, PSNR evidence, or permission to train. Formal candidates start from scratch.')
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
