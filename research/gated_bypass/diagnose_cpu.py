"""CPU-only fixed-checkpoint bypass audit; real LR images required for content claims."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), std=float(a.std()), min=float(a.min()), max=float(a.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backup', type=Path, required=True)
    p.add_argument('--lr-root', type=Path, default=Path('/home/featurize/datasets/DIV2K/DIV2K_valid_LR_bicubic/X4'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = args.backup.parent
    output = args.output or root / ('GB-CPU-DIAG-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid()))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    matches = list((args.backup/'runs').glob('*-LG6-SiLU-GB-s10'))
    if len(matches) != 1:
        raise ValueError(f'Expected one combined run, got {matches}')
    run = matches[0]
    history = sorted([json.loads(s) for s in (run/'epochs.jsonl').read_text().splitlines() if s.strip()], key=lambda r:r['update'])
    selected = max(history, key=lambda r:r['metrics']['DIV2K_valid']['psnr'])
    checkpoint = run / f'epoch{selected["epoch"]:05d}_u{selected["update"]}.pt'
    info = json.loads((run/'manifest.json').read_text())
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['fingerprint'] != info['fingerprint']:
        raise ValueError('Checkpoint/manifest fingerprint mismatch')
    params = {k:state['model'][k].detach().float().contiguous() for k in
              ('image_gate.weight','image_gate.bias','image_residual.weight','image_residual.bias')}
    del state
    for value in params.values():
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite bypass parameters')
    gw,gb,rw,rb = [params[k] for k in ('image_gate.weight','image_gate.bias','image_residual.weight','image_residual.bias')]
    if gw.shape != (1,3,3,3) or rw.shape != (48,3,3,3):
        raise ValueError('Unexpected architecture')
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), selected_update=selected['update'],
                  source_commit=info.get('commit'), device='cpu', output=str(output), parameters={}, images=[])
    for k,v in params.items():
        report['parameters'][k] = dict(shape=list(v.shape), l2=float(v.norm()), absmax=float(v.abs().max()), mean=float(v.mean()))
    report['gate_bias_only_value'] = float(2*torch.sigmoid(gb)[0])
    report['gate_rgb_kernel_l2'] = gw.square().sum((0,2,3)).sqrt().tolist()
    report['gate_kernel'] = gw.tolist()
    print('CHECKPOINT PARAMETERS\n'+json.dumps(report,indent=2),flush=True)
    expected = {Path(r['lr']).name:r for r in info['data']['DIV2K_valid']}
    if len(expected) != 100:
        raise ValueError('Expected 100 original validation LR files')
    files = [args.lr_root / name for name in sorted(expected)]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        report.update(status='NEEDS_REAL_LR_IMAGES', missing_count=len(missing), expected_lr_root=str(args.lr_root),
                      note='Weight norms cannot prove content adaptivity. Restore only the 100 original DIV2K validation LR x4 PNGs; HR/training images/GPU are not required.')
        (output/'report.json').write_text(json.dumps(report,indent=2))
        print('DIAGNOSTIC WAITING FOR DATA\n'+json.dumps(report,indent=2),flush=True)
        return
    import cv2
    kernels = (gw,gb,rw,rb)
    samples=[]; pooled_n=0; pooled_g=0.;pooled_g2=0.;pooled_c2=0.;pooled_i2=0.;pooled_r2=0.
    with torch.inference_mode():
        for index,path in enumerate(files,1):
            if sha(path) != expected[path.name]['lr_sha256']:
                raise ValueError(f'Original validation LR hash mismatch: {path}')
            bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
            if bgr is None:raise ValueError(f'Decode failed: {path}')
            image=torch.from_numpy(np.ascontiguousarray(bgr[:,:,::-1].transpose(2,0,1))).float().unsqueeze(0)/255
            gate=2*torch.sigmoid(F.conv2d(image,gw,gb,padding=1))
            residual=F.conv2d(image,rw,rb,padding=1)
            correction=gate*residual
            # Interior statistics exclude zero-padding effects, align both 3x3 branches.
            g=gate[:,:,1:-1,1:-1];r=residual[:,:,1:-1,1:-1];c=correction[:,:,1:-1,1:-1];im=image[:,:,1:-1,1:-1]
            if not torch.isfinite(c).all():raise FloatingPointError(f'Nonfinite correction: {path}')
            constant=image.mean((2,3),keepdim=True).expand_as(image)
            constant_gate=(2*torch.sigmoid(F.conv2d(constant,gw,gb,padding=1)))[:,:,1:-1,1:-1]
            rms=lambda t:float(t.square().mean().sqrt())
            row=dict(image=path.name, height=image.shape[2],width=image.shape[3],gate_mean=float(g.mean()),
                gate_spatial_std=float(g.std(unbiased=False)),gate_min=float(g.min()),gate_max=float(g.max()),
                gate_fraction_below_0_1=float((g<.1).float().mean()),gate_fraction_above_1_9=float((g>1.9).float().mean()),
                gate_delta_vs_same_mean_constant_rms=rms(g-constant_gate),input_rms=rms(im),
                residual_rms=rms(r),correction_rms=rms(c),correction_mean_abs=float(c.abs().mean()),
                correction_absmax=float(c.abs().max()),correction_rms_8bit_equivalent=255*rms(c),
                correction_rms_over_input_rms=rms(c)/max(rms(im),1e-12),
                correction_change_if_gate_one_rms=rms((g-1)*r),
                correction_change_if_gate_image_mean_rms=rms((g-g.mean())*r))
            report['images'].append(row)
            n=g.numel();pooled_n+=n;pooled_g+=float(g.sum());pooled_g2+=float(g.square().sum())
            pooled_c2+=float(c.square().mean())*n;pooled_i2+=float(im.square().mean())*n;pooled_r2+=float(r.square().mean())*n
            flat=g.flatten();stride=max(1,(flat.numel()+2047)//2048);samples.append(flat[::stride].numpy())
            if index%20==0:print(f'CPU bypass audit {index}/100',flush=True)
    fields=('gate_mean','gate_spatial_std','gate_fraction_below_0_1','gate_fraction_above_1_9',
            'gate_delta_vs_same_mean_constant_rms','correction_rms','correction_mean_abs','correction_absmax',
            'correction_rms_8bit_equivalent','correction_rms_over_input_rms',
            'correction_change_if_gate_one_rms','correction_change_if_gate_image_mean_rms')
    report['per_image_statistics']={k:summary([r[k] for r in report['images']]) for k in fields}
    report['pooled_statistics']=dict(gate_mean=pooled_g/pooled_n,
        gate_std=max(0,pooled_g2/pooled_n-(pooled_g/pooled_n)**2)**.5,
        gate_quantiles_01_05_50_95_99_sampled=np.quantile(np.concatenate(samples),[.01,.05,.5,.95,.99]).tolist(),
        residual_rms=(pooled_r2/pooled_n)**.5,correction_rms=(pooled_c2/pooled_n)**.5,
        correction_rms_over_input_rms=(pooled_c2/max(pooled_i2,1e-30))**.5)
    report.update(status='COMPLETE',image_count=100,
        note='Real LR hashes verified; no full backbone inference or PSNR. Statistics exclude 1 LR pixel borders; 8bit-equivalent is pre-clamp correction scale, not image error. Ratios use input RGB, not main-branch output. Pooled statistics are spatial-site weighted; per-image summaries give each image equal weight. Intervention magnitudes do not prove PSNR benefit or independent retraining effects.')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    compact={k:v for k,v in report.items() if k!='images'}
    compact['largest_spatial_variation_images']=sorted(report['images'],key=lambda r:r['gate_spatial_std'],reverse=True)[:5]
    compact['largest_correction_images']=sorted(report['images'],key=lambda r:r['correction_rms'],reverse=True)[:5]
    (output/'summary.json').write_text(json.dumps(compact,indent=2))
    print('DIAGNOSTIC COMPLETE\n'+json.dumps(compact,indent=2),flush=True)


if __name__=='__main__':main()
