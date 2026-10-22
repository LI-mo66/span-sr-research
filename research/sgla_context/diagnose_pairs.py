"""Read-only CPU paired error diagnosis; new output directory, no training.

Uses fixed uniformly spaced DIV2K-valid images, all four validation-selected
weights, original quantized Y/crop4 metrics. No checkpoint or source rewrite.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
from datetime import datetime, timezone
import cv2
import numpy as np
import torch
from run import existing
from sgla_model import build_model

ROOT=Path(__file__).resolve().parents[2]
MODES=('SG6','context','original','local')
REFERENCE_SHA='9c16f95c1c265e628914160f7929ef7b7f0b07b7c6a8af81b6277f6bc252c86b'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''): h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(path.read_text(encoding='utf8'))
def write(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def selected(folder):
    rows=[json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines() if line.strip()]
    rows=sorted(rows,key=lambda r:r['update'])
    if [r['update'] for r in rows]!=list(range(1000,50001,1000)):
        raise ValueError('Complete unambiguous 50K history required: '+str(folder))
    return max(rows,key=lambda r:r['metrics']['DIV2K_valid']['psnr'])

def reference_run(archive):
    matches=[]
    for p in (archive/'runs').glob('*/manifest.json'):
        if read(p).get('config',{}).get('experiment_version')=='LG6-SiLU-GB-SpatialGate6-v1': matches.append(p.parent)
    if len(matches)!=1: raise ValueError('Exactly one SG6 reference required')
    return matches[0]

def locate_runs(records,reference,group):
    found={'SG6':(reference_run(reference),reference)}
    for mode in MODES[1:]:
        archives=sorted(records.glob(f'{group}-SGLA-{mode}-s10-*'))
        matches=[]
        for archive in archives:
            if not (archive/'BACKUP_COMPLETE').is_file(): continue
            metadata=read(archive/'backup-manifest.json')
            if metadata.get('training_exit_code')!=0 or metadata.get('comparison_exit_code')!=0: continue
            run=archive/'runs'/f'{group}-SGLA-{mode}-s10'
            if (run/'manifest.json').is_file(): matches.append((run,archive))
        if len(matches)!=1:
            raise ValueError(f'{mode}: expected one successful persistent archive, found {len(matches)}')
        found[mode]=matches[0]
    return found

def region_masks(gt):
    gx=cv2.Sobel(gt.astype(np.float32),cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(gt.astype(np.float32),cv2.CV_32F,0,1,ksize=3)
    magnitude=cv2.magnitude(gx,gy)
    q1,q2=np.quantile(magnitude,[1/3,2/3])
    return dict(all=np.ones(gt.shape,bool),low_gradient=magnitude<=q1,
                medium_gradient=(magnitude>q1)&(magnitude<=q2),high_gradient=magnitude>q2)

def compare_errors(base,candidate,masks):
    result={}
    for name,mask in masks.items():
        count=int(mask.sum())
        a,b=base[mask],candidate[mask]
        if count==0:
            result[name]=dict(pixels=0); continue
        gain=float(np.maximum(a-b,0).sum()); harm=float(np.maximum(b-a,0).sum())
        result[name]=dict(pixels=count,SG6_sse=float(a.sum()),candidate_sse=float(b.sum()),
                          delta_mse=float((b-a).mean()),gain_sse=gain,harm_sse=harm,
                          improved_fraction=float((b<a).mean()),worsened_fraction=float((b>a).mean()),
                          unchanged_fraction=float((a==b).mean()))
    return result

def save_image(path,image):
    if not cv2.imwrite(str(path),image): raise IOError('Image write failed: '+str(path))

def patch_panels(folder,hr,base,candidate,delta,mode,border):
    h,w=delta.shape; size=min(64,h,w)
    if size<1: return []
    cells=[]
    # Nonoverlapping fixed raster grid; partial edge cells excluded.
    for y in range(0,h-size+1,size):
        for x in range(0,w-size+1,size): cells.append((float(delta[y:y+size,x:x+size].mean()),y,x))
    examples=[]
    for label,cell in [('largest_harm',max(cells)),('largest_gain',min(cells))]:
        value,y,x=cell; yy,xx=y+border,x+border
        crops=[v[yy:yy+size,xx:xx+size] for v in (hr,base,candidate)]
        panel=np.concatenate(crops,axis=1)
        panel=cv2.resize(panel,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
        canvas=np.full((panel.shape[0]+28,panel.shape[1],3),255,np.uint8)
        canvas[28:]=panel
        cv2.putText(canvas,'HR | SG6 | '+mode,(4,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        save_image(folder/f'{mode}-{label}.png',canvas)
        examples.append(dict(label=label,HR_y=yy,HR_x=xx,size=size,delta_mse=value,
                             scope='posthoc within-image illustration, not independent evidence'))
    return examples

def heat(delta,limit):
    # BGR: red=more error, blue=less error; identical scale across candidates.
    z=np.clip(delta/max(limit,1e-12),-1,1)
    out=np.full((*z.shape,3),255,dtype=np.float32)
    out[...,0]-=255*np.maximum(z,0); out[...,1]-=255*np.abs(z); out[...,2]-=255*np.maximum(-z,0)
    return out.astype(np.uint8)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--records',type=Path,default=Path('/home/featurize/work/span-x4-research/records'))
    p.add_argument('--reference',type=Path,default=Path('/home/featurize/work/span-x4-research/records/FEAT-LG6-SG6-50K-01-1792264356078917269'))
    p.add_argument('--group',default='FEAT-SG6-SGLA-3ARM-50K-02')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not args.group or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.group): raise ValueError('Invalid group')
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',scope='fixed12 validation images; exploratory paired errors, no causal inference',
                created_utc=datetime.now(timezone.utc).isoformat(),torch=torch.__version__,device='CPU',
                metric='original clamp/rint uint8, BGR-to-Y, crop4, whole image, no ensemble',
                numerical_note='CPU FP32 fused inference; compare against archived GPU per-image scores; warnings are not silently waived',
                provenance={},images=[],warnings=[])
    report_path=args.output/'report.json'
    write(report_path,report)
    try:
        runs=locate_runs(args.records,args.reference,args.group)
        arch,metrics=existing.fs_runtime.components()
        from deploy import fused_copy
        from basicsr.metrics.metric_util import to_y_channel
        models={}; archived={}; manifests={}; weights={}
        for mode,(folder,archive) in runs.items():
            manifest=read(folder/'manifest.json'); manifests[mode]=manifest
            protected=['basicsr','research/spatial_gate6/sg_model.py','research/gated_bypass/gb_model.py',
                       'research/linear_gate/lg_model.py','research/deploy.py','research/runtime.py']
            if mode!='SG6': protected+=['research/sgla_context/attention.py','research/sgla_context/sgla_model.py']
            changed=subprocess.check_output(['git','diff','--name-only',manifest['commit'],'HEAD','--',*protected],cwd=ROOT,text=True).strip()
            if changed: raise ValueError('Model source differs from checkpoint provenance: '+changed)
            cfg=manifest['config']; row=selected(folder)
            if cfg['crop_border']!=4 or cfg['test_y_channel'] is not True or cfg['self_ensemble']: raise ValueError('Unexpected metric protocol')
            actual=cfg.get('sgla',{}).get('mode','off')
            if actual!=('off' if mode=='SG6' else mode): raise ValueError('Model label mismatch')
            filename=f'epoch{row["epoch"]:05d}_u{row["update"]}.pt'; weight=folder/filename
            weight_hash=sha(weight)
            metadata=read(archive/'backup-manifest.json')
            key=str(weight.relative_to(archive)).replace('\\','/')
            recorded=metadata.get('files',{}).get(key)
            if mode=='SG6':
                if weight_hash!=REFERENCE_SHA: raise ValueError('SG6 weight hash differs')
            elif recorded!=weight_hash:
                raise ValueError('Archived candidate weight hash absent/mismatched: '+mode)
            model=build_model(arch,cfg).cpu().eval()
            state=torch.load(weight,map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True)
            if state['update']!=row['update']: raise ValueError('Checkpoint update mismatch')
            del state
            # No GPU access, optimizer or weight mutation on disk.
            models[mode]=fused_copy(model,arch.Conv3XC).cpu().eval(); del model
            assert all(v.device.type=='cpu' for v in models[mode].parameters())
            scorefile=folder/f'DIV2K_valid_epoch{row["epoch"]:05d}_u{row["update"]}.json'
            archived[mode]={r['image']:r for r in read(scorefile)}
            weights[mode]=(weight,weight_hash)
            report['provenance'][mode]=dict(run=str(folder),checkpoint=filename,sha256=weight_hash,
                                            source_commit=manifest.get('commit'),selected_update=row['update'])
        data=manifests['SG6']['data']['DIV2K_valid']
        if len(data)!=100: raise ValueError('Expected DIV2K_valid100')
        indices=np.linspace(0,99,12,dtype=int).tolist()
        report['fixed_indices']=indices; report['fixed_images']=[Path(data[i]['hr']).name for i in indices]
        for mode in MODES[1:]:
            other=manifests[mode]['data']['DIV2K_valid']
            if len(other)!=100 or any((a['hr_sha256'],a['lr_sha256'])!=(b['hr_sha256'],b['lr_sha256']) for a,b in zip(data,other)):
                raise ValueError('Ordered validation data mismatch: '+mode)
        for number,index in enumerate(indices,1):
            item=data[index]; hp,lp=Path(item['hr']),Path(item['lr']); name=hp.name
            print(f'[{number}/12] {name} CPU whole-image inference',flush=True)
            for path,key in [(hp,'hr_sha256'),(lp,'lr_sha256')]:
                if sha(path)!=item[key]: raise ValueError('Image hash differs: '+str(path))
            hr=cv2.imread(str(hp)); lr=cv2.imread(str(lp))
            if hr is None or lr is None: raise ValueError('Image decode failed')
            h,w=lr.shape[0]*4,lr.shape[1]*4
            if not (0<=hr.shape[0]-h<4 and 0<=hr.shape[1]-w<4): raise ValueError('Unaligned x4')
            hr=hr[:h,:w]; x=torch.from_numpy(np.ascontiguousarray(lr[:,:,::-1].transpose(2,0,1))).float()[None]/255
            gt=to_y_channel(hr[4:-4,4:-4])[...,0].astype(np.float64)
            masks=region_masks(gt); outputs={}; errors={}; scores={}
            folder=args.output/hp.stem; folder.mkdir()
            for mode in MODES:
                start=time.perf_counter()
                with torch.inference_mode(): prediction=models[mode](x)
                if not torch.isfinite(prediction).all(): raise FloatingPointError('Nonfinite prediction')
                sr=np.rint(prediction[0].clamp(0,1).numpy().transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8)
                outputs[mode]=sr
                error=to_y_channel(sr[4:-4,4:-4])[...,0].astype(np.float64)-gt
                errors[mode]=error**2
                psnr=metrics.calculate_psnr(sr,hr,crop_border=4,test_y_channel=True)
                old=archived[mode][name]['psnr']; deviation=float(psnr-old)
                if abs(deviation)>2e-4: report['warnings'].append(dict(image=name,mode=mode,CPU_minus_archive_psnr=deviation,threshold=2e-4))
                scores[mode]=dict(CPU_psnr=float(psnr),archive_psnr=old,CPU_minus_archive_psnr=deviation,
                                  CPU_y_bias=float(error.mean()),seconds=time.perf_counter()-start)
                del prediction
            del x
            differences={m:errors[m]-errors['SG6'] for m in MODES[1:]}
            sample=np.concatenate([v.ravel()[::16] for v in differences.values()])
            limit=float(np.quantile(np.abs(sample),.99))
            image=dict(image=name,HR_shape=list(hr.shape),scores=scores,heat_scale_Y_squared=limit,candidates={})
            for mode in MODES[1:]:
                image['candidates'][mode]=dict(regions=compare_errors(errors['SG6'],errors[mode],masks),
                    examples=patch_panels(folder,hr,outputs['SG6'],outputs[mode],differences[mode],mode,4))
                save_image(folder/f'{mode}-delta-error-red-harm-blue-gain.png',heat(differences[mode],limit))
            report['images'].append(image); write(report_path,report)
        summary={}
        for mode in MODES[1:]:
            summary[mode]={}
            for region in ('all','low_gradient','medium_gradient','high_gradient'):
                rows=[v['candidates'][mode]['regions'][region] for v in report['images']]
                counts=sum(v['pixels'] for v in rows); valid=[v for v in rows if v['pixels']]
                summary[mode][region]=dict(pixels=counts,pooled_delta_mse=sum(v['candidate_sse']-v['SG6_sse'] for v in valid)/counts if counts else None,
                    total_gain_sse=sum(v['gain_sse'] for v in valid),total_harm_sse=sum(v['harm_sse'] for v in valid),
                    image_mean_delta_mse=float(np.mean([v['delta_mse'] for v in valid])) if valid else None)
        for weight,value in weights.values():
            if sha(weight)!=value: raise ValueError('Checkpoint changed during diagnosis')
        report.update(status='COMPLETE',summary=summary,checkpoint_post_hashes_verified=True,
                      notes=['12 fixed images are diagnostic, not the complete100-image benchmark.',
                             'Gradient regions describe HR edges, not semantic texture classes or CAMD priors.',
                             'Patch extrema are posthoc illustrations and must not select architectures.',
                             'GPU archive metrics remain the formal experiment results; CPU numerical warnings limit map interpretation.'])
        write(report_path,report)
        print(json.dumps(dict(status=report['status'],summary=summary,numerical_warnings=report['warnings']),indent=2),flush=True)
        print('REPORT:',report_path,flush=True)
    except Exception as e:
        report.update(status='FAILED',error=repr(e)); write(report_path,report); raise

if __name__=='__main__': main()
