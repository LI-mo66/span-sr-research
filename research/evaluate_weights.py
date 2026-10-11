"""Evaluate released standard SPAN x4 weights; no training or self-ensemble."""
import argparse
import json
import platform
import subprocess
import sys
import time
import cv2
import numpy as np
import torch
from runtime import ROOT, components
from train import aligned, digest, pairs, tensor

class Tee:
    def __init__(self, stream, file): self.stream,self.file=stream,file
    def write(self,text): self.stream.write(text);self.file.write(text);self.file.flush()
    def flush(self): self.stream.flush();self.file.flush()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',default='weights/official/spanx4_ch48.pth')
    p.add_argument('--key',default='params',choices=['params','params_ema'])
    p.add_argument('--paths',default='research/configs/paths.local.json')
    p.add_argument('--id',required=True)
    p.add_argument('--save-images',action='store_true')
    args=p.parse_args()
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.id): raise ValueError('Unsafe ID')
    out=ROOT/'results'/args.id
    out.mkdir(parents=True,exist_ok=False)
    log=open(out/'console.log','w',encoding='utf8')
    sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    paths=json.loads((ROOT/args.paths).read_text(encoding='utf-8-sig'))
    arch,metrics=components()
    model=arch.SPAN(3,3,feature_channels=48,upscale=4,img_range=255.).cuda().eval()
    weight=ROOT/args.weights
    state=torch.load(weight,map_location='cpu',weights_only=True)
    print(model.load_state_dict(state[args.key],strict=True),flush=True)
    names=('Set5','Set14','B100','Urban100','Manga109')
    dataset={name:pairs(paths['validation'][name]) for name in names}
    protocol=dict(id=args.id,checkpoint=str(weight),sha256=digest(weight),param_key=args.key,upstream_commit='c77a5917759f09e66fbc7124220c5afc5ee221e5',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),model='SPAN x4 ch48 unmodified upstream',input='RGB NCHW float32 [0,1]',output='raw forward then clamp [0,1] and round uint8',crop_border=4,test_y_channel=True,metrics='pinned upstream BasicSR PSNR/SSIM',self_ensemble=False,inference='whole image; no chop; batch1; original eval including per-forward fusion',torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),device=torch.cuda.get_device_name(),data={name:[dict(hr=str(h),lr=str(l),hr_sha256=digest(h),lr_sha256=digest(l)) for h,l in values] for name,values in dataset.items()})
    (out/'manifest.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    summary={};start=time.perf_counter()
    with torch.inference_mode():
        for name,data in dataset.items():
            rows=[]
            for i,(hp,lp) in enumerate(data):
                hr,lr=aligned(hp,lp)
                output=model(tensor(lr).unsqueeze(0).cuda())[0]
                sr=np.rint(output.clamp(0,1).cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8)
                if sr.shape!=hr.shape: raise ValueError(f'Shape mismatch: {hp}')
                kw=dict(crop_border=4,test_y_channel=True)
                rows.append(dict(image=hp.name,psnr=metrics.calculate_psnr(sr,hr,**kw),ssim=metrics.calculate_ssim(sr,hr,**kw),raw_min=float(output.min()),raw_max=float(output.max())))
                if args.save_images or i==0:
                    dest=out/'images'/name;dest.mkdir(parents=True,exist_ok=True)
                    cv2.imencode('.png',sr)[1].tofile(dest/hp.name)
                if (i+1)%25==0: print(f'{name}: {i+1}/{len(data)}',flush=True)
            (out/f'{name}_per_image.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
            summary[name]=dict(count=len(rows),psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])))
            print(name,summary[name],flush=True)
    summary['elapsed_seconds']=time.perf_counter()-start
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('Results:',out,flush=True)

if __name__=='__main__':main()
