"""Disposable real-data profile: data/transfer/compute/Adam/evaluation split, no weights saved."""
import argparse,gc,hashlib,json,os,random,statistics,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import cv2,numpy as np,torch
from bootstrap import components,trainer,ROOT
from image_io import enable_cache
from sg_model import build_model as sg6
from add_model import build_model
from frozen import load_frozen_reference

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',type=Path,required=True);a=p.parse_args()
    frozen=a.group/'frozen';audit=json.loads((frozen/'reuse-audit.json').read_text());cfg=json.loads((frozen/'ADD31.json').read_text())
    paths=json.loads(Path(cfg['paths_file']).read_text());specs=dict(train=paths['train'],**paths['validation'])
    pairs={k:trainer.pairs(v) for k,v in specs.items()}
    manifests={k:[dict(hr=str(h),lr=str(l),hr_sha256=trainer.digest(h),lr_sha256=trainer.digest(l)) for h,l in v] for k,v in pairs.items()}
    enable_cache(manifests,ROOT/'datasets/image_cache')
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    arch,metrics=components()
    report=dict(status='RUNNING',synthetic_only=False,disposable=True,results={},shape=dict(micro=16,accumulation=4,lr_patch=64),
        environment=dict(torch=str(torch.__version__),device=torch.cuda.get_device_name(),torch_threads=torch.get_num_threads(),cv2_threads=cv2.getNumThreads(),
            cpu_affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            load=os.getloadavg() if hasattr(os,'getloadavg') else None,OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS')))
    reference_state=torch.load(audit['reference_weight'],map_location='cpu',weights_only=False)['model']
    for label,factory in [('SG6',sg6),('ADD31',build_model)]:
        torch.manual_seed(10);model=factory(arch,cfg).cuda().train()
        opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=2e-4,betas=(.9,.999),eps=1e-8)
        rng=random.Random(10);samples=[]
        for step in range(5):
            if step==2:torch.cuda.reset_peak_memory_stats()
            row=dict(data=0.,transfer=0.,forward_backward=0.,optimizer=0.);start=time.perf_counter();opt.zero_grad(set_to_none=True)
            for _ in range(4):
                t=time.perf_counter();batch=[trainer.sample(rng.choice(pairs['train']),256,rng=rng) for _ in range(16)]
                x=torch.stack([b[0] for b in batch]);target=torch.stack([b[1] for b in batch]);row['data']+=time.perf_counter()-t
                t=time.perf_counter();x=x.cuda();target=target.cuda();torch.cuda.synchronize();row['transfer']+=time.perf_counter()-t
                t=time.perf_counter();loss=(model(x)-target).abs().mean()/4
                if not torch.isfinite(loss):raise FloatingPointError('Profile loss nonfinite')
                loss.backward();torch.cuda.synchronize();row['forward_backward']+=time.perf_counter()-t
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise FloatingPointError('Profile gradient missing/nonfinite')
            t=time.perf_counter();opt.step();torch.cuda.synchronize();row['optimizer']=time.perf_counter()-t
            row['total']=time.perf_counter()-start
            if step>=2:samples.append(row)
        memory=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        del opt,x,target,batch,loss;gc.collect();torch.cuda.empty_cache()
        # Separate frozen evaluation diagnostic; these weights NEVER enter the formal trainer.
        if label=='SG6':model.load_state_dict(reference_state,strict=True)
        else:
            report['strict_frozen_coverage']=load_frozen_reference(model,reference_state)
        model.eval();evaluation=[];data=pairs['DIV2K_valid'];indices=[0,len(data)//3,2*len(data)//3,len(data)-1]
        with torch.inference_mode():
            for index in indices:
                hp,lp=data[index];t=time.perf_counter();hr,lr=trainer.aligned(hp,lp);read_seconds=time.perf_counter()-t
                t=time.perf_counter();input_tensor=trainer.tensor(lr).unsqueeze(0).cuda();torch.cuda.synchronize();transfer=time.perf_counter()-t
                t=time.perf_counter();prediction=model(input_tensor)[0];torch.cuda.synchronize();forward=time.perf_counter()-t
                if not torch.isfinite(prediction).all():raise FloatingPointError('Profile eval nonfinite')
                t=time.perf_counter();sr=prediction.clamp(0,1).cpu().numpy().transpose(1,2,0)[:,:,::-1];sr=np.rint(sr*255).astype(np.uint8);conversion=time.perf_counter()-t
                t=time.perf_counter();kw=dict(crop_border=4,test_y_channel=True);metrics.calculate_psnr(sr,hr,**kw);metrics.calculate_ssim(sr,hr,**kw);metric_seconds=time.perf_counter()-t
                evaluation.append(dict(image=hp.name,lr_shape=list(lr.shape),read=read_seconds,transfer=transfer,forward=forward,conversion=conversion,metrics=metric_seconds,
                    output_sha256=hashlib.sha256(sr.tobytes()).hexdigest()))
        report['results'][label]=dict(update_seconds_median=statistics.median(r['total'] for r in samples),train_samples=samples,
            phase_medians={k:statistics.median(r[k] for r in samples) for k in samples[0]},train_peak_bytes=memory,frozen_evaluation=evaluation)
        del model,prediction,input_tensor
        gc.collect();torch.cuda.empty_cache()
        (a.group/'profile.json').write_text(json.dumps(report,indent=2))
    report['update_ratio']=report['results']['ADD31']['update_seconds_median']/report['results']['SG6']['update_seconds_median']
    compute=lambda label:statistics.median(r['forward_backward']+r['optimizer'] for r in report['results'][label]['train_samples'])
    report['compute_ratio']=compute('ADD31')/compute('SG6')
    report['gate_ratio']=max(report['update_ratio'],report['compute_ratio'])
    report['frozen_eval_outputs_identical']=[r['output_sha256'] for r in report['results']['SG6']['frozen_evaluation']]==[r['output_sha256'] for r in report['results']['ADD31']['frozen_evaluation']]
    if not report['frozen_eval_outputs_identical']:raise ValueError('Zero-output additive frozen mapping differs from original SG6')
    report['status']='PASS';report['limitations']='Five disposable real-data updates per model, two warmup and three measured. Four frozen eval images; no candidate PSNR claim or guarantee for 50K walltime. Phase synchronization is diagnostic only. Sequential data-cache/order effects may affect total time; compute and total ratios are both reported and the larger is the cost gate.'
    (a.group/'profile.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
