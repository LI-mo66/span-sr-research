"""Same-process complete-path fused FP32 latency/memory; component timings separate."""
import argparse,gc,json,statistics,time
from pathlib import Path
import torch
from run import HERE,existing
from candidate import build_model,build_sg6

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    arch,_=existing.fs_runtime.components()
    from deploy import fused_copy
    cfg=json.loads((HERE/'config.json').read_text());report={'precision':'FP32 TF32 off','device':a.device,
        'hardware':torch.cuda.get_device_name() if a.device=='cuda' else 'CPU','warmup':10,'repeats':20,'sizes':{},
        'scope':'random weights; full fused path; component timings synchronize separately and do not sum to full latency'}
    def sync():
        if a.device=='cuda':torch.cuda.synchronize()
    from reference_candidate import build_model as build_combo
    from reference_pair_model import build_model as build_pair
    combo_cfg=json.loads((HERE/'reference_config.json').read_text())
    pair_cfg=json.loads((HERE/'reference_pair_config.json').read_text())
    factories=(('SG6',build_sg6,cfg),('P24',build_pair,pair_cfg),
               ('P24_LKSA5',build_combo,combo_cfg),('candidate',build_model,cfg))
    report['reference_scope']='same-process random-weight original P24 and prior combo factories; not accuracy or trained-weight latency evidence'
    for h,w in ((64,64),(180,320)):
        rows={}
        for label,factory,model_cfg in factories:
            torch.manual_seed(10);model=fused_copy(factory(arch,model_cfg),arch.Conv3XC).to(a.device).eval()
            x=torch.rand(1,3,h,w,device=a.device)
            with torch.inference_mode():
                for _ in range(10):model(x)
                sync()
                row={'deployed_parameters':sum(p.numel() for p in model.parameters())}
                if a.device=='cuda':
                    torch.cuda.reset_peak_memory_stats();allocated=torch.cuda.memory_allocated();model(x);sync()
                    row.update(peak_allocated_increment_bytes=torch.cuda.max_memory_allocated()-allocated,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                times=[]
                for _ in range(20):
                    sync();start=time.perf_counter();model(x);sync();times.append(1000*(time.perf_counter()-start))
                row.update(median_ms=statistics.median(times),raw_ms=times)
                if label=='candidate':
                    captured={};handles=[]
                    def capture(name):
                        def hook(module,args):captured[name]=args
                        return hook
                    for name,module in model.named_children():
                        if name.startswith(('context_','lksa_','plain_')):handles.append(module.register_forward_pre_hook(capture(name)))
                    model(x)
                    for handle in handles:handle.remove()
                    components={}
                    for name,feature in captured.items():
                        module=getattr(model,name);sample=[]
                        for _ in range(10):module(*feature)
                        for _ in range(20):
                            sync();start=time.perf_counter();module(*feature);sync();sample.append(1000*(time.perf_counter()-start))
                        components[name]={'median_ms':statistics.median(sample),'raw_ms':sample}
                    row['isolated_adapter_timings']=components
            if label=='candidate':del captured,handles,feature,module
            rows[label]=row;print(h,w,label,row['deployed_parameters'],row['median_ms'],flush=True)
            del model,x;gc.collect()
            if a.device=='cuda':torch.cuda.empty_cache()
        report['sizes'][f'{h}x{w}']=rows
    if a.device=='cuda':
        # Check the actual microbatch/patch memory before any real50K job.
        # Synthetic random data; this model is discarded immediately afterward.
        torch.backends.cudnn.allow_tf32=True
        train=build_model(arch,cfg).to(a.device).train()
        opt=torch.optim.Adam((p for p in train.parameters() if p.requires_grad),lr=cfg['lr'])
        lr=cfg['hr_patch']//4
        image=torch.rand(cfg['micro_batch'],3,lr,lr,device=a.device)
        target=torch.rand(cfg['micro_batch'],3,cfg['hr_patch'],cfg['hr_patch'],device=a.device)
        sync();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for _ in range(2):
            opt.zero_grad(set_to_none=True);loss=(train(image)-target).abs().mean()
            if not torch.isfinite(loss):raise ValueError('Synthetic production-shape loss is not finite')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in train.parameters()):raise ValueError('Nonfinite production-shape gradient')
            opt.step()
        sync()
        report['synthetic_training_memory']={'micro_batch':cfg['micro_batch'],'lr_hw':[lr,lr],
            'effective_batch':cfg['effective_batch'],'micro_steps':2,'optimizer':'Adam',
            'precision':'FP32 cudnnTF32=True matmulTF32=False',
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'elapsed_seconds':time.perf_counter()-start,'scope':'synthetic memory/startup check, not real training or effective-batch performance'}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
