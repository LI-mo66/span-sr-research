"""Complete deployed models, synchronized FP32 timings; no real-data metrics."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from run import existing
from sgla_model import build_model

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    torch.set_num_threads(2); torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    arch,_=existing.fs_runtime.components()
    from deploy import fused_copy
    here=Path(__file__).resolve().parent
    models={}
    for mode in ('off','original','context','local'):
        cfg=json.loads((here/'configs'/('SG6-'+mode+'.json')).read_text())
        torch.manual_seed(10); models[mode]=fused_copy(build_model(arch,cfg),arch.Conv3XC).cuda().eval()
    report=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),precision='FP32 TF32 off',batch=1,scope='random tensors; no PSNR; MACs omit norm/activation/rope/softmax/elementwise',sizes={})
    for h,w in ((64,64),(180,320)):
        x=torch.rand(1,3,h,w,device='cuda'); rows={}
        with torch.inference_mode():
            for m in models.values():
                for _ in range(20): m(x)
            torch.cuda.synchronize()
            for mode,m in models.items():
                torch.cuda.reset_peak_memory_stats(); start=torch.cuda.memory_allocated(); m(x); torch.cuda.synchronize()
                row=dict(parameters=sum(v.numel() for v in m.parameters()),peak_allocated_increment=torch.cuda.max_memory_allocated()-start,peak_reserved=torch.cuda.max_memory_reserved(),round_median_ms=[])
                total=[0]
                def count(module,args,y): total[0]+=y.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
                handles=[v.register_forward_hook(count) for v in m.modules() if isinstance(v,torch.nn.Conv2d)]
                m(x)
                for handle in handles: handle.remove()
                row['conv_MACs']=total[0]
                row['attention_matmul_MACs']=h*w*(2*52*52+2*52) if mode in ('original','context') else 0
                rows[mode]=row
            for cycle in range(3):
                keys=list(models); keys=keys[cycle:]+keys[:cycle]
                for mode in keys:
                    times=[]
                    for _ in range(20):
                        e,f=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        e.record(); models[mode](x); f.record(); f.synchronize(); times.append(e.elapsed_time(f))
                    rows[mode]['round_median_ms'].append(statistics.median(times))
            for row in rows.values(): row['median_ms']=statistics.median(row['round_median_ms'])
        report['sizes'][f'{h}x{w}']=rows
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
