import argparse,json,statistics,time,torch
from pathlib import Path
from runtime import components
from modules.candidates import build_model,PartialContext
from deploy import fused_copy

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--height',type=int,default=180);p.add_argument('--width',type=int,default=320);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());arch,_=components();torch.manual_seed(cfg['seed'])
    original=build_model(arch,cfg).cuda().eval();deployed=fused_copy(original,arch.Conv3XC)
    report=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,precision='FP32',lr_hw=[args.height,args.width],warmup=5,repeats=20,stored_parameters=sum(p.numel() for p in original.parameters()),trainable_parameters=sum(p.numel() for p in original.parameters() if p.requires_grad),deployed_parameters=sum(p.numel() for p in deployed.parameters()),checkpoint='random initialization: efficiency evidence only',timings={})
    x=torch.rand(1,3,args.height,args.width,device='cuda')
    with torch.inference_mode():
        for name,model in [('original',original),('prefused',deployed)]:
            for _ in range(5):model(x)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(20):
                torch.cuda.synchronize();start=time.perf_counter();model(x);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            report['timings'][name]=dict(median_ms=statistics.median(times),mean_ms=statistics.mean(times),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),resident_models='original and deployed both resident; same procedure for every variant')
        macs=[0]
        def hook(m,inputs,output):macs[0]+=output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        hooks=[m.register_forward_hook(hook) for m in deployed.modules() if isinstance(m,torch.nn.Conv2d)]
        deployed(x)
        for h in hooks:h.remove()
    dynamic_macs=sum(m.p*9*args.height*args.width for m in deployed.modules() if isinstance(m,PartialContext))
    report.update(conv_module_MACs=macs[0],functional_dynamic_conv_MACs=dynamic_macs,all_convolution_MACs=macs[0]+dynamic_macs,convolution_FLOPs_2_per_MAC=2*(macs[0]+dynamic_macs),cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,coverage='All convolution MACs including analytical functional dynamic convolution. Pooling, sigmoid, softmax, pointwise/layout costs excluded. Not full all-operator FLOPs.')
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
