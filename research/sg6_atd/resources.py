"""Exploratory full-network FP32 timing; randomized weights, not trained PSNR."""
import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from run import atd_runtime as runtime
from model import build_model
from deploy import fused_copy
import torch


def compare_execution(models,x,report_path=None):
    """Separate routing/semantic changes from small FP32 output rounding."""
    report=dict(status='RUNNING',fp32_full_atol=1e-5,fp32_full_rtol=1e-5,
                fp32_adapter_atol=2e-6,fp32_adapter_rtol=1e-5,
                fp64_adapter_atol=1e-10,fp64_adapter_rtol=1e-10,variants={})
    def save():
        if report_path:
            Path(report_path).write_text(json.dumps(report,indent=2))
    def capture(model,value):
        features=[]; labels=[]; outputs=[]
        handles=[model.content_tab.register_forward_pre_hook(lambda m,i:features.append(i[0].clone())),
                 model.content_tab.tdca.register_forward_hook(lambda m,i,o:labels.append(o[1].clone())),
                 model.content_tab.register_forward_hook(lambda m,i,o:outputs.append(o.clone()))]
        try:
            prediction=model(value)
        finally:
            for handle in handles:handle.remove()
        return prediction,features[0],torch.cat(labels,1),outputs[0]
    with torch.inference_mode():
        reference,features,labels,adapter_output=capture(models['ATD_unchunked'],x)
        core64=copy.deepcopy(models['ATD_unchunked'].content_tab).double()
        features64=features.double()
        labels64=[]
        handle=core64.tdca.register_forward_hook(lambda m,i,o:labels64.append(o[1].clone()))
        try:
            reference64=core64(features64)
        finally:
            handle.remove()
        labels64=torch.cat(labels64,1)
        del core64
        for name,model in models.items():
            if name=='SG6':continue
            actual,current_features,current_labels,current_core=capture(model,x)
            row=dict(backbone_features_exact=torch.equal(features,current_features),
                     fp32_route_mismatches=int((labels!=current_labels).sum()),
                     fp32_full_max_abs=float((actual-reference).abs().max()),
                     fp32_full_relative_l2=float((actual-reference).norm()/reference.norm().clamp_min(1e-12)),
                     fp32_adapter_max_abs=float((current_core-adapter_output).abs().max()),
                     old_tolerance_mismatches=int(((actual-reference).abs()>(2e-6+1e-5*reference.abs())).sum()))
            # FP64 audit only the changed attention core; the backbone is already exact.
            core64=copy.deepcopy(model.content_tab).double()
            current_labels64=[]
            handle=core64.tdca.register_forward_hook(lambda m,i,o:current_labels64.append(o[1].clone()))
            try:
                actual64=core64(features64)
            finally:
                handle.remove()
            row.update(fp64_route_mismatches=int((labels64!=torch.cat(current_labels64,1)).sum()),
                       reference_fp32_vs_fp64_route_mismatches=int((labels!=labels64).sum()),
                       fp64_adapter_max_abs=float((actual64-reference64).abs().max()))
            report['variants'][name]=row
            save()
            try:
                if not row['backbone_features_exact'] or row['fp32_route_mismatches'] or row['fp64_route_mismatches']:
                    raise AssertionError('Backbone or category routing changed')
                torch.testing.assert_close(actual64,reference64,atol=1e-10,rtol=1e-10)
                torch.testing.assert_close(current_core,adapter_output,atol=2e-6,rtol=1e-5)
                # A full reconstruction adds floating point error after the core.
                # This tolerance is used only after exact routing and FP64 checks.
                torch.testing.assert_close(actual,reference,atol=1e-5,rtol=1e-5)
            except AssertionError as exc:
                report['status']='FAILED';report['failed_variant']=name;report['error']=str(exc);save()
                raise
            del core64,actual64,actual,current_features,current_labels,current_core
    report['status']='PASS'
    report['scope']='Random-weight execution equivalence; no claim of identical uint8 metrics or trained PSNR'
    save()
    return report


def benchmark(cfg,height,width,repeats=20,warmup=1.0,equivalence_output=None):
    arch,_=runtime.components()
    torch.manual_seed(cfg['seed'])
    baseline=fused_copy(build_model(arch,dict(cfg,atd_attention=False)).cuda(),arch.Conv3XC)
    torch.manual_seed(cfg['seed'])
    candidate=build_model(arch,cfg).cuda().eval()
    native_cfg=dict(cfg,atd_settings=dict(position='tail',query_chunk=10**9,group_chunk=10**9))
    unchunked=build_model(arch,native_cfg).cuda().eval()
    unchunked.load_state_dict(candidate.state_dict(),strict=True)
    candidate=fused_copy(candidate,arch.Conv3XC)
    unchunked=fused_copy(unchunked,arch.Conv3XC)
    models=dict(SG6=baseline,ATD_selected=candidate,ATD_unchunked=unchunked)
    x=torch.rand(1,3,height,width,device='cuda')
    values={k:[] for k in models}; peaks={k:0 for k in models}; reserved={k:0 for k in models}
    with torch.inference_mode():
        equivalence=compare_execution(models,x,equivalence_output)
        torch.cuda.empty_cache()  # Discard diagnostic FP64 cache before warmup/timing.
        start=time.monotonic()
        while time.monotonic()-start<warmup:
            for model in models.values():
                output=model(x); del output
            torch.cuda.synchronize()
        keys=list(models)
        for iteration in range(repeats):
            for offset in range(len(keys)):
                name=keys[(iteration+offset)%len(keys)]
                torch.cuda.synchronize()
                resident=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter(); output=models[name](x); torch.cuda.synchronize()
                values[name].append((time.perf_counter()-start)*1000)
                peaks[name]=max(peaks[name],torch.cuda.max_memory_allocated()-resident)
                reserved[name]=max(reserved[name],torch.cuda.max_memory_reserved())
                del output
    return dict(input=[1,3,height,width],device=torch.cuda.get_device_name(),torch=torch.__version__,
                repeats=repeats,warmup_seconds=warmup,precision='FP32',prefused=True,
                execution_equivalence=equivalence,
                results={name:dict(median_ms=statistics.median(values[name]),
                                  incremental_peak_allocated=peaks[name],
                                  process_peak_reserved=reserved[name]) for name in models},
                caveat='All three models co-resident. Reserved is process-wide; random-weight exploration, not trained deployment or full FLOPs.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(Path(__file__).parent/'configs/A1-sg6-atd-lr2e4.json'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--height',type=int,default=180)
    parser.add_argument('--width',type=int,default=320)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads(Path(args.config).read_text())
    result=benchmark(cfg,args.height,args.width,equivalence_output=str(Path(args.output).with_suffix('.equivalence.json')))
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
