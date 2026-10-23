"""Two synthetic effective-batch64 updates per arm; local cost/startup only."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from run import HERE, existing, validate_config
from pair_model import build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    arch,_=existing.fs_runtime.components()
    report=dict(scope='random images; two effective64 updates, micro16, HR256; no accuracy evidence',
                torch=torch.__version__,gpu=torch.cuda.get_device_name(),precision='FP32 TF32 off',arms={})
    for mode in ('off','independent','reuse'):
        cfg=json.loads((HERE/'configs'/('SG6-UCAN-'+mode+'.json')).read_text())
        validate_config(cfg)
        torch.manual_seed(10)
        model=build_model(arch,cfg).cuda().train()
        opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=cfg['lr'],
                             betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'])
        x=torch.rand(16,3,64,64,device='cuda')
        target=torch.rand(16,3,256,256,device='cuda')
        rows=[]
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            value=0.
            for _ in range(4):
                loss=torch.nn.functional.l1_loss(model(x),target)/4
                assert torch.isfinite(loss)
                loss.backward()
                value+=float(loss.detach())
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            opt.step()
            torch.cuda.synchronize()
            rows.append(dict(step=step+1,synthetic_loss=value,seconds=time.perf_counter()-start,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        report['arms'][mode]=rows
        print(mode,rows,flush=True)
        del model,opt,x,target,loss
        gc.collect();torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
