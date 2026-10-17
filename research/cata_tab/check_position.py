"""Matched random-start signal path audit, not a PSNR ranking of positions."""
import importlib.util
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cata_runtime',HERE/'runtime.py')
runtime=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
import torch
from model import build_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--height',type=int,default=64)
    parser.add_argument('--width',type=int,default=64)
    args=parser.parse_args()
    h,w=args.height,args.width
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    arch,_=runtime.components()
    cfg=json.loads((HERE/'configs/C1-mid-lr2e4.json').read_text())
    report=dict(scope='paired random-start signal transmission only; no PSNR or learned-use claim',lr_hw=[h,w],results=[])
    original=torch.nn.functional.scaled_dot_product_attention
    for seed in (10,20,30):
        torch.manual_seed(seed+1)
        image=torch.rand(2,3,h,w,device='cuda')
        target=torch.rand(2,3,h*4,w*4,device='cuda')
        initial=None
        for position in ('mid','tail'):
            option=dict(cfg,seed=seed,cata_settings=dict(position=position,chunk_groups=128))
            torch.manual_seed(seed)
            model=build_model(arch,option).cuda()
            if initial is None:
                initial={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:
                assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
            features=[]
            handle=model.content_tab.register_forward_pre_hook(
                lambda _,inputs:features.append(inputs[0].detach().clone()))
            model.train()
            loss=torch.nn.functional.l1_loss(model(image),target)
            loss.backward()
            handle.remove()
            gradients={name:float(p.grad.norm()) for name,p in model.content_tab.named_parameters()}
            model.eval()
            with torch.no_grad():
                full=model(image)
                _,stats=model.content_tab.forward_with_diagnostics(features[0])
            changes={}
            for disabled in ('group','center'):
                calls=[0]
                def bypass(*args,**kwargs):
                    output=original(*args,**kwargs)
                    branch='group' if calls[0]%2==0 else 'center'
                    calls[0]+=1
                    return torch.zeros_like(output) if branch==disabled else output
                with torch.no_grad(),patch('torch.nn.functional.scaled_dot_product_attention',bypass):
                    changes[disabled]=float((model(image)-full).norm()/full.norm().clamp_min(1e-12))
            report['results'].append(dict(seed=seed,position=position,
                output_change_ratio=changes,parameter_gradient_norms=gradients,tab=stats))
            del model
    (HERE/'position-audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
