"""Frozen shared trainer entry; only the model factory and 50K stop option differ."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from bootstrap import ROOT,trainer,components
from strip_model import build_model,validate_variant

def validate_config(cfg):
    validate_variant(cfg)
    fixed=json.loads((ROOT/'research/spatial_gate6/configs/LG6-SiLU-SG6.json').read_text())
    if set(cfg)-set(fixed)-{'partial_strip','engineering_only'}:
        raise ValueError('Unknown configuration fields or mixed candidate')
    exceptions={'experiment_version','paths_file'}
    if cfg.get('engineering_only'):
        exceptions|={'total_updates','record_updates','screen_epochs','micro_batch','effective_batch','hr_patch','image_backend'}
    for key,value in fixed.items():
        if key not in exceptions and cfg.get(key)!=value:
            raise ValueError('SG6 protocol differs: '+key)
    if cfg.get('engineering_only') and os.environ.get('SG6_STRIP_ENGINEERING')!='1':
        raise ValueError('Synthetic fixture requires SG6_STRIP_ENGINEERING=1')

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',required=True)
    parser.add_argument('--id',required=True)
    parser.add_argument('--resume',action='store_true')
    args,_=parser.parse_known_args()
    cfg=json.loads(Path(args.config).read_text())
    validate_config(cfg)
    import torch
    from monitor import Monitor
    # Preserve the archived SG6 production precision policy explicitly.
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    start=0
    run=ROOT/'runs'/args.id
    if args.resume and (run/'latest.pt').exists():
        state=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        start=state['update'];del state
        path=run/'feature_updates.jsonl'
        if path.exists():
            rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['update']<=start))
    monitor=Monitor(run/'feature_updates.jsonl',start)
    def factory(arch,cfg):
        model=build_model(arch,cfg);monitor.bind(model);return model
    trainer.build_model=factory
    adam=torch.optim.Adam
    torch.optim.Adam=monitor.optimizer(adam)
    original=argparse.ArgumentParser.add_argument
    def add_argument(self,*flags,**kwargs):
        if '--screen-updates' in flags and kwargs.get('choices'):
            kwargs['choices']=tuple(sorted(set(kwargs['choices'])|{50000}))
        return original(self,*flags,**kwargs)
    argparse.ArgumentParser.add_argument=add_argument
    try:trainer.main()
    finally:
        argparse.ArgumentParser.add_argument=original
        torch.optim.Adam=adam

if __name__=='__main__':main()
