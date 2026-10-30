"""Fresh F0 long run; immutable training protocol, separate execution stage cap."""
import importlib.util
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research/fusion_wu'))
import fusion
sys.path.insert(0,str(HERE))

def main():
    import argparse
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--config',required=True);p.add_argument('--id',required=True)
    p.add_argument('--until-update',type=int,required=True)
    a,_=p.parse_known_args()
    cfg=json.loads(Path(a.config).read_text())
    fusion.validate(cfg)
    if cfg.get('engineering_only'):
        if os.environ.get('F0_LONG_ENGINEERING')!='1':raise ValueError('Engineering flag missing')
    else:
        frozen=json.loads((ROOT/'logs/f0-long/frozen/config.json').read_text())
        if cfg!=frozen:raise ValueError('Configuration differs from immutable frozen protocol')
        if a.id!='FEAT-SG6-F0-X8-S10-1M-01':raise ValueError('Wrong formal run ID')
        if a.until_update not in (100000,250000,500000,750000,1000000):raise ValueError('Unregistered stage')
        if cfg['total_updates']!=1000000 or cfg['record_updates']!=1000 or cfg['self_ensemble'] is not True:
            raise ValueError('Long protocol mismatch')
    import torch
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    spec=importlib.util.spec_from_file_location('f0_long_trainer',HERE/'trainer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main()

if __name__=='__main__':main()
