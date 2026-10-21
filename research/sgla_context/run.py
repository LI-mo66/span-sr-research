"""Use SG6's existing trainer/data/metric implementation unchanged."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parent
sys.path.insert(1,str(RESEARCH/'spatial_gate6'))
from entry import existing
sys.path.insert(0,str(HERE))
from sgla_model import build_model, validate_variant

def validate_config(cfg):
    validate_variant(cfg)
    if cfg.get('engineering_only'):
        if os.environ.get('SGLA_ENGINEERING')!='1':
            raise ValueError('Synthetic fixtures require SGLA_ENGINEERING=1')
        return
    reference=json.loads((RESEARCH/'spatial_gate6/configs/LG6-SiLU-SG6.json').read_text())
    # Only adapter metadata, version/name and paths differ from the fixed SG6.
    for key,value in reference.items():
        if key not in ('paths_file','experiment_version') and cfg.get(key)!=value:
            raise ValueError('SG6 protocol mismatch: '+key)
    if cfg.get('experiment_version')!='SG6-SGLA-QKDW-v1':
        raise ValueError('Unknown experiment version')

def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--config',required=True)
    args,_=p.parse_known_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf8'))
    validate_config(cfg)
    existing.validate_data(cfg)
    existing.fs_runtime.trainer.build_model=build_model
    original=argparse.ArgumentParser.add_argument
    def add(parser,*flags,**kwargs):
        if '--screen-updates' in flags:
            kwargs['choices']=(1000,5000,20000,50000)
        return original(parser,*flags,**kwargs)
    argparse.ArgumentParser.add_argument=add
    try: existing.fs_runtime.trainer.main()
    finally: argparse.ArgumentParser.add_argument=original

if __name__=='__main__': main()
