"""Unchanged trainer with the explicitly frozen fusion model factory."""
import argparse
import json
import os
import sys
from pathlib import Path
import torch
from fusion import ROOT, build_model, trainer, validate

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',required=True)
    parser.add_argument('--id',required=True)
    args,_ = parser.parse_known_args()
    cfg = json.loads(Path(args.config).read_text())
    validate(cfg)
    frozen = json.loads((ROOT/'logs/fusion-WU/frozen/config.json').read_text())
    if not cfg.get('engineering_only') and cfg != frozen:
        raise ValueError('Configuration differs from frozen config')
    if cfg.get('engineering_only'):
        if os.environ.get('FUSION_ENGINEERING') != '1':
            raise ValueError('Engineering permission flag missing')
    elif '--resume' in sys.argv or args.id != 'FEAT-SG6-WU-LKSA3-'+cfg['fusion_order']+'-50K-01' or '--screen' not in sys.argv:
        raise ValueError('This release is one scratch50K fusion screen only')
    if not cfg.get('engineering_only'):
        index=sys.argv.index('--screen-updates') if '--screen-updates' in sys.argv else -1
        if index<0 or sys.argv[index+1]!='50000':
            raise ValueError('Explicit cumulative50K cap required')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    trainer.build_model = build_model
    original = argparse.ArgumentParser.add_argument
    def add_argument(parser,*flags,**kwargs):
        if '--screen-updates' in flags:
            kwargs['choices'] = (1000,5000,20000,50000)
        return original(parser,*flags,**kwargs)
    argparse.ArgumentParser.add_argument = add_argument
    try:
        trainer.main()
    finally:
        argparse.ArgumentParser.add_argument = original

if __name__ == '__main__':
    main()
