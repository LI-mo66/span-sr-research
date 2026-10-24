"""Unchanged SG6 trainer with arm-specific factory and fixed scratch50K cap."""
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent
sys.path.insert(1,str(HERE.parent/'spatial_gate6'))
from entry import existing
sys.path.insert(0,str(HERE))
from candidate import build_model,validate_variant

def validate_config(cfg):
    validate_variant(cfg)
    if cfg.get('engineering_only'):
        if os.environ.get('UCAN_PAIR_ENGINEERING')!='1':raise ValueError('Engineering flag required')
        return
    reference=json.loads((HERE.parent/'spatial_gate6/configs/LG6-SiLU-SG6.json').read_text())
    for k,v in reference.items():
        if k not in ('paths_file','experiment_version') and cfg.get(k)!=v:raise ValueError('Protocol drift: '+k)
    if set(cfg)!=set(reference)|{'ucan_sweep'}:raise ValueError('Unknown config fields')
    locked=json.loads((HERE/'config.json').read_text())
    if cfg['experiment_version']!=locked['experiment_version']:raise ValueError('Wrong experiment version')

def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--config',required=True);p.add_argument('--id',required=True)
    p.add_argument('--screen',action='store_true');p.add_argument('--screen-updates',type=int)
    a,_=p.parse_known_args();cfg=json.loads(Path(a.config).read_text());validate_config(cfg)
    import torch
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    if not cfg.get('engineering_only'):
        prefix='FEAT-SG6-UCAN-'+cfg['ucan_sweep']['arm'].upper()+'-'
        if not a.id.startswith(prefix) or not a.screen or a.screen_updates!=50000:raise ValueError('Wrong ID or scratch50K cap')
    existing.validate_config(cfg);existing.validate_data(cfg)
    existing.fs_runtime.trainer.build_model=build_model
    original=argparse.ArgumentParser.add_argument
    def add_argument(parser,*flags,**kwargs):
        if '--screen-updates' in flags:kwargs['choices']=(1000,5000,20000,50000)
        return original(parser,*flags,**kwargs)
    argparse.ArgumentParser.add_argument=add_argument
    try:existing.fs_runtime.trainer.main()
    finally:argparse.ArgumentParser.add_argument=original

if __name__=='__main__':main()
