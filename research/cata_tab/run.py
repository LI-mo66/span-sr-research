"""Candidate-only entry; shared B0 trainer and evaluation stay unchanged."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cata_runtime', HERE / 'runtime.py')
cata_runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cata_runtime)
from model import build_model


def validate_config(cfg):
    if cfg.get('cata_tab') is not True or cfg.get('moe_fs') or cfg.get('easa') or cfg.get('variants'):
        raise ValueError('This entry permits only the ContentTAB candidate')
    if set(cfg.get('cata_settings', {})) - {'chunk_groups'}:
        raise ValueError('ContentTAB-v1 structural settings are fixed')
    chunk = cfg.get('cata_settings', {}).get('chunk_groups', 128)
    if type(chunk) is not int or chunk < 1:
        raise ValueError('chunk_groups must be a positive integer')
    if cfg.get('engineering_only'):
        return
    reference = json.loads((cata_runtime.RESEARCH / 'configs/baseline_x4.json').read_text())
    for key in ('model','total_updates','hr_patch','effective_batch','adam_betas',
                'adam_eps','milestones','gamma','crop_border','test_y_channel',
                'self_ensemble','selection_dataset'):
        if cfg.get(key) != reference[key]:
            raise ValueError('Common B0 protocol differs: ' + key)
    if cfg.get('lr') != 2e-4 or cfg.get('training_protocol') != 'multistep-lr2e4-v1':
        raise ValueError('ContentTAB screen requires the matched LR2e-4 protocol')
    if cfg.get('micro_batch') != 16 or cfg.get('record_updates') != 1000 or cfg.get('seed') != 10:
        raise ValueError('Archived B0 reuse requires microbatch16/eval1000/seed10')


def validate_data(cfg):
    paths = json.loads(Path(cfg['paths_file']).read_text(encoding='utf-8-sig'))
    datasets = {'train': cata_runtime.trainer.pairs(paths['train'])}
    datasets.update({k:cata_runtime.trainer.pairs(v) for k,v in paths['validation'].items()})
    expected = dict(train=3450,DIV2K_valid=100,Set5=5,Set14=14,B100=100,Urban100=100,Manga109=109)
    if not cfg.get('engineering_only'):
        if set(datasets) != set(expected) or any(len(datasets[k]) != n for k,n in expected.items()):
            raise ValueError('Dataset names/counts differ from B0')
    for name,pairs in datasets.items():
        for hp,lp in pairs:
            hr,lr = cata_runtime.trainer.aligned(hp,lp)
            if name == 'train' and min(lr.shape[:2]) < cfg['hr_patch']//4:
                raise ValueError('Training image too small: ' + str(lp))
            if name != 'train' and min(hr.shape[:2])-2*cfg['crop_border'] < 11:
                raise ValueError('Evaluation image too small: ' + str(hp))
        print('Data audit:',name,len(pairs),'pairs decoded, x4 aligned',flush=True)


def train_main():
    original = argparse.ArgumentParser.add_argument
    def add_argument(parser,*args,**kwargs):
        if '--screen-updates' in args:
            kwargs['choices'] = [5000,20000,50000,100000]
        return original(parser,*args,**kwargs)
    argparse.ArgumentParser.add_argument = add_argument
    try:
        cata_runtime.trainer.main()
    finally:
        argparse.ArgumentParser.add_argument = original


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        train_main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',required=True)
    args,_ = parser.parse_known_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf8'))
    validate_config(cfg)
    validate_data(cfg)
    cata_runtime.trainer.build_model = build_model
    train_main()


if __name__ == '__main__':
    main()
