import os,sys,json,argparse
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[0]))
sys.path.insert(0,str(HERE))
import fusion
import trainer
if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--config',required=True);p.add_argument('--id',required=True)
    p.add_argument('--until-update',type=int,required=True)
    args,_=p.parse_known_args()
    cfg=json.loads(Path(args.config).read_text())
    if not cfg.get('engineering_only'):
        scale=cfg['model']['upscale'];arm=cfg['arm']
        assert scale in (2,3) and arm in ('B0','F0')
        frozen=HERE.parents[1]/f'logs/multiscale50k/x{scale}-{arm}.json'
        assert cfg==json.loads(frozen.read_text())
        assert args.id==f'MS-X{scale}-{arm}-S10-50K-01'
        assert args.until_update==50000 and cfg['total_updates']==50000
        assert cfg['hr_patch']==64*scale and cfg['crop_border']==scale
        assert cfg['effective_batch']==64 and cfg['micro_batch']==16
        assert cfg['selection_dataset']=='DIV2K_valid' and cfg['self_ensemble'] is True
    trainer.main()
