"""Freeze four matched configurations and audit decoded ordered pairs."""
import copy,hashlib,json,sys
from pathlib import Path
import cv2
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(1,str(ROOT/'research'))
import fusion,trainer

def main():
    import platform,torch,numpy
    assert platform.python_version()=='3.11.8',platform.python_version()
    assert torch.__version__=='2.2.2+cu121',torch.__version__
    assert torch.version.cuda=='12.1',torch.version.cuda
    assert numpy.__version__=='1.26.4',numpy.__version__
    assert cv2.__version__=='4.10.0',cv2.__version__
    assert '4090' in torch.cuda.get_device_name(),torch.cuda.get_device_name()
    group=ROOT/'logs/multiscale50k';group.mkdir(exist_ok=False,parents=True)
    data=Path('/home/featurize/datasets');base=json.loads((HERE/'template.json').read_text())
    from deploy import fused_copy
    arch,_=fusion.components()
    train_hr=set();audit=[]
    for scale in (2,3):
        def spec(hr,lr):return dict(hr=str(hr),lr=str(lr),lr_suffix=f'x{scale}')
        paths=dict(train=[spec(data/'DIV2K/DIV2K_train_HR',data/f'DIV2K/DIV2K_train_LR_bicubic/X{scale}'),spec(data/'downloads/Flickr2K/Flickr2K_HR',data/f'downloads/Flickr2K/Flickr2K_LR_bicubic/X{scale}')],validation={})
        paths['validation']['DIV2K_valid']=spec(data/'DIV2K/DIV2K_valid_HR',data/f'DIV2K/DIV2K_valid_LR_bicubic/X{scale}')
        for name in ('Set5','Set14','B100','Urban100','Manga109'):
            folder=data/'benchmark'/name
            if not folder.is_dir():
                matches=[p for p in (data/'benchmark').iterdir() if p.is_dir() and p.name.lower()==name.lower()]
                assert len(matches)==1;folder=matches[0]
            paths['validation'][name]=spec(folder/'HR',folder/f'LR_bicubic/X{scale}')
        trainer.SCALE=scale
        ordered={'train':trainer.pairs(paths['train'])}
        ordered.update({k:trainer.pairs(v) for k,v in paths['validation'].items()})
        expected=dict(train=3450,DIV2K_valid=100,Set5=5,Set14=14,B100=100,Urban100=100,Manga109=109)
        hashes={}
        for name,pairs in ordered.items():
            assert len(pairs)==expected[name],(scale,name,len(pairs))
            rows=[]
            for index,(hp,lp) in enumerate(pairs,1):
                h,l=trainer.aligned(hp,lp)
                assert h.dtype==l.dtype and str(h.dtype)=='uint8'
                if name=='train':assert min(l.shape[:2])>=64
                rows.append(dict(hr=str(hp),lr=str(lp),hr_sha256=trainer.digest(hp),lr_sha256=trainer.digest(lp)))
                if index%500==0:print('AUDIT',scale,name,index,flush=True)
            hashes[name]=rows
            print('PASS data',scale,name,len(rows),flush=True)
        train_hr={r['hr_sha256'] for r in hashes['train']}
        for name,rows in hashes.items():
            if name!='train':assert not train_hr & {r['hr_sha256'] for r in rows}
        (group/f'x{scale}-data-manifest.json').write_text(json.dumps(hashes,indent=2))
        pf=group/f'x{scale}-paths.json';pf.write_text(json.dumps(paths,indent=2))
        for arm in ('B0','F0'):
            cfg=copy.deepcopy(base)
            cfg['model']['upscale']=scale
            cfg.update(arm=arm,hr_patch=64*scale,crop_border=scale,paths_file=str(pf),
                seed=10,effective_batch=64,micro_batch=16,lr=2e-4,
                total_updates=50000,record_updates=1000,selection_dataset='DIV2K_valid',
                loader_workers=0,image_backend='npy_mmap')
            baseline=arch.SPAN(**cfg['model'])
            cfg['deployed_baseline_parameters']=sum(p.numel() for p in fused_copy(baseline,arch.Conv3XC).parameters())
            cfg['max_extra_deployed_parameters']=71687-1372+84*scale**2+28
            (group/f'x{scale}-{arm}.json').write_text(json.dumps(cfg,indent=2))
    (group/'DATA_PASS').write_text('Decoded/hash/overlap/scale checks passed.\n')
if __name__=='__main__':main()
