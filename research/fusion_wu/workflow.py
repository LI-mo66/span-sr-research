"""Audit references, freeze protocol, report one validation-selected checkpoint."""
import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from fusion import ROOT, HERE, trainer

BASE='e731483d4ce4e325f3d64dbbc80add342b46f7f1'
WCOMMIT='78cc1dda90a5bafcefa6c62225197f6eb95c92e2'
UCOMMIT='d42b73fef5b2ec3e088f340ab033a5c4ac13a190'
GROUP=ROOT/'logs/fusion-WU'
RUNID='FEAT-SG6-W-UCAN-P24-LKSA5-FUSION-V1-50K-01'
RECORDS=Path('/home/featurize/work/span-x4-research/records')

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2)+'\n')
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def history(run):
    rows=sorted([json.loads(s) for s in (run/'epochs.jsonl').read_text().splitlines() if s.strip()],key=lambda r:r['update'])
    if [r['update'] for r in rows] != list(range(1000,50001,1000)):
        raise ValueError('Incomplete or duplicate 50K evaluation history: '+str(run))
    return rows,min(rows,key=lambda r:(-r['metrics']['DIV2K_valid']['psnr'],r['update']))

def prepare(wrun,urun):
    if git('status','--porcelain'): raise ValueError('Dirty source')
    for name,source in read(HERE/'source-provenance.json').items():
        raw=(HERE/name).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob==source['git_blob_sha'],'Pinned source differs: '+name
    if any(not p.startswith('research/fusion_wu/') for p in git('diff','--name-only',BASE,'HEAD').splitlines()):
        raise ValueError('Protected baseline changed')
    wm,um=read(wrun/'manifest.json'),read(urun/'manifest.json')
    assert wm['commit']==WCOMMIT and um['commit']==UCOMMIT
    oldaudit=read(wrun.parents[1]/'logs/sg6_mid_window/FEAT-SG6-MID-W16-PC26-AC-LR2E4-50K-01/audit/reuse-audit.json')
    srun=Path(oldaudit['reference_run']);sm=read(srun/'manifest.json')
    fixed=read(ROOT/'research/spatial_gate6/configs/LG6-SiLU-SG6.json')
    assert all(sm['config'].get(k)==v for k,v in fixed.items() if k!='paths_file'),'SG6 reference config differs from fixed base'
    assert (srun.parents[1]/'BACKUP_COMPLETE').is_file()
    assert (urun.parents[1]/'BACKUP_COMPLETE').is_file()
    refs={'SG6':srun,'W':wrun,'U':urun}
    selected={};final={}
    for name,folder in refs.items():
        rows,best=history(folder);selected[name]=best;final[name]=rows[-1]
        assert best['update']==48000
        assert (folder/f'epoch{best["epoch"]:05d}_u{best["update"]}.pt').is_file()
    assert trainer.digest(srun/'epoch00048_u48000.pt')=='9c16f95c1c265e628914160f7929ef7b7f0b07b7c6a8af81b6277f6bc252c86b'
    common=('seed','model','total_updates','hr_patch','effective_batch','micro_batch','lr',
            'adam_betas','adam_eps','milestones','gamma','crop_border','test_y_channel',
            'self_ensemble','record_updates','training_protocol','image_backend',
            'loader_workers','loader_prefetch','cpu_threads','selection_dataset')
    for key in common:
        assert sm['config'].get(key)==wm['config'].get(key)==um['config'].get(key),key
    assert sm['config']['seed']==10 and sm['config']['lr']==2e-4
    import torch,numpy,cv2,scipy,platform
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    env=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
             deterministic=True,cudnn_allow_tf32=True,matmul_allow_tf32=False)
    assert env==sm['environment']==wm['environment']==um['environment']
    assert platform.python_version()=='3.11.8'
    assert (numpy.__version__,cv2.__version__,scipy.__version__)==('1.26.4','4.10.0','1.13.1')
    def ordered(m):
        return {k:[(Path(r['hr']).name,Path(r['lr']).name,r['hr_sha256'],r['lr_sha256']) for r in v] for k,v in m['data'].items()}
    assert ordered(sm)==ordered(wm)==ordered(um)
    protected=['basicsr','research/moe_fs/trainer.py','research/runtime.py','research/image_io.py',
               'research/parallel_loader.py','research/deploy.py','research/spatial_gate6/sg_model.py',
               'research/gated_bypass/gb_model.py','research/linear_gate/lg_model.py']
    for m in (sm,wm,um):
        assert not git('diff','--name-only',m['commit'],BASE,'--',*protected),'Shared source differs'
    paths=read(Path(wm['config']['paths_file']))
    data={'train':trainer.pairs(paths['train'])}
    data.update({n:trainer.pairs(spec) for n,spec in paths['validation'].items()})
    for name,pairs in data.items():
        old=wm['data'][name];assert len(pairs)==len(old)
        print('Auditing ordered hashes:',name,len(pairs),flush=True)
        for (hp,lp),row in zip(pairs,old):
            assert hp.name==Path(row['hr']).name and lp.name==Path(row['lr']).name
            assert trainer.digest(hp)==row['hr_sha256'] and trainer.digest(lp)==row['lr_sha256'],str(hp)
            trainer.aligned(hp,lp)
    out=GROUP/'frozen';out.mkdir(parents=True,exist_ok=False)
    write(out/'paths.json',paths)
    cfg=copy.deepcopy(sm['config'])
    cfg.update(paths_file=str(out/'paths.json'),experiment_version='SG6-W-UCAN-P24-LKSA5-FUSION-v1',
               max_extra_deployed_parameters=71687,
               fusion_settings=dict(positions=[2,3,4,5],window=16,window_channels=26,window_heads=2,
                   window_initial_scale=.03,lksa_kernel=35,context='independent',initialization='unchanged-W-U'))
    write(out/'config.json',cfg)
    write(out/'audit.json',dict(commit=git('rev-parse','HEAD'),config=cfg,environment=env,
        source_provenance=read(HERE/'source-provenance.json'),
        reference_runs={k:str(v) for k,v in refs.items()},selected=selected,final=final,
        note='Scratch50K; all reference weights hashed/read-only; never used for initialization'))
    print('PASS: references, common protocol, data hashes, environment; scratch fusion only',flush=True)

def finish():
    import numpy as np
    audit=read(GROUP/'frozen/audit.json');run=ROOT/'runs'/RUNID
    meta=read(run/'manifest.json')
    assert meta['commit']==audit['commit'] and meta['config']==audit['config']
    assert meta['environment']==audit['environment']
    rows,best=history(run)
    checkpoint=f'epoch{best["epoch"]:05d}_u{best["update"]}.pt'
    assert (run/checkpoint).is_file()
    result=dict(checkpoint=checkpoint,rule='DIV2K best in50K, earliest tie, same weight all tests',
                selected={},fixed50000={},scope='Single seed10 50K; no guaranteed synergy or final-convergence claim')
    paired={}
    for label,refrows,candidate in (('selected',audit['selected'],best),('fixed50000',audit['final'],rows[-1])):
        for name,scores in candidate['metrics'].items():
            reference={k:v['metrics'][name] for k,v in refrows.items()}
            result[label][name]=dict(fusion=scores,references=reference,
                delta={k:{m:scores[m]-v[m] for m in ('psnr','ssim')} for k,v in reference.items()})
    for name in best['metrics']:
        def per(folder,row):
            values=read(folder/f'{name}_epoch{row["epoch"]:05d}_u{row["update"]}.json')
            assert len({v['image'] for v in values})==len(values)
            for metric in ('psnr','ssim'):
                assert abs(float(np.mean([v[metric] for v in values]))-row['metrics'][name][metric])<1e-9
            return {v['image']:v for v in values}
        actual=per(run,best);paired[name]={}
        for label,path in audit['reference_runs'].items():
            old=per(Path(path),audit['selected'][label]);assert actual.keys()==old.keys()
            diffs=[dict(image=k,delta_psnr=actual[k]['psnr']-old[k]['psnr'],
                        delta_ssim=actual[k]['ssim']-old[k]['ssim']) for k in sorted(actual)]
            numbers=np.array([v['delta_psnr'] for v in diffs])
            paired[name][label]=dict(mean=float(numbers.mean()),median=float(np.median(numbers)),
                wins=int((numbers>0).sum()),losses=int((numbers<0).sum()),rows=diffs)
    result['target_all_above_W_and_U']=all(result['selected'][n]['delta'][r]['psnr']>0
        for n in ('B100','Urban100','Manga109') for r in ('W','U'))
    result['all_six_above_W_and_U']=all(result['selected'][n]['delta'][r]['psnr']>0
        for n in best['metrics'] for r in ('W','U'))
    write(GROUP/'comparison.json',result);write(GROUP/'paired.json',paired)
    print(json.dumps(result,indent=2),flush=True)
    files=[]
    for source,relative in ((GROUP,Path('logs')),(run,Path('runs')/RUNID)):
        for p in source.rglob('*'):
            assert not p.is_symlink()
            if p.is_file(): files.append((p,relative/p.relative_to(source)))
    assert shutil.disk_usage(RECORDS).free>sum(p.stat().st_size for p,_ in files)+1024**3
    destination=RECORDS/(RUNID+'-'+str(time.time_ns()));destination.mkdir()
    entries=[]
    for source,relative in files:
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        before=trainer.digest(source);shutil.copy2(source,target)
        assert before==trainer.digest(target)==trainer.digest(source)
        entries.append(dict(path=str(relative),sha256=before))
    write(destination/'backup-manifest.json',dict(files=entries,training_exit_code=0))
    (destination/'BACKUP_COMPLETE').write_text('Files SHA256 verified; no shutdown requested.\n')
    (GROUP/'backup-path.txt').write_text(str(destination)+'\n')
    print('BACKUP_COMPLETE:',destination,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','finish'))
    p.add_argument('--wrun',type=Path);p.add_argument('--urun',type=Path);a=p.parse_args()
    prepare(a.wrun,a.urun) if a.mode=='prepare' else finish()
