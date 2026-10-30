"""Audit references, freeze protocol, report one validation-selected checkpoint."""
import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fusion_wu'))
from fusion import ROOT, HERE, trainer

BASE='e731483d4ce4e325f3d64dbbc80add342b46f7f1'
WCOMMIT='78cc1dda90a5bafcefa6c62225197f6eb95c92e2'
UCOMMIT='d42b73fef5b2ec3e088f340ab033a5c4ac13a190'
GROUP=ROOT/'logs/f0-long'
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
    if any(not p.startswith(('research/fusion_wu/','research/f0_long/')) for p in git('diff','--name-only',BASE,'HEAD').splitlines()):
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
    oldconfig=copy.deepcopy(cfg)
    write(ROOT/'logs/fusion-WU/frozen/config.json',oldconfig)
    cfg.update(self_ensemble=True,selection_dataset='DIV2K_valid_x8',
        long_protocol='scratch-F0-DF2K-FP32-1M-dual-eval-v1',
        evaluation_modes=['plain','x8'],anomaly_policy='finite-and-loss-stop-v1')
    write(out/'config.json',cfg)
    write(out/'audit.json',dict(commit=git('rev-parse','HEAD'),config=cfg,environment=env,
        source_provenance=read(HERE/'source-provenance.json'),
        reference_runs={k:str(v) for k,v in refs.items()},selected=selected,final=final,
        note='Formal training starts from scratch; historical weights used only by isolated reproduction process'))
    print('PASS: DF2K and six held-out sets audited; scratch F0 total1M, dual plain+x8 evaluation',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wrun',type=Path,required=True);p.add_argument('--urun',type=Path,required=True)
    a=p.parse_args();prepare(a.wrun,a.urun)
