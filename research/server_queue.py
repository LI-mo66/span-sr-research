"""Frozen staged queue. Test metrics are recorded but never rank candidates."""
import argparse,copy,json,re,subprocess,sys
from pathlib import Path
from runtime import ROOT

VARIANTS={'B0':[],'A-fusion':[{'name':'fusion','position':'fusion'}],'B-mid':[{'name':'context','position':'mid'}],'B-tail':[{'name':'context','position':'tail'}],'C-mid':[{'name':'frequency','position':'mid'}],'C-tail':[{'name':'frequency','position':'tail'}]}
def put_immutable(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError(f'Frozen configuration differs: {path}. Use the same data/microbatch/backend or choose a new group ID.')
    else:path.write_text(json.dumps(value,indent=2),encoding='utf8')
def prepare(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.group):raise ValueError('Invalid group ID')
    root=Path(args.data_root).resolve();bench=root/'benchmark'
    paths=dict(train=[dict(hr=str(root/'DIV2K/DIV2K_train_HR'),lr=str(root/'DIV2K/DIV2K_train_LR_bicubic/X4'),lr_suffix='x4'),dict(hr=str(root/'Flickr2K/Flickr2K_HR'),lr=str(root/'Flickr2K/Flickr2K_LR_bicubic/X4'),lr_suffix='x4')],validation={})
    paths['validation']['DIV2K_valid']=dict(hr=str(root/'DIV2K/DIV2K_valid_HR'),lr=str(root/'DIV2K/DIV2K_valid_LR_bicubic/X4'),lr_suffix='x4')
    for name in ('Set5','Set14','B100','Urban100','Manga109'):
        folder=bench/name
        if name=='Manga109' and not folder.is_dir():folder=bench/'manga109'
        paths['validation'][name]=dict(hr=str(folder/'HR'),lr=str(folder/'LR_bicubic/X4'),lr_suffix='x4')
    for spec in paths['train']+list(paths['validation'].values()):
        for side in ('hr','lr'):
            if not Path(spec[side]).is_dir():raise FileNotFoundError(spec[side])
    group=ROOT/'logs/server_jobs'/args.group;group.mkdir(parents=True,exist_ok=True)
    put_immutable(group/'paths.local.json',paths)
    return group
def run_variant(args,group,name,seed,updates):
    config=copy.deepcopy(json.loads((ROOT/'research/configs/baseline_x4.json').read_text()))
    config.update(seed=seed,micro_batch=args.micro_batch,paths_file=str(group/'paths.local.json'),image_backend=args.image_backend,variants=VARIANTS[name])
    config_file=group/f'{name}-s{seed}.local.json';put_immutable(config_file,config)
    profile=group/f'{name}-resource.json'
    if not profile.exists():
        result=subprocess.run([sys.executable,'-u','research/profile_candidate.py','--config',str(config_file),'--output',str(profile)],cwd=ROOT)
        if result.returncode:
            if name=='B0':raise RuntimeError('Baseline resource profile failed')
            with open(group/'failures.jsonl','a') as f:f.write(json.dumps(dict(variant=name,stage='resource_profile',exit_code=result.returncode))+'\n')
            return False
    run_id=f'{args.group}-{name}-s{seed}';out=ROOT/'runs'/run_id
    command=[sys.executable,'-u','research/train.py','--config',str(config_file),'--id',run_id,'--screen','--screen-updates',str(updates)]
    if (out/'latest.pt').exists():command.append('--resume')
    print(f'QUEUE {run_id} -> {updates} updates',flush=True)
    result=subprocess.run(command,cwd=ROOT)
    if result.returncode:
        with open(group/'failures.jsonl','a') as f:f.write(json.dumps(dict(run=run_id,updates=updates,exit_code=result.returncode))+'\n')
        if name=='B0':raise RuntimeError('Baseline failed; candidate comparison aborted')
        return False
    return True
def latest_score(args,name,seed,updates):
    f=ROOT/'runs'/f'{args.group}-{name}-s{seed}'/'epochs.jsonl'
    if not f.exists():return None
    rows=[json.loads(line) for line in f.read_text().splitlines()]
    rows=[row for row in rows if row['update']<=updates]
    if not rows or rows[-1]['update']!=updates:return None
    tail=rows[-3:]
    return {metric:sum(r['metrics']['DIV2K_valid'][metric] for r in tail)/len(tail) for metric in ('psnr','ssim')}
def rank(args,group,seed,updates,names):
    baseline=latest_score(args,'B0',seed,updates)
    if baseline is None:raise ValueError('Required baseline results missing')
    ranked=[]
    for name in names:
        score=latest_score(args,name,seed,updates)
        if score is None:continue
        ranked.append(dict(name=name,seed=seed,updates=updates,delta_psnr=score['psnr']-baseline['psnr'],delta_ssim=score['ssim']-baseline['ssim'],validation=score))
    ranked.sort(key=lambda r:r['delta_psnr'],reverse=True)
    (group/f'ranking-s{seed}-u{updates}.json').write_text(json.dumps(ranked,indent=2))
    print(json.dumps(ranked,indent=2),flush=True);return ranked
def select_families(ranked):
    result=[];families=set()
    for row in ranked:
        family=row['name'][0]
        if row['delta_psnr']>0 and row['delta_ssim']>=0 and family not in families:
            result.append(row['name']);families.add(family)
        if len(result)==2:break
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--data-root',required=True);p.add_argument('--phase',choices=['initial','confirm','fusion'],default='initial');p.add_argument('--micro-batch',type=int,choices=[4,8,16,32,64],default=16);p.add_argument('--image-backend',choices=['png','npy_mmap'],default='npy_mmap');p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    group=prepare(args)
    if args.prepare_only:print('Prepared',group);return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Source must be clean before launching')
    if args.phase=='initial':
        alive=list(VARIANTS)
        for updates in (5000,20000):
            alive=[name for name in alive if run_variant(args,group,name,10,updates)]
            rank(args,group,10,updates,[n for n in alive if n!='B0'])
        choices=select_families(rank(args,group,10,20000,[n for n in alive if n!='B0']))
        (group/'selected-initial.json').write_text(json.dumps(choices));print('Provisional choices:',choices,'Not stable-gain evidence',flush=True)
    elif args.phase=='confirm':
        choices=json.loads((group/'selected-initial.json').read_text())
        if not choices:print('No positive PSNR/nondecreasing SSIM candidate admitted; preserve failed/gray runs.');return
        results={}
        for seed in (10,20):
            run_variant(args,group,'B0',seed,100000)
            for name in choices:run_variant(args,group,name,seed,100000)
            results[seed]=rank(args,group,seed,100000,choices)
        stable=[name for name in choices if all(any(r['name']==name and r['delta_psnr']>0 and r['delta_ssim']>=0 for r in results[seed]) for seed in results)]
        (group/'selected-confirm.json').write_text(json.dumps(stable));print('Two-seed provisional survivors:',stable,'Million-update confirmation still required',flush=True)
    else:
        choices=json.loads((group/'selected-confirm.json').read_text())
        if len(choices)<2:print('Fewer than2 independent mechanism survivors; fusion skipped.');return
        specifications=[spec for name in choices for spec in VARIANTS[name]];VARIANTS['FUSION']=specifications
        for seed in (10,20):
            if latest_score(args,'B0',seed,100000) is None:raise ValueError('Confirm phase baseline missing')
            for updates in (5000,20000,100000):
                if not run_variant(args,group,'FUSION',seed,updates):break
            rank(args,group,seed,100000,['FUSION']+choices)
    print('QUEUE COMPLETE; full-training/ablation decisions require review.',flush=True)
if __name__=='__main__':main()
