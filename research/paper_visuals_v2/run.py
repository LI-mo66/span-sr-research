"""Preflight, unified recorded metrics and verified persistent archive."""
import argparse
import csv
import datetime
import io
import importlib.metadata
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common import *


def git(*args):return subprocess.check_output(['git','-C',str(PROJECT),*args],text=True).strip()


def preflight():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if git('rev-parse','HEAD')!='82296152efc423d34e8c89ad6166ae30bf01e26d':raise ValueError('Unexpected final experiment source commit')
    dirty=git('diff','--name-only','HEAD','--','research','basicsr')
    if dirty:raise ValueError('Tracked model/evaluation source changed: '+dirty)
    swin=ROOT/'external/SwinIR'
    if subprocess.check_output(['git','-C',str(swin),'rev-parse','HEAD'],text=True).strip()!='6545850fbf8df298df73d81f3e8cba638787c8bd':
        raise ValueError('SwinIR source commit differs from previously verified version')
    if subprocess.check_output(['git','-C',str(swin),'diff','--name-only','HEAD','--','models/network_swinir.py'],text=True).strip():
        raise ValueError('SwinIR model source modified')
    if shutil.disk_usage(BASE).free<3*1024**3:raise RuntimeError('Less than 3GiB free; expand storage before running')
    # Pin all old PNG inputs to the previously reviewed report.
    report=read_json(ROOT/'step8-roi-review/roi_ranking.json');verified=0
    for row in report['inputs']:
        for method,expected in row['sha256'].items():
            p=ROOT/'sr_outputs'/row['dataset']/method/row['image']
            if sha(p)!=expected:raise ValueError('Old PNG differs from reviewed input: '+str(p))
            verified+=1
    refs={};common_cfg=None
    keys=['seed','model','hr_patch','effective_batch','micro_batch','lr','adam_betas','adam_eps','milestones','gamma','crop_border','test_y_channel','self_ensemble','record_updates']
    for name,(commit,relative) in RUNS.items():
        run=BASE/'records'/relative;manifest=read_json(run/'manifest.json');cfg=manifest['config']
        if manifest['commit']!=commit:raise ValueError('Historical source commit mismatch: '+name)
        history=[json.loads(s) for s in (run/'epochs.jsonl').read_text().splitlines() if s.strip()]
        eligible=[r for r in history if r['update']<=50000]
        if [r['update'] for r in eligible]!=list(range(1000,50001,1000)):raise ValueError('Historical 50K history incomplete: '+name)
        selected=min(eligible,key=lambda r:(-r['metrics']['DIV2K_valid']['psnr'],r['update']))
        if selected['update']!=48000:raise ValueError('Selected historical checkpoint is not 48K: '+name)
        match={k:cfg.get(k) for k in keys}
        if common_cfg is None:common_cfg=match
        elif common_cfg!=match:raise ValueError('Ablation protocol mismatch: '+name)
        if cfg['seed']!=10 or cfg['self_ensemble'] or cfg['lr']!=2e-4:raise ValueError('Wrong ablation protocol')
        weight=run/'epoch00048_u48000.pt'
        refs[name]=dict(run=str(run),checkpoint=str(weight),sha256=sha(weight),commit=commit,selected_update=48000)
        source=OUT/'source50'/name
        if not source.exists():
            try:subprocess.run(['git','-C',str(PROJECT),'cat-file','-e',commit+'^{commit}'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                subprocess.run(['git','-C',str(PROJECT),'fetch','origin',commit],check=True)
            source.mkdir(parents=True)
            archive=OUT/f'source-{name}.tar'
            subprocess.run(['git','-C',str(PROJECT),'archive','--format=tar','--output',str(archive),commit],check=True)
            with tarfile.open(archive) as tar:
                for member in tar:
                    if not member.isfile():continue
                    target=source/member.name
                    if not target.resolve().is_relative_to(source.resolve()):raise ValueError('Unsafe source archive member')
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(member).read())
        print('PREFLIGHT_50K_PASS',name,flush=True)
    cfg=read_json(PROJECT/'logs/f0-finetune-five-01/FT2/config.json')
    final=BASE/'records/F0-X4-FT2-20K-FINAL-ARCHIVE-01/assets/F0-x4-FT2-20K-full.pt'
    bound=BASE/'records/F0-FEATURE-BOUND8-1M-FINAL-01/candidate/feature_bound.py'
    if sha(final)!=FINAL_SHA or sha(bound)!=BOUND_SHA:raise ValueError('Final weight or Bound8 hash mismatch')
    weights=OUT/'weights-used';weights.mkdir(exist_ok=True)
    for name,source in [('FT2-20K-RAW.pt',final),('feature_bound.py',bound),*[(n+'-48K.pt',Path(v['checkpoint'])) for n,v in refs.items()]]:
        dest=weights/name
        if not dest.exists():shutil.copy2(source,dest)
        if sha(dest)!=sha(source):raise ValueError('Persistent weight-copy hash mismatch')
    sources={}
    for path in Path(__file__).resolve().parent.iterdir():
        if path.suffix in ['.py','.sh','.txt'] and path.is_file() and path.name not in ['probe_official.py']:
            dest=OUT/'suite-code'/path.name;dest.parent.mkdir(exist_ok=True)
            if not dest.exists():shutil.copy2(path,dest)
            if sha(dest)!=sha(path):raise ValueError('Suite code changed since first execution')
            sources[path.name]=sha(path)
    archive=OUT/'final-model-source-8229615.tar'
    if not archive.exists():subprocess.run(['git','-C',str(PROJECT),'archive','--format=tar','--output',str(archive),'82296152efc423d34e8c89ad6166ae30bf01e26d'],check=True)
    packages={}
    for package in ['numpy','Pillow','opencv-python','opencv-python-headless','matplotlib','timm','einops','scipy']:
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    write_json(OUT/'preflight.json',dict(status='PASS',old_PNGs_verified=verified,ablation=refs,ablation_common_protocol=common_cfg,
        final=dict(checkpoint=str(final),sha256=FINAL_SHA,bound=str(bound),bound_sha256=BOUND_SHA,config=cfg),suite_sources=sources,
        environment=dict(python=sys.version,torch=torch.__version__,device=torch.cuda.get_device_name(),cuda=torch.version.cuda,packages=packages),
        candidate_selection='DDistill-SR and IMDN: official weights with exact git-blob verification; all methods single pass',
        alternatives=dict(SRConvNet_L='No official pretrained link found in inspected repository/releases/issues',ESRT='No official pretrained link found in inspected repository/releases/issues',CNMC='No pretrained weights or link found in inspected official main tree/README; mamba_ssm1.0.1 dependency',LBNet='Direct official weight found; current architecture has four unmatched top-level parameter keys: norm.weight, norm.bias, proj.weight, proj.bias. Not filtered for this suite.')))
    print('PREFLIGHT_COMPLETE',flush=True)


def merge():
    rows=[];seen={};files=[]
    for parent in [ROOT/'sr_outputs',OUT/'sr_outputs']:
        for p in sorted(parent.glob('*/*/*.png')):
            ds,method,image=p.relative_to(parent).parts
            if method=='GT':continue
            key=(ds,method,image)
            if key in seen:
                if sha(p)!=seen[key]:raise ValueError('Conflicting duplicate PNG')
                continue
            gt,_,inputs=pair(ds,image);sr=np.array(Image.open(p).convert('RGB'))
            if sr.shape!=gt.shape:raise ValueError('Merged PNG shape mismatch: '+str(p))
            score=metrics(sr,gt);seen[key]=sha(p)
            rows.append(dict(dataset=ds,image_name=image,method=method,psnr_y=score['psnr'],ssim_y=score['ssim'],width=sr.shape[1],height=sr.shape[0],mode='x8' if method=='ours_x8' else 'single',scope='50K ablation' if '_50k_' in method else 'released/final',png_sha256=sha(p),png=str(p),lr_sha256=inputs['lr_sha256'],hr_sha256=inputs['hr_sha256']))
            if len(rows)%100==0:print('MERGE_METRICS',len(rows),flush=True)
    path=OUT/'metrics_per_image.csv'
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    raw=buffer.getvalue().encode('utf-8-sig')
    if path.exists():
        if path.read_bytes()!=raw:raise FileExistsError('Existing merged metrics differ')
    else:path.write_bytes(raw)
    grouped={}
    for row in rows:grouped.setdefault((row['dataset'],row['method']),[]).append(row)
    means=[]
    for (ds,method),items in sorted(grouped.items()):
        means.append(dict(dataset=ds,method=method,images=len(items),psnr_y=float(np.mean([r['psnr_y'] for r in items])),ssim_y=float(np.mean([r['ssim_y'] for r in items])),scope='two_selected_images_only' if '_50k_' in method else 'full_dataset'))
    summary=OUT/'dataset-means.csv'
    if not summary.exists():
        with summary.open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(means[0]));writer.writeheader();writer.writerows(means)
    figures={name:read_json(OUT/f'{name}-v2.json') for name in ['fig3','fig4','fig5','fig6']}
    manifest=dict(status='COMPUTED_PENDING_LOCAL_VISUAL_REVIEW_AND_DOWNLOAD_VERIFICATION',suite_directory=str(OUT),
        sources=read_json(OUT/'preflight.json'),models=[read_json(OUT/x) for x in ['ddistill-results.json','imdn-results.json']],
        figures=figures,metrics_file=str(path),metric_rows=len(rows),
        new_fig3='All eight displayed blocks use single pass; no x8 or tiling',
        fig4='Historical independently trained same-protocol 50K models selected at48K; no late Bound8',
        LAM=[read_json(OUT/f'lam-{n}.json') for n in ['SPAN52','SwinIR-light','Ours']],features=read_json(OUT/'features.json'),
        limitations=['B100 external whole-dataset inference not added by this suite','Newest competitors require a separate literature comparison','Local rendering review and downloaded-archive verification still required before shutdown'])
    write_json(OUT/'manifest.json',manifest)
    # Fill absent root entrypoints; never replace a previous version.
    for name in ['metrics_per_image.csv','manifest.json']:
        dest=ROOT/name
        if dest.exists() and sha(dest)!=sha(OUT/name):dest=ROOT/(Path(name).stem+'-v2'+Path(name).suffix)
        if dest.exists() and sha(dest)!=sha(OUT/name):raise FileExistsError(dest)
        if not dest.exists():shutil.copy2(OUT/name,dest)
    review=OUT/'paper-visuals-v2-review.zip'
    review_files=[(p,'figures_paper_ready/'+p.name) for p in sorted((OUT/'figures_paper_ready').iterdir()) if p.is_file() and p.suffix in ['.png','.pdf']]
    review_files.extend((p,p.name) for p in sorted(OUT.glob('*.json')) if p.name!='MERGE_COMPLETE.json')
    review_files.extend((p,p.name) for p in sorted(OUT.glob('*.csv')))
    if not review.exists():
        partial=review.with_name(review.name+'.part')
        with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p,name in review_files:z.write(p,name)
        partial.replace(review)
    with zipfile.ZipFile(review) as z:
        if z.testzip() is not None:raise ValueError('Review ZIP CRC error')
        for p,name in review_files:
            if hashlib.sha256(z.read(name)).hexdigest()!=sha(p):raise ValueError('Review ZIP source mismatch')
    (OUT/'paper-visuals-v2-review.zip.sha256').write_text(sha(review)+'  '+review.name+'\n')
    write_json(OUT/'MERGE_COMPLETE.json',dict(metrics_rows=len(rows),review_zip=str(review),review_sha256=sha(review)))
    print('MERGE_COMPLETE',len(rows),flush=True)


def archive():
    read_json(OUT/'manifest.json')
    now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest=BASE/'records'/('PAPER-VISUALS-FINAL-V2-'+now);dest.mkdir()
    exclusions={OUT/'queue.nohup.log',OUT/'archive.log',OUT/'status.txt',OUT/'suite.pid'}
    entries=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p in exclusions or any(x in ['.git','__pycache__','deps','python_deps'] for x in p.relative_to(ROOT).parts):continue
        if p.suffix in ['.part','.tmp']:continue
        entries.append((p,'paper_visuals_20261126/'+p.relative_to(ROOT).as_posix()))
    for ds in ['Set5','Set14','Urban100','Manga109']:
        root=dataset(ds)
        for relative in ['HR','LR_bicubic/X4']:
            for p in sorted((root/relative).glob('*.png')):entries.append((p,'benchmark/'+ds+'/'+relative+'/'+p.name))
    expected={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p,name in entries}
    payload=sum(r['bytes'] for r in expected.values())
    if shutil.disk_usage(dest).free<payload+1024**3:raise RuntimeError('Expand persistent storage before final archive')
    target=dest/'paper-visuals-final-v2.tar.gz.part'
    with tarfile.open(target,'w:gz',compresslevel=4,dereference=True) as tar:
        for i,(p,name) in enumerate(entries,1):
            tar.add(p,arcname=name,recursive=False)
            if i%200==0:print('ARCHIVED',i,'/',len(entries),flush=True)
        raw=json.dumps(expected,ensure_ascii=False,indent=2).encode('utf-8');info=tarfile.TarInfo('archive-file-sha256.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    checked=0
    with tarfile.open(target,'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name=='archive-file-sha256.json':continue
            h=hashlib.sha256()
            stream=tar.extractfile(member)
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            if member.size!=expected[member.name]['bytes'] or h.hexdigest()!=expected[member.name]['sha256']:raise ValueError('Archive member mismatch: '+member.name)
            checked+=1
    if checked!=len(expected):raise ValueError('Archive member count mismatch')
    final=dest/'paper-visuals-final-v2.tar.gz';target.replace(final)
    write_json(dest/'archive-verification.json',dict(status='PASS',files_verified=checked,bytes=final.stat().st_size,sha256=sha(final),note='Server archive verified; download and local SHA256 check still pending. No shutdown authorized by this marker.'))
    (dest/'paper-visuals-final-v2.tar.gz.sha256').write_text(sha(final)+'  paper-visuals-final-v2.tar.gz\n')
    (dest/'SERVER_BACKUP_COMPLETE').write_text('All payload members verified; local download/visual review pending.\n')
    (OUT/'backup-path.txt').write_text(str(dest)+'\n')
    print('SERVER_BACKUP_COMPLETE',dest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','merge','archive']);a=p.parse_args();globals()[a.action]()
