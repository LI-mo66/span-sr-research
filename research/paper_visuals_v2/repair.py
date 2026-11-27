"""Resume a stopped suite after correcting the historical W factory filename."""
import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
import torch
from common import *

HERE=Path(__file__).resolve().parent
REPAIR=OUT/'repair-W-entry-01'


def fig3_package():
    record=read_json(OUT/'fig3-v2.json')
    if record.get('all_methods_mode')!='single, no self-ensemble, no tiling':raise ValueError('Wrong completed Figure3 mode')
    files=[(Path(record['figure'][kind]),'figures_paper_ready/'+Path(record['figure'][kind]).name) for kind in ['png','pdf']]
    files.extend((OUT/name,name) for name in ['fig3-v2.json','fig3-roi-metrics.csv','ddistill-results.json','imdn-results.json'])
    for scene in record['scenes']:
        for method,expected in scene['sha256'].items():
            if sha(image_path(scene['dataset'],method,scene['image']))!=expected:raise ValueError('Completed Figure3 input changed')
    target=OUT/'fig3-v2-completed.zip'
    notice='Completed Figure3 stage only. Figures4-6 and the complete final archive are not included. All Figure3 methods use single-pass inference.\n'
    if not target.exists():
        part=target.with_name(target.name+'.part')
        with zipfile.ZipFile(part,'w',zipfile.ZIP_DEFLATED) as z:
            for path,name in files:z.write(path,name)
            z.writestr('PACKAGE_STATUS.txt',notice)
        part.replace(target)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise ValueError('Figure3 ZIP CRC mismatch')
        for path,name in files:
            if hashlib.sha256(z.read(name)).hexdigest()!=sha(path):raise ValueError('Figure3 ZIP payload changed')
    (OUT/'fig3-v2-completed.zip.sha256').write_text(sha(target)+'  '+target.name+'\n')
    print('FIG3_PARTIAL_PACKAGE_READY',target,flush=True)
    return dict(path=str(target),sha256=sha(target),scope='Completed Figure3 stage only')


def probe(name):
    torch.set_num_threads(2)
    if name in ['W','U','F0']:
        model,run,provenance=ablation_model(name,'cpu')
    elif name=='Ours':model,provenance=final_model('cpu')
    else:model,provenance=public_model(name,'cpu')
    # Actual image input, no random-noise stress fixture and no optimization.
    _,lr,_=pair('Urban100','img004.png')
    with torch.no_grad():
        y=model(tensor(lr[:31,:37],'cpu'))
    if y.shape!=(1,3,124,148) or not torch.isfinite(y).all():raise ValueError('Repaired-entry smoke failure: '+name)
    write_json(REPAIR/(name+'-strict-load.json'),dict(model=name,strict_load=True,forward_shape=list(y.shape),finite=True,weight=provenance,scope='CPU actual-checkpoint loading and odd-rectangle smoke only; full historical/image reproduction remains in main suite'))
    print('REPAIR_STRICT_LOAD_PASS',name,flush=True)


def audit():
    REPAIR.mkdir(parents=True,exist_ok=True)
    original=read_json(OUT/'preflight.json')
    expected_completed=['preflight.json','ddistill-results.json','imdn-results.json','fig3-v2.json','ablation-B0.json','ablation-SG6.json']
    complete={name:sha(OUT/name) for name in expected_completed}
    for name in expected_completed:read_json(OUT/name)
    package=fig3_package()
    factories={}
    for name in ['W','U','F0']:
        path,blob=check_ablation_entry(name,OUT/'source50'/name)
        weight=original['ablation'][name]
        if sha(Path(weight['checkpoint']))!=weight['sha256']:raise ValueError('Historical checkpoint changed')
        factories[name]=dict(path=str(path),git_blob=blob,source_commit=RUNS[name][0])
    code={}
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ['.py','.sh','.txt','.json']:
            destination=REPAIR/'code'/p.name;destination.parent.mkdir(exist_ok=True)
            if destination.exists() and sha(destination)!=sha(p):raise ValueError('Repair code changed within same repair ID')
            if not destination.exists():destination.write_bytes(p.read_bytes())
            code[p.name]=sha(p)
    for name in ['W','U','F0','SPAN52','SwinIR-light','Ours']:
        marker=REPAIR/(name+'-strict-load.json')
        if not marker.exists():subprocess.run([sys.executable,str(HERE/'repair.py'),'probe','--name',name],check=True)
        read_json(marker)
    # Completed stage artifacts and their markers must remain untouched.
    for name,expected in complete.items():
        if sha(OUT/name)!=expected:raise ValueError('Previously completed artifact changed')
    commit=subprocess.check_output(['git','-C',str(HERE.parents[1]),'rev-parse','HEAD'],text=True).strip()
    write_json(REPAIR/'repair-provenance.json',dict(status='PASS',cause='W historical entry is model.py, not candidate.py',
        change='Factory path selection only; model architecture, weights, Bound8, metric implementation and completed scientific outputs unchanged',
        original_preflight_sha256=sha(OUT/'preflight.json'),original_suite_commit=original['suite_commit'],repair_suite_commit=commit,
        archived_factories=factories,repair_code_sha256=code,completed_artifacts_preserved=complete,fig3_package=package,
        strict_load_reports=[str(REPAIR/(name+'-strict-load.json')) for name in ['W','U','F0','SPAN52','SwinIR-light','Ours']],
        continuation='Skip completed stages. Resume W/U/F0, then Figure4, LAM, features and final archive. No shutdown.'))
    print('REPAIR_AUDIT_PASS; continuing the existing suite from first incomplete stage',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','probe','package']);p.add_argument('--name');a=p.parse_args()
    if a.action=='audit':audit()
    elif a.action=='package':fig3_package()
    else:probe(a.name)
