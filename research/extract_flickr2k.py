"""Extract only official HR and x4 LR PNGs; refuse traversal and overwrites."""
import argparse
import json
import shutil
import tarfile
import time
from runtime import ROOT

def main():
    p=argparse.ArgumentParser(); p.add_argument('--wait-download',action='store_true'); a=p.parse_args()
    archive=ROOT/'datasets/downloads/Flickr2K.tar'
    if a.wait_download:
        print('Waiting for completed official archive; no training will be launched',flush=True)
        while not archive.exists(): time.sleep(10)
    if not archive.exists(): raise FileNotFoundError(archive)
    root=(ROOT/'datasets').resolve(); counts={'HR':0,'LR_X4':0}
    with tarfile.open(archive,'r:') as source:
        for m in source:
            if not m.isfile() or not m.name.endswith('.png'): continue
            if m.name.startswith('Flickr2K/Flickr2K_HR/'): key='HR'
            elif m.name.startswith('Flickr2K/Flickr2K_LR_bicubic/X4/'): key='LR_X4'
            else: continue
            target=(root/m.name).resolve()
            if not target.is_relative_to(root): raise ValueError('Archive path escapes dataset directory')
            if target.exists(): raise FileExistsError(f'Refusing overwrite {target}; inspect prior partial extraction')
            target.parent.mkdir(parents=True,exist_ok=True)
            with source.extractfile(m) as src, open(target,'xb') as dst: shutil.copyfileobj(src,dst)
            counts[key]+=1
    if counts!={'HR':2650,'LR_X4':2650}: raise ValueError(f'Unexpected counts {counts}')
    (root/'Flickr2K/extraction.json').write_text(json.dumps(counts,indent=2))
    print(f'Extraction completed: {counts}; pair/provenance preflight still required',flush=True)
if __name__=='__main__': main()
