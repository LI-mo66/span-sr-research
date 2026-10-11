"""Build portable Git-history bundle plus x4-only data archive; no credentials."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',action='store_true');a=p.parse_args()
    out=ROOT/'server-upload';out.mkdir(exist_ok=True)
    bundle=out/'span-source.bundle'
    if bundle.exists():raise FileExistsError('Bundle exists; preserve or rename prior export before re-export')
    subprocess.run(['git','bundle','create',str(bundle),'--all'],cwd=ROOT,check=True)
    manifest={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'bundle_sha256':sha(bundle)}
    if a.data:
        specs=[('DIV2K/DIV2K_train_HR',800),('DIV2K/DIV2K_train_LR_bicubic/X4',800),('DIV2K/DIV2K_valid_HR',100),('DIV2K/DIV2K_valid_LR_bicubic/X4',100),('Flickr2K/Flickr2K_HR',2650),('Flickr2K/Flickr2K_LR_bicubic/X4',2650)]
        for name,count in [('Set5',5),('Set14',14),('B100',100),('Urban100',100),('manga109',109)]:
            specs.extend([(f'benchmark/{name}/HR',count),(f'benchmark/{name}/LR_bicubic/X4',count)])
        files=[]
        for folder,expected in specs:
            paths=sorted((ROOT/'datasets'/folder).glob('*.png'))
            if len(paths)!=expected:raise ValueError(f'{folder}: expected {expected}, found {len(paths)}')
            files.extend(paths)
        target=out/'df2k-benchmarks-x4.tar'
        if target.exists():raise FileExistsError(target)
        print(f'Packaging {len(files)} PNGs, {sum(x.stat().st_size for x in files)/1024**3:.2f} GiB. Archives/cache/x2/x3 excluded.',flush=True)
        with tarfile.open(target,'w') as tar:
            for index,path in enumerate(files):
                tar.add(path,arcname=path.relative_to(ROOT/'datasets').as_posix(),recursive=False)
                if (index+1)%500==0:print(index+1,flush=True)
        manifest['data_tar_sha256']=sha(target)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Exported to',out,flush=True)
if __name__=='__main__':main()
