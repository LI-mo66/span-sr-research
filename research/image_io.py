"""Optional exact uint8 decode cache: same image sampling, faster crop reads."""
import functools,os,shutil,struct
from pathlib import Path
import cv2
import numpy as np

_mapping={}
def decode(p):
    a=cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR)
    if a is None:raise ValueError(f'Unreadable image {p}')
    return a
@functools.lru_cache(maxsize=128)
def mapped(p):return np.load(p,mmap_mode='r',allow_pickle=False)
def read(p):
    target=_mapping.get(str(Path(p).resolve()))
    return mapped(str(target)) if target else decode(p)
def enable_cache(manifests,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    sources={}
    for records in manifests.values():
        for record in records:
            for side in ('hr','lr'):sources[str(Path(record[side]).resolve())]=record[side+'_sha256']
    needed=0
    for path,sha in sources.items():
        if (root/(sha+'.npy')).exists():continue
        with open(path,'rb') as f:header=f.read(24)
        if header[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('npy_mmap backend currently requires PNG inputs')
        w,h=struct.unpack('>II',header[16:24]);needed+=w*h*3+256
    if shutil.disk_usage(root).free<needed+2*1024**3:raise OSError(f'Insufficient cache disk space: need {needed/1024**3:.2f} GiB plus2GiB reserve; choose image_backend=png or a larger cache disk')
    for index,(path,sha) in enumerate(sources.items()):
        target=root/(sha+'.npy')
        if not target.exists():
            tmp=target.with_suffix('.tmp')
            with open(tmp,'wb') as f:np.save(f,decode(path),allow_pickle=False)
            os.replace(tmp,target)
        _mapping[path]=target
        if (index+1)%250==0:print(f'Image cache {index+1}/{len(sources)}',flush=True)
    mapped.cache_clear()
