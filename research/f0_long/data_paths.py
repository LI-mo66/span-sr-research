"""Restore missing historical Flickr alias or extract original archive; never overwrite."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
root=Path('/home/featurize/datasets');target=root/'Flickr2K'
def complete(path):return (path/'Flickr2K_HR').is_dir() and (path/'Flickr2K_LR_bicubic/X4').is_dir()
if not complete(target):
    if target.exists() or target.is_symlink():raise RuntimeError('Incomplete Flickr2K path; refusing overwrite')
    alternate=root/'downloads/Flickr2K'
    if complete(alternate):target.symlink_to(alternate,target_is_directory=True)
    else:
        assert (root/'downloads/Flickr2K.tar').is_file(),'Missing original Flickr2K.tar; no training started'
        sys.path.insert(0,str(ROOT/'research'));import extract_flickr2k
        extract_flickr2k.ROOT=Path('/home/featurize');sys.argv=['extract_flickr2k.py'];extract_flickr2k.main()
assert complete(target)
print('Flickr2K path ready; ordered SHA256 audit still required',flush=True)
