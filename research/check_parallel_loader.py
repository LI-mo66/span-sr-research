"""Check real PNG crops, worker ordering and restart at a microbatch boundary."""
import torch
from train import sample
from parallel_loader import PrefetchBatches
from pathlib import Path
from runtime import ROOT

if __name__ == '__main__':
    hr=ROOT/'datasets/DIV2K/DIV2K_train_HR'
    lr=ROOT/'datasets/DIV2K/DIV2K_train_LR_bicubic/X4'
    pairs=[(p,lr/(p.stem+'x4.png')) for p in sorted(hr.glob('*.png'))[:3]]
    assert len(pairs)==3 and all(l.exists() for _,l in pairs)
    def collect(workers,start=0):
        loader=PrefetchBatches(pairs,256,2,10,start,4,sample,workers,2)
        try: return list(loader)
        finally: loader.close()
    a,b,c=collect(1),collect(4),collect(4,2)
    for left,right in [(a,b),(a[2:],c)]:
        for x,y in zip(left,right):
            for xs,ys in zip(x,y):
                assert all(torch.equal(u,v) for u,v in zip(xs,ys))
    print('PASS: real PNG tensors identical across worker counts and indexed restart')
