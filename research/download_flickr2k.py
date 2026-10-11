"""Resumable official archive download; TLS verification is never disabled."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from runtime import ROOT

URL='https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar'
def main():
    folder=ROOT/'datasets'/'downloads'; folder.mkdir(parents=True,exist_ok=True)
    target=folder/'Flickr2K.tar'; partial=folder/'Flickr2K.tar.part'
    if target.exists(): print('Complete archive already exists; refusing overwrite',flush=True); return
    offset=partial.stat().st_size if partial.exists() else 0
    req=urllib.request.Request(URL,headers={'Range':f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(req,timeout=60) as response:
        if offset and response.status!=206: raise RuntimeError('Server did not honor resume range')
        total=offset+int(response.headers['Content-Length']); start=time.monotonic(); count=offset; last=offset
        print(f'Downloading official archive: {total} bytes; resuming at {offset}',flush=True)
        with open(partial,'ab') as output:
            while True:
                block=response.read(4*1024*1024)
                if not block: break
                output.write(block); count+=len(block)
                if count-last>=64*1024*1024:
                    output.flush(); last=count; print(f'{count}/{total} bytes ({100*count/total:.2f}%); {(count-offset)/max(1,time.monotonic()-start)/1048576:.2f} MiB/s',flush=True)
    if count!=total: raise RuntimeError(f'Incomplete archive {count}/{total}')
    h=hashlib.sha256()
    with open(partial,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    partial.rename(target)
    (folder/'Flickr2K.source.json').write_text(json.dumps(dict(url=URL,bytes=count,sha256=h.hexdigest(),status='downloaded_not_extracted'),indent=2),encoding='utf8')
    print('Download complete; extraction and pair audit required before training',flush=True)
if __name__=='__main__': main()
