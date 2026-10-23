"""Synthetic tests of delivery selection, relocation, comparison and backup."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import prepare
from run import HERE


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def main():
    from release_guard import successful
    good=dict(success=True,training_exit_code=0,comparison_exit_code=0,plotting_exit_code=0,backup_exit_code=0,exit_code=0)
    assert successful(good)
    for key in good:
        assert not successful(dict(good,**{key:False if key=='success' else 1}))
    assert not successful(dict(good,training_exit_code=None))
    root=HERE.parents[1]
    (root/'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root/'tmp',prefix='ucan-delivery-test-') as temp:
        work=Path(temp)
        # Preserve train source ordering; only a candidate-local path changes.
        rows=[dict(hr='/home/featurize/datasets/DIV2K/HR/0001.png',lr='/home/featurize/datasets/DIV2K/LR/0001x4.png'),
              dict(hr='/home/featurize/datasets/Flickr2K/Flickr2K_HR/000001.png',lr='/home/featurize/datasets/Flickr2K/Flickr2K_LR_bicubic/X4/000001x4.png')]
        data={'train':rows,'DIV2K_valid':rows[:1]}
        (work/'data/downloads/Flickr2K/Flickr2K_HR').mkdir(parents=True)
        (work/'data/downloads/Flickr2K/Flickr2K_LR_bicubic/X4').mkdir(parents=True)
        paths=prepare.relocate_paths(prepare.paths_from_manifest(data),work/'data')
        assert len(paths['train'])==2
        assert Path(paths['train'][0]['hr']).as_posix().endswith('DIV2K/HR')
        assert '/downloads/Flickr2K/' in paths['train'][1]['hr'].replace('\\','/')
        assert paths['train'][1]['lr_suffix']=='x4'
        # Select by validation only, retain earliest tie, use that same row
        # for every test set even when its own peak is elsewhere.
        datasets=('DIV2K_valid','Set5','Set14','B100','Urban100','Manga109')
        reference,candidate=work/'reference',work/'candidate'
        reference.mkdir();candidate.mkdir()
        base_rows=[];new_rows=[]
        for u in range(1000,50001,1000):
            b=dict(epoch=u//1000,update=u,metrics={k:dict(psnr=30.+u/1e6,ssim=.8) for k in datasets})
            if u==48000:b['metrics']['DIV2K_valid']['psnr']=30.1
            n=json.loads(json.dumps(b))
            n['metrics']['DIV2K_valid']['psnr']=30.2 if u in (47000,48000) else 30.+u/1e6
            for k in datasets[1:]:n['metrics'][k]['psnr']+=.03
            base_rows.append(b);new_rows.append(n)
        for folder,history in ((reference,base_rows),(candidate,new_rows)):
            (folder/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf8')
        _,bb=prepare.history(reference);_,nb=prepare.history(candidate)
        assert bb['update']==48000 and nb['update']==47000
        (candidate/'epoch00047_u47000.pt').write_bytes(b'fixture-checkpoint-not-a-real-weight')
        for name in datasets:
            for folder,row in ((reference,bb),(candidate,nb)):
                save(folder/f'{name}_epoch{row["epoch"]:05d}_u{row["update"]}.json',
                     [dict(image='fixture.png',psnr=row['metrics'][name]['psnr'])])
        cfg=dict(ucan_pair=dict(mode='independent'))
        audit=dict(candidate_source='fixture',candidate_config=cfg,environment={},reference_run=str(reference),
                   reference_selected=bb,reference_final=base_rows[-1])
        save(work/'audit.json',audit)
        save(candidate/'manifest.json',dict(commit='fixture',config=cfg,environment={}))
        result=work/'comparison.json'
        subprocess.run([sys.executable,str(HERE/'compare.py'),'--audit',str(work/'audit.json'),
                        '--run',str(candidate),'--output',str(result)],check=True,stdout=subprocess.DEVNULL)
        compared=json.loads(result.read_text())
        assert compared['selected_checkpoint'].endswith('epoch00047_u47000.pt')
        assert compared['validation_selected']['Set5']['candidate']==nb['metrics']['Set5']
        # Refuse incomplete/duplicate timelines, not just a final 50K row.
        bad=work/'bad';bad.mkdir()
        (bad/'epochs.jsonl').write_text(json.dumps(new_rows[-1])+'\n')
        try:prepare.history(bad)
        except ValueError:pass
        else:raise AssertionError('Incomplete timeline accepted')
        # Backup never overwrites, records a failed run honestly, and verifies
        # every copied payload and the exact source.zip archive.
        logs=work/'logs';logs.mkdir();(logs/'status.json').write_text('{"success":false}')
        dest=work/'archive'
        command=[sys.executable,str(HERE/'backup.py'),'--run',str(candidate),'--logs',str(logs),
                 '--destination',str(dest),'--training-exit','1','--comparison-exit','-1']
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
        backup=json.loads((dest/'backup-manifest.json').read_text())
        assert backup['training_exit_code']==1 and backup['comparison_exit_code']==-1
        from backup import sha
        assert all(sha(dest/k)==v for k,v in backup['files'].items())
        assert subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
        assert (dest/'BACKUP_COMPLETE').is_file()
        print('PASS path order/fallback, validation selection/ties, same-weight six-set report, incomplete history rejection, failed-run backup hashes, no-overwrite')


if __name__=='__main__':
    main()
