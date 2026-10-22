"""Synthetic CPU end-to-end test; fake archives cannot be used as PSNR evidence."""
import json
from pathlib import Path
import sys
import tempfile
import cv2
import numpy as np
import torch
import diagnose_pairs as diag

def main():
    torch.set_num_threads(2)
    arch,metrics=diag.existing.fs_runtime.components()
    from deploy import fused_copy
    fixture=Path(tempfile.mkdtemp(prefix='paired-diagnostic-',dir=diag.ROOT/'tmp'))
    records=fixture/'records'; records.mkdir(); images=fixture/'images'; images.mkdir()
    lr=np.random.default_rng(10).integers(0,256,(8,10,3),dtype=np.uint8)
    hr=np.random.default_rng(11).integers(0,256,(32,40,3),dtype=np.uint8)
    data=[]
    for i in range(100):
        hp,lp=images/f'{i:04d}-HR.png',images/f'{i:04d}-LR.png'
        diag.save_image(hp,hr); diag.save_image(lp,lr)
        data.append(dict(hr=str(hp),lr=str(lp),hr_sha256=diag.sha(hp),lr_sha256=diag.sha(lp)))
    group='SMOKE'; reference=records/'reference'; reference.mkdir()
    ref_sha=None
    for mode in diag.MODES:
        label='off' if mode=='SG6' else mode
        cfg=diag.read(diag.ROOT/'research/sgla_context/configs'/f'SG6-{label}.json')
        if mode=='SG6': cfg.pop('sgla'); cfg['experiment_version']='LG6-SiLU-GB-SpatialGate6-v1'
        archive=reference if mode=='SG6' else records/f'{group}-SGLA-{mode}-s10-fixture'
        archive.mkdir(exist_ok=True); run=archive/'runs'/f'{group}-SGLA-{mode}-s10'; run.mkdir(parents=True)
        manifest=dict(commit='6f38f9757a474a4fde34dfbfa3a46554075bbb14',config=cfg,data={'DIV2K_valid':data})
        diag.write(run/'manifest.json',manifest)
        rows=[dict(update=u,epoch=u//1000,metrics={'DIV2K_valid':dict(psnr=1. if u==48000 else 0.)}) for u in range(1000,50001,1000)]
        (run/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        torch.manual_seed(10); model=diag.build_model(arch,cfg).cpu().eval()
        weight=run/'epoch00048_u48000.pt'; torch.save(dict(model=model.state_dict(),update=48000),weight)
        checkpoint_sha=diag.sha(weight)
        if mode=='SG6': ref_sha=checkpoint_sha
        with torch.inference_mode():
            x=torch.from_numpy(np.ascontiguousarray(lr[:,:,::-1].transpose(2,0,1))).float()[None]/255
            y=fused_copy(model,arch.Conv3XC)(x)
        sr=np.rint(y[0].clamp(0,1).numpy().transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8)
        score=metrics.calculate_psnr(sr,hr,crop_border=4,test_y_channel=True)
        diag.write(run/'DIV2K_valid_epoch00048_u48000.json',[dict(image=Path(r['hr']).name,psnr=score) for r in data])
        key=str(weight.relative_to(archive)).replace('\\','/')
        # SG6's historical server.py used [{path,sha256,bytes}], not a mapping.
        files=[dict(path=key,sha256=checkpoint_sha,bytes=weight.stat().st_size)] if mode=='SG6' else {key:checkpoint_sha}
        diag.write(archive/'backup-manifest.json',dict(training_exit_code=0,comparison_exit_code=0,files=files))
        (archive/'BACKUP_COMPLETE').write_text('synthetic fixture only')
    # Fixture-only substitution in this test process; no CLI bypass in production.
    diag.REFERENCE_SHA=ref_sha
    sys.argv=['diagnose_pairs.py','--records',str(records),'--reference',str(reference),'--group',group,'--output',str(fixture/'output')]
    diag.main()
    report=diag.read(fixture/'output/report.json')
    assert report['status']=='COMPLETE' and len(report['images'])==12
    assert report['checkpoint_post_hashes_verified'] and not report['warnings']
    for mode in diag.MODES[1:]:
        assert report['summary'][mode]['all']['pooled_delta_mse']==0
    masks=diag.region_masks(np.zeros((20,20)))
    r=diag.compare_errors(np.ones((20,20)),np.full((20,20),2.),masks)
    assert r['all']['harm_sse']==400 and r['all']['gain_sse']==0
    assert r['high_gradient']['pixels']==0
    assert diag.heat(np.array([[1.,-1.]]),1).tolist()==[[[0,0,255],[255,0,0]]]
    h='a'*64
    assert diag.archive_file_hash({'files':[dict(path='runs/a.pt',sha256=h,bytes=12)]},'runs/a.pt')==h
    assert diag.archive_file_hash({'files':{'runs/a.pt':h}},'runs/a.pt')==h
    assert diag.archive_file_hash({'files':{'runs/a.pt':h}},'missing.pt') is None
    for invalid in ({'files':None},{'files':[dict(path='a',sha256=h),dict(path='a',sha256=h)]},{'files':{'a':'bad'}}):
        try: diag.archive_file_hash(invalid,'a')
        except ValueError: pass
        else: raise AssertionError('Invalid manifest was accepted')
    print('PASS full CPU archive/hash/strict-load/four-model/fixed12/metrics/maps pipeline:',fixture)

if __name__=='__main__': main()
