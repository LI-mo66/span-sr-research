"""Real branch runner on tiny fixtures, interruption/resume, plots and backup."""
import json
import os
import tempfile
from pathlib import Path
os.environ['SG6_FINETUNE_ENGINEERING']='1'
from common import *
import ft_runner as run
import ft_queue as server_queue


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for runner integration')
    ROOT.joinpath('tmp').mkdir(exist_ok=True)
    fixture=Path(tempfile.mkdtemp(prefix='ft-integration-',dir=ROOT/'tmp'))
    import cv2
    gen=np.random.default_rng(321)
    hp,lp=fixture/'hr.png',fixture/'lr.png'
    cv2.imwrite(str(hp),gen.integers(0,256,(48,64,3),dtype=np.uint8))
    cv2.imwrite(str(lp),gen.integers(0,256,(12,16,3),dtype=np.uint8))
    data={k:[(hp,lp)] for k in ('train','DIV2K_valid','Set5','Set14','B100','Urban100','Manga109')}
    # Intentional synthetic shared data; production audit rejects train/validation overlap.
    arch,metrics=sg_entry.existing.fs_runtime.components()
    cfg=json.loads((ROOT/'research/spatial_gate6/configs/LG6-SiLU-SG6.json').read_text())
    cfg.update(hr_patch=32,effective_batch=2,micro_batch=1)
    seed(10);model=build_model(arch,cfg).cuda();initial=state_copy(model,arch)
    small=dict(updates=2,evaluate_every=1,average_updates=[1,2])
    direct=fixture/'continuous';direct.mkdir()
    resumed=fixture/'resumed';resumed.mkdir()
    signature='engineering-only-not-a-real-reference'
    result=run.branch('A-L1',initial,cfg,data,arch,metrics,direct,signature,small)
    real_step=run.step
    count=0
    def interrupt(*args,**kwargs):
        nonlocal count
        count+=1
        if count==2:raise RuntimeError('Intentional engineering interruption')
        return real_step(*args,**kwargs)
    run.step=interrupt
    try:
        run.branch('A-L1',initial,cfg,data,arch,metrics,resumed,signature,small)
        raise AssertionError('Expected interruption')
    except RuntimeError as exc:
        assert str(exc)=='Intentional engineering interruption'
    finally:run.step=real_step
    with (resumed/'A-L1/loss_steps.jsonl').open('a') as f:f.write('{"partial":')
    recovered=run.branch('A-L1',initial,cfg,data,arch,metrics,resumed,signature,small)
    for name in ('raw_u00002.pt','ema_u00002.pt','average_6k_10k.pt'):
        a=torch.load(direct/'A-L1'/name,map_location='cpu',weights_only=False)['model']
        b=torch.load(resumed/'A-L1'/name,map_location='cpu',weights_only=False)['model']
        assert all(torch.equal(v,b[k]) for k,v in a.items())
    assert result['selected']==recovered['selected']
    assert (direct/'A-L1/loss_steps.jsonl').read_text()==(resumed/'A-L1/loss_steps.jsonl').read_text()
    run.branch('B-MSE',initial,cfg,data,arch,metrics,resumed,signature,small)
    write(resumed/'R0.json',dict(metrics={'DIV2K_valid':{'psnr':9.,'ssim':.01}}))
    server_queue.plots(resumed)
    assert len(list((resumed/'curves').glob('*.png')))==2
    logs=fixture/'logs';logs.mkdir();write(logs/'status.json',dict(exit_code=0))
    records=fixture/'records';records.mkdir()
    dest=server_queue.backup(resumed,logs,records,'ENGINEERING',0)
    for e in json.loads((dest/'backup-manifest.json').read_text())['files']:
        assert sha(dest/e['path'])==e['sha256']
    assert (dest/'BACKUP_COMPLETE').is_file()
    print('PASS real branch runner: both arms, partial JSON recovery, exact raw/EMA/average resume, selected metrics, figures and SHA backup')
    print('Synthetic only. Fixture:',fixture)


if __name__=='__main__':main()
