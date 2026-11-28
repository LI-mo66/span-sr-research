"""External CPU manager; execute the unchanged6e71b81 candidate runner in1K stages."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

PIN='6e71b81cdcb53479233b674d35132f7233f96e64'


def expected_lr(update):
    return 2e-4 * .5**sum(update >= m for m in (200000,400000,600000,800000))


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def finite(value):
    if torch.is_tensor(value):return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
    if isinstance(value,float):return math.isfinite(value)
    return True


def quality_stop(rows,reference):
    current=rows[-1]['metrics']
    for name,value in current.items():
        delta=value['psnr']-reference[name]['psnr']
        if delta<-.20:return dict(rule='single_evaluation_drop_over_0p20dB',dataset=name,delta=delta)
    if len(rows)>=2:
        last=rows[-2:]
        if all(any(r['metrics'][name]['psnr']-reference[name]['psnr']<-.05
                   for name in ('DIV2K_valid','DIV2K_valid_x8')) for r in last):
            return dict(rule='two_consecutive_DIV2K_drop_over_0p05dB')
        if all(r['metrics']['Manga109']['psnr']-reference['Manga109']['psnr']<-.15
               or r['metrics']['Manga109_x8']['psnr']-reference['Manga109_x8']['psnr']<-.10
               for r in last):
            return dict(rule='two_consecutive_Manga_plain_drop_over_0p15_or_x8_over_0p10dB')
    return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--source-project',type=Path,required=True)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    project,stage=args.project.resolve(),args.stage.resolve()
    run=project/'runs/F0-X3-FRONTCAP5E4-S10-V1'
    assert not (stage/'protocol.json').exists()
    def clean():
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()==PIN
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=project,text=True).strip()
    clean()
    previous=read(project/'logs/f0_x3_update_audit/FRONTCAP-300K-310K-06/report.json')
    assert previous['status']=='COMPLETE_310K_REVIEW_REQUIRED_NO_SHUTDOWN'
    assert previous['update']==310000 and previous['stop_reason'] is None
    protocol=read(run/'protocol.json')
    assert protocol['rho']==.0005 and protocol['groups']==['conv_1','block_1']
    assert protocol['gradient_clip']==.1 and protocol['optimizer_reset'] is False
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    reference=read(run/'reference295k.json')
    def audit(update):
        assert not (run/'ANOMALY.json').exists()
        assert read(run/f'COMPLETE-{update}.json')['update']==update
        comparison=read(run/f'comparison-{update}.json')
        assert comparison['update']==update and comparison['source_unchanged'] is True
        receipt=read(run/f'verification-{update}.json')
        assert receipt['status']=='PASS' and receipt['as_of_update']==update
        with (run/'latest.pt').open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        assert digest==receipt['file_sha256']['latest.pt']
        state=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        assert state['format']=='F0-X3-frontcap-candidate-v1'
        assert state['update']==update and state['fingerprint']==fingerprint
        assert state['scheduler']['last_epoch']==update
        assert state['scheduler']['gamma']==.5
        assert state['scheduler']['base_lrs']==[2e-4] and state['scheduler']['_last_lr']==[expected_lr(update)]
        assert dict(state['scheduler']['milestones'])=={200000:1,400000:1,600000:1,800000:1}
        assert [g['lr'] for g in state['optimizer']['param_groups']]==[expected_lr(update)]
        assert finite(state['model']) and finite(state['optimizer'])
        assert set(state['rng'])=={'python','numpy','torch','cuda'}
        saved_record=state['epoch_record']
        del state
        rows=[json.loads(s) for s in (run/'epochs.jsonl').read_text().splitlines() if s.strip()]
        assert [r['update'] for r in rows]==list(range(296000,update+1,1000))
        assert all(set(r['metrics'])==set(reference) and finite(r['metrics']) for r in rows)
        assert saved_record==rows[-1]
        return rows
    header=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
    start=header['update'];del header
    assert start in range(310000,400001,1000), start
    history=audit(start)
    write(stage/'protocol.json',dict(project=str(project),pinned_code=PIN,start=start,maximum_stop=400000,
        stage_updates=1000,candidate_fingerprint=fingerprint,rho=.0005,lr_at_start=expected_lr(start),
        lr_schedule="Update310001..400000 uses1e-4; scheduler sets5e-5 after update400000; update400001 would use5e-5.",
        manager_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stopping_rules='Any plain/x8 dataset drop>0.20dB once; DIV plain/x8 drop>0.05dB for2 consecutive evals; Manga plain drop>0.15dB OR x8 drop>0.10dB for2 consecutive evals. All versus original295K.',
        scope='Engineering budget stop rules, not statistical significance. No automatic rescue or stage beyond400K.'))
    reason=quality_stop(history,reference)
    last=start
    if reason is None:
        for target in range(start+1000,400001,1000):
            clean()
            (stage/'status.txt').write_text(f'TRAINING_FRONTCAP_{last}_TO_{target}\n')
            command=[sys.executable,'-u','research/f0_x3_update_audit/frontcap_train.py',
                '--source-project',str(args.source_project.resolve()),'--output',str(run),
                '--cache',str(args.cache.resolve()),'--resume','--until-update',str(target)]
            print('START_STAGE',last,'TO',target,flush=True)
            subprocess.run(command,cwd=project,check=True)
            history=audit(target);last=target
            reason=quality_stop(history,reference)
            write(stage/f'gate-{target}.json',dict(update=target,stop=reason,
                comparison=read(run/f'comparison-{target}.json')))
            print('QUALITY_GATE',target,json.dumps(reason),flush=True)
            if reason is not None:break
    status='STOPPED_QUALITY_REVIEW_REQUIRED' if reason else 'COMPLETE_400K_REVIEW_REQUIRED_NO_SHUTDOWN'
    write(stage/'report.json',dict(status=status,update=last,stop_reason=reason,
        checkpoint=str(run/f'epoch{last//1000:05d}_u{last}.pt'),
        evaluations=history,scope='Candidate staged continuation only; not proof of long stability or recovered Manga quality.'))
    (stage/'status.txt').write_text(status+'\n')
    print('MANAGER_FINISHED',status,last,json.dumps(reason),flush=True)


if __name__=='__main__':main()
