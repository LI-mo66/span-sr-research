"""Synthetic independent acceptance check for selection, pairing, and small return ZIP."""
import json
from pathlib import Path
import tempfile
import zipfile
from bootstrap import ROOT,trainer
from compare import compare
from collect import collect

def main():
    with tempfile.TemporaryDirectory(prefix='strip-report-',dir=ROOT/'tmp') as tmp:
        top=Path(tmp);ref=top/'reference';run=top/'candidate';group=top/'group';frozen=group/'frozen'
        for p in (ref,run,frozen):p.mkdir(parents=True)
        names=('DIV2K_valid','Set5','Set14','B100','Urban100','Manga109')
        for directory,peak,gain in ((ref,48,0),(run,47,.1)):
            history=[]
            for epoch in range(1,51):
                score=30+gain-abs(epoch-peak)*.01
                metrics={name:dict(psnr=score,ssim=.8) for name in names}
                history.append(dict(epoch=epoch,update=epoch*1000,metrics=metrics))
                for name in names:
                    values=[dict(image='a.png',psnr=score-.1,ssim=.8),dict(image='b.png',psnr=score+.1,ssim=.8)]
                    (directory/f'{name}_epoch{epoch:05d}_u{epoch*1000}.json').write_text(json.dumps(values))
            (directory/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
            (directory/'manifest.json').write_text(json.dumps(dict(commit='synthetic')))
        audit=dict(reference_run=str(ref),candidate_commit='synthetic',
            reference_manifest_sha256=trainer.digest(ref/'manifest.json'),reference_history_sha256=trainer.digest(ref/'epochs.jsonl'))
        (frozen/'reuse-audit.json').write_text(json.dumps(audit))
        result=compare(frozen,run,group/'comparison.json')
        assert result['reference_selected_update']==48000 and result['candidate_selected_update']==47000
        assert abs(result['datasets']['DIV2K_valid']['selected_delta_psnr']-.1)<1e-10
        assert abs(result['datasets']['DIV2K_valid']['final_delta_psnr']-.09)<1e-10
        path=run/'DIV2K_valid_epoch00047_u47000.json'
        original=path.read_text();path.write_text(json.dumps(list(reversed(json.loads(original)))))
        try:compare(frozen,run,group/'invalid.json')
        except ValueError:pass
        else:raise AssertionError('Reordered image pairing must fail')
        path.write_text(original)
        (run/'latest.pt').write_bytes(b'WEIGHTS_MUST_NOT_BE_RETURNED')
        (group/'queue-status.json').write_text(json.dumps(dict(K15=dict(status='COMPLETE',run=str(run)))))
        archive=collect(group)
        with zipfile.ZipFile(archive) as z:
            assert not any(n.endswith('.pt') for n in z.namelist())
            hashes=json.loads(z.read('return-hashes.json'))
            import hashlib
            assert all(hashlib.sha256(z.read(name)).hexdigest()==r['sha256'] for name,r in hashes.items())
        print('REPORTING PASS: validation selection, final endpoint, paired-order rejection, return excludes weights, all return hashes')

if __name__=='__main__':main()
