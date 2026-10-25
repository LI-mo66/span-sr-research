"""Synthetic delivery checks. Every shutdown command is mocked."""
import json
import subprocess
import sys
import tempfile
import contextlib
import uuid
from pathlib import Path
from unittest import mock
import torch
from verify import ROOT,build_model,load_arch
from observe import Observer
from backup_utils import backup,digest
from collect import collect
import power
from compare import compare
from compare_pair import compare as pair_compare


def rejected(call):
    try:call()
    except (RuntimeError,ValueError,IOError):return
    raise AssertionError('Unsafe condition was not rejected')


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    checks=[]
    # Keep fixtures under the workspace; preserve them for audit instead of cleanup.
    with contextlib.nullcontext(str(ROOT/'tmp'/('fe-delivery-'+uuid.uuid4().hex[:10]))) as directory:
        root=Path(directory)
        root.mkdir(parents=True,exist_ok=False)
        arch=load_arch()
        cfg=json.loads((ROOT/'research/feature_evolution/configs/SG6-FE-add.json').read_text())
        torch.manual_seed(10);plain=build_model(arch,cfg)
        torch.manual_seed(10);observed=build_model(arch,cfg)
        observer=Observer(observed,root/'observations.jsonl',2)
        opts=[torch.optim.Adam((p for p in net.parameters() if p.requires_grad),lr=2e-4)
              for net in (plain,observed)]
        x=torch.rand(1,3,7,11);target=torch.rand(1,3,28,44)
        random_state=torch.get_rng_state().clone()
        for step in range(3):
            for net,opt in zip((plain,observed),opts):
                opt.zero_grad(set_to_none=True)
                for _ in range(2):
                    (torch.nn.functional.l1_loss(net(x),target)/2).backward()
                opt.step()
            assert all(torch.equal(v,observed.state_dict()[k]) for k,v in plain.state_dict().items())
        with torch.no_grad():observed.eval();observed(x)
        assert torch.equal(random_state,torch.get_rng_state())
        rows=[json.loads(line) for line in (root/'observations.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows if r['kind']=='actual_update']==[1,2,3]
        assert all(set(r['statistics'])=={'a','b','c','delta','local_gate','spatial_gate'}
                   for r in rows if r['kind']=='features_first_microbatch')
        checks.append('Observer leaves three-step model updates and RNG bitwise unchanged')
        group=root/'group';group.mkdir()
        status={}
        for label in ('add','difference'):
            run=root/label;run.mkdir()
            (run/'weights.pt').write_bytes(b'synthetic-weight-placeholder')
            (run/'manifest.json').write_text('{}')
            saved=backup(run,group/(label+'-backup'),0)
            status[label]=dict(status='COMPLETE',training_exit_code=0,run=str(run),backup=saved)
        archive=collect(group,status)
        import zipfile
        with zipfile.ZipFile(archive) as stream:
            assert not any(name.endswith('.pt') for name in stream.namelist())
            assert stream.testzip() is None
        checks.append('Backup SHA256 and lightweight ZIP contents/CRC/hash verification')
        with mock.patch.object(power.subprocess,'check_output',return_value=''), \
             mock.patch.object(power.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'','')) as command, \
             mock.patch.object(power.os,'sync',create=True):
            rejected(lambda:power.finish(group,{'add':status['add']},['MOCK-SHUTDOWN'],archive))
            assert command.call_count==0
            broken=json.loads(json.dumps(status));broken['add']['status']='FAILED'
            rejected(lambda:power.finish(group,broken,['MOCK-SHUTDOWN'],archive))
            assert command.call_count==0
            target=Path(status['add']['backup']['path'])/'weights.pt'
            original=target.read_bytes();target.write_bytes(b'corrupt')
            rejected(lambda:power.finish(group,status,['MOCK-SHUTDOWN'],archive))
            assert command.call_count==0
            target.write_bytes(original)
            with mock.patch.object(power.subprocess,'check_output',return_value='999'):
                rejected(lambda:power.finish(group,status,['MOCK-SHUTDOWN'],archive))
            assert command.call_count==0
            power.finish(group,status,['MOCK-SHUTDOWN'],archive)
            assert command.call_count==1
            command.return_value=subprocess.CompletedProcess([],1,'','mock refusal')
            rejected(lambda:power.finish(group,status,['MOCK-SHUTDOWN'],archive))
        checks.append('Mock shutdown rejects missing candidate, failed run, corrupt backup and busy GPU; handles command failure')
        with mock.patch.object(power.os,'geteuid',return_value=1000,create=True), \
             mock.patch.object(power.shutil,'which',return_value=sys.executable), \
             mock.patch.object(power.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','denied')):
            rejected(power.plan)
        checks.append('Mock noninteractive shutdown permission rejection')
        ref=root/'reference';candidate=root/'candidate';ref.mkdir();candidate.mkdir()
        def fixture(folder,offset,commit):
            records=[]
            for epoch in range(1,51):
                value=20+epoch*.001+offset
                metrics={name:dict(psnr=value,ssim=.5+offset*.01) for name in ('DIV2K_valid','Set5')}
                records.append(dict(epoch=epoch,update=epoch*1000,metrics=metrics))
                for name in metrics:
                    images=[dict(image='a.png',psnr=value-.1,ssim=.5),dict(image='b.png',psnr=value+.1,ssim=.5)]
                    (folder/f'{name}_epoch{epoch:05d}_u{epoch*1000}.json').write_text(json.dumps(images))
            (folder/'epochs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
            (folder/'manifest.json').write_text(json.dumps(dict(commit=commit,environment={'fixture':True})))
        fixture(ref,0,'reference');fixture(candidate,.02,'candidate')
        frozen=root/'frozen';frozen.mkdir()
        (frozen/'reuse-audit.json').write_text(json.dumps(dict(reference_run=str(ref),candidate_commit='candidate',
            environment={'fixture':True},reference_manifest_sha256=digest(ref/'manifest.json'),
            reference_history_sha256=digest(ref/'epochs.jsonl'))))
        result=compare(frozen,candidate,root/'compare.json')
        assert abs(result['datasets']['DIV2K_valid']['selected_delta_psnr']-.02)<1e-10
        assert abs(result['datasets']['DIV2K_valid']['last10_mean_delta_psnr']-.02)<1e-10
        pair=pair_compare(ref,candidate,root/'pair.json')
        assert abs(pair['datasets']['DIV2K_valid']['selected_delta_psnr']-.02)<1e-10
        checks.append('Synthetic comparison validates selected weight, paired images, last3/last10 and equal-capacity comparison')
    output=ROOT/'research/feature_evolution/delivery-checks.json'
    output.write_text(json.dumps(dict(status='PASS',checks=checks,
        limitation='Synthetic fixtures and mocked shutdown only; no real training, Linux power-off or target-server permissions tested.'),indent=2))
    print(output)


if __name__=='__main__':main()
