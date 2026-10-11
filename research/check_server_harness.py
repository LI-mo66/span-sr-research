"""Offline checks: no training gain claims, no actual shutdown invoked."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
import numpy as np,cv2
from runtime import ROOT
from image_io import enable_cache,read,mapped
from train import digest
from server_queue import rank,select_families

def posix(p):
    s=str(Path(p).resolve()).replace('\\','/')
    return '/'+s[0].lower()+s[2:] if len(s)>1 and s[1]==':' else s
def main():
    p=argparse.ArgumentParser();p.add_argument('--bash',default='bash');a=p.parse_args()
    (ROOT/'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
        t=Path(temp);image=t/'original.png';pixels=np.random.default_rng(1).integers(0,256,(17,23,3),dtype=np.uint8);cv2.imwrite(str(image),pixels)
        manifest={'fixture':[dict(hr=str(image),lr=str(image),hr_sha256=digest(image),lr_sha256=digest(image))]}
        enable_cache(manifest,t/'cache');assert np.array_equal(read(image),pixels)
        mapped.cache_clear()
        # Deliberately contradictory TEST scores must not change validation ranking.
        class Args:group='ENGINEERING-RANK-CHECK'
        group=ROOT/'logs/server_jobs'/Args.group;group.mkdir(parents=True,exist_ok=True)
        for name,valid,test in [('B0',30.,30.),('A-fusion',29.,99.),('B-tail',30.1,1.),('B-mid',30.05,2.)]:
            out=ROOT/'runs'/f'{Args.group}-{name}-s10';out.mkdir(parents=True,exist_ok=True)
            rows=[dict(update=u,metrics={'DIV2K_valid':dict(psnr=valid,ssim=.9),'Set5':dict(psnr=test,ssim=.9)}) for u in (10000,15000,20000)]
            (out/'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        assert select_families(rank(Args,group,10,20000,['A-fusion','B-tail','B-mid']))==['B-tail']
        fake=t/'bin';fake.mkdir()
        (fake/'flock').write_text('#!/usr/bin/env bash\nexit "${TEST_LOCK_CODE:-0}"\n')
        python=fake/'fake_python';python.write_text('#!/usr/bin/env bash\nexit "${TEST_EXIT_CODE:-0}"\n')
        shutdown=fake/'fake_shutdown';shutdown.write_text('#!/usr/bin/env bash\n[[ -f "$TEST_JOB/status.txt" ]] || exit 99\nprintf "mock shutdown\\n" > "$TEST_JOB/shutdown.called"\n')
        for f in fake.iterdir():f.chmod(0o755)
        for exit_code,lock_code in [(0,0),(7,0),(0,1)]:
            job=t/f'case-{exit_code}-{lock_code}';job.mkdir()
            env=dict(os.environ,PYTHON_BIN=posix(python),SHUTDOWN_BIN=posix(shutdown),TEST_JOB=posix(job),TEST_EXIT_CODE=str(exit_code),TEST_LOCK_CODE=str(lock_code))
            runner=t/'runner.sh';runner.write_text(f'#!/usr/bin/env bash\nexport PATH="{posix(fake)}:$PATH"\nexec bash research/scripts/run_server_worker.sh "{posix(job)}"\n')
            result=subprocess.run([a.bash,posix(runner)],cwd=ROOT,env=env)
            assert result.returncode==(75 if lock_code else exit_code)
            assert (job/'shutdown.called').exists()==(lock_code==0)
            if not lock_code:assert f'exit_code={exit_code}' in (job/'status.txt').read_text()
    print('PASS: cache pixel equality, validation-only selection, worker success/failure shutdown ordering, lock refusal without shutdown. All shutdown calls mocked.')
if __name__=='__main__':main()
