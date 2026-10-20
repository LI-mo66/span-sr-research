"""Reuse unchanged successful checks, repair only failed resource equivalence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
PREVIOUS_SOURCE='bbec93f20556efff09cf2ea12a416b7275a1dc3a'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-run',type=Path,default=Path('/home/featurize/work/span-x4-research/records/FEAT-MOEFS-G4-LR2E4-20K-01-1791983354063378494/runs/FEAT-MOEFS-G4-LR2E4-20K-01-B0-s10'))
    args=parser.parse_args()
    protected=['basicsr','research/moe_fs/trainer.py','research/runtime.py','research/deploy.py','research/image_io.py',
               'research/parallel_loader.py','research/atd_attention/adapter.py',
               'research/atd_attention/model.py','research/atd_attention/run.py',
               'research/atd_attention/runtime.py','research/atd_attention/verify.py',
               'research/atd_attention/check_reference.py','research/atd_attention/server_smoke.py',
               'research/atd_attention/vendor','research/atd_attention/server_check.py']
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Diagnostic source must be clean')
    changed=subprocess.check_output(['git','diff',PREVIOUS_SOURCE,'HEAD','--',*protected],cwd=ROOT,text=True).strip()
    if changed:raise ValueError('Previous checks cannot be reused: protected code changed')
    files=['reference-cuda.json','engineering-cuda.json','server-smoke.json']
    for name in files:
        value=json.loads((args.previous/name).read_text())
        if value.get('status')!='PASS':raise ValueError('Previous stage did not pass: '+name)
    cfg=json.loads((args.previous/'config.json').read_text())
    if cfg.get('atd_attention') is not True or cfg.get('engineering_only'):
        raise ValueError('Expected the original real-server ATD engineering config')
    from run import validate_config
    validate_config(cfg)
    smoke=json.loads((args.previous/'server-smoke.json').read_text())
    if smoke.get('micro_batch')!=cfg['micro_batch'] or smoke.get('accumulation')!=cfg['effective_batch']//cfg['micro_batch']:
        raise ValueError('Previous smoke batch differs from saved config')
    engineering=json.loads((args.previous/'engineering-cuda.json').read_text())
    if engineering.get('extra_parameters')!=21813 or engineering.get('deployed_parameters')!=520021:
        raise ValueError('Previous engineering parameter count differs')
    import torch
    baseline=json.loads((args.baseline_run/'manifest.json').read_text())
    env=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
             deterministic=True,cudnn_allow_tf32=True,matmul_allow_tf32=False)
    if env!=baseline['environment']:raise ValueError('CUDA environment differs from B0')
    args.output.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name in files+['config.json']:
        source=args.previous/name
        hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source,args.output/name)
        if hashlib.sha256((args.output/name).read_bytes()).hexdigest()!=hashes[name]:
            raise IOError('Copied prior evidence hash mismatch')
    provenance=dict(previous=str(args.previous.resolve()),previous_source=PREVIOUS_SOURCE,
                    new_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    copied_sha256=hashes,
                    environment=env,
                    reused='Three previous PASS stages; protected model/training/check source unchanged',
                    rerun='Only resource equivalence and timing; no dataset training or release')
    (args.output/'recovery-provenance.json').write_text(json.dumps(provenance,indent=2))
    subprocess.run([sys.executable,'-u',str(Path(__file__).parent/'resources.py'),
                    '--config',str(args.output/'config.json'),
                    '--output',str(args.output/'resources.json')],cwd=ROOT,check=True)
    result=json.loads((args.output/'resources.json').read_text())
    if result['execution_equivalence']['status']!='PASS':raise ValueError('Execution equivalence failed')
    (args.output/'COMPLETE').write_text('Three prior checks reused with source checks; new routing/FP64/FP32 equivalence and resource checks passed.\n')
    print('COMPLETE: resource recovery; no training or automatic release',flush=True)


if __name__=='__main__':main()
