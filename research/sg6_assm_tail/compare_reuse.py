"""SG6 is the direct comparator; a single validation-selected weight per model."""
import argparse
import json
from pathlib import Path
from prepare_reuse import history


def compare(base,candidate):
    if set(base['metrics'])!=set(candidate['metrics']):raise ValueError('Dataset set differs')
    return {name:dict(SG6=metrics,SG6_ASSM=candidate['metrics'][name],
                     delta_vs_SG6={m:candidate['metrics'][name][m]-metrics[m] for m in ('psnr','ssim')})
            for name,metrics in base['metrics'].items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit=json.loads(args.audit.read_text());manifest=json.loads((args.run/'manifest.json').read_text())
    if manifest['commit']!=audit['candidate_source'] or manifest['config']!=audit['candidate_config']:
        raise ValueError('Candidate source/config differs from frozen audit')
    rows,best=history(args.run)
    checkpoint=f'epoch{best["epoch"]:05d}_u{best["update"]}.pt'
    if not (args.run/checkpoint).is_file():raise FileNotFoundError(checkpoint)
    result=dict(direct_reference='SG6',fixed_base=audit['fixed_base'],seed=10,budget=50000,
                reference_checkpoint=audit['reference_weight'],candidate_checkpoint=checkpoint,
                rule='DIV2K_valid best within 50k, earliest tie; all datasets share that weight',
                validation_selected=compare(audit['reference_selected'],best),
                fixed_50000=compare(audit['reference_final'],rows[-1]),
                scope='single-seed 50k screen; no convergence/stability claim')
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
