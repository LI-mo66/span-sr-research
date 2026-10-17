"""One validation-selected checkpoint per model within the common 50k cap."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', required=True)
    parser.add_argument('--candidate-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text())
    run = Path(args.candidate_run)
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['commit'] != audit['candidate_source']:
        raise ValueError('Candidate source differs from audited source')
    history = sorted([json.loads(line) for line in (run / 'epochs.jsonl').read_text().splitlines()],
                     key=lambda r: r['update'])
    if [r['update'] for r in history] != list(range(1000, 50001, 1000)):
        raise ValueError('Candidate must finish all 50k updates and 50 evaluations')
    selected = max(history, key=lambda r: r['metrics']['DIV2K_valid']['psnr'])
    checkpoint = run / f'epoch{selected["epoch"]:05d}_u{selected["update"]}.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    def comparison(base, candidate):
        return {name: dict(B0=values, ContentTAB=candidate['metrics'][name],
                           delta={metric: candidate['metrics'][name][metric] - values[metric]
                                  for metric in ('psnr', 'ssim')})
                for name, values in base['metrics'].items()}
    report = dict(budget=50000, seed=10, baseline_record=audit['baseline_record'],
                  baseline_checkpoint='epoch00048_u48000.pt',
                  candidate_checkpoint=checkpoint.name,
                  checkpoint_rule='highest DIV2K_valid PSNR within 50k; earliest tie; same selected weight on all tests',
                  validation_selected=comparison(audit['baseline_selected'], selected),
                  fixed_50000=comparison(audit['baseline_final'], history[-1]),
                  scope='single-seed 50k screening, not final convergence/seed stability')
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
