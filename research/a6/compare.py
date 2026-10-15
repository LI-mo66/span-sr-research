"""Read-only archived B0 comparison: selected, fixed final, curves and last3."""
import argparse
import csv
import hashlib
import html
import json
import statistics
from pathlib import Path


def history(path):
    rows = sorted([json.loads(line) for line in Path(path).read_text().splitlines()], key=lambda r: r['update'])
    if [r['update'] for r in rows] != list(range(1000, 50001, 1000)):
        raise ValueError('Complete, unique 1k..50k evaluations required')
    return rows


def metrics_delta(base, candidate):
    if set(base) != set(candidate):
        raise ValueError('Evaluation dataset sets differ')
    return {name: dict(B0=values, A6=candidate[name],
                       delta={m: candidate[name][m] - values[m] for m in ('psnr', 'ssim')})
            for name, values in base.items()}


def paired(base_run, base_row, run, row):
    result = {}
    for name in row['metrics']:
        def filename(root, r):
            return root / f'{name}_epoch{r["epoch"]:05d}_u{r["update"]}.json'
        left, right = filename(base_run, base_row), filename(run, row)
        if not left.is_file() or not right.is_file():
            result[name] = dict(status='unavailable', missing=[str(f) for f in (left, right) if not f.is_file()])
            continue
        a = {r['image']: r for r in json.loads(left.read_text())}
        b = {r['image']: r for r in json.loads(right.read_text())}
        if set(a) != set(b):
            raise ValueError('Per-image identifiers differ: ' + name)
        rows = [dict(image=k, psnr_delta=b[k]['psnr']-a[k]['psnr'],
                     ssim_delta=b[k]['ssim']-a[k]['ssim']) for k in sorted(a)]
        ds = [r['psnr_delta'] for r in rows]
        result[name] = dict(status='available', count=len(ds), mean=statistics.mean(ds),
                            median=statistics.median(ds), win_rate=sum(v > 0 for v in ds)/len(ds),
                            worst=min(rows, key=lambda r: r['psnr_delta']), images=rows)
    return result


def curves_svg(base, candidate):
    names = list(base[0]['metrics'])
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="%s">' % (285*((len(names)+1)//2)),
           '<rect width="100%" height="100%" fill="white"/>']
    for n, name in enumerate(names):
        x0, y0 = 60+(n % 2)*500, 40+(n//2)*285
        vals = [r['metrics'][name]['psnr'] for rows in (base, candidate) for r in rows]
        lo, hi = min(vals)-.01, max(vals)+.01
        svg += [f'<text x="{x0}" y="{y0-12}" font-family="sans-serif">{html.escape(name)} | PSNR (Y/crop4)</text>',
                f'<path d="M{x0} {y0} V{y0+205} H{x0+420}" fill="none" stroke="#888"/>']
        for label, color, rows in (('B0', '#777777', base), ('A6', '#007aaf', candidate)):
            points = ' '.join(f'{x0+(r["update"]-1000)/49000*420:.2f},{y0+205-(r["metrics"][name]["psnr"]-lo)/(hi-lo)*205:.2f}' for r in rows)
            svg += [f'<polyline points="{points}" stroke="{color}" fill="none" stroke-width="2"/>']
        svg += [f'<text x="{x0}" y="{y0+225}" font-size="12" font-family="sans-serif">1k → 50k optimizer updates; B0 gray / A6 blue</text>',
                f'<text x="{x0}" y="{y0+243}" font-size="12" font-family="sans-serif">PSNR range: {lo:.4f}–{hi:.4f} dB</text>']
    return '\n'.join(svg+['</svg>'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audit', required=True)
    p.add_argument('--candidate-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit = json.loads(Path(args.audit).read_text())
    run, base_run = args.candidate_run.resolve(), Path(audit['baseline_run'])
    manifest = json.loads((run/'manifest.json').read_text())
    data_hash = hashlib.sha256(json.dumps(manifest['data'], sort_keys=True).encode()).hexdigest()
    if (manifest['commit'] != audit['candidate_source'] or manifest['config'] != audit['candidate_config'] or
            manifest['environment'] != audit['environment'] or data_hash != audit['data_fingerprint']):
        raise ValueError('Candidate provenance differs from B0 audit')
    base, candidate = history(base_run/'epochs.jsonl'), history(run/'epochs.jsonl')
    if base != audit['baseline_history']:
        raise ValueError('Archived B0 history changed after audit')
    selected = max(candidate, key=lambda r: r['metrics']['DIV2K_valid']['psnr'])
    selected_base = audit['baseline_selected']
    checkpoint = f'epoch{selected["epoch"]:05d}_u{selected["update"]}.pt'
    if not (run/checkpoint).is_file():
        raise FileNotFoundError(run/checkpoint)
    def average(rows):
        return {name: {m: statistics.mean(r['metrics'][name][m] for r in rows[-3:])
                       for m in ('psnr', 'ssim')} for name in rows[-1]['metrics']}
    report = dict(budget_updates=50000, seed=10, baseline_record='b546036',
                  baseline_checkpoint='epoch00048_u48000.pt', candidate_checkpoint=checkpoint,
                  selection_rule='Max DIV2K_valid PSNR within 50k, earliest tie; one weight for all datasets',
                  validation_selected=metrics_delta(selected_base['metrics'], selected['metrics']),
                  fixed_50000=metrics_delta(base[-1]['metrics'], candidate[-1]['metrics']),
                  last3_evaluation_mean=metrics_delta(average(base), average(candidate)),
                  paired_validation_selected=paired(base_run, selected_base, run, selected),
                  paired_fixed_50000=paired(base_run, base[-1], run, candidate[-1]),
                  scope='single-seed short screen; not final convergence or stability evidence',
                  sharing_reminder='Discuss attention sharing after this experiment. A6 uses fresh QK in all six blocks. '
                                   'If gains persist at fixed/last3 points, validate a repeated seed and consider a matched fresh/shared comparison; '
                                   'if PSNR does not improve, do not use sharing to claim success.')
    args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    args.output.with_suffix('.svg').write_text(curves_svg(base, candidate), encoding='utf8')
    with args.output.with_suffix('.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f); writer.writerow(['update', 'dataset', 'B0_psnr', 'A6_psnr', 'delta_psnr', 'B0_ssim', 'A6_ssim'])
        for a, b in zip(base, candidate):
            for name, values in a['metrics'].items():
                v = b['metrics'][name]
                writer.writerow([a['update'], name, values['psnr'], v['psnr'], v['psnr']-values['psnr'], values['ssim'], v['ssim']])
    lines = ['# A6 50k screening result', '',
             'B0: epoch00048_u48000.pt; A6: '+checkpoint, '',
             '|Dataset|Selected ΔPSNR|Fixed50k ΔPSNR|Last3 ΔPSNR|', '|---|---:|---:|---:|']
    for name in report['fixed_50000']:
        ds = [report[k][name]['delta']['psnr'] for k in ('validation_selected', 'fixed_50000', 'last3_evaluation_mean')]
        lines.append('|'+name+'|'+'|'.join(f'{v:+.6f}' for v in ds)+'|')
    lines += ['', '## 实验后的共享提醒', '',
              '本次六处均独立重算QK，没有共享。先判断固定50k、末3点、完整曲线和逐图结果是否支持涨分；',
              '若有效，再讨论重复种子及fresh/shared同协议对照；若无效，共享省成本不能替代PSNR目标。',
              '单seed短筛不能宣称稳定提升；没有自动判定成功或启动下一轮训练。']
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
