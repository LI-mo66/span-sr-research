"""Task 7: Merge 5-benchmark metrics, update 720p efficiency Avg5 columns, build review ZIP and verified full archive."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from common_sci import (
    ALL_DATASETS,
    BASE,
    OLD_OUT,
    OUT,
    ROOT,
    dataset,
    metrics,
    pair,
    read_json,
    sha,
    write_json,
)
from compute_sci import METHOD_DIR_MAP


def merge():
    rows = []
    seen = {}
    # Order preference: OUT -> OLD_OUT -> ROOT
    for parent in [OUT / 'sr_outputs', OLD_OUT / 'sr_outputs', ROOT / 'sr_outputs']:
        if not parent.is_dir():
            continue
        for p in sorted(parent.glob('*/*/*.png')):
            ds, method, image = p.relative_to(parent).parts
            if method == 'GT':
                continue
            key = (ds, method, image)
            if key in seen:
                if sha(p) != seen[key]:
                    print(f'WARN_DUPLICATE_DIFF {key}: keeping first ({parent})', flush=True)
                continue
            gt, _, inp = pair(ds, image)
            sr = np.array(Image.open(p).convert('RGB'))
            if sr.shape != gt.shape:
                raise ValueError(f'Merged PNG shape mismatch: {p} ({sr.shape} vs {gt.shape})')
            score = metrics(sr, gt)
            h = sha(p)
            seen[key] = h
            rows.append(dict(
                dataset=ds,
                image_name=image,
                method=method,
                psnr_y=round(score['psnr'], 6),
                ssim_y=round(score['ssim'], 6),
                width=sr.shape[1],
                height=sr.shape[0],
                mode='x8' if method == 'ours_x8' else 'single',
                scope='50K_ablation' if '_50k_' in method else 'released_or_final',
                png_sha256=h,
                png=str(p),
                lr_sha256=inp['lr_sha256'],
                hr_sha256=inp['hr_sha256'],
            ))
            if len(rows) % 250 == 0:
                print('MERGE_METRICS', len(rows), flush=True)

    metrics_csv = OUT / 'metrics_per_image.csv'
    with metrics_csv.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    grouped = {}
    for r in rows:
        grouped.setdefault((r['dataset'], r['method']), []).append(r)

    # Full 5-benchmark pivot summary per method
    all_methods = sorted({r['method'] for r in rows})
    summary_rows = []
    method_avg5 = {}
    for m in all_methods:
        rec = dict(method=m)
        psnrs, ssims = [], []
        complete5 = True
        for ds, expected_cnt in ALL_DATASETS:
            items = grouped.get((ds, m), [])
            if len(items) == expected_cnt:
                mp = float(np.mean([x['psnr_y'] for x in items]))
                ms = float(np.mean([x['ssim_y'] for x in items]))
                rec[f'{ds}_PSNR'] = round(mp, 4)
                rec[f'{ds}_SSIM'] = round(ms, 6)
                psnrs.append(mp)
                ssims.append(ms)
            else:
                rec[f'{ds}_PSNR'] = f'({len(items)}/{expected_cnt})' if items else ''
                rec[f'{ds}_SSIM'] = ''
                complete5 = False
        if complete5:
            rec['Avg5_PSNR'] = round(float(np.mean(psnrs)), 4)
            rec['Avg5_SSIM'] = round(float(np.mean(ssims)), 6)
            method_avg5[m] = (rec['Avg5_PSNR'], rec['Avg5_SSIM'])
        else:
            rec['Avg5_PSNR'] = ''
            rec['Avg5_SSIM'] = ''
        summary_rows.append(rec)

    means_csv = OUT / 'dataset_means_5bench.csv'
    with means_csv.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Backfill Avg5_PSNR / Avg5_SSIM into efficiency_720p_rtx4090.csv & json if present
    eff_json_path = OUT / 'efficiency_720p_rtx4090.json'
    if eff_json_path.is_file():
        eff_data = read_json(eff_json_path)
        for r in eff_data['rows']:
            if r['Scale'] == 'x4' and r['Method'] in METHOD_DIR_MAP:
                mdir = METHOD_DIR_MAP[r['Method']]
                if mdir in method_avg5:
                    r['Avg5_PSNR'], r['Avg5_SSIM'] = method_avg5[mdir]
        write_json(eff_json_path, eff_data, overwrite=True)
        eff_csv_path = OUT / 'efficiency_720p_rtx4090.csv'
        cols = ['Method', 'Scale', 'Params_K', 'MACs_G', 'FLOPs_G', 'Latency_ms', 'PeakMem_MB', 'Avg5_PSNR', 'Avg5_SSIM']
        with eff_csv_path.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in eff_data['rows']:
                w.writerow({k: r[k] for k in cols})

    # Collect manifests
    manifests = {}
    for name in ['fig3_v2_manifest.json', 'fig4_manifest.json', 'fig5_manifest.json', 'fig6_manifest.json',
                 'fig6_stats.json', 'fig_ablation_convergence_50k.json', 'efficiency_720p_rtx4090.json']:
        p = OUT / name
        if p.is_file():
            manifests[name] = read_json(p)

    master = dict(
        status='COMPUTED_PENDING_LOCAL_VISUAL_REVIEW_AND_TNR_RERENDER',
        suite_directory=str(OUT),
        metric_rows=len(rows),
        metrics_file=str(metrics_csv),
        dataset_means_file=str(means_csv),
        five_benchmark_means=summary_rows,
        manifests=manifests,
    )
    write_json(OUT / 'manifest.json', master, overwrite=True)

    # Non-destructively publish root entrypoints (never overwrite existing differing files)
    for name in ['metrics_per_image.csv', 'manifest.json']:
        dest = ROOT / name
        if dest.exists() and sha(dest) != sha(OUT / name):
            dest = ROOT / f'{Path(name).stem}-sci-final{Path(name).suffix}'
        shutil.copy2(OUT / name, dest)

    # Write SHA256SUMS.txt for all top-level CSV/JSON & figures
    sha_entries = []
    fig_dir = OUT / 'figures_paper_ready'
    if fig_dir.is_dir():
        for p in sorted(fig_dir.iterdir()):
            if p.is_file() and p.suffix in ['.png', '.pdf']:
                sha_entries.append((p, 'figures_paper_ready/' + p.name))
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.suffix in ['.csv', '.json'] and p.name != 'MERGE_COMPLETE.json':
            sha_entries.append((p, p.name))
    portable_dir = OUT / 'portable_render_assets'
    if portable_dir.is_dir():
        for p in sorted(portable_dir.rglob('*')):
            if p.is_file():
                sha_entries.append((p, 'portable_render_assets/' + p.relative_to(portable_dir).as_posix()))

    sha_lines = [f'{sha(p)}  {arcname}' for p, arcname in sha_entries]
    sha_txt = OUT / 'SHA256SUMS.txt'
    sha_txt.write_text('\n'.join(sha_lines) + '\n', encoding='utf-8')
    sha_entries.append((sha_txt, 'SHA256SUMS.txt'))

    # Build lightweight verified review + portable re-render ZIP
    review = OUT / 'paper-visuals-sci-final-review.zip'
    part = review.with_name(review.name + '.part')
    with zipfile.ZipFile(part, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p, arcname in sha_entries:
            z.write(p, arcname)
    part.replace(review)

    with zipfile.ZipFile(review) as z:
        if z.testzip() is not None:
            raise ValueError('Review ZIP CRC error')
        for p, arcname in sha_entries:
            if hashlib.sha256(z.read(arcname)).hexdigest() != sha(p):
                raise ValueError(f'Review ZIP member mismatch: {arcname}')

    (OUT / 'paper-visuals-sci-final-review.zip.sha256').write_text(
        f'{sha(review)}  {review.name}\n', encoding='utf-8'
    )
    write_json(
        OUT / 'MERGE_COMPLETE.json',
        dict(
            metrics_rows=len(rows),
            review_zip=str(review),
            review_bytes=review.stat().st_size,
            review_sha256=sha(review),
            members_verified=len(sha_entries),
        ),
        overwrite=True,
    )
    print(f'MERGE_COMPLETE rows={len(rows)} zip={review} size={review.stat().st_size/1024**2:.1f}MB', flush=True)


def archive():
    read_json(OUT / 'MERGE_COMPLETE.json')
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest = BASE / 'records' / f'PAPER-VISUALS-SCI-FINAL-{now}'
    dest.mkdir(parents=True, exist_ok=True)
    exclusions = {OUT / 'queue.nohup.log', OUT / 'archive.log', OUT / 'status.txt', OUT / 'suite.pid'}

    entries = []
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p in exclusions:
            continue
        rel_parts = p.relative_to(ROOT).parts
        if any(x in ['.git', '__pycache__', 'deps', 'python_deps', 'ngswin_deps'] for x in rel_parts):
            continue
        if p.suffix in ['.part', '.tmp']:
            continue
        entries.append((p, 'paper_visuals_20261126/' + p.relative_to(ROOT).as_posix()))

    for ds, _ in ALL_DATASETS:
        droot = dataset(ds)
        for rel in ['HR', 'LR_bicubic/X4']:
            for p in sorted((droot / rel).glob('*.png')):
                entries.append((p, f'benchmark/{ds}/{rel}/{p.name}'))

    expected = {arc: dict(bytes=p.stat().st_size, sha256=sha(p)) for p, arc in entries}
    payload = sum(v['bytes'] for v in expected.values())
    if shutil.disk_usage(dest).free < payload + 1024 ** 3:
        raise RuntimeError('Insufficient disk space (< payload + 1GiB) for full server archive')

    part = dest / 'paper-visuals-sci-final.tar.gz.part'
    with tarfile.open(part, 'w:gz', compresslevel=4, dereference=True) as tar:
        for i, (p, arc) in enumerate(entries, 1):
            tar.add(p, arcname=arc, recursive=False)
            if i % 500 == 0:
                print(f'ARCHIVED {i}/{len(entries)}', flush=True)
        raw = json.dumps(expected, ensure_ascii=False, indent=2).encode('utf-8')
        info = tarfile.TarInfo('archive-file-sha256.json')
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))

    checked = 0
    with tarfile.open(part, 'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name == 'archive-file-sha256.json':
                continue
            h = hashlib.sha256()
            stream = tar.extractfile(member)
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
            if member.size != expected[member.name]['bytes'] or h.hexdigest() != expected[member.name]['sha256']:
                raise ValueError('Archive member mismatch: ' + member.name)
            checked += 1
    if checked != len(expected):
        raise ValueError('Archive member count mismatch')

    final = dest / 'paper-visuals-sci-final.tar.gz'
    part.replace(final)
    final_sha = sha(final)
    (dest / 'paper-visuals-sci-final.tar.gz.sha256').write_text(
        f'{final_sha}  {final.name}\n', encoding='utf-8'
    )
    write_json(
        dest / 'archive-verification.json',
        dict(status='PASS', files_verified=checked, bytes=final.stat().st_size, sha256=final_sha,
             note='Server archive verified; download review ZIP and verify locally before releasing instance.'),
        overwrite=True,
    )
    (dest / 'SERVER_BACKUP_COMPLETE').write_text('All payload members verified; no auto shutdown.\n', encoding='utf-8')
    (OUT / 'backup-path.txt').write_text(str(dest) + '\n', encoding='utf-8')
    print(f'SERVER_BACKUP_COMPLETE {dest} size={final.stat().st_size/1024**2:.1f}MB', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['merge', 'archive'])
    args = p.parse_args()
    globals()[args.action]()


if __name__ == '__main__':
    main()
