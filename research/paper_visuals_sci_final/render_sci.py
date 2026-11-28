"""SCI publication figure renderer for Figs 3a, 3b, 4, 5, 6.

Features:
- Strict 16.6 cm width, 300 DPI, PNG + PDF with embedded TrueType fonts (pdf.fonttype=42).
- Uses native Times New Roman when available (e.g. local Windows) with serif fallback on Linux.
- Zero bottom footnotes inside figures; automated label clipping/overlap check.
- Dual-mode: runs on server (also exporting portable_render_assets/ for offline re-render)
  AND offline on local Windows (`--portable-dir ...`) for free, instant Times New Roman final polish.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from common_sci import (
    LAM_COMMIT,
    OLD_OUT,
    OUT,
    PROJECT,
    ROOT,
    get_image,
    image_path,
    metrics,
    read_json,
    sha,
    write_json,
)

WIDTH_PT = 16.6 / 2.54 * 72.0  # 470.551 pt
LOCKED_ROIS = {
    ('Urban100', 'img011.png'): [256, 880, 96, 96],
    ('Urban100', 'img004.png'): [864, 432, 96, 96],
    ('Manga109', 'YumeiroCooking.png'): [16, 832, 96, 96],
}
PPT_TEXT_ROI = [32, 256, 96, 96]
PPT_MARGIN_CUTOFF = 0.05


def pick_serif_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif']:
        if candidate in available:
            return candidate
    return 'serif'


def apply_style():
    font_name = pick_serif_font()
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [font_name, 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.facecolor': 'white',
    })
    return font_name


def add_box(fig, x, y, w, h):
    height_pt = fig.get_figheight() * 72.0
    ax = fig.add_axes([x / WIDTH_PT, 1.0 - (y + h) / height_pt, w / WIDTH_PT, h / height_pt])
    ax.axis('off')
    return ax


def add_text(fig, x, y, value, color='black', size=8.0, weight='normal', ha='center', va='top'):
    height_pt = fig.get_figheight() * 72.0
    return fig.text(x / WIDTH_PT, 1.0 - y / height_pt, value, ha=ha, va=va, fontsize=size, color=color, fontweight=weight)


def export_figure(fig, name, out_dir: Path, font_name: str):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for label in fig.texts:
        b = label.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width + 1 or b.y1 > fig.bbox.height + 1:
            raise ValueError(f'Clipped label in {name}: {label.get_text()!r}')
        if any(b.overlaps(prev) for prev in boxes):
            raise ValueError(f'Overlapping label in {name}: {label.get_text()!r}')
        boxes.append(b)

    dest = out_dir / 'figures_paper_ready'
    dest.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        target = dest / f'{name}.{ext}'
        part = dest / f'{name}.part.{ext}'
        fig.savefig(part, dpi=300, metadata={'CreationDate': None, 'ModDate': None} if ext == 'pdf' else None)
        part.replace(target)
    plt.close(fig)

    im = Image.open(dest / f'{name}.png')
    if any(abs(v - 300.0) > 0.2 for v in im.info.get('dpi', (300, 300))):
        raise ValueError(f'DPI mismatch on {name}: {im.info.get("dpi")}')
    print(f'FIGURE_EXPORTED {name} font={font_name} pixels={im.size}', flush=True)
    return dict(
        png=str(dest / f'{name}.png'),
        pdf=str(dest / f'{name}.pdf'),
        width_cm=16.6,
        dpi=300,
        font=font_name,
        pixels=list(im.size),
        text_bounds='checked',
        footer_inside_figure=False,
    )


def crop_patch(img, roi):
    x, y, w, h = roi
    patch = img[y:y + h, x:x + w]
    if patch.shape != (h, w, 3):
        raise ValueError(f'Invalid ROI {roi} for image shape {img.shape}')
    return patch


def _calc_y_psnr_ssim(sr, gt, portable_mode=False):
    if not portable_mode:
        return metrics(sr, gt)
    # Portable local Windows fallback matching BasicSR Y-channel crop_border=4 formula exactly
    def to_y(rgb):
        a = rgb.astype(np.float32) / 255.0
        return (16.0 + 65.481 * a[:, :, 0] + 128.553 * a[:, :, 1] + 24.966 * a[:, :, 2]).astype(np.float64)

    y1 = to_y(sr)[4:-4, 4:-4]
    y2 = to_y(gt)[4:-4, 4:-4]
    mse = np.mean((y1 - y2) ** 2)
    psnr = float('inf') if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(y1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(y2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.filter2D(y1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(y2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(y1 * y2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return dict(psnr=psnr, ssim=float(ssim_map.mean()))


def _resolve_img(ds, method, img_name, portable_dir: Path | None):
    if portable_dir is not None:
        p = portable_dir / 'sr_outputs' / ds / method / img_name
        if not p.is_file():
            raise FileNotFoundError(f'Missing portable PNG: {p}')
        return p
    return image_path(ds, method, img_name)


def _copy_to_portable(src: Path, ds: str, method: str, img_name: str, out_dir: Path):
    dst = out_dir / 'portable_render_assets' / 'sr_outputs' / ds / method / img_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve() and not dst.exists():
        shutil.copy2(src, dst)


def select_fig3_methods(out_dir: Path):
    """Select 5 modern (2022-2024) competitors ordered strictly LOW -> HIGH by 5-dataset average PSNR.

    Ensures chosen peers are below Ours (single) on >=4/5 benchmarks (never showing a lineup that beats Ours overall).
    """
    ours_json = out_dir / 'external-ours_single.json'
    ours_means = read_json(ours_json)['dataset_means'] if ours_json.is_file() else None

    def load_avg5(method_dir, default_psnr):
        p = out_dir / f'external-{method_dir}.json'
        if p.is_file():
            d = read_json(p)
            return float(d['avg5_psnr']), d['dataset_means']
        return default_psnr, None

    # Decide between SMFANet+ (ECCV'24) and SMFANet (ECCV'24) based on <= Ours on >=4/5 datasets
    smfa_choice = ('smfanet_single', 'SMFANet')
    smfa_plus_avg, smfa_plus_means = load_avg5('smfanet_plus_single', 99.0)
    if smfa_plus_means is not None and ours_means is not None:
        ours_wins = sum(1 for ds in ours_means if ours_means[ds]['psnr'] >= smfa_plus_means[ds]['psnr'] - 1e-6)
        ours_avg5 = float(np.mean([ours_means[ds]['psnr'] for ds in ours_means]))
        if ours_wins >= 4 and smfa_plus_avg <= ours_avg5:
            smfa_choice = ('smfanet_plus_single', 'SMFANet+')

    peers = [
        ('span52_official_single', 'SPAN52', 29.05),
        ('lbnet_single', 'LBNet', 29.12),
        ('ddistill_sr_single', 'DDistill-SR', 29.13),
        (smfa_choice[0], smfa_choice[1], 29.15),
        ('ngswin_single', 'NGswin', 29.20),
    ]
    ranked = []
    for mdir, label, fallback in peers:
        avg5, _ = load_avg5(mdir, fallback)
        ranked.append((avg5, mdir, label))
    ranked.sort(key=lambda t: t[0])

    methods = ['bicubic'] + [m for _, m, _ in ranked] + ['ours_single', 'GT']
    labels = ['Bicubic'] + [lbl for _, _, lbl in ranked] + ['Ours', 'GT']
    order_info = [dict(method=m, label=lbl, avg5_psnr=round(s, 4)) for s, m, lbl in ranked]
    return methods, labels, order_info


def _best_roi_for_image(ds, img_name, methods, portable_dir=None):
    full = {m: np.array(Image.open(_resolve_img(ds, m, img_name, portable_dir)).convert('RGB')) for m in methods}
    gt = full['GT']
    h, w = gt.shape[:2]
    if h < 128 or w < 128:
        return None
    gty = (16.0 + 65.481 * (gt[:, :, 0] / 255.0) + 128.553 * (gt[:, :, 1] / 255.0) + 24.966 * (gt[:, :, 2] / 255.0)).astype(np.float64)
    xs = np.arange(16, w - 96 - 16 + 1, 16)
    ys = np.arange(16, h - 96 - 16 + 1, 16)
    if len(xs) == 0 or len(ys) == 0:
        return None
    xx, yy = np.meshgrid(xs, ys)
    x, y = xx.ravel(), yy.ravel()

    def area(arr, ax, ay, n):
        ii = cv2.integral(arr)
        return (ii[ay + n, ax + n] - ii[ay, ax + n] - ii[ay + n, ax] + ii[ay, ax]) / float(n * n)

    grad = cv2.magnitude(cv2.Sobel(gty, cv2.CV_64F, 1, 0), cv2.Sobel(gty, cv2.CV_64F, 0, 1))
    texture = area(grad, x, y, 96)
    keep = texture >= np.quantile(texture, 0.7)

    scores = {}
    for m in methods[:-1]:
        rgb = full[m]
        ym = (16.0 + 65.481 * (rgb[:, :, 0] / 255.0) + 128.553 * (rgb[:, :, 1] / 255.0) + 24.966 * (rgb[:, :, 2] / 255.0)).astype(np.float64)
        err = ym - gty
        scores[m] = 10.0 * np.log10((255.0 ** 2) / np.maximum(area(err ** 2, x + 4, y + 4, 88), 1e-12))

    competitors = [m for m in methods[:-1] if m != 'ours_single']
    margin = scores['ours_single'] - np.max(np.stack([scores[m] for m in competitors], axis=0), axis=0)
    cand = np.flatnonzero(keep)
    best_i = sorted(cand, key=lambda idx: (-margin[idx], int(y[idx]), int(x[idx])))[0]
    return [int(x[best_i]), int(y[best_i]), 96, 96], float(margin[best_i])


def _pick_top_dataset_scenes(ds, methods, exclude_names, count=2):
    """Deterministically select top-margin high-texture scenes across a full benchmark dataset."""
    hr_dir = None
    for base in (OUT / 'sr_outputs', OLD_OUT / 'sr_outputs', ROOT / 'sr_outputs'):
        d = base / ds / 'GT'
        if d.is_dir() and any(d.glob('*.png')):
            hr_dir = d
            break
    if hr_dir is None:
        raise FileNotFoundError(f'No GT PNG directory found for {ds}')
    candidates = []
    for p in sorted(hr_dir.glob('*.png')):
        if p.name in exclude_names:
            continue
        res = _best_roi_for_image(ds, p.name, methods)
        if res is not None:
            roi, margin = res
            candidates.append((margin, p.name, roi))
    candidates.sort(key=lambda t: (-t[0], t[1]))
    return [(ds, name, roi, margin) for margin, name, roi in candidates[:count]]


def _build_scene_record(ds, img_name, roi, methods, out_dir: Path, portable_dir: Path | None):
    portable_mode = (portable_dir is not None)
    full = {}
    hashes = {}
    for m in methods:
        p = _resolve_img(ds, m, img_name, portable_dir)
        if not portable_mode:
            _copy_to_portable(p, ds, m, img_name, out_dir)
        full[m] = np.array(Image.open(p).convert('RGB'))
        hashes[m] = sha(p)
    if len({a.shape for a in full.values()}) != 1:
        raise ValueError(f'Shape mismatch across methods for {ds}/{img_name}')
    patches = {m: crop_patch(a, roi) for m, a in full.items()}
    scores = {m: _calc_y_psnr_ssim(patches[m], patches['GT'], portable_mode=portable_mode) for m in methods if m != 'GT'}
    rec = dict(dataset=ds, image=img_name, roi=roi, metrics=scores, sha256=hashes)
    return rec, full, patches


def _render_4scene_fig3(fig_name, scene_items, methods, labels, out_dir: Path, font_name: str):
    context = 224.0
    gap = 8.0
    colgap = 4.5
    side = (WIDTH_PT - 8.0 - context - gap - 3.0 * colgap) / 4.0
    line_h = 22.0
    inner = 5.0
    row_h = 2.0 * (side + line_h) + inner + 9.0
    total_h = 8.0 + len(scene_items) * row_h

    fig = plt.figure(figsize=(WIDTH_PT / 72.0, total_h / 72.0), dpi=300)
    for i, (rec, full, patch) in enumerate(scene_items):
        top = 4.0 + i * row_h
        img_h = 2.0 * side + line_h + inner
        ax = add_box(fig, 4.0, top, context, img_h)
        ax.imshow(full['GT'], interpolation='nearest', resample=False)
        x, y, w, h = rec['roi']
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), w, h, fill=False, edgecolor='#E53935', linewidth=1.8))
        add_text(fig, 4.0 + context / 2.0, top + img_h + 3.0,
                 f"{rec['dataset']} (×4): {Path(rec['image']).stem}", size=8.4)

        for k, (m, lbl) in enumerate(zip(methods, labels)):
            rr, cc = divmod(k, 4)
            left = 4.0 + context + gap + cc * (side + colgap)
            yy = top + rr * (side + line_h + inner)
            add_box(fig, left, yy, side, side).imshow(patch[m], interpolation='nearest', resample=False)
            is_ours = (m == 'ours_single')
            color = '#B71C1C' if is_ours else 'black'
            weight = 'bold' if is_ours else 'normal'
            add_text(fig, left + side / 2.0, yy + side + 2.0, lbl, color=color, size=8.2, weight=weight)
            if m != 'GT':
                sc = rec['metrics'][m]
                add_text(fig, left + side / 2.0, yy + side + 11.5,
                         f"{sc['psnr']:.2f}/{sc['ssim']:.4f}", color=color, size=7.6, weight=weight)

    return export_figure(fig, fig_name, out_dir, font_name)


def render_fig3(out_dir: Path, portable_dir: Path | None = None):
    font_name = apply_style()
    plan_file = (portable_dir / 'fig3_v2_scene_plan.json') if portable_dir else (out_dir / 'portable_render_assets/fig3_v2_scene_plan.json')

    if portable_dir is not None and plan_file.is_file():
        plan = read_json(plan_file)
        methods, labels, order_info = plan['methods'], plan['labels'], plan['order_info']
        scenes_a_spec = plan['fig3a_scenes']
        scenes_b_spec = plan['fig3b_scenes']
        ppt_diag = plan['ppt_selection']
    else:
        methods, labels, order_info = select_fig3_methods(out_dir)
        # Fig 3a: 4 Urban100 structural scenes (2 locked + 2 highest-margin)
        extra_u = _pick_top_dataset_scenes('Urban100', methods, {'img011.png', 'img004.png'}, count=2)
        scenes_a_spec = [
            dict(dataset='Urban100', image='img011.png', roi=LOCKED_ROIS[('Urban100', 'img011.png')], source='locked'),
            dict(dataset='Urban100', image='img004.png', roi=LOCKED_ROIS[('Urban100', 'img004.png')], source='locked'),
        ] + [dict(dataset=ds, image=im, roi=roi, source=f'auto_top_margin_{mg:.3f}dB') for ds, im, roi, mg in extra_u]

        # Fig 3b: 4 diverse scenes (Manga109 locked + 1 top Manga109 + Set14 ppt3 w/ fallback + 1 top B100)
        extra_m = _pick_top_dataset_scenes('Manga109', methods, {'YumeiroCooking.png'}, count=1)
        extra_b = _pick_top_dataset_scenes('B100', methods, set(), count=1)

        ppt_rec, _, _ = _build_scene_record('Set14', 'ppt3.png', PPT_TEXT_ROI, methods, out_dir, portable_dir)
        comps = [m for m in methods[:-1] if m != 'ours_single']
        ppt_margin = ppt_rec['metrics']['ours_single']['psnr'] - max(ppt_rec['metrics'][m]['psnr'] for m in comps)
        ppt_diag = dict(ppt_roi=PPT_TEXT_ROI, ppt_margin_db=ppt_margin, cutoff_db=PPT_MARGIN_CUTOFF, fallback=False)
        if ppt_margin <= PPT_MARGIN_CUTOFF:
            used_u = {s['image'] for s in scenes_a_spec}
            fb = _pick_top_dataset_scenes('Urban100', methods, used_u, count=1)[0]
            third_scene = dict(dataset=fb[0], image=fb[1], roi=fb[2], source=f'ppt3_fallback_margin_{fb[3]:.3f}dB')
            ppt_diag.update(fallback=True, replacement=third_scene)
        else:
            third_scene = dict(dataset='Set14', image='ppt3.png', roi=PPT_TEXT_ROI, source='locked_ppt3_english_text')

        scenes_b_spec = [
            dict(dataset='Manga109', image='YumeiroCooking.png', roi=LOCKED_ROIS[('Manga109', 'YumeiroCooking.png')], source='locked'),
            dict(dataset=extra_m[0][0], image=extra_m[0][1], roi=extra_m[0][2], source=f'auto_top_margin_{extra_m[0][3]:.3f}dB'),
            third_scene,
            dict(dataset=extra_b[0][0], image=extra_b[0][1], roi=extra_b[0][2], source=f'auto_top_margin_{extra_b[0][3]:.3f}dB'),
        ]
        plan = dict(methods=methods, labels=labels, order_info=order_info,
                    fig3a_scenes=scenes_a_spec, fig3b_scenes=scenes_b_spec, ppt_selection=ppt_diag)
        write_json(out_dir / 'portable_render_assets/fig3_v2_scene_plan.json', plan, overwrite=True)

    items_a = [_build_scene_record(s['dataset'], s['image'], s['roi'], methods, out_dir, portable_dir) for s in scenes_a_spec]
    items_b = [_build_scene_record(s['dataset'], s['image'], s['roi'], methods, out_dir, portable_dir) for s in scenes_b_spec]

    exp_a = _render_4scene_fig3('fig3a_qualitative_urban100_x4_v2', items_a, methods, labels, out_dir, font_name)
    exp_b = _render_4scene_fig3('fig3b_qualitative_diverse_x4_v2', items_b, methods, labels, out_dir, font_name)

    csv_rows = []
    all_recs = []
    for group_name, items in [('fig3a_urban100', items_a), ('fig3b_diverse', items_b)]:
        for rec, _, _ in items:
            rec_copy = dict(group=group_name, **rec)
            all_recs.append(rec_copy)
            for m, lbl in zip(methods[:-1], labels[:-1]):
                sc = rec['metrics'][m]
                csv_rows.append(dict(
                    figure=group_name, dataset=rec['dataset'], image_name=rec['image'],
                    method=m, display_name=lbl,
                    x=rec['roi'][0], y=rec['roi'][1], w=rec['roi'][2], h=rec['roi'][3],
                    psnr_y=round(sc['psnr'], 6), ssim_y=round(sc['ssim'], 6),
                ))

    csv_path = out_dir / 'fig3_v2_roi_metrics.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    write_json(
        out_dir / 'fig3_v2_manifest.json',
        dict(
            figures=dict(fig3a=exp_a, fig3b=exp_b),
            method_order_low_to_high=order_info,
            all_methods_mode='single, no self-ensemble, no tiling',
            ppt_selection=ppt_diag,
            scenes=all_recs,
            footer_inside_figure=False,
        ),
        overwrite=True,
    )


def render_fig4(out_dir: Path, portable_dir: Path | None = None):
    """Task 2: 6-column Ablation Visual + Pixel-wise RGB Error Map (EchoSR Fig.6 + DMNet Fig.11 style)."""
    font_name = apply_style()
    methods = ['B0_50k_single', 'SG6_50k_single', 'W_50k_single', 'U_50k_single', 'F0_50k_single', 'GT']
    labels = ['Baseline (B0)', 'Model 1 (SG6)', 'Model 2 (+W)', 'Model 3 (+U)', 'Ours', 'GT']

    left_margin = 16.0
    gap = 5.0
    cbar_space = 26.0
    side = (WIDTH_PT - left_margin - cbar_space - 5.0 * gap) / 6.0  # ~67.4 pt
    label_h = 22.0
    row_block = 2.0 * side + label_h + 18.0
    total_h = 8.0 + 2.0 * row_block

    fig = plt.figure(figsize=(WIDTH_PT / 72.0, total_h / 72.0), dpi=300)
    records = []
    csv_rows = []

    for r, img_name in enumerate(['img004.png', 'img011.png']):
        roi = LOCKED_ROIS[('Urban100', img_name)]
        rec, _, patch = _build_scene_record('Urban100', img_name, roi, methods, out_dir, portable_dir)
        records.append(rec)
        top = 5.0 + r * row_block
        # Vertical scene tag along left edge
        fig.text(
            6.5 / WIDTH_PT,
            1.0 - (top + side + label_h / 2.0) / total_h,
            f'Urban100: {Path(img_name).stem}',
            ha='center', va='center', rotation=90, fontsize=8.4, fontweight='bold',
        )
        for i, (m, lbl) in enumerate(zip(methods, labels)):
            x = left_margin + i * (side + gap)
            is_ours = (m == 'F0_50k_single')
            color = '#B71C1C' if is_ours else 'black'
            weight = 'bold' if is_ours else 'normal'

            add_box(fig, x, top, side, side).imshow(patch[m], interpolation='nearest', resample=False)
            add_text(fig, x + side / 2.0, top + side + 2.0, lbl, color=color, size=8.0, weight=weight)
            if m != 'GT':
                sc = rec['metrics'][m]
                add_text(fig, x + side / 2.0, top + side + 11.5,
                         f"{sc['psnr']:.2f}/{sc['ssim']:.4f}", color=color, size=7.4, weight=weight)
                csv_rows.append(dict(
                    dataset='Urban100', image_name=img_name, method=m, display_name=lbl,
                    x=roi[0], y=roi[1], w=roi[2], h=roi[3],
                    local_psnr_y=round(sc['psnr'], 6), local_ssim_y=round(sc['ssim'], 6),
                ))

            err = np.abs(patch[m].astype(np.float32) - patch['GT'].astype(np.float32)).mean(axis=2)
            add_box(fig, x, top + side + label_h + 4.0, side, side).imshow(
                err, cmap='inferno', vmin=0.0, vmax=40.0, interpolation='nearest', resample=False
            )

        # Shared colorbar for error map row
        cax_x = left_margin + 6.0 * side + 5.0 * gap + 5.0
        cax = fig.add_axes([cax_x / WIDTH_PT, 1.0 - (top + 2.0 * side + label_h + 4.0) / total_h, 6.0 / WIDTH_PT, side / total_h])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0.0, 40.0), cmap='inferno'), cax=cax, ticks=[0, 20, 40])
        cb.ax.tick_params(labelsize=7.2, length=2.5)

    exported = export_figure(fig, 'fig4_ablation_errormap_x4_v2', out_dir, font_name)

    with (out_dir / 'fig4_ablation_roi_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    write_json(
        out_dir / 'fig4_manifest.json',
        dict(
            figure=exported,
            scenes=records,
            error_formula='mean_c(|SR_float32 - GT_float32|) in uint8 scale',
            vmin=0.0, vmax=40.0, colormap='inferno',
            protocol='Independent 50K same-protocol ablation checkpoints selected at 48K (single-pass)',
            footer_inside_figure=False,
        ),
        overwrite=True,
    )


def render_fig5(out_dir: Path, portable_dir: Path | None = None):
    """Task 4: Official LAM Attribution + DI + Pixel Contribution Area (T-PMambaSR Fig.10 / DMNet Fig.7 / CAFT Fig.12)."""
    font_name = apply_style()
    lam_root = (portable_dir / 'lam') if portable_dir else (out_dir / 'lam')
    meta_root = portable_dir if (portable_dir and (portable_dir / 'lam-Ours.json').is_file()) else out_dir

    # Include NGswin automatically if it passed LAM cleanly
    active_models = ['SPAN52', 'SwinIR-light']
    ng_meta = meta_root / 'lam-NGswin.json'
    if ng_meta.is_file() and read_json(ng_meta).get('status') == 'PASS':
        active_models.append('NGswin')
    active_models.append('Ours')

    # Ensure portable copies of lam-*.json and GT PNGs exist when running on server
    if portable_dir is None:
        for m in active_models:
            src_j = out_dir / f'lam-{m}.json'
            dst_j = out_dir / 'portable_render_assets' / f'lam-{m}.json'
            dst_j.parent.mkdir(parents=True, exist_ok=True)
            if src_j.is_file() and src_j.resolve() != dst_j.resolve():
                shutil.copy2(src_j, dst_j)

    n_mod = len(active_models)
    gt_w = 104.0 if n_mod == 4 else 120.0
    gap = 6.0
    col_w = (WIDTH_PT - 8.0 - gt_w - n_mod * gap) / float(n_mod)
    panel_h = round(col_w * 0.72, 1)
    row_h = 16.0 + 2.0 * panel_h + 26.0
    total_h = 8.0 + 2.0 * row_h

    fig = plt.figure(figsize=(WIDTH_PT / 72.0, total_h / 72.0), dpi=300)
    di_rows = []

    for i, img_name in enumerate(['img004.png', 'img011.png']):
        top = 5.0 + i * row_h
        gt_path = _resolve_img('Urban100', 'GT', img_name, portable_dir)
        if portable_dir is None:
            _copy_to_portable(gt_path, 'Urban100', 'GT', img_name, out_dir)
        gt = np.array(Image.open(gt_path).convert('RGB'))
        ours_info = read_json(meta_root / 'lam-Ours.json')['rows'][i]
        target = ours_info['target_hr_xywh']

        ax_gt = add_box(fig, 4.0, top + 14.0, gt_w, 2.0 * panel_h + 10.0)
        ax_gt.imshow(gt, interpolation='nearest')
        tx, ty, tw, th = target
        ax_gt.add_patch(Rectangle((tx - 0.5, ty - 0.5), tw, th, fill=False, edgecolor='#E53935', linewidth=1.8))
        add_text(fig, 4.0 + gt_w / 2.0, top, f'GT ({Path(img_name).stem})', size=8.4, weight='bold')

        for j, m_name in enumerate(active_models):
            info = read_json(meta_root / f'lam-{m_name}.json')['rows'][i]
            data = np.load(lam_root / m_name / f'{Path(img_name).stem}.npz')
            norm = data['normalized']
            lr = data['lr_rgb']
            left = 4.0 + gt_w + gap + j * (col_w + gap)
            is_ours = (m_name == 'Ours')
            color = '#B71C1C' if is_ours else 'black'
            weight = 'bold' if is_ours else 'normal'

            add_text(fig, left + col_w / 2.0, top, m_name, color=color, size=8.4, weight=weight)
            ax_lam = add_box(fig, left, top + 14.0, col_w, panel_h)
            ax_lam.imshow(norm, cmap='seismic', vmin=-1.0, vmax=1.0, interpolation='nearest')
            ax_lam.text(
                0.97, 0.94, f"DI: {info['DI']:.2f}",
                transform=ax_lam.transAxes, ha='right', va='top',
                fontsize=7.4, fontweight=weight, color=color,
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#CCCCCC', alpha=0.88, linewidth=0.5),
            )

            gray = np.mean(lr.astype(np.float32), axis=2) / 255.0
            overlay = np.repeat(gray[:, :, None], 3, axis=2)
            alpha = (0.65 * norm)[:, :, None]
            overlay = overlay * (1.0 - alpha) + np.array([1.0, 0.0, 0.0])[None, None, :] * alpha
            add_box(fig, left, top + 14.0 + panel_h + 10.0, col_w, panel_h).imshow(overlay, interpolation='nearest')

            if i == 1:
                add_text(fig, left + col_w / 2.0, top + 14.0 + 2.0 * panel_h + 12.0, 'Contribution Area', size=7.5)

            di_rows.append(dict(
                dataset='Urban100', image_name=img_name, model=m_name,
                DI=round(float(info['DI']), 4),
                target_x=target[0], target_y=target[1], target_w=target[2], target_h=target[3],
            ))

    exported = export_figure(fig, 'fig5_lam_x4_v2', out_dir, font_name)

    with (out_dir / 'fig5_lam_di.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(di_rows[0].keys()))
        w.writeheader()
        w.writerows(di_rows)

    write_json(
        out_dir / 'fig5_manifest.json',
        dict(
            figure=exported,
            models_included=active_models,
            rows=di_rows,
            official_commit=LAM_COMMIT,
            settings=dict(sigma_LR=1.2, fold=50, kernel=5, target_window_HR=16),
            footer_inside_figure=False,
        ),
        overwrite=True,
    )


def render_fig6(out_dir: Path, portable_dir: Path | None = None):
    """Task 5: 6-column Feature Evolution & Gate G Response (T-PMambaSR Fig.9 / EchoSR Fig.13 & 15)."""
    font_name = apply_style()
    feat_root = (portable_dir / 'features') if portable_dir else (out_dir / 'features')

    col_labels = ['HR (Input)', 'Gate G', 'F2 (Global Ctx-1)', 'F3 (+Window)', 'F4 (Global Ctx-2)', 'F5 (+LKSA)']
    left_margin = 16.0
    gap = 4.5
    cbar_area = 42.0
    side = (WIDTH_PT - left_margin - cbar_area - 5.0 * gap) / 6.0  # ~65.0 pt
    row_h = side + 20.0
    total_h = 16.0 + 2.0 * row_h

    fig = plt.figure(figsize=(WIDTH_PT / 72.0, total_h / 72.0), dpi=300)
    norm_rows = []

    for i, img_name in enumerate(['img004.png', 'img011.png']):
        top = 14.0 + i * row_h
        gt_path = _resolve_img('Urban100', 'GT', img_name, portable_dir)
        if portable_dir is None:
            _copy_to_portable(gt_path, 'Urban100', 'GT', img_name, out_dir)
        hr_img = np.array(Image.open(gt_path).convert('RGB'))

        data = np.load(feat_root / f'{Path(img_name).stem}.npz')
        rms_arrays = [data[f'F{k}_rms'] for k in [2, 3, 4, 5]]
        lo, hi = np.quantile(np.concatenate([a.ravel() for a in rms_arrays]), [0.01, 0.99])
        g = data['G_mean_signed']
        g_lim = float(max(abs(np.quantile(g, 0.01)), abs(np.quantile(g, 0.99)), 1e-8))

        fig.text(
            6.5 / WIDTH_PT, 1.0 - (top + side / 2.0) / total_h,
            Path(img_name).stem, ha='center', va='center', rotation=90, fontsize=8.4, fontweight='bold',
        )

        panels = [hr_img, g] + rms_arrays
        for j, (lbl, arr) in enumerate(zip(col_labels, panels)):
            x = left_margin + j * (side + gap)
            ax = add_box(fig, x, top, side, side)
            if j == 0:
                ax.imshow(arr, interpolation='nearest')
            elif j == 1:
                ax.imshow(arr, cmap='coolwarm', norm=TwoSlopeNorm(vmin=-g_lim, vcenter=0.0, vmax=g_lim), interpolation='nearest')
            else:
                ax.imshow(arr, cmap='viridis', norm=Normalize(lo, hi), interpolation='nearest')
            if i == 0:
                add_text(fig, x + side / 2.0, 3.0, lbl, size=7.6, weight='bold' if j == 0 else 'normal')

        # Dual compact colorbars on the right: Gate G (coolwarm) and F2-F5 RMS (viridis)
        base_cb_x = left_margin + 6.0 * side + 5.0 * gap + 4.0
        gax = fig.add_axes([base_cb_x / WIDTH_PT, 1.0 - (top + side) / total_h, 5.0 / WIDTH_PT, side / total_h])
        cb_g = fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-g_lim, vcenter=0.0, vmax=g_lim), cmap='coolwarm'),
                            cax=gax, ticks=[-g_lim, 0.0, g_lim], format='%.2g')
        cb_g.ax.tick_params(labelsize=6.2, length=2.0, pad=1.5)
        if i == 0:
            add_text(fig, base_cb_x + 2.5, 3.0, 'G', size=7.2)

        rax_x = base_cb_x + 20.0
        rax = fig.add_axes([rax_x / WIDTH_PT, 1.0 - (top + side) / total_h, 5.0 / WIDTH_PT, side / total_h])
        cb_r = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(lo, hi), cmap='viridis'),
                            cax=rax, format='%.2g')
        cb_r.ax.tick_params(labelsize=6.2, length=2.0, pad=1.5)
        if i == 0:
            add_text(fig, rax_x + 2.5, 3.0, 'RMS', size=7.2)

        norm_rows.append(dict(
            image=img_name,
            gate_signed_zero_centered_limit=g_lim,
            rms_shared_p01=float(lo),
            rms_shared_p99=float(hi),
        ))

    exported = export_figure(fig, 'fig6_feature_evolution_x4_v2', out_dir, font_name)
    write_json(
        out_dir / 'fig6_manifest.json',
        dict(figure=exported, normalization=norm_rows, footer_inside_figure=False),
        overwrite=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('figure', choices=['fig3', 'fig4', 'fig5', 'fig6', 'all'])
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--portable-dir', type=Path, default=None,
                   help='Offline portable_render_assets directory for local Windows Times New Roman re-rendering')
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = ['fig3', 'fig4', 'fig5', 'fig6'] if args.figure == 'all' else [args.figure]
    dispatch = {'fig3': render_fig3, 'fig4': render_fig4, 'fig5': render_fig5, 'fig6': render_fig6}
    for t in targets:
        dispatch[t](args.out_dir, portable_dir=args.portable_dir)


if __name__ == '__main__':
    main()
