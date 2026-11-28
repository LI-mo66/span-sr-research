"""Task 3: 50K ablation validation PSNR convergence curve (DMNet Fig.9a / EchoSR Fig.14 style).

Runs identically on the Linux server (reading the 5 historical 50K runs and copying their
epochs.jsonl into portable_render_assets/) and on local Windows (re-rendering from the
copied logs with native Times New Roman embedded fonts).
"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(os.environ.get('PAPER_BASE', '/home/featurize/work/span-x4-research'))
ROOT = Path(os.environ.get('PAPER_ROOT', str(BASE / 'records/paper_visuals_20261126')))
OUT = Path(os.environ.get('PAPER_OUT', str(ROOT / 'step11-paper-sci-final-01')))

RUNS = {
    'B0': 'FEAT-MOEFS-G4-LR2E4-20K-01-1791983354063378494/runs/FEAT-MOEFS-G4-LR2E4-20K-01-B0-s10',
    'SG6': 'FEAT-LG6-SG6-50K-01-1792264356078917269/runs/FEAT-LG6-SG6-50K-01-LG6-SiLU-SG6-s10',
    'W': 'FEAT-SG6-MID-W16-PC26-AC-LR2E4-50K-01-INTERRUPTED-20261026T033650Z-20587/runs/FEAT-SG6-MID-W16-PC26-AC-LR2E4-50K-01-A-s10',
    'U': 'FEAT-SG6-UCAN-P24-LKSA5-V1-50K-01-20261025T125002Z/runs/FEAT-SG6-UCAN-P24-LKSA5-V1-50K-01',
    'F0': 'FEAT-SG6-W-UCAN-P24-LKSA5-FUSION-V1-50K-01-1793142163941762977/runs/FEAT-SG6-W-UCAN-P24-LKSA5-FUSION-V1-50K-01',
}

CURVE_SPEC = [
    ('B0', 'Baseline (B0)', '#6B7280', 1.45, 2),
    ('SG6', 'Model 1 (SG6)', '#2563EB', 1.55, 3),
    ('W', 'Model 2 (+W)', '#059669', 1.55, 4),
    ('U', 'Model 3 (+U)', '#D97706', 1.55, 5),
    ('F0', 'Ours', '#DC2626', 2.25, 6),
]

ALL_DATASETS = ['DIV2K_valid', 'Set5', 'Set14', 'B100', 'Urban100', 'Manga109']


def pick_serif_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif']:
        if candidate in available:
            return candidate
    return 'serif'


def load_epochs(logs_dir: Path | None, out_dir: Path):
    portable_dir = out_dir / 'portable_render_assets' / 'ablation_logs'
    portable_dir.mkdir(parents=True, exist_ok=True)
    series = {}
    for key, rel in RUNS.items():
        if logs_dir is not None:
            src = logs_dir / f'{key}.jsonl'
        else:
            src = BASE / 'records' / rel / 'epochs.jsonl'
        if not src.is_file():
            raise FileNotFoundError(f'Missing epochs.jsonl for {key}: {src}')
        dst = portable_dir / f'{key}.jsonl'
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        rows = [json.loads(line) for line in src.read_text(encoding='utf-8').splitlines() if line.strip()]
        eligible = sorted([r for r in rows if 1000 <= int(r['update']) <= 50000], key=lambda r: int(r['update']))
        updates = [int(r['update']) for r in eligible]
        if updates != list(range(1000, 50001, 1000)):
            raise ValueError(f'Incomplete 1K-50K schedule for {key}: got {len(updates)} points')
        series[key] = eligible
    return series


def export_csvs(series, out_dir: Path):
    updates = list(range(1000, 50001, 1000))
    main_csv = out_dir / 'ablation_50k_curves.csv'
    with main_csv.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['update', 'B0', 'SG6', 'W', 'U', 'Ours'])
        for idx, upd in enumerate(updates):
            writer.writerow([
                upd,
                f"{series['B0'][idx]['metrics']['DIV2K_valid']['psnr']:.6f}",
                f"{series['SG6'][idx]['metrics']['DIV2K_valid']['psnr']:.6f}",
                f"{series['W'][idx]['metrics']['DIV2K_valid']['psnr']:.6f}",
                f"{series['U'][idx]['metrics']['DIV2K_valid']['psnr']:.6f}",
                f"{series['F0'][idx]['metrics']['DIV2K_valid']['psnr']:.6f}",
            ])

    all6_csv = out_dir / 'ablation_50k_curves_all6.csv'
    with all6_csv.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        header = ['update', 'dataset', 'B0_psnr', 'SG6_psnr', 'W_psnr', 'U_psnr', 'Ours_psnr',
                  'B0_ssim', 'SG6_ssim', 'W_ssim', 'U_ssim', 'Ours_ssim']
        writer.writerow(header)
        for ds in ALL_DATASETS:
            for idx, upd in enumerate(updates):
                if all(ds in series[k][idx]['metrics'] for k in ['B0', 'SG6', 'W', 'U', 'F0']):
                    writer.writerow([
                        upd, ds,
                        *[f"{series[k][idx]['metrics'][ds]['psnr']:.6f}" for k in ['B0', 'SG6', 'W', 'U', 'F0']],
                        *[f"{series[k][idx]['metrics'][ds]['ssim']:.6f}" for k in ['B0', 'SG6', 'W', 'U', 'F0']],
                    ])
    return main_csv, all6_csv


def render_convergence(series, out_dir: Path):
    font_name = pick_serif_font()
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [font_name, 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.facecolor': 'white',
        'axes.linewidth': 0.85,
    })

    width_in = 16.6 / 2.54
    height_in = 8.4 / 2.54
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=300)

    x_k = np.arange(1, 51, dtype=np.float64)
    all_vals = []
    for key, label, color, lw, z in CURVE_SPEC:
        y = np.array([row['metrics']['DIV2K_valid']['psnr'] for row in series[key]], dtype=np.float64)
        all_vals.append(y)
        ax.plot(x_k, y, label=label, color=color, linewidth=lw, linestyle='-', zorder=z)

    y_cat = np.concatenate(all_vals)
    y_min = float(np.floor((y_cat.min() - 0.05) * 10.0) / 10.0)
    y_max = float(np.ceil((y_cat.max() + 0.06) * 10.0) / 10.0)

    ax.set_xlim(0, 51)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r'Iterations ($\times 10^3$)', fontsize=10.5, labelpad=4)
    ax.set_ylabel('DIV2K PSNR (dB)', fontsize=10.5, labelpad=4)
    ax.tick_params(axis='both', which='major', labelsize=9.5, direction='in', length=4, width=0.8)
    ax.grid(True, linestyle='--', linewidth=0.6, color='#BDBDBD', alpha=0.65, zorder=0)

    leg = ax.legend(
        loc='lower right',
        frameon=True,
        facecolor='white',
        edgecolor='#CCCCCC',
        framealpha=0.94,
        fontsize=9.2,
        handlelength=2.2,
        borderpad=0.5,
        labelspacing=0.35,
    )
    leg.get_frame().set_linewidth(0.7)

    fig.tight_layout(pad=0.6)
    fig_dir = out_dir / 'figures_paper_ready'
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / 'fig_ablation_convergence_50k.png'
    pdf_path = fig_dir / 'fig_ablation_convergence_50k.pdf'
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)

    summary = {
        'figure': {'png': str(png_path), 'pdf': str(pdf_path), 'width_cm': 16.6, 'dpi': 300, 'font': font_name},
        'final_50k_div2k_psnr': {k: float(series[k][-1]['metrics']['DIV2K_valid']['psnr']) for k in ['B0', 'SG6', 'W', 'U', 'F0']},
        'best_48k_div2k_psnr': {k: float(series[k][47]['metrics']['DIV2K_valid']['psnr']) for k in ['B0', 'SG6', 'W', 'U', 'F0']},
        'footer_inside_figure': False,
    }
    (out_dir / 'fig_ablation_convergence_50k.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(f'CONVERGENCE_EXPORTED font={font_name} png={png_path}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logs-dir', type=Path, default=None, help='Optional directory containing B0/SG6/W/U/F0.jsonl')
    p.add_argument('--out-dir', type=Path, default=OUT)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    series = load_epochs(args.logs_dir, args.out_dir)
    export_csvs(series, args.out_dir)
    render_convergence(series, args.out_dir)


if __name__ == '__main__':
    main()
