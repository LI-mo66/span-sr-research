"""Plot saved measurements; no interpolation of missing evaluation points."""
import argparse
import json
from pathlib import Path


def records(folder):
    values = {}
    path = folder / 'epochs.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf8').splitlines():
            try:
                row = json.loads(line)
                values[row['update']] = row
            except (ValueError, KeyError):
                continue
    return [values[u] for u in sorted(values)]


def generate(runs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    data = {label: records(folder) for label, folder in runs.items()}
    data = {label: rows for label, rows in data.items() if rows}
    if not data:
        raise ValueError('No complete evaluation records available')
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / (name + '.png'), dpi=180)
        fig.savefig(output / (name + '.svg'))
        plt.close(fig)
    for label, rows in data.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([r['update'] for r in rows], [r['train_l1_mean'] for r in rows], marker='o')
        ax.set(xlabel='Optimizer updates', ylabel='Mean training RGB L1', title=label)
        ax.grid(alpha=.25)
        save(fig, label + '-loss-period-mean')
        step_path = runs[label] / 'loss_steps.jsonl'
        if step_path.exists():
            steps = {}
            last = rows[-1]['update']
            for line in step_path.read_text(encoding='utf8').splitlines():
                try:
                    row = json.loads(line)
                    if row['update'] <= last:
                        steps[row['update']] = row['train_l1']
                except (ValueError, KeyError):
                    continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(sorted(steps), [steps[u] for u in sorted(steps)], lw=.6)
            ax.set(xlabel='Optimizer updates', ylabel='Training RGB L1', title=label + ' raw steps')
            ax.grid(alpha=.25)
            save(fig, label + '-loss-steps')
        for name in sorted(rows[-1]['metrics']):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, metric in zip(axes, ('psnr', 'ssim')):
                available = [r for r in rows if name in r['metrics']]
                ax.plot([r['update'] for r in available],
                        [r['metrics'][name][metric] for r in available], marker='o')
                ax.set(xlabel='Optimizer updates', ylabel='PSNR (dB)' if metric == 'psnr' else 'SSIM')
                ax.grid(alpha=.25)
            fig.suptitle(label + ' / ' + name)
            save(fig, label + '-' + name)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, rows in data.items():
        ax.plot([r['update'] for r in rows], [r['train_l1_mean'] for r in rows], marker='o', label=label)
    ax.set(xlabel='Optimizer updates', ylabel='Mean training RGB L1', title='Training loss comparison')
    ax.legend(); ax.grid(alpha=.25)
    save(fig, 'comparison-loss')
    if 'B0' in data and 'A1-G4' in data:
        baseline = {r['update']: r for r in data['B0']}
        candidate = {r['update']: r for r in data['A1-G4']}
        common = sorted(baseline.keys() & candidate.keys())
        datasets = sorted(set(data['B0'][-1]['metrics']) & set(data['A1-G4'][-1]['metrics']))
        for name in datasets:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            for col, metric in enumerate(('psnr', 'ssim')):
                for label, rows in data.items():
                    available = [r for r in rows if name in r['metrics']]
                    axes[0, col].plot([r['update'] for r in available],
                                      [r['metrics'][name][metric] for r in available], marker='o', label=label)
                points = [u for u in common if name in baseline[u]['metrics'] and name in candidate[u]['metrics']]
                axes[1, col].plot(points, [candidate[u]['metrics'][name][metric] - baseline[u]['metrics'][name][metric]
                                         for u in points], marker='o')
                axes[1, col].axhline(0, color='gray', ls='--')
                for row in (0, 1):
                    axes[row, col].set(xlabel='Optimizer updates', ylabel=('Delta ' if row else '') + metric.upper())
                    axes[row, col].grid(alpha=.25)
                axes[0, col].legend()
            fig.suptitle(name + ' / A1-G4 minus B0 at matched updates')
            save(fig, 'comparison-' + name)
    (output / 'source-data.json').write_text(json.dumps(data, indent=2), encoding='utf8')
    print('Curves saved:', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate({'B0': args.baseline, 'A1-G4': args.candidate}, args.output)
