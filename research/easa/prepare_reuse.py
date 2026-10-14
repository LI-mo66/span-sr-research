"""Audit the archived B0, freeze EASA only; never start or initialize from B0."""
import argparse
import importlib.util
import json
import platform
import subprocess
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('easa_runtime', HERE / 'runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
import torch

BASE_SOURCE = 'd280dc5ebcffc46c50983dfe27e81c6096d90dcf'
RECORD = 'b546036'
RECORD_DIR = 'research/docs/module-study/results/FEAT-MOEFS-G4-LR2E4-20K-01'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=runtime.ROOT, text=True)


def reconstructed_paths(data):
    def specs(rows):
        groups = {}
        for row in rows:
            hr, lr = PurePosixPath(row['hr']), PurePosixPath(row['lr'])
            if not lr.stem.startswith(hr.stem):
                raise ValueError('Cannot reconstruct LR suffix from B0 manifest')
            suffix = lr.stem[len(hr.stem):]
            key = (str(hr.parent), str(lr.parent), suffix)
            groups[key] = dict(hr=key[0], lr=key[1], lr_suffix=suffix)
        values = list(groups.values())
        return values[0] if len(values) == 1 else values
    return dict(train=specs(data['train']),
                validation={name: specs(rows) for name, rows in data.items() if name != 'train'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-run', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if git('status', '--porcelain').strip():
        raise RuntimeError('Checkout must be clean and committed')
    common = ['basicsr', 'research/moe_fs/trainer.py', 'research/runtime.py',
              'research/image_io.py', 'research/parallel_loader.py']
    changed = git('diff', '--name-only', BASE_SOURCE, 'HEAD', '--', *common).strip()
    if changed:
        raise RuntimeError('Common training/evaluation source changed: ' + changed)
    baseline = Path(args.baseline_run).resolve()
    manifest = json.loads((baseline / 'manifest.json').read_text())
    cfg = manifest['config']
    recorded = json.loads(git('show', RECORD + ':' + RECORD_DIR + '/B0-effective-config.json'))
    keys = ('model', 'seed', 'total_updates', 'record_updates', 'hr_patch',
            'effective_batch', 'micro_batch', 'lr', 'adam_betas', 'adam_eps',
            'milestones', 'gamma', 'crop_border', 'test_y_channel',
            'self_ensemble', 'selection_dataset', 'image_backend')
    for key in keys:
        if cfg.get(key) != recorded.get(key):
            raise ValueError('Server B0 differs from recorded protocol: ' + key)
    if manifest.get('commit') != BASE_SOURCE or cfg.get('moe_fs') is not False:
        raise ValueError('Not the recorded original B0 run')
    if cfg.get('variants') or cfg.get('easa') or cfg.get('engineering_only'):
        raise ValueError('Baseline contains candidate/engineering settings')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    actual_environment = dict(torch=torch.__version__, cuda=torch.version.cuda,
                              device=torch.cuda.get_device_name(), deterministic=True,
                              cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                              matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    if actual_environment != manifest['environment']:
        raise ValueError(f'B0 environment mismatch: {actual_environment} != {manifest["environment"]}')
    if platform.python_version() != '3.11.8':
        raise ValueError('Recorded B0 requires Python 3.11.8')
    import numpy, cv2, scipy
    if (numpy.__version__, cv2.__version__, scipy.__version__) != ('1.26.4', '4.10.0', '1.13.1'):
        raise ValueError('Recorded NumPy/OpenCV/SciPy versions differ')
    # Same ordered image hashes preserve the sampler's image-index meaning.
    paths = reconstructed_paths(manifest['data'])
    datasets = dict(train=runtime.trainer.pairs(paths['train']))
    datasets.update({n: runtime.trainer.pairs(s) for n, s in paths['validation'].items()})
    counts = {}
    expected_counts = dict(train=3450, DIV2K_valid=100, Set5=5, Set14=14,
                           B100=100, Urban100=100, Manga109=109)
    if set(datasets) != set(expected_counts):
        raise ValueError('Recorded B0 dataset names differ')
    for name, pairs in datasets.items():
        old = manifest['data'][name]
        if len(pairs) != len(old) or len(old) != expected_counts[name]:
            raise ValueError('B0 image count differs: ' + name)
        print(f'Auditing ordered data hashes: {name}, {len(pairs)} pairs', flush=True)
        for i, ((hr, lr), row) in enumerate(zip(pairs, old), 1):
            if runtime.trainer.digest(hr) != row['hr_sha256'] or runtime.trainer.digest(lr) != row['lr_sha256']:
                raise ValueError(f'B0 ordered data hash mismatch: {name}/{i}')
        counts[name] = len(pairs)
    # Validate the recorded 48k selection from the full 50k history.
    history = [json.loads(line) for line in (baseline / 'epochs.jsonl').read_text().splitlines()]
    history = sorted(history, key=lambda r: r['update'])
    if [r['update'] for r in history] != list(range(1000, 50001, 1000)):
        raise ValueError('B0 complete 1k..50k history required')
    best = max(history, key=lambda r: r['metrics']['DIV2K_valid']['psnr'])
    if best['update'] != 48000:
        raise ValueError('B0 validation-selected checkpoint differs from record')
    weight = baseline / 'epoch00048_u48000.pt'
    weight_hash = runtime.trainer.digest(weight)
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError('Use a new EASA output group: ' + str(out))
    out.mkdir(parents=True)
    path_file = out / 'paths.json'
    path_file.write_text(json.dumps(paths, indent=2), encoding='utf8')
    candidate = dict(cfg)
    candidate.pop('moe_fs', None)
    candidate.update(easa=True, experiment_version='EASAResidual-v1',
                     easa_settings=dict(position='mid', down_scale=8, residual_init=0.01),
                     paths_file=str(path_file))
    (out / 'E1-config.json').write_text(json.dumps(candidate, indent=2), encoding='utf8')
    audit = dict(status='PASS', baseline_run=str(baseline), baseline_source=BASE_SOURCE,
                 baseline_record=RECORD, candidate_source=git('rev-parse', 'HEAD').strip(),
                 executed_updates=50000, configured_horizon=candidate['total_updates'],
                 counts=counts, environment=actual_environment, baseline_selected=best,
                 baseline_final=history[-1],
                 baseline_weight_sha256=weight_hash,
                 initialization='random, seed10; B0 weights not loaded',
                 evidence='protocol/data audit; no candidate PSNR measured')
    (out / 'reuse-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf8')
    print('PASS: EASA only, from scratch, microbatch16, fixed cap50000. Config:', out / 'E1-config.json')


if __name__ == '__main__':
    main()
