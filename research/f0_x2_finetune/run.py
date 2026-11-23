"""Independent loss fine-tuning from the verified x2 982K checkpoint."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'research/multiscale_f0'), str(ROOT / 'research')]
import fusion
import trainer
from image_io import enable_cache

EXPECTED_SHA = '3dae5537ca672813dc4e9870f97276d173eda6f15ef8ffaf9aaf1c8406ef3068'
ARMS = {'l1_lr5': ('l1', 5e-6), 'mse_lr5': ('mse', 5e-6)}
TARGETS = ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109')


def objective(prediction, target, kind, step, total):
    residual = prediction - target
    l1 = residual.abs().mean()
    mse = residual.square().mean()
    use_mse = kind == 'mse' or (kind == 'staged' and step > total // 2)
    return (mse if use_mse else l1), l1, mse


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def json_write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def manifest_pairs(cfg, expected):
    paths = json.loads(Path(cfg['paths_file']).read_text(encoding='utf-8-sig'))
    pairs = dict(train=trainer.pairs(paths['train']))
    pairs.update({name: trainer.pairs(spec) for name, spec in paths['validation'].items()})
    assert set(pairs) == set(expected)
    for name, data in pairs.items():
        assert len(data) == len(expected[name]), name
        for (hr, lr), row in zip(data, expected[name]):
            assert str(hr) == row['hr'] and str(lr) == row['lr'], (name, hr)
    return pairs


def preflight(source, group):
    metadata = json.loads((source / 'manifest.json').read_text())
    cfg = metadata['config']
    assert cfg['arm'] == 'F0' and cfg['model']['upscale'] == 2
    assert cfg['seed'] == 10 and cfg['hr_patch'] == 128 and cfg['crop_border'] == 2
    assert cfg['effective_batch'] == 64 and cfg['micro_batch'] == 16
    assert cfg['self_ensemble'] is True and cfg['loader_workers'] == 0
    checkpoint = source / 'epoch00982_u982000.pt'
    assert trainer.digest(checkpoint) == EXPECTED_SHA
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['update'] == 982000 and state['fingerprint'] == metadata['fingerprint']
    assert state['optimizer']['param_groups'][0]['lr'] == 1.25e-5
    trainer.SCALE = 2
    data = manifest_pairs(cfg, metadata['data'])
    for name, items in data.items():
        for (hr, lr), row in zip(items, metadata['data'][name]):
            assert trainer.digest(hr) == row['hr_sha256'], hr
            assert trainer.digest(lr) == row['lr_sha256'], lr
            high, low = trainer.aligned(hr, lr)
            if name == 'train':
                assert min(low.shape[:2]) >= 96, (hr, lr, low.shape)
        print('PASS ordered data hashes and geometry', name, len(items), flush=True)
    assert len(data['train']) == 3450
    train_hashes = {row['hr_sha256'] for row in metadata['data']['train']}
    for name, items in metadata['data'].items():
        if name != 'train':
            assert not train_hashes & {row['hr_sha256'] for row in items}
    import shutil
    assert shutil.disk_usage(ROOT).free > 4 * 1024**3, 'Need4GiB free for checkpoints and logs'
    json_write(group / 'preflight.json', dict(status='PASS', source_sha256=EXPECTED_SHA,
               counts={name: len(items) for name, items in data.items()},
               training='DF2K only', test_selection='exploratory five-set reporting'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    source, group = args.source_run.resolve(), args.group.resolve()
    assert group.is_dir()
    if args.preflight:
        preflight(source, group)
        return
    assert args.arm
    assert json.loads((group / 'preflight.json').read_text())['status'] == 'PASS'
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    metadata = json.loads((source / 'manifest.json').read_text())
    cfg = dict(metadata['config'], hr_patch=192)
    trainer.SCALE = 2
    checkpoint = source / 'epoch00982_u982000.pt'
    assert trainer.digest(checkpoint) == EXPECTED_SHA
    original = torch.load(checkpoint, map_location='cpu', weights_only=False)
    data = manifest_pairs(cfg, metadata['data'])
    enable_cache(metadata['data'], ROOT / 'datasets/image_cache')
    arch, metrics = fusion.components()
    model = fusion.build_model(arch, cfg).cuda()
    model.load_state_dict(original['model'], strict=True)
    kind, rate = ARMS[args.arm]
    # All arms reset Adam equally to avoid transferring L1 moments into MSE.
    optimizer = torch.optim.Adam(model.parameters(), lr=rate,
        betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'])
    total, interval = 10000, 1000
    out = ROOT / 'runs' / ('FT-X2-982K-PC96-' + args.arm + '-10K-v1')
    protocol = dict(arm=args.arm, source_sha256=EXPECTED_SHA, objective=kind,
        lr=rate, adam='fresh for every arm', steps=total, seed=10, clip=0.1,
        model='unchanged F0 x2 bound8', lr_patch=96, hr_patch=192, cfg=cfg,
        code_sha256=trainer.digest(Path(__file__)), data=metadata['data'])
    protocol['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    step = 0
    restore_rng(dict(python=original['python_rng'], numpy=original['numpy_rng'],
                     torch=original['torch_rng'], cuda=original['cuda_rng']))
    if args.resume:
        state = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        assert state['fingerprint'] == fingerprint
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        restore_rng(state['rng'])
        history = out / 'epochs.jsonl'
        saved_rows = [json.loads(line) for line in history.open() if line.strip()]
        saved_rows = [row for row in saved_rows if row['step'] <= step]
        assert [row['step'] for row in saved_rows] == list(range(interval, step + 1, interval))
        history.write_text(''.join(json.dumps(row) + '\n' for row in saved_rows))
        steps_path = out / 'steps.jsonl'
        saved = [json.loads(line) for line in steps_path.open() if line.strip()]
        saved = [row for row in saved if row['step'] <= step]
        assert [row['step'] for row in saved] == list(range(1, step + 1))
        steps_path.write_text(''.join(json.dumps(row) + '\n' for row in saved))
    else:
        json_write(out / 'protocol.json', protocol)
        # Verify the exact starting weight through the ordinary x8 entry point once.
        if args.arm == 'l1_lr5':
            print('Verifying starting982K six-set plain+x8 metrics', flush=True)
            before_rng = rng_state()
            observed = trainer.evaluate(model, {k: v for k, v in data.items() if k != 'train'},
                                        metrics, cfg, out, 0, 0)
            expected = original['epoch_record']['metrics']
            for name, values in observed.items():
                for key, value in values.items():
                    assert abs(value - expected[name][key]) < 2e-6, (name, key, value)
            json_write(group / 'baseline-verified.json', observed)
            restore_rng(before_rng)
        assert (group / 'baseline-verified.json').is_file()
        (out / 'epochs.jsonl').touch()
        (out / 'steps.jsonl').touch()
        trainer.atomic_save(dict(fingerprint=fingerprint, step=0, model=model.state_dict(),
            optimizer=optimizer.state_dict(), rng=rng_state()), out / 'latest.pt')
    del original
    assert step < total
    model.train()
    log = (out / 'steps.jsonl').open('a', buffering=1)
    try:
        while step < total:
            optimizer.zero_grad(set_to_none=True)
            values = [0.0, 0.0, 0.0]
            for micro in range(4):
                samples = [trainer.sample(random.choice(data['train']), 192) for _ in range(16)]
                low = torch.stack([item[0] for item in samples]).cuda()
                high = torch.stack([item[1] for item in samples]).cuda()
                prediction = model(low)
                assert prediction.shape == high.shape
                loss, l1, mse = objective(prediction, high, kind, step + 1, total)
                if not all(bool(torch.isfinite(v)) for v in (loss, l1, mse)) or float(l1.detach()) > 1.0:
                    json_write(out / 'ANOMALY.json', dict(next_step=step + 1,
                        micro=micro, action='stopped before Adam; latest complete checkpoint preserved'))
                    raise FloatingPointError('Nonfinite objective or microbatch RGB L1 above1')
                (loss / 4).backward()
                for i, value in enumerate((loss, l1, mse)):
                    values[i] += float(value.detach()) / 4
            assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)
            optimizer.step()
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            step += 1
            record = dict(step=step, objective=values[0], rgb_l1=values[1], rgb_mse=values[2],
                          gradient_norm_before_clip=float(norm), clip_applied=float(norm) > 0.1,
                          lr=optimizer.param_groups[0]['lr'])
            log.write(json.dumps(record) + '\n')
            if step == 1 or step % 100 == 0:
                print(args.arm, step, '/', total, record, flush=True)
            if step % interval == 0:
                scores = trainer.evaluate(model, {k: v for k, v in data.items() if k != 'train'},
                                           metrics, cfg, out, step // interval, step)
                assert all(math.isfinite(v) for row in scores.values() for v in row.values())
                epoch_record = dict(step=step, metrics=scores)
                state = dict(fingerprint=fingerprint, step=step, model=model.state_dict(),
                             optimizer=optimizer.state_dict(), rng=rng_state(), record=epoch_record)
                trainer.atomic_save(state, out / f'step{step:05d}.pt')
                # History first; latest.pt is the commit point for exact resume.
                with (out / 'epochs.jsonl').open('a') as stream:
                    stream.write(json.dumps(epoch_record) + '\n')
                trainer.atomic_save(state, out / 'latest.pt')
                print('EVALUATED', args.arm, step, scores, flush=True)
    finally:
        log.close()
    json_write(out / 'COMPLETE.json', dict(step=step, arm=args.arm, fingerprint=fingerprint))
    print('COMPLETE', args.arm, flush=True)


if __name__ == '__main__':
    main()
