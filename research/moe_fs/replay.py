"""Read-only checkpoint replay with diagnostics; never repairs the baseline."""
import argparse
import importlib.util
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fs_runtime', HERE / 'runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
import numpy as np
import torch
from model import build_model


def stats(tensor):
    tensor = tensor.detach()
    finite = torch.isfinite(tensor)
    return dict(shape=list(tensor.shape), finite=bool(finite.all()),
                minimum=float(tensor.min()), maximum=float(tensor.max()),
                absmax=float(tensor.abs().max()), mean=float(tensor.mean()))


def crop(pair, patch):
    hr, lr = runtime.trainer.aligned(*pair)
    n = patch // 4
    y = random.randint(0, lr.shape[0] - n)
    x = random.randint(0, lr.shape[1] - n)
    lr, hr = lr[y:y+n, x:x+n], hr[y*4:y*4+patch, x*4:x*4+patch]
    flips = []
    for axis in (1, 0):
        flag = random.random() < .5
        flips.append(flag)
        if flag:
            lr, hr = np.flip(lr, axis), np.flip(hr, axis)
    transpose = random.random() < .5
    if transpose:
        lr, hr = lr.transpose(1, 0, 2), hr.transpose(1, 0, 2)
    return runtime.trainer.tensor(lr), runtime.trainer.tensor(hr), dict(
        hr=str(pair[0]), lr=str(pair[1]), crop_y=y, crop_x=x,
        horizontal_flip=flips[0], vertical_flip=flips[1], transpose=transpose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='epoch00004_u4000.pt')
    parser.add_argument('--data-root', type=Path, default=Path('/home/featurize/data/span-x4/datasets'))
    parser.add_argument('--original-data-root', type=Path, default=Path('/home/featurize/data/span-x4/datasets'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--until', type=int, default=4305)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    info = json.loads((args.run / 'manifest.json').read_text(encoding='utf8'))
    cfg = info['config']
    if cfg.get('moe_fs') or cfg.get('variants') or cfg.get('loader_workers', 0):
        raise ValueError('Replay currently supports only baseline with serial sampling')
    environment = info['environment']
    if torch.__version__ != environment['torch'] or torch.version.cuda != environment['cuda']:
        raise ValueError(f'Environment mismatch: need torch={environment["torch"]}, CUDA runtime={environment["cuda"]}')
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != environment['device']:
        raise ValueError('Original CUDA device type required for matched replay')
    checkpoint = args.run / args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    old_root = args.original_data_root
    pairs = []
    for row in info['data']['train']:
        mapped = {}
        for side in ('hr', 'lr'):
            original = Path(row[side])
            path = args.data_root / original.relative_to(old_root)
            if not path.is_file():
                raise FileNotFoundError(f'Restore original dataset first: {path}')
            if runtime.trainer.digest(path) != row[side + '_sha256']:
                raise ValueError(f'Data hash mismatch: {path}')
            mapped[side] = path
        pairs.append((mapped['hr'], mapped['lr']))
    print('Training data hashes and environment match; pairs:', len(pairs), flush=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['fingerprint'] != info['fingerprint']:
        raise ValueError('Checkpoint/manifest fingerprint mismatch')
    start = state['update']
    if args.check_only:
        print('CHECK PASS: checkpoint update=', start, '; no training started')
        return
    if not args.output or args.output.exists():
        raise ValueError('A new --output directory is required; existing output is never overwritten')
    if not start < args.until <= start + 1000:
        raise ValueError('Replay cap must be above checkpoint and within 1000 updates')
    args.output.mkdir(parents=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = environment['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32 = environment['matmul_allow_tf32']
    torch.use_deterministic_algorithms(True)
    arch, _ = runtime.components()
    model = build_model(arch, cfg).cuda().train()
    model.load_state_dict(state['model'], strict=True)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg['lr'], betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg['milestones'], gamma=cfg['gamma'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    log_file = args.output / 'steps.jsonl'
    expected_losses = {}
    for line in (args.run / 'loss_steps.jsonl').read_text(encoding='utf8').splitlines():
        try:
            row = json.loads(line)
            expected_losses[row['update']] = row['train_l1']
        except (ValueError, KeyError):
            continue
    layer_stats = {}
    def hook(name):
        def record(module, inputs, outputs):
            tensors = outputs if isinstance(outputs, tuple) else (outputs,)
            layer_stats[name] = [stats(t) for t in tensors if isinstance(t, torch.Tensor)]
        return record
    hooks = [module.register_forward_hook(hook(name)) for name, module in model.named_modules()
             if name in ('conv_1', 'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'block_6', 'conv_2', 'conv_cat', 'upsampler')]
    (args.output / 'replay-manifest.json').write_text(json.dumps(dict(
        source_run=str(args.run), checkpoint=str(checkpoint), checkpoint_sha256=runtime.trainer.digest(checkpoint),
        start=start, until=args.until, config=cfg, environment=environment,
        note='Diagnostic replay only; no clipping/LR change/AMP/optimizer change'), indent=2), encoding='utf8')
    # Restore RNG only after construction and diagnostics initialization.
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    accum = cfg['effective_batch'] // cfg['micro_batch']
    with log_file.open('w', encoding='utf8', buffering=1) as stream:
        for update in range(start + 1, args.until + 1):
            optimizer.zero_grad(set_to_none=True)
            micros = []
            total = 0.0
            before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
            for micro in range(accum):
                samples = [crop(random.choice(pairs), cfg['hr_patch']) for _ in range(cfg['micro_batch'])]
                lr = torch.stack([s[0] for s in samples]).cuda()
                hr = torch.stack([s[1] for s in samples]).cuda()
                prediction = model(lr)
                loss = torch.nn.functional.l1_loss(prediction, hr) / accum
                total += loss.item()
                micros.append(dict(micro=micro, loss=loss.item()*accum, input=stats(lr), target=stats(hr),
                                   output=stats(prediction), layers=dict(layer_stats), samples=[s[2] for s in samples]))
                if not torch.isfinite(loss):
                    stream.write(json.dumps(dict(update=update, nonfinite_loss=True, micros=micros))+'\n')
                    raise FloatingPointError('Nonfinite loss; diagnostic record saved')
                loss.backward()
            gradients = {name: dict(norm=float(p.grad.norm()), absmax=float(p.grad.abs().max()), finite=bool(torch.isfinite(p.grad).all()))
                         for name, p in model.named_parameters() if p.requires_grad and p.grad is not None}
            if not all(g['finite'] for g in gradients.values()):
                stream.write(json.dumps(dict(update=update, train_l1=total, micros=micros, gradients=gradients))+'\n')
                raise FloatingPointError('Nonfinite gradient; diagnostic record saved')
            optimizer.step(); scheduler.step()
            changes = {name: dict(parameter_norm=float(p.detach().norm()), update_norm=float((p.detach()-before[name]).norm()))
                       for name, p in model.named_parameters() if p.requires_grad}
            row = dict(update=update, train_l1=total, lr=scheduler.get_last_lr()[0], micros=micros,
                       gradients=gradients, parameter_changes=changes)
            row['original_train_l1'] = expected_losses.get(update)
            row['matches_original_loss'] = (math.isclose(total, expected_losses[update], rel_tol=1e-6, abs_tol=1e-8)
                                            if update in expected_losses else None)
            stream.write(json.dumps(row)+'\n')
            if row['matches_original_loss'] is False:
                raise RuntimeError(f'Replay loss differs at update {update}; diagnostics saved. Do not interpret as matched replay.')
            if update % 10 == 0 or total > .2:
                print(f'update={update} loss={total:.8g}', flush=True)
    for handle in hooks:
        handle.remove()
    # Normal epoch snapshots follow evaluation, which refreshes derived frozen
    # Conv3XC deployment kernels. Match that state without changing training.
    for module in model.modules():
        if isinstance(module, arch.Conv3XC):
            module.update_params()
    torch.save(dict(update=args.until, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict()), args.output / 'diagnostic-final.pt')
    print('Replay complete; diagnostics:', args.output, flush=True)


if __name__ == '__main__':
    main()
