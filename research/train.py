"""Single GPU research harness. Upstream architecture and metrics remain untouched."""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from runtime import ROOT, components

def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def pairs(spec):
    if isinstance(spec, list):
        return [pair for source in spec for pair in pairs(source)]
    hr, lr = Path(spec['hr']), Path(spec['lr'])
    if not hr.is_dir() or not lr.is_dir():
        raise FileNotFoundError(f'Missing HR/LR directory: {hr}, {lr}')
    low = {p.stem: p for p in lr.iterdir() if p.suffix.lower() in ('.png', '.bmp', '.jpg', '.jpeg')}
    result = []
    for p in sorted(hr.iterdir()):
        if p.suffix.lower() not in ('.png', '.bmp', '.jpg', '.jpeg'):
            continue
        key = p.stem + spec.get('lr_suffix', '')
        if key not in low:
            raise ValueError(f'Unpaired HR: {p}; expected LR stem {key}')
        result.append((p, low[key]))
    if not result or len(result) != len(low):
        raise ValueError('Empty dataset or unmatched LR images')
    return result

def read(p):
    a = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if a is None:
        raise ValueError(f'Unreadable image {p}')
    return a

def tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a[:, :, ::-1].transpose(2, 0, 1))).float() / 255.

def aligned(hp, lp):
    hr, lr = read(hp), read(lp)
    h, w = lr.shape[0]*4, lr.shape[1]*4
    if not (0 <= hr.shape[0]-h < 4 and 0 <= hr.shape[1]-w < 4):
        raise ValueError(f'Nonaligned x4 pair: {hp}, {lp}')
    return hr[:h, :w], lr

def sample(pair, patch):
    hp, lp = pair
    hr, lr = aligned(hp, lp)
    n = patch // 4
    y, x = random.randint(0, lr.shape[0] - n), random.randint(0, lr.shape[1] - n)
    lr, hr = lr[y:y+n, x:x+n], hr[y*4:y*4+patch, x*4:x*4+patch]
    if random.random() < .5:
        lr, hr = lr[:, ::-1], hr[:, ::-1]
    if random.random() < .5:
        lr, hr = lr[::-1], hr[::-1]
    if random.random() < .5:
        lr, hr = lr.transpose(1, 0, 2), hr.transpose(1, 0, 2)
    return tensor(lr), tensor(hr)

def evaluate(model, datasets, metrics, cfg, out, epoch, update):
    model.eval()
    summary = {}
    with torch.inference_mode():
        for name, data in datasets.items():
            rows = []
            for hp, lp in data:
                hr, lr = aligned(hp, lp)
                prediction = model(tensor(lr).unsqueeze(0).cuda())[0]
                sr = prediction.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
                sr = np.rint(sr * 255).astype(np.uint8)
                if sr.shape != hr.shape:
                    raise ValueError(f'Evaluation shape mismatch: {hp}')
                kw = dict(crop_border=cfg['crop_border'], test_y_channel=cfg['test_y_channel'])
                rows.append(dict(image=hp.name, psnr=metrics.calculate_psnr(sr, hr, **kw), ssim=metrics.calculate_ssim(sr, hr, **kw)))
            summary[name] = {m: float(np.mean([r[m] for r in rows])) for m in ('psnr', 'ssim')}
            (out / f'{name}_epoch{epoch:05d}_u{update}.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
    model.train()
    return summary

def atomic_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='research/configs/baseline_x4.json')
    parser.add_argument('--id', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not args.id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.id):
        raise ValueError('ID must contain only letters, digits, hyphen and underscore')
    cfg = json.loads(Path(args.config).read_text(encoding='utf8'))
    if cfg['self_ensemble'] or cfg['model']['upscale'] != 4:
        raise ValueError('This development harness supports non-ensemble x4 only')
    paths = json.loads(Path(cfg['paths_file']).read_text(encoding='utf-8-sig'))
    train = pairs(paths['train'])
    val = {k: pairs(v) for k, v in paths['validation'].items()}
    required = {'Set5', 'Set14', 'B100', 'Urban100', 'Manga109', cfg['selection_dataset']}
    if not required <= set(val):
        raise ValueError(f'Missing evaluation datasets {required-set(val)}')
    # Reject exact overlap across train and held-out HR images (including copied files).
    manifests = {}
    for name, data in {'train': train, **val}.items():
        manifests[name] = [dict(hr=str(h), lr=str(l), hr_sha256=digest(h), lr_sha256=digest(l)) for h,l in data]
    train_hashes = {r['hr_sha256'] for r in manifests['train']}
    for name in val:
        if train_hashes & {r['hr_sha256'] for r in manifests[name]}:
            raise ValueError(f'Train/evaluation overlap in {name}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if cfg['effective_batch'] % cfg['micro_batch']:
        raise ValueError('Micro batch must divide effective batch')
    if args.preflight:
        print(json.dumps({k: len(v) for k,v in manifests.items()}, indent=2))
        return
    if subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit source changes before training')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    fingerprint = hashlib.sha256(json.dumps(dict(config=cfg, manifests=manifests, commit=commit, screen=args.screen), sort_keys=True).encode()).hexdigest()
    out = ROOT / 'runs' / args.id
    if out.exists() and not args.resume:
        raise FileExistsError(f'Run ID already exists: {out}')
    if args.resume and not (out / 'latest.pt').exists():
        raise FileNotFoundError('No resume checkpoint')
    out.mkdir(parents=True, exist_ok=True)
    class Tee:
        def __init__(self, stream, file): self.stream, self.file = stream, file
        def write(self, s): self.stream.write(s); self.file.write(s); self.file.flush()
        def flush(self): self.stream.flush(); self.file.flush()
    log = open(out / 'console.log', 'a', encoding='utf8')
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    arch, metrics = components()
    model = arch.SPAN(**cfg['model']).cuda()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg['lr'], betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, cfg['milestones'], gamma=cfg['gamma'])
    updates_per_epoch = math.ceil(len(train) / cfg['effective_batch'])
    stop = min(cfg['total_updates'], cfg['screen_epochs'] * updates_per_epoch) if args.screen else cfg['total_updates']
    update, epoch, best = 0, 0, {}
    if args.resume:
        state = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        if state['fingerprint'] != fingerprint:
            raise ValueError('Resume protocol/source/data fingerprint mismatch')
        model.load_state_dict(state['model'], strict=True); opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler'])
        update, epoch, best = state['update'], state['epoch'], state['best']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        metadata = dict(id=args.id, fingerprint=fingerprint, commit=commit, config=cfg, data=manifests, updates_per_epoch=updates_per_epoch, screen=args.screen, torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(), parameters_total=sum(p.numel() for p in model.parameters()), parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        metadata['python'] = sys.version
        metadata['dependencies'] = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
        (out / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    print(f'Run {args.id}; protocol {fingerprint}; stop {stop} updates; {updates_per_epoch} updates per research epoch', flush=True)
    while update < stop:
        epoch += 1
        start = time.perf_counter()
        for _ in range(min(updates_per_epoch, stop-update)):
            opt.zero_grad(set_to_none=True)
            loss_value = 0.
            for _ in range(cfg['effective_batch']//cfg['micro_batch']):
                samples = [sample(random.choice(train), cfg['hr_patch']) for _ in range(cfg['micro_batch'])]
                lr = torch.stack([s[0] for s in samples]).cuda(); hr = torch.stack([s[1] for s in samples]).cuda()
                loss = torch.nn.functional.l1_loss(model(lr), hr) / (cfg['effective_batch']//cfg['micro_batch'])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                loss.backward(); loss_value += loss.item()
            opt.step(); sched.step(); update += 1
            if update % 100 == 0: print(f'epoch={epoch} update={update} L1={loss_value:.8f} lr={sched.get_last_lr()[0]:.8g}', flush=True)
        scores = evaluate(model, val, metrics, cfg, out, epoch, update)
        selected = False
        for name, row in scores.items():
            for metric, value in row.items():
                key = name + '/' + metric
                if key not in best or value > best[key]['value']:
                    best[key] = dict(value=value, epoch=epoch, update=update)
                    if name == cfg['selection_dataset'] and metric == 'psnr': selected = True
            print(f'{name}: PSNR={row["psnr"]:.8f} SSIM={row["ssim"]:.8f}; best={best[name+"/psnr"]}, {best[name+"/ssim"]}', flush=True)
        record = dict(epoch=epoch, update=update, metrics=scores, best=best, elapsed_seconds=time.perf_counter()-start)
        with open(out/'epochs.jsonl','a',encoding='utf8') as f: f.write(json.dumps(record)+'\n')
        state = dict(fingerprint=fingerprint, epoch=epoch, update=update, best=best, model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        atomic_save(state, out/'latest.pt')
        if selected: atomic_save(state, out/'best_validation.pt')
    print('Training complete. No stability or paper-value claim follows from a single run.', flush=True)

if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
