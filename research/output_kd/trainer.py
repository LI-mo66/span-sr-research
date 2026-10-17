# Independent copy of research/moe_fs/trainer.py; output KD and separate loss recording.
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
from teacher import build_teacher, teacher_prediction, output_loss

def build_model(arch, cfg):
    return arch.SPAN(**cfg["model"])
from image_io import read,enable_cache
from parallel_loader import PrefetchBatches

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

def tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a[:, :, ::-1].transpose(2, 0, 1))).float() / 255.

def aligned(hp, lp):
    hr, lr = read(hp), read(lp)
    h, w = lr.shape[0]*4, lr.shape[1]*4
    if not (0 <= hr.shape[0]-h < 4 and 0 <= hr.shape[1]-w < 4):
        raise ValueError(f'Nonaligned x4 pair: {hp}, {lp}')
    return hr[:h, :w], lr

def sample(pair, patch, rng=None):
    rng = random if rng is None else rng
    hp, lp = pair
    hr, lr = aligned(hp, lp)
    n = patch // 4
    y, x = rng.randint(0, lr.shape[0] - n), rng.randint(0, lr.shape[1] - n)
    lr, hr = lr[y:y+n, x:x+n], hr[y*4:y*4+patch, x*4:x*4+patch]
    if rng.random() < .5:
        lr, hr = lr[:, ::-1], hr[:, ::-1]
    if rng.random() < .5:
        lr, hr = lr[::-1], hr[::-1]
    if rng.random() < .5:
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
    parser.add_argument('--screen-updates', type=int, choices=[5000,20000,50000,100000])
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not args.id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.id):
        raise ValueError('ID must contain only letters, digits, hyphen and underscore')
    cfg = json.loads(Path(args.config).read_text(encoding='utf8'))
    from config import validate_config
    validate_config(cfg)
    workers = cfg.get('loader_workers', 0)
    if workers < 0 or cfg.get('log_updates', 100) < 1:
        raise ValueError('Invalid loader/log settings')
    if workers and cfg.get('image_backend', 'png') != 'png':
        raise ValueError('Threaded loader currently supports png only')
    if workers:
        cv2.setNumThreads(1)
        torch.set_num_threads(cfg.get('cpu_threads', 4))
    print('Checking dataset hashes and train/validation separation...', flush=True)
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
    # Match the recorded B0 FP32/TF32 policy explicitly, including on newer torch.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher = build_teacher(cfg, 'cuda')
    if args.preflight:
        print(json.dumps({k: len(v) for k,v in manifests.items()}, indent=2))
        return
    if subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit source changes before training')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    # Screen/full are execution stopping caps, not different optimizer protocols.
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),deterministic=True,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    fingerprint = hashlib.sha256(json.dumps(dict(config=cfg, manifests=manifests, commit=commit,environment=environment), sort_keys=True).encode()).hexdigest()
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
    if cfg.get('image_backend','png')=='npy_mmap':enable_cache(manifests,ROOT/'datasets'/'image_cache')
    elif cfg.get('image_backend','png')!='png':raise ValueError('Unknown image backend')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    arch, metrics = components()
    model = build_model(arch,cfg).cuda()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg['lr'], betas=tuple(cfg['adam_betas']), eps=cfg['adam_eps'])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, cfg['milestones'], gamma=cfg['gamma'])
    updates_per_epoch = cfg.get('record_updates', math.ceil(len(train) / cfg['effective_batch']))
    if args.screen_updates is not None and not args.screen:
        raise ValueError('--screen-updates requires --screen')
    stop = min(cfg['total_updates'], args.screen_updates or cfg['screen_epochs'] * updates_per_epoch) if args.screen else cfg['total_updates']
    update, epoch, best = 0, 0, {}
    if args.resume:
        state = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        if state['fingerprint'] != fingerprint:
            raise ValueError('Resume protocol/source/data fingerprint mismatch')
        model.load_state_dict(state['model'], strict=True); opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler'])
        update, epoch, best = state['update'], state['epoch'], state['best']
        history=out/'epochs.jsonl'
        records={}
        if history.exists():
            for line in history.read_text(encoding='utf8').splitlines():
                try:row=json.loads(line)
                except json.JSONDecodeError:continue  # Interrupted append, recovered below.
                if row['update']<=update:records[row['update']]=row
        if state.get('epoch_record'):records[update]=state['epoch_record']
        restored=''.join(json.dumps(records[u])+'\n' for u in sorted(records))
        if history.exists() and history.read_text(encoding='utf8')!=restored:
            (out/f'epochs_before_resume_{time.time_ns()}.jsonl').write_text(history.read_text(encoding='utf8'),encoding='utf8')
        tmp=history.with_suffix('.tmp');tmp.write_text(restored,encoding='utf8');os.replace(tmp,history)
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        metadata = dict(id=args.id, fingerprint=fingerprint, commit=commit, config=cfg, data=manifests, updates_per_epoch=updates_per_epoch, screen=args.screen, torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(), parameters_total=sum(p.numel() for p in model.parameters()), parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        metadata['teacher'] = dict(cfg['distillation'], frozen=True, eval_mode=True, inference_only_during_training=True)
        metadata['python'] = sys.version
        metadata['environment'] = environment
        metadata['extra_deployed_parameters'] = getattr(model,'extra_parameters',0)
        metadata['dependencies'] = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
        (out / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    # Independent recorder: truncate uncommitted step records on resume.
    step_history = out / 'loss_steps.jsonl'
    if step_history.exists():
        recovered = {}
        for line in step_history.read_text(encoding='utf8').splitlines():
            try:
                row = json.loads(line)
                if row['update'] <= update:
                    recovered[row['update']] = row
            except (ValueError, KeyError):
                continue
        atomic_text = step_history.with_suffix('.tmp')
        atomic_text.write_text(''.join(json.dumps(recovered[u])+'\n' for u in sorted(recovered)), encoding='utf8')
        os.replace(atomic_text, step_history)
    step_log = open(step_history, 'a', encoding='utf8', buffering=1)
    print(f'Run {args.id}; protocol {fingerprint}; stop {stop} updates; {updates_per_epoch} updates per research epoch', flush=True)
    accum = cfg['effective_batch']//cfg['micro_batch']
    loader = PrefetchBatches(train, cfg['hr_patch'], cfg['micro_batch'], cfg['seed'],
                             update*accum, stop*accum, sample, workers,
                             cfg.get('loader_prefetch', 2)) if workers else None
    print(f'Loader workers={workers}; log every {cfg.get("log_updates",100)} updates; evaluation every {updates_per_epoch}', flush=True)
    run_start, run_update = time.perf_counter(), update
    while update < stop:
        epoch += 1
        start = time.perf_counter()
        epoch_loss_sum, epoch_updates = 0., 0
        epoch_rec_sum = epoch_kd_sum = 0.
        for _ in range(min(updates_per_epoch, stop-update)):
            step_start = time.perf_counter()
            data_seconds = 0.
            opt.zero_grad(set_to_none=True)
            loss_value = rec_value = kd_value = 0.
            for _ in range(cfg['effective_batch']//cfg['micro_batch']):
                data_start = time.perf_counter()
                samples = next(loader) if loader else [sample(random.choice(train), cfg['hr_patch']) for _ in range(cfg['micro_batch'])]
                data_seconds += time.perf_counter()-data_start
                lr = torch.stack([s[0] for s in samples]).cuda(); hr = torch.stack([s[1] for s in samples]).cuda()
                target = teacher_prediction(teacher, lr, cfg['distillation']['teacher_chunk'])
                total, reconstruction, distillation = output_loss(model(lr), hr, target, cfg['distillation']['weight'])
                loss = total / accum
                rec_value += reconstruction.item() / accum
                kd_value += distillation.item() / accum
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                loss.backward(); loss_value += loss.item()
            opt.step(); sched.step(); update += 1
            step_log.write(json.dumps(dict(update=update, train_l1=rec_value, distill_l1=kd_value, total_loss=loss_value, lr=sched.get_last_lr()[0]))+'\n')
            epoch_loss_sum += loss_value
            epoch_rec_sum += rec_value
            epoch_kd_sum += kd_value
            epoch_updates += 1
            if update == run_update+1 or update % cfg.get('log_updates',100) == 0:
                elapsed = time.perf_counter()-run_start
                eta = elapsed/(update-run_update)*(stop-update)
                print(f'epoch={epoch} update={update}/{stop} progress={100*update/stop:.2f}% total={loss_value:.8f} rec_L1={rec_value:.8f} kd_L1={kd_value:.8f} lr={sched.get_last_lr()[0]:.8g} step_s={time.perf_counter()-step_start:.3f} data_wait_s={data_seconds:.3f} elapsed_min={elapsed/60:.1f} ETA_train_min={eta/60:.1f} (evaluation overhead not forecast)', flush=True)
        print(f'Evaluation starting at update={update}...', flush=True)
        scores = evaluate(model, val, metrics, cfg, out, epoch, update)
        epoch_loss = epoch_loss_sum / epoch_updates
        print(f'epoch={epoch} updates_in_epoch={epoch_updates} total_update={update} mean_total={epoch_loss:.8f} mean_rec_L1={epoch_rec_sum/epoch_updates:.8f} mean_kd_L1={epoch_kd_sum/epoch_updates:.8f}', flush=True)
        selected = False
        for name, row in scores.items():
            for metric, value in row.items():
                key = name + '/' + metric
                if key not in best or value > best[key]['value']:
                    best[key] = dict(value=value, epoch=epoch, update=update)
                    if name == cfg['selection_dataset'] and metric == 'psnr': selected = True
            print(f'{name}: PSNR={row["psnr"]:.8f} SSIM={row["ssim"]:.8f}; best={best[name+"/psnr"]}, {best[name+"/ssim"]}', flush=True)
        record = dict(epoch=epoch, update=update, updates_in_epoch=epoch_updates, samples_in_epoch=epoch_updates*cfg['effective_batch'], train_l1_mean=epoch_rec_sum/epoch_updates, train_l1_last=rec_value, distill_l1_mean=epoch_kd_sum/epoch_updates, total_loss_mean=epoch_loss, metrics=scores, best=best, elapsed_seconds=time.perf_counter()-start)
        state = dict(fingerprint=fingerprint, epoch=epoch, update=update, best=best, epoch_record=record, model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        atomic_save(state, out/'latest.pt')
        atomic_save(state, out/f'epoch{epoch:05d}_u{update}.pt')
        if selected: atomic_save(state, out/'best_validation.pt')
        with open(out/'epochs.jsonl','a',encoding='utf8') as f: f.write(json.dumps(record)+'\n')
    step_log.close()
    if loader: loader.close()
    print('Training complete. No stability or paper-value claim follows from a single run.', flush=True)

if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
