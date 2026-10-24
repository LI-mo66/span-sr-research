"""SG6 unchanged model, explicit trainable-parameter averaging and exact metrics."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research/spatial_gate6'))
import entry as sg_entry
import numpy as np
import torch
from sg_model import build_model
from deploy import fused_copy

trainer = sg_entry.existing.fs_runtime.trainer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf8')
    os.replace(tmp, path)


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def trainable(model):
    return {k: v for k, v in model.named_parameters() if v.requires_grad}


def refresh(model, arch):
    for module in model.modules():
        if isinstance(module, arch.Conv3XC):
            module.update_params()


def state_copy(model, arch):
    refresh(model, arch)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def install(model, values, arch):
    params = trainable(model)
    if set(params) != set(values):
        raise ValueError('Trainable parameter keys differ')
    with torch.no_grad():
        for name, p in params.items():
            v = values[name]
            if v.shape != p.shape or not torch.isfinite(v).all():
                raise ValueError(f'Bad averaged parameter: {name}')
            p.copy_(v)
    refresh(model, arch)


def update_ema(shadow, model, decay=.999):
    with torch.no_grad():
        for name, p in trainable(model).items():
            shadow[name].mul_(decay).add_(p.detach(), alpha=1-decay)


def average(models, names):
    keys = set(models[0])
    if any(set(m) != keys for m in models):
        raise ValueError('Checkpoint structure mismatch')
    result = {}
    for name in names:
        values = [m[name] for m in models]
        if any(v.shape != values[0].shape or v.dtype != values[0].dtype or not torch.isfinite(v).all() for v in values):
            raise ValueError(f'Invalid checkpoint parameter: {name}')
        result[name] = sum(v.double() for v in values).div(len(values)).to(values[0].dtype)
    return result


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def step(model, opt, scheduler, shadow, data, cfg, objective, device):
    model.train(); opt.zero_grad(set_to_none=True)
    accumulation = cfg['effective_batch'] // cfg['micro_batch']
    total = common_l1 = 0.
    for _ in range(accumulation):
        samples = [trainer.sample(random.choice(data), cfg['hr_patch']) for _ in range(cfg['micro_batch'])]
        low = torch.stack([s[0] for s in samples]).to(device)
        high = torch.stack([s[1] for s in samples]).to(device)
        prediction = model(low)
        l1 = torch.nn.functional.l1_loss(prediction, high)
        loss = l1 if objective == 'L1' else torch.nn.functional.mse_loss(prediction, high)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite fine-tuning loss')
        (loss / accumulation).backward()
        total += float(loss.detach()) / accumulation
        common_l1 += float(l1.detach()) / accumulation
    for p in trainable(model).values():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise FloatingPointError('Missing or nonfinite gradient')
    opt.step(); scheduler.step(); update_ema(shadow, model)
    return dict(objective=total, train_l1=common_l1, lr_next=opt.param_groups[0]['lr'])


def evaluate(model, pairs, metrics, cfg, path, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for index, (hp, lp) in enumerate(pairs, 1):
            hr, lr = trainer.aligned(hp, lp)
            prediction = model(trainer.tensor(lr).unsqueeze(0).to(device))[0]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f'Nonfinite evaluation: {lp}')
            sr = np.rint(prediction.clamp(0,1).cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8)
            if sr.shape != hr.shape:
                raise ValueError('Evaluation shape mismatch')
            kw = dict(crop_border=cfg['crop_border'], test_y_channel=cfg['test_y_channel'])
            rows.append(dict(image=hp.name, psnr=metrics.calculate_psnr(sr,hr,**kw), ssim=metrics.calculate_ssim(sr,hr,**kw)))
            if index % 20 == 0:
                print(f'{path.stem}: {index}/{len(pairs)}', flush=True)
    write(path, rows)
    return {m: float(np.mean([r[m] for r in rows])) for m in ('psnr','ssim')}


def deployment_check(model, arch, device):
    model.eval(); deployed = fused_copy(model, arch.Conv3XC)
    # Fixed input must not consume the training RNG.
    image = torch.linspace(0,1,3*13*17,device=device).reshape(1,3,13,17)
    with torch.inference_mode():
        a, b = model(image), deployed(image)
        error = float((a-b).abs().max())
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    count = sum(p.numel() for p in deployed.parameters())
    if count != 535772:
        raise ValueError('SG6 deployment parameter count changed')
    return dict(parameters=count, absolute_max_error=error)
