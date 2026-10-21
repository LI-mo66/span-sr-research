"""Disposable candidate-only FP32 memory/backend check; never reused for training."""
import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from model import build_model
from run import assm_runtime as runtime, validate_config
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    validate_config(cfg)
    torch.set_num_threads(cfg.get('cpu_threads', 4))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(cfg['seed'])
    arch, _ = runtime.components()
    model = build_model(arch, cfg).cuda().train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                           betas=cfg['adam_betas'], eps=cfg['adam_eps'])
    torch.cuda.reset_peak_memory_stats()
    paths = json.loads(Path(cfg['paths_file']).read_text())
    training = runtime.trainer.pairs(paths['train'])
    rng = random.Random(10)
    # Matched sampling and accumulation on real images; discard this model.
    accumulation = cfg['effective_batch'] // cfg['micro_batch']
    for _ in range(accumulation):
        samples = [runtime.trainer.sample(training[rng.randrange(len(training))], cfg['hr_patch'], rng)
                   for _ in range(cfg['micro_batch'])]
        lr = torch.stack([row[0] for row in samples]).cuda()
        hr = torch.stack([row[1] for row in samples]).cuda()
        loss = torch.nn.functional.l1_loss(model(lr), hr) / accumulation
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite smoke loss')
        loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all()
           for p in model.parameters() if p.requires_grad):
        raise ValueError('Missing/nonfinite smoke gradients')
    opt.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite parameters after real crop update')
    torch.cuda.synchronize()
    training_peak = torch.cuda.max_memory_allocated()
    opt.zero_grad(set_to_none=True)
    del opt, lr, hr, loss
    model.eval()
    paths = json.loads(Path(cfg['paths_file']).read_text())
    largest = None
    area = -1
    for spec in paths['validation'].values():
        for _, lp in runtime.trainer.pairs(spec):
            image = runtime.trainer.read(lp)
            size = image.shape[0] * image.shape[1]
            if size > area:
                largest, area = lp, size
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        value = runtime.trainer.tensor(runtime.trainer.read(largest)).unsqueeze(0).cuda()
        output = model(value)
        if list(output.shape)!=[1,3,value.shape[-2]*4,value.shape[-1]*4]:
            raise ValueError('Whole-image x4 size mismatch')
        if not torch.isfinite(output).all():
            raise ValueError('Nonfinite whole-image smoke prediction')
    torch.cuda.synchronize()
    report = dict(status='PASS', micro_batch=cfg['micro_batch'],
                  accumulation=accumulation, training_peak_allocated=training_peak,
                  largest_lr=str(largest), eval_shape=list(value.shape),
                  eval_peak_allocated=torch.cuda.max_memory_allocated(),
                  evidence='Disposable real-crop candidate smoke, largest-area validation whole image; weights discarded, no PSNR conclusion')
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
