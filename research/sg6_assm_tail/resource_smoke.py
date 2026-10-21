"""Disposable full-model micro16 accumulation smoke, synthetic data only."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from model import build_model
import runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((Path(__file__).parent/'configs/M2-tail-rms-lr2e4.json').read_text())
    assert cfg['assm_settings']['backend'] == 'cuda_isolated'
    arch, _ = runtime.components()
    torch.manual_seed(10)
    model = build_model(arch, cfg).cuda().train()
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(.9,.999), eps=1e-8)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    opt.zero_grad(set_to_none=True)
    for _ in range(4):
        x = torch.rand(16,3,64,64,device='cuda')
        target = torch.rand(16,3,256,256,device='cuda')
        y = model(x)
        assert y.shape == target.shape and torch.isfinite(y).all()
        loss = torch.nn.functional.l1_loss(y,target)/4
        assert torch.isfinite(loss)
        loss.backward()
        losses.append(float(loss.detach())*4)
        del x,target,y,loss
    for name,p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    torch.cuda.synchronize()
    report = dict(status='PASS_SYNTHETIC_MICRO16',backend='cuda_isolated',
        lr_shape=[16,3,64,64],hr_shape=[16,3,256,256],accumulations=4,updates=1,
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),losses=losses,
        scope='Disposable synthetic update; actual dataset hashes, largest whole-image evaluation and trainer integration pending')
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
