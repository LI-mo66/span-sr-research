"""Inspect a trained checkpoint's branch use; bypasses are not training ablations."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cata_runtime', HERE / 'runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
import numpy as np
import torch
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--lr-image',required=True)
    parser.add_argument('--hr-image')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads(Path(args.config).read_text())
    arch,metrics = runtime.components()
    torch.manual_seed(cfg['seed'])
    model = build_model(arch,cfg).to(args.device).eval()
    state = torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    if not model.content_tab.initted.item():
        raise ValueError('Checkpoint centers have not been initialized by training')
    image = runtime.trainer.read(Path(args.lr_image))
    value = runtime.trainer.tensor(image).unsqueeze(0).to(args.device)
    features = []
    handle = model.content_tab.register_forward_pre_hook(
        lambda _,inputs:features.append(inputs[0].detach().clone()))
    buffers = {k:v.clone() for k,v in model.content_tab.named_buffers()}
    with torch.inference_mode():
        full = model(value)
    handle.remove()
    original = torch.nn.functional.scaled_dot_product_attention
    predictions = {'full':full}
    for disabled in ('group','center'):
        calls = [0]
        def bypass(*inputs,**kwargs):
            output = original(*inputs,**kwargs)
            branch = 'group' if calls[0] % 2 == 0 else 'center'
            calls[0] += 1
            return torch.zeros_like(output) if branch == disabled else output
        with torch.inference_mode(), patch('torch.nn.functional.scaled_dot_product_attention',bypass):
            predictions[disabled+'_disabled'] = model(value)
    with torch.inference_mode():
        _,stats = model.content_tab.forward_with_diagnostics(features[0])
    assert all(torch.equal(v,buffers[k]) for k,v in model.content_tab.named_buffers())
    report = dict(checkpoint=args.checkpoint,update=state.get('update'),
                  image=args.lr_image,tab=stats,variants={},eval_buffers_unchanged=True,
                  scope='same-checkpoint dependence only; not independently trained ablations or proof of PSNR gain')
    hr = None
    if args.hr_image:
        hr = runtime.trainer.read(Path(args.hr_image))
        h,w = image.shape[0]*4,image.shape[1]*4
        if not (0 <= hr.shape[0]-h < 4 and 0 <= hr.shape[1]-w < 4):
            raise ValueError('HR/LR do not follow the common x4 alignment protocol')
        hr = hr[:h,:w]
    for name,prediction in predictions.items():
        row = dict(output_change_ratio=float((prediction-full).norm()/full.norm().clamp_min(1e-12)))
        if hr is not None:
            sr = prediction[0].clamp(0,1).cpu().numpy().transpose(1,2,0)[:,:,::-1]
            sr = np.rint(sr*255).astype(np.uint8)
            settings = dict(crop_border=cfg['crop_border'],test_y_channel=cfg['test_y_channel'])
            row.update(psnr=metrics.calculate_psnr(sr,hr,**settings),
                       ssim=metrics.calculate_ssim(sr,hr,**settings))
        report['variants'][name] = row
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
