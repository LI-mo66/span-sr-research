"""GPU compute suite: 5-benchmark inference, 50K ablations, official LAM, feature hooks, and 720p RTX 4090 efficiency."""
import argparse
import ast
import csv
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.flop_counter import FlopCounterMode

from common_sci import *


METHOD_DIR_MAP = {
    'Bicubic': 'bicubic',
    'SPAN52': 'span52_official_single',
    'LBNet': 'lbnet_single',
    'DDistill-SR': 'ddistill_sr_single',
    'SMFANet': 'smfanet_single',
    'SMFANet+': 'smfanet_plus_single',
    'IMDN': 'imdn_single',
    'SwinIR-light': 'swinir_light_single',
    'NGswin': 'ngswin_single',
    'Ours': 'ours_single',
}


def bicubic_sr(lr_rgb):
    """Exact MATLAB-compatible bicubic upsampling matching BasicSR."""
    sys.path.insert(0, str(PROJECT))
    from basicsr.utils.matlab_functions import imresize
    t = torch.from_numpy(np.ascontiguousarray(lr_rgb.transpose(2, 0, 1))).float() / 255.0
    out = imresize(t, scale=4.0)
    return np.rint(out.clamp(0.0, 1.0).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)


def run_external(method_name, device):
    method_dir = METHOD_DIR_MAP[method_name]
    model, meta = None, dict(method=method_name, mode='single')

    def ensure_model():
        nonlocal model, meta
        if model is None:
            if method_name == 'Ours':
                model, meta = final_model(device, deploy_fold=False)
            elif method_name != 'Bicubic':
                model, meta = public_model(method_name, device, deploy_fold=False)
                # Odd-rectangle sanity check before full dataset inference
                with torch.no_grad():
                    for h, w in [(32, 40), (31, 37)]:
                        y = model(torch.zeros(1, 3, h, w, device=device))
                        if y.shape != (1, 3, h * 4, w * 4) or not torch.isfinite(y).all():
                            raise ValueError(f'{method_name} forward sanity check failed on {(h, w)}')
                # Pixel-level reproduction check if an already-verified Urban100 img004 PNG exists
                existing_check = find_existing_png('Urban100', method_dir, 'img004.png')
                if existing_check is not None:
                    _, lr0, _ = pair('Urban100', 'img004.png')
                    with torch.no_grad():
                        obs = quantize(model(tensor(lr0, device)))
                    ref = np.array(Image.open(existing_check).convert('RGB'))
                    max_diff = int(np.max(np.abs(obs.astype(np.int16) - ref.astype(np.int16))))
                    print(f'REPRO_CHECK {method_name} Urban100/img004 max_uint8_diff={max_diff}', flush=True)
                    if max_diff > 1:
                        raise ValueError(f'{method_name} forward diverges from existing verified PNG (max_diff={max_diff})')

    rows = []
    reused = 0
    computed = 0
    with torch.no_grad():
        for ds, count in ALL_DATASETS:
            images = sorted((dataset(ds) / 'HR').glob('*.png'))
            if len(images) != count:
                raise ValueError(f'Unexpected image count in {ds}: {len(images)} != {count}')
            started = time.monotonic()
            for i, hp in enumerate(images, 1):
                gt, lr, inp = pair(ds, hp.name)
                # Ensure aligned GT exists for every dataset (including Set5 & B100)
                if find_existing_png(ds, 'GT', hp.name) is None:
                    save_png(OUT / 'sr_outputs' / ds / 'GT' / hp.name, gt)
                existing = find_existing_png(ds, method_dir, hp.name)
                if existing is not None:
                    sr_path = existing
                    reused += 1
                else:
                    if method_name == 'Bicubic':
                        sr = bicubic_sr(lr)
                    else:
                        ensure_model()
                        sr = quantize(model(tensor(lr, device)))
                    if sr.shape != gt.shape:
                        raise ValueError(f'{method_name} shape mismatch on {ds}/{hp.name}: {sr.shape} vs {gt.shape}')
                    sr_path = OUT / 'sr_outputs' / ds / method_dir / hp.name
                    save_png(sr_path, sr)
                    computed += 1
                sr_read = np.array(Image.open(sr_path).convert('RGB'))
                if sr_read.shape != gt.shape:
                    raise ValueError(f'Shape mismatch on readback {sr_path}')
                score = metrics(sr_read, gt)
                rows.append(dict(
                    dataset=ds,
                    image_name=hp.name,
                    method=method_dir,
                    display_name=method_name,
                    psnr_y=score['psnr'],
                    ssim_y=score['ssim'],
                    width=gt.shape[1],
                    height=gt.shape[0],
                    png_sha256=sha(sr_path),
                    png=str(sr_path),
                    lr_sha256=inp['lr_sha256'],
                    hr_sha256=inp['hr_sha256'],
                ))
                if i % 25 == 0 or i == count:
                    print(f'EXT_PASS {method_name} {ds} {i}/{count} reused={reused} computed={computed} '
                          f'elapsed={time.monotonic()-started:.1f}s', flush=True)

    means = {}
    for ds, _ in ALL_DATASETS:
        sub = [r for r in rows if r['dataset'] == ds]
        means[ds] = dict(
            images=len(sub),
            psnr=float(np.mean([r['psnr_y'] for r in sub])),
            ssim=float(np.mean([r['ssim_y'] for r in sub])),
        )
    avg5_psnr = float(np.mean([means[ds]['psnr'] for ds, _ in ALL_DATASETS]))
    avg5_ssim = float(np.mean([means[ds]['ssim'] for ds, _ in ALL_DATASETS]))
    print(f'EXT_SUMMARY {method_name} Avg5={avg5_psnr:.4f}/{avg5_ssim:.6f} '
          + ' '.join(f"{ds}={means[ds]['psnr']:.2f}" for ds, _ in ALL_DATASETS), flush=True)

    slug = method_dir
    write_json(
        OUT / f'external-{slug}.json',
        dict(
            method=method_name,
            method_dir=method_dir,
            mode='single',
            reused_pngs=reused,
            computed_pngs=computed,
            metadata=meta,
            dataset_means=means,
            avg5_psnr=avg5_psnr,
            avg5_ssim=avg5_ssim,
            rows=rows,
        ),
        overwrite=True,
    )


def run_ablation(name, device):
    """Run or reuse 50K historical 48K ablation checkpoints on Urban100 img004 & img011."""
    old_marker = OLD_OUT / f'ablation-{name}.json'
    if old_marker.is_file():
        old_rec = read_json(old_marker)
        ok = True
        for r in old_rec.get('rows', []):
            p = find_existing_png('Urban100', f'{name}_50k_single', r['image'])
            if p is None or sha(p) != r['png_sha256']:
                ok = False
                break
        if ok and len(old_rec.get('rows', [])) == 2:
            print(f'ABLATION_REUSE_PASS {name} from {old_marker}', flush=True)
            write_json(OUT / f'ablation-{name}.json', old_rec, overwrite=True)
            return

    model, run, provenance = ablation_model(name, device)
    rows = []
    with torch.no_grad():
        for image in ['img004.png', 'img011.png']:
            gt, lr, inp = pair('Urban100', image)
            sr = quantize(model(tensor(lr, device)))
            if sr.shape != gt.shape:
                raise ValueError(f'Ablation output shape mismatch: {name} {image}')
            score = metrics(sr, gt)
            historical = read_json(run / 'Urban100_epoch00048_u48000.json')
            match = [r for r in historical if r['image'] == image]
            if len(match) != 1:
                raise ValueError(f'Historical per-image result missing: {name} {image}')
            error = {k: score[k] - float(match[0][k]) for k in ['psnr', 'ssim']}
            print(f'ABLATION_DELTA {name} {image} score={score} hist_delta={error}', flush=True)
            if abs(error['psnr']) > 5e-4 or abs(error['ssim']) > 1e-5:
                raise ValueError(f'Historical reproduction failure {name} {image}: {error}')
            # Save into OUT (avoiding collision if OLD_OUT had an incomplete/different file)
            dest = OUT / 'sr_outputs/Urban100' / f'{name}_50k_single' / image
            existing = find_existing_png('Urban100', f'{name}_50k_single', image)
            if existing is not None and np.array_equal(np.array(Image.open(existing).convert('RGB')), sr):
                dest = existing
            else:
                save_png(dest, sr)
            rows.append(dict(
                image=image,
                score=score,
                historical=dict(psnr=float(match[0]['psnr']), ssim=float(match[0]['ssim'])),
                historical_delta=error,
                png=str(dest),
                png_sha256=sha(dest),
                input=inp,
            ))
            print('ABLATION_REPRODUCTION_PASS', name, image, score, flush=True)

    write_json(
        OUT / f'ablation-{name}.json',
        dict(
            model=name,
            checkpoint=provenance,
            training_budget=50000,
            selected_update=48000,
            mode='single',
            rows=rows,
            scope='historical independent 50K same-protocol training selected at 48K; no late Bound8',
        ),
        overwrite=True,
    )


def official_lam_functions():
    folder = OUT / 'external/LAM_Demo'
    files = [
        ('SaliencyModel/BackProp.py', '1401bc836cf88b2eb255565e9c00e66b6ada4ce2'),
        ('SaliencyModel/utils.py', 'dc9d8f35c214eaa51b19b93bb921e47592b2c587'),
        ('SaliencyModel/attributes.py', 'abda92e9eeb6bc0a7684f5dee116aacd883e6c50'),
    ]
    keep = {'GaussianBlurPath', 'isotropic_gaussian_kernel', 'grad_abs_norm', 'gini', 'attr_grad', 'reduce_func'}
    body = []
    sources = []
    for rel, blob in files:
        path = folder / rel
        sources.append(fetch(f'https://raw.githubusercontent.com/X-Lowlevel-Vision/LAM_Demo/{LAM_COMMIT}/{rel}',
                             path, blob=blob))
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        body.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in keep)
    namespace = dict(np=np, torch=torch, cv2=cv2)
    exec(compile(ast.Module(body=body, type_ignores=[]), 'official_LAM_selected_functions', 'exec'), namespace)
    if not keep.issubset(namespace):
        raise ValueError('Missing pinned official LAM functions')
    return namespace, sources


def stable_objective(prediction, y, x, window=16):
    a = prediction[:, :, y:y + window, x:x + window]
    dy = a - prediction[:, :, y + 1:y + window + 1, x:x + window]
    dx = a - prediction[:, :, y:y + window, x + 1:x + window + 1]
    return torch.linalg.vector_norm(torch.stack([dy, dx]), dim=0).sum()


def run_lam(name, device, allow_skip=False):
    try:
        fn, sources = official_lam_functions()
        model, weight = final_model(device) if name == 'Ours' else public_model(name, device)
        model.requires_grad_(False)
        method_dir = METHOD_DIR_MAP[name]
        records = []
        for image, target in [('img004.png', [904, 472, 16, 16]), ('img011.png', [296, 920, 16, 16])]:
            gt, lr, inputs = pair('Urban100', image)
            x, y, w, h = target
            if x < 0 or y < 0 or x + w >= gt.shape[1] or y + h >= gt.shape[0]:
                raise ValueError('LAM target outside output-gradient domain')
            with torch.no_grad():
                observed = quantize(model(tensor(lr, device)))
            ref = get_image('Urban100', method_dir, image)
            if not np.array_equal(observed, ref):
                max_diff = int(np.max(np.abs(observed.astype(np.int16) - ref.astype(np.int16))))
                if max_diff > 1:
                    raise ValueError(f'LAM model forward differs from verified PNG: {name} {image} max_diff={max_diff}')
            print('LAM_MODEL_REPRODUCTION_PASS', name, image, flush=True)
            raw = lr.astype(np.float32) / 255.0
            path, velocity = fn['GaussianBlurPath'](1.2, 50, l=5)(raw)
            attribution = np.zeros_like(path[0])
            objectives = []
            started = time.monotonic()
            for i in range(50):
                value = torch.from_numpy(path[i:i + 1]).to(device).requires_grad_(True)
                prediction = model(value)
                objective = stable_objective(prediction, y, x, window=16)
                gradient, = torch.autograd.grad(objective, value)
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f'LAM nonfinite input gradient: {name} {image} fold{i+1}')
                attribution += gradient[0].detach().cpu().numpy() * velocity[i] / 50.0
                objectives.append(float(objective.detach()))
                del gradient, objective, prediction, value
                if (i + 1) % 10 == 0:
                    print(f'LAM_PROGRESS {name} {image} {i+1}/50 elapsed={time.monotonic()-started:.1f}s', flush=True)
            norm = fn['grad_abs_norm'](attribution)
            if not np.isfinite(norm).all() or norm.max() <= 0:
                raise ValueError('Empty or invalid LAM map')
            di = float((1.0 - fn['gini'](norm.copy())) * 100.0)
            if not (0.0 <= di <= 100.0):
                raise ValueError(f'Invalid diffusion index: {di}')
            for dest_base in (OUT / 'lam' / name, OUT / 'portable_render_assets/lam' / name):
                dest_base.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    dest_base / (Path(image).stem + '.npz'),
                    attribution=attribution,
                    normalized=norm,
                    lr_rgb=lr,
                    objectives=np.asarray(objectives),
                )
            records.append(dict(
                image=image,
                target_hr_xywh=target,
                DI=di,
                input=inputs,
                lr_shape=list(lr.shape),
                seconds=time.monotonic() - started,
            ))
            print('LAM_PASS', name, image, f'DI={di:.4f}', flush=True)
        write_json(
            OUT / f'lam-{name}.json',
            dict(
                status='PASS',
                model=name,
                weight=weight,
                official_commit=LAM_COMMIT,
                sources=sources,
                settings=dict(sigma_LR=1.2, fold=50, kernel=5, target_window_HR=16),
                mode='single',
                rows=records,
            ),
            overwrite=True,
        )
    except Exception as exc:
        if not allow_skip:
            raise
        print(f'LAM_SKIPPED {name}: {type(exc).__name__}: {exc}', flush=True)
        write_json(
            OUT / f'lam-{name}.json',
            dict(status='SKIPPED_INCOMPATIBLE', model=name, error=f'{type(exc).__name__}: {exc}'),
            overwrite=True,
        )


def run_features(device):
    model, weight = final_model(device, deploy_fold=False)
    captured = {}
    handles = []
    rows = []

    def stage_hook(label):
        def callback(module, args, output):
            captured[label] = output[0].detach().float().cpu().clone().numpy()[0]
        return callback

    for i in [2, 3, 4, 5]:
        handles.append(getattr(model, f'block_{i}').register_forward_hook(stage_hook(f'F{i}')))

    def gate_hook(module, args, output):
        captured['G'] = output[2].detach().float().cpu().clone().numpy()[0]

    handles.append(model.block_3.original.register_forward_hook(gate_hook))
    try:
        for image in ['img004.png', 'img011.png']:
            captured.clear()
            gt, lr, inputs = pair('Urban100', image)
            with torch.no_grad():
                sr = quantize(model(tensor(lr, device)))
            ref = get_image('Urban100', 'ours_single', image)
            max_err = float(np.max(np.abs(sr.astype(np.float32) - ref.astype(np.float32))))
            if max_err != 0.0:
                raise ValueError(f'Feature-hook inference differs from verified Ours PNG: max_err={max_err}')
            if set(captured) != {'G', 'F2', 'F3', 'F4', 'F5'}:
                raise ValueError(f'Missing captured feature tensors: {set(captured)}')
            if any(a.shape != (52, *lr.shape[:2]) or not np.isfinite(a).all() for a in captured.values()):
                raise ValueError('Feature/gate shape or numerical mismatch')
            maps = {k: np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=0)) for k, a in captured.items() if k != 'G'}
            maps['G_mean_signed'] = captured['G'].mean(0)
            for dest_dir in (OUT / 'features', OUT / 'portable_render_assets/features'):
                dest_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    dest_dir / (Path(image).stem + '.npz'),
                    **captured,
                    **{k + '_rms': v for k, v in maps.items() if k != 'G_mean_signed'},
                    G_mean_signed=maps['G_mean_signed'],
                )
            stats = {}
            for k, a in captured.items():
                stats[k] = dict(
                    shape=list(a.shape),
                    mean=float(a.mean()),
                    rms=float(np.sqrt(np.mean(a.astype(np.float64) ** 2))),
                    min=float(a.min()),
                    max=float(a.max()),
                    p01=float(np.quantile(a, 0.01)),
                    p99=float(np.quantile(a, 0.99)),
                )
            rows.append(dict(image=image, hook_max_abs_diff_vs_ours_png=max_err, inputs=inputs, statistics=stats))
            print('FEATURES_PASS', image, 'hook_max_err=0.0', flush=True)
    finally:
        for h in handles:
            h.remove()

    write_json(
        OUT / 'fig6_stats.json',
        dict(
            weight=weight,
            gate='True SG6 block_3.original output[2], 52-channel signed gate; displayed as channel mean with zero-centered scale (no extra sigmoid)',
            stages='F2/F3/F4/F5 captured after post-block UCAN/W/UCAN/LKSA adapters with Bound8 installed',
            rows=rows,
        ),
        overwrite=True,
    )


def _measure_one(method_label, scale, model, lr_h, lr_w, device, warmup=20, repeats=100, note=''):
    model = model.to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    x = torch.rand(1, 3, lr_h, lr_w, device=device, dtype=torch.float32)

    # Exact FLOPs via PyTorch built-in FlopCounterMode (1 MAC = 2 FLOPs)
    with torch.inference_mode():
        fc = FlopCounterMode(display=False)
        with fc:
            y = model(x)
        flops = int(fc.get_total_flops())
        if y.shape != (1, 3, lr_h * scale, lr_w * scale):
            raise ValueError(f'{method_label} unexpected 720p output shape: {y.shape}')

    # Warmup & timed CUDA-synchronized runs
    times_ms = []
    peak_mem_mb = 0.0
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        with torch.inference_mode():
            for _ in range(warmup):
                _ = model(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(repeats):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(x)
                torch.cuda.synchronize()
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    else:
        with torch.inference_mode():
            for _ in range(3):
                _ = model(x)
            for _ in range(10):
                t0 = time.perf_counter()
                _ = model(x)
                times_ms.append((time.perf_counter() - t0) * 1000.0)

    del x, y, model
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    rec = dict(
        Method=method_label,
        Scale=f'x{scale}',
        Params_Exact=params,
        Params_K=round(params / 1000.0, 2),
        MACs_G=round(flops / 2.0 / 1e9, 4),
        FLOPs_G=round(flops / 1e9, 4),
        Latency_ms=round(float(statistics.mean(times_ms)), 3),
        Latency_median_ms=round(float(statistics.median(times_ms)), 3),
        PeakMem_MB=round(float(peak_mem_mb), 2),
        Avg5_PSNR='',
        Avg5_SSIM='',
        Note=note,
    )
    print(f"EFF_720P {method_label} x{scale}: Params={rec['Params_K']}K "
          f"MACs={rec['MACs_G']}G FLOPs={rec['FLOPs_G']}G "
          f"Lat={rec['Latency_ms']}ms PeakMem={rec['PeakMem_MB']}MB", flush=True)
    return rec


def run_efficiency(device, warmup=20, repeats=100):
    """Task 6: Unified 1280x720 output complexity & latency benchmark on RTX 4090."""
    gpu_name = torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU'
    rows = []

    # Verify Ours deploy folding matches unfolded output exactly before benchmarking
    ours_raw, _ = final_model(device, deploy_fold=False)
    ours_dep, dep_meta = final_model(device, deploy_fold=True)
    with torch.inference_mode():
        probe = torch.rand(1, 3, 32, 40, device=device)
        diff = float((ours_raw(probe) - ours_dep(probe)).abs().max().item())
    if diff > 1e-5:
        raise ValueError(f'Ours deploy folding mismatch: max_diff={diff}')
    if dep_meta['parameters'] != 569895:
        raise ValueError(f"Expected 569895 deploy params for Ours x4, got {dep_meta['parameters']}")
    del ours_raw
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    rows.append(_measure_one(
        'Ours', 4, ours_dep, 180, 320, device, warmup=warmup, repeats=repeats,
        note=f'Deployed folded (569,895) + Bound8; fold_max_diff={diff:.2e}',
    ))

    # Structural x2 deployed model (no x2 checkpoint on server; exact architecture Params/FLOPs + runtime)
    ours_x2, x2_meta = structural_ours_x2_deploy(device)
    rows.append(_measure_one(
        'Ours', 2, ours_x2, 360, 640, device, warmup=warmup, repeats=repeats,
        note=f"Structural x2 deployed ({x2_meta['parameters']}) + Bound8; no x2 checkpoint on server",
    ))

    # Competitors (x4)
    for name in ['SPAN52', 'SMFANet', 'SMFANet+', 'LBNet', 'DDistill-SR', 'IMDN', 'SwinIR-light', 'NGswin']:
        try:
            m, _ = public_model(name, device, deploy_fold=(name == 'SPAN52'))
            rows.append(_measure_one(
                name, 4, m, 180, 320, device, warmup=warmup, repeats=repeats,
                note='Deployed folded' if name == 'SPAN52' else 'Official x4 checkpoint',
            ))
        except Exception as exc:
            print(f'EFF_SKIP {name}: {type(exc).__name__}: {exc}', flush=True)
            rows.append(dict(
                Method=name, Scale='x4', Params_Exact='', Params_K='', MACs_G='', FLOPs_G='',
                Latency_ms='', Latency_median_ms='', PeakMem_MB='', Avg5_PSNR='', Avg5_SSIM='',
                Note=f'SKIPPED: {type(exc).__name__}: {exc}',
            ))

    # Attach Avg5_PSNR/SSIM if external summary JSONs already exist
    for r in rows:
        if r['Scale'] == 'x4' and r['Method'] in METHOD_DIR_MAP:
            ext_json = OUT / f"external-{METHOD_DIR_MAP[r['Method']]}.json"
            if ext_json.is_file():
                d = read_json(ext_json)
                r['Avg5_PSNR'] = round(float(d['avg5_psnr']), 4)
                r['Avg5_SSIM'] = round(float(d['avg5_ssim']), 6)

    csv_path = OUT / 'efficiency_720p_rtx4090.csv'
    fieldnames = ['Method', 'Scale', 'Params_K', 'MACs_G', 'FLOPs_G', 'Latency_ms', 'PeakMem_MB', 'Avg5_PSNR', 'Avg5_SSIM']
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    write_json(
        OUT / 'efficiency_720p_rtx4090.json',
        dict(
            hardware=gpu_name,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            precision='FP32',
            batch_size=1,
            output_resolution='1280x720',
            flops_convention='1 MAC = 2 FLOPs (PyTorch FlopCounterMode)',
            warmup=warmup,
            repeats=repeats,
            rows=rows,
        ),
        overwrite=True,
    )
    print(f'EFFICIENCY_COMPLETE csv={csv_path}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['external', 'ablation', 'lam', 'features', 'efficiency'])
    p.add_argument('--name', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--allow-skip', action='store_true')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=100)
    args = p.parse_args()

    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')

    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == 'external':
        run_external(args.name, args.device)
    elif args.action == 'ablation':
        run_ablation(args.name, args.device)
    elif args.action == 'lam':
        run_lam(args.name, args.device, allow_skip=args.allow_skip)
    elif args.action == 'features':
        run_features(args.device)
    elif args.action == 'efficiency':
        run_efficiency(args.device, warmup=args.warmup, repeats=args.repeats)


if __name__ == '__main__':
    main()
