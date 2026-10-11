# Baseline source and protocol audit

Evidence date: 2026-10-11 (Asia/Shanghai).

## Verified source facts

- Upstream commit c77a5917759f09e66fbc7124220c5afc5ee221e5; six SPABs, 48 channels, ×4, Conv3XC training reparameterization. Standard SPAN, not SPAN-F or 28-channel challenge entry.
- Paper section 4.1: DF2K 3450 images; bicubic degradation; 256 HR patches; effective batch 64; flip/rotation; Adam L1; LR 5e-4 halved every 200k updates; 1M updates on four A100s. Y-channel PSNR/SSIM over the five named benchmarks (text mistakenly says four).
- Paper says repeat training twice after loading trained weights. Exact stage interpretation and optimizer reset are not supplied. Current configuration covers the first stage only; paper-table reproduction remains pending.
- Repository ships EDSR options, not an official SPAN training option. Adam betas/epsilon, seed and exact official metric details are not all established by this source. Our defaults are declared assumptions.
- Upstream forward subtracts mean and multiplies by img_range, then returns upsampler output without inverse normalization. Released params were now tested on all five benchmarks and work directly with [0,1] output quantization. Preserve forward verbatim; do not add inverse normalization. Full accuracy comparison is in official-weights-test.md.
- Eval Conv3XC performs update_params on every forward. Parameter totals include frozen eval_conv weights; trainable count differs from deployment count. Profile currently reports both totals and measured original eval cost. Conv-only MAC/FLOP subtotal excludes fusion and pointwise work.
- Proposed metric convention: quantize RGB output [0,1] to uint8, use original BasicSR BGR/Y conversion and metric implementations, crop four HR pixels, whole-image inference, self-ensemble off. Crop/quantization details need comparison with the official reported evaluation protocol.

## Interfaces

| Path | Shape/layout | Range/state/gradient |
| --- | --- | --- |
| Input | NCHW RGB, float32, CUDA | [0,1], aligned LR pixels |
| Train HR | NCHW RGB, exactly 4H × 4W | [0,1], augmentation shared with LR |
| SPAB | N×48×H×W, preserves spatial order | no external normalization/mask, residual `(out3+x)*attention`; attention sigmoid minus .5 |
| Conv3XC train | same spatial grid | trainable branches, padded LR input; gradients into branches |
| Conv3XC eval | same grid | fused frozen eval kernel refreshed every forward, detached caches |
| Output | N×3×4H×4W | raw upstream output, unclamped for loss; clamp/round only for metrics |

Engineering verification covers odd rectangular dimensions, finite full gradients, real optimizer update, strict checkpoint reload, and local train/eval fusion equivalence. It does not prove PSNR, dataset correctness or convergence.

## Current blockers

User supplied datasets/; DIV2K and five benchmark pair audit completed. Flickr2K official archive downloaded and2650 HR/2650 LR x4 images extracted; combined DF2K audit remains pending. User-supplied benchmark bicubic provenance still unverified. Released-weight evaluation completed, from-scratch baseline training not started. GitHub independent private repository LI-mo66/span-sr-research created and pushed with upstream history. Local GPU RTX4060 8GB; PyTorch2.9.1+cu128 available in dl environment. Synthetic resource profiles are in engineering-evidence.md; real-data long-run time and training memory remain unknown.
