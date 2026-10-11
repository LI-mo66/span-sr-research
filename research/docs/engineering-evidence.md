# Engineering evidence, 2026-10-11

Synthetic CPU check PASSED: odd rectangular input 13×17 -> 52×68, finite loss and gradients for all trainable parameters, actual Adam weight update, strict serialized reload with identical eval output, and Conv3XC train/eval equivalence within specified tolerance. These checks do not show any accuracy improvement.

Original eval path profiled on RTX4060 Laptop GPU, PyTorch2.9.1+cu128, FP32, random weights, LR64×64, warmup5/repetitions20: median8.5852ms, mean8.8135ms, peak allocated24,801,792 bytes, peak reserved50,331,648 bytes. Total stored parameters2,236,728; trainable1,840,488. Conv module subtotal1,741,750,272 MACs /3,483,500,544 FLOPs (2 per MAC). This excludes kernel fusion and pointwise costs and is not a total FLOPs count. Small synthetic shape timing is preliminary, not publication-ready full-image timing.

Failures retained in this record: script was initially named profile.py and shadowed Python's standard profile module, blocking Adam import; renamed benchmark.py. TemporaryDirectory on sandbox's system temporary path blocked torch.save; changed verification fixture to repository tmp/. No model architecture or production optimizer was altered to bypass these failures.

Data location supplied by user: datasets/. DIV2K800 train/100 valid, Set5 5, Set14 14, B100100, Urban100100, Manga109109 present. Pair alignment and content-overlap audit runs separately. Existing bicubic provenance remains unverified. Standard modcrop trims only 0–3 trailing HR rows/columns to align with x4 LR, and rejects other mismatch.

Flickr2K official archive: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar, Content-Length21,542,973,440. Windows curl TLS handshake failed; Python urllib with default TLS verification succeeded. A resumable download is running to datasets/downloads/Flickr2K.tar.part; full completion, SHA256 and extraction are not yet established. No incomplete archive is admitted to training.

GitHub CLI remains unauthenticated. The connector identifies LI-mo66 but cannot create repositories through its currently supplied tools. Remote creation/push is pending authentication. Username alone is not an authentication credential.

Synthetic CUDA integration run ENGINEERING-ONLY-7b42c83912 PASSED one real optimizer update, all six evaluation streams, per-image files, best/latest checkpoint serialization and fingerprint-checked resume. Its random image PSNR/SSIM values are deliberately excluded from research accuracy tables.

Pre-fused deployment export PASSED full-network CPU equality within atol1e-6/rtol1e-5 on the odd rectangle. Deployed parameters426,288; first-round +10% budget permits at most42,628 extra deployed parameters. Preliminary deployed FP32 LR256×256 GPU benchmark, warmup20/repetitions100: median12.06095ms, mean12.132295ms, p95 12.7074ms; allocated254,945,792 bytes, reserved306,184,192 bytes. Convolution subtotal27,868,004,352 MACs. These are synthetic efficiency measurements, not trained-model accuracy or complete FLOPs. The two recorded latency paths use different image sizes and must not be directly compared.

Existing data audit completed: all provided DIV2K and five benchmark pairs have x4-compatible dimensions; no exact-HR SHA256 overlap between DIV2K training and validation/test data. See data-audit.json. This cannot establish the original bicubic resampling kernel.
