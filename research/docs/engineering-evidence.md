# Engineering evidence, 2026-10-11

Synthetic CPU check PASSED: odd rectangular input 13×17 -> 52×68, finite loss and gradients for all trainable parameters, actual Adam weight update, strict serialized reload with identical eval output, and Conv3XC train/eval equivalence within specified tolerance. These checks do not show any accuracy improvement.

Original eval path profiled on RTX4060 Laptop GPU, PyTorch2.9.1+cu128, FP32, random weights, LR64×64, warmup5/repetitions20: median8.5852ms, mean8.8135ms, peak allocated24,801,792 bytes, peak reserved50,331,648 bytes. Total stored parameters2,236,728; trainable1,840,488. Conv module subtotal1,741,750,272 MACs /3,483,500,544 FLOPs (2 per MAC). This excludes kernel fusion and pointwise costs and is not a total FLOPs count. Small synthetic shape timing is preliminary, not publication-ready full-image timing.

Failures retained in this record: script was initially named profile.py and shadowed Python's standard profile module, blocking Adam import; renamed benchmark.py. TemporaryDirectory on sandbox's system temporary path blocked torch.save; changed verification fixture to repository tmp/. No model architecture or production optimizer was altered to bypass these failures.

Data location supplied by user: datasets/. DIV2K800 train/100 valid, Set5 5, Set14 14, B100100, Urban100100, Manga109109 present. Pair alignment and content-overlap audit runs separately. Existing bicubic provenance remains unverified. Standard modcrop trims only 0–3 trailing HR rows/columns to align with x4 LR, and rejects other mismatch.

Flickr2K official archive: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar, Content-Length21,542,973,440. Windows curl TLS handshake failed; Python urllib with default TLS verification succeeded. A resumable download is running to datasets/downloads/Flickr2K.tar.part; full completion, SHA256 and extraction are not yet established. No incomplete archive is admitted to training.

GitHub CLI remains unauthenticated. The connector identifies LI-mo66 but cannot create repositories through its currently supplied tools. Remote creation/push is pending authentication. Username alone is not an authentication credential.
