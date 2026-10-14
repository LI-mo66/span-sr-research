# Reusable B0 baseline: SPAN52 x4, seed 10, 50000 updates

Recorded 2026-10-14. Source: user-provided server reports archived alongside this document. Checkpoint hashes and backup manifests remain on the server and have not been independently audited. This is a research baseline, not an exact claim of official paper reproduction.

## Training protocol

- B0 original SPAN52 x4, 498208 deployed parameters, RGB input/output, feature channels 52, img_range 255.
- Random initialization, seed 10; no external pretrained weights. RGB L1 loss. No teacher or EMA in this training entry.
- DF2K: DIV2K training 800 + Flickr2K 2650 paired bicubic x4 images. Fixed DIV2K validation 100; five public benchmarks.
- HR crop 256x256 / LR crop 64x64; effective batch 64, micro batch 16, four accumulated micro batches per optimizer update. Random crop, horizontal/vertical flips and transpose.
- Adam lr 0.0002, betas (0.9, 0.999), eps 1e-8, weight_decay default 0. MultiStepLR milestones [200000,400000,600000,800000], gamma 0.5. Learning rate stays 0.0002 through update 50000. Full configured horizon 1000000; executed budget 50000.
- Started from zero to 20000, resumed latest to 50000 including optimizer, scheduler and Python/NumPy/Torch/CUDA RNG states. Evaluate/save every 1000 optimizer updates. These counts are updates, not dataset epochs.
- FP32, deterministic algorithms, cuDNN benchmark false, cuDNN TF32 true, matmul TF32 false.
- Server: RTX4090; Python3.11.8, torch2.2.2+cu121, NumPy1.26.4, OpenCV4.10.0, SciPy1.13.1, matplotlib3.8.4. Complete pip freeze is in server manifest.json.
- Training commit d280dc5ebcffc46c50983dfe27e81c6096d90dcf. Runtime argparse wrapper only admitted stop cap 50000; no model/optimizer protocol change.

## Evaluation and selection

Full-image inference, batch 1, no self ensemble; output clamped [0,1], rounded to uint8 [0,255]; BasicSR PSNR/SSIM on Y channel, border 4. Each dataset score averages its images. No average across checkpoints or datasets.

Primary selection: highest DIV2K_valid PSNR within updates 1000..50000, earliest tie. B0 selected epoch00048_u48000.pt; all six B0 dataset maxima also occur at 48000. A1 primary checkpoint is 47000; its per-dataset maxima use distinct checkpoints.

## Results: per-dataset maximum PSNR

| Dataset | B0 PSNR | B0 SSIM at PSNR peak | B0 update | A1 PSNR | A1 update | A1 minus B0 dB |
|---|---:|---:|---:|---:|---:|---:|
| DIV2K_valid | 30.159891247 | 0.831052143 | 48000 | 30.136644038 | 47000 | -0.023247209 |
| Set5 | 31.820548063 | 0.889614702 | 48000 | 31.768135532 | 46000 | -0.052412531 |
| Set14 | 28.367292797 | 0.776058048 | 48000 | 28.358239350 | 46000 | -0.009053447 |
| B100 | 27.419095383 | 0.730811953 | 48000 | 27.407023390 | 47000 | -0.012071992 |
| Urban100 | 25.572203489 | 0.768094719 | 48000 | 25.572889014 | 49000 | +0.000685525 |
| Manga109 | 29.772182273 | 0.897528826 | 48000 | 29.731499262 | 48000 | -0.040683011 |

## Server locations

- Code: /home/featurize/work/span-x4-research/code/span-sr-research
- Python: /environment/miniconda3/bin/python3
- Data: /home/featurize/datasets; Flickr2K links to /home/featurize/datasets/downloads/Flickr2K
- Persistent completed 50k backup: /home/featurize/work/span-x4-research/records/FEAT-MOEFS-G4-LR2E4-20K-01-1791983354063378494
- B0 run inside backup: runs/FEAT-MOEFS-G4-LR2E4-20K-01-B0-s10
- B0 selected weight inside run: epoch00048_u48000.pt
- latest.pt is the 50000-update continuation state; do not substitute it for the selected 48000 weight.
- manifest.json: effective config, environment, source commit, dataset hashes and fingerprint.
- epochs.jsonl: per-1000-update loss/PSNR/SSIM; loss_steps.jsonl: per-update loss/LR.
- Per-image metrics: DATASET_epoch00048_u48000.json.
- Curves inside backup: curves/seed-10/B0-overview.png and A1-G4-overview.png.
- backup-manifest.json: copied-file SHA256 inventory; BACKUP_COMPLETE: backup completion marker.
- Temporary live artifacts: /home/featurize/span-x4-runtime/runs and logs. Persistent backup is preferred for reuse.
- Earlier 20k backup: /home/featurize/work/span-x4-research/records/FEAT-MOEFS-G4-LR2E4-20K-01-1791958015802799076.

## Baseline reuse rule and decision

New candidates may train alone from random initialization under this exact common protocol, seed and 50000-update budget, then compare against archived B0. Preserve original baseline initialization and data RNG stream when adding parameters. Record candidate source and show that common trainer/evaluation behavior is unchanged; a different commit alone is not a training protocol mismatch. Changes to data, degradation, initialization, common trainer, precision, loss, batch, crop, optimizer, schedule or evaluation require a new matched B0 or an explicitly justified common reevaluation.

Do not initialize a candidate from B0 48000/50000 and compare its extra training against the archived from-scratch B0: that changes initialization and budget. Keep per-dataset peak checkpoints identified; public-set peaks are descriptive and are not used to tune candidates. Single seed is the user-approved budget limit and does not establish seed robustness.

A1-G4 (+11024 deployed parameters, about 2.21%) did not show consistent PSNR benefit at 50k. Stop training this version. This result does not reject all MoE architectures. Inference gate diagnostics remain pending.

GitHub contains textual result evidence and configs only. Model weights, actual plots, full per-step/per-image records and original dataset files remain on the server; they were not uploaded as part of this record.
