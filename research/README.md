# SPAN ×4 research

Status: framework implemented; baseline accuracy reproduction NOT started. No measured PSNR gains.

Upstream: https://github.com/hongyuanyu/SPAN, commit c77a5917759f09e66fbc7124220c5afc5ee221e5. `upstream-baseline` preserves the complete upstream history. `research/main` adds an isolated harness. The original README, LICENSE directory, LICENSE.txt, architecture, and metrics are unchanged. Cite Wan et al., CVPR Workshops 2024, pp. 6246–6256, as in the original README.

## Layout

`research/configs`: versioned protocol and example paths; local paths ignored.
`research/modules`: future isolated single-change candidates.
`research`: train, verification and resource profiling scripts.
`research/docs`: source audit, evidence and experiment rules.
`research/paper`: citation and eventual manuscript material.
`runs/<unique-ID>`: immutable initial manifest, console log, per-image metrics, epoch summary, latest resumable checkpoint, independent-validation best checkpoint. Runs ignored by Git; archive them separately.
`datasets`, `weights`, `logs`, `results`: ignored local bulk assets; reviewed small summaries belong in research/docs.

## Commands (from repository root)

Use `E:/anaconda/envs/dl/python.exe`. This existing environment has PyTorch 2.9.1+cu128; it differs from upstream's 1.11/CUDA11.3. Do not present environment equivalence as verified.

1. Copy `research/configs/paths.example.json` to `paths.local.json` and enter paired directories. Configure `lr_suffix` explicitly (empty if HR and LR stems match). Do not generate LR with an arbitrary OpenCV/Pillow resize; provenance must establish MATLAB-compatible bicubic degradation.
2. `python research/verify.py`
3. `python research/train.py --id B0-x4-s10-YYYYMMDD-unique --preflight`
4. Commit all source/config changes. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python when launching programmatically after CUDA has already initialized.
5. `python research/train.py --id B0-x4-s10-YYYYMMDD-unique --screen` for 20 research epochs. Full training uses the same config without `--screen` and a new ID. Screen stops early without shortening the LR schedule.
6. Continue the SAME run with identical arguments plus `--resume`. Resumes only at completed epoch checkpoints, restoring optimizer/scheduler/best/RNG state. An interruption loses unfinished epoch work; that epoch repeats. Latest checkpoint writes are atomic. Source/data/config fingerprint must match. Never change a screen run into a full run under the same ID.
7. `python research/benchmark.py --output results/B0-x4-profile.json`

Every research epoch comprises ceil(N/effective_batch) updates with paired images sampled with replacement, independently of BasicSR's dataset-enlargement epoch convention. Effective batch 64 is accumulated in microbatches of 4 to fit 8GB; hardware suitability must be measured. This sampler and accumulation are explicit reproduction deviations. No AMP, EMA, teacher, self-ensemble, or pretrained initialization.

Five public benchmarks are monitored every epoch as requested, but ONLY the separate holdout PSNR selects checkpoints or candidates. Per-dataset bests are descriptive; never select separate test-optimal checkpoints for the principal comparison. An independent holdout needs a fixed, documented source and exclusion from training. Preserve all failed runs. Training data and metric provenance must be approved by audit before interpreting accuracy.

Remote creation, when CLI is authenticated:
`gh repo create LI-mo66/span-sr-research --private --source . --remote origin`
`git push origin research/main:main upstream-baseline`
This creates an independent repository, retaining upstream commit ancestry. Do not squash away upstream history or initialize a new unrelated Git history.

