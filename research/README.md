# SPAN ×4 research

Complete Linux server toolkit: docs/server-full-run.zh.md. Use scripts/launch_nohup.sh for initial/confirm/fusion batches with persisted logs and shutdown on batch success/failure. Independent adapters now implemented and engineering-checked; no candidate accuracy gains measured. Image decode cache preserves exact pixel values and sampling. Current SSH server has not accepted a working connection; actual remote shutdown is not tested.

Current decision (2026-10-11): main baseline is SPAN52, deployed498,208 params; previous48-channel runs are historical SPAN-S references and are not the main baseline. Original upstream defaults/source remain untouched. baseline_x4.json now uses52 channels, a recording epoch of5000 optimizer updates, full budget1M updates=200 cycles; screen20 cycles=100k updates. Stage screening accepts --screen-updates5000/20000/100000. A SAME --screen run may resume to a larger screening cap with config/source/data unchanged; scheduler never compresses. Epoch snapshots are retained for reuse. Earlier54-updates/epoch paragraphs below describe the historical harness and are superseded by this paragraph. See docs/SPAN52-research-plan.zh.md and docs/candidate-cards.zh.md for current strategy.

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
5. `python research/train.py --id B0-SPAN52-x4-s10-unique --screen --screen-updates 5000` starts screening. Extend to20000/100000 with the same --screen arguments plus --resume. Screen stops early without shortening the LR schedule.
6. Continue the SAME run with --resume and without --screen to reach1M updates. Screen/full flags change only the stopping cap, not the protocol fingerprint. Resumes only at completed epoch checkpoints, restoring optimizer/scheduler/best/RNG state. An interruption loses unfinished epoch work; that epoch repeats. Latest checkpoint writes are atomic; numbered epoch snapshots retained. Source/data/config fingerprint must match.
7. `python research/benchmark.py --output results/B0-x4-profile.json`

Every current recording epoch comprises5000 optimizer updates with paired images sampled with replacement, independently of BasicSR's dataset-enlargement epoch convention. Historical48 experiments usedceil(N/effective_batch); these are not the current main baseline. Effective batch64 is accumulated in microbatches of4 by default; server microbatch must be measured and frozen for all comparisons. This sampler and accumulation are explicit reproduction deviations. No AMP, EMA, teacher, self-ensemble, or pretrained initialization.

Five public benchmarks are monitored every epoch as requested, but ONLY the separate holdout PSNR selects checkpoints or candidates. Per-dataset bests are descriptive; never select separate test-optimal checkpoints for the principal comparison. An independent holdout needs a fixed, documented source and exclusion from training. Preserve all failed runs. Training data and metric provenance must be approved by audit before interpreting accuracy.

Independent private remote repository already created: https://github.com/LI-mo66/span-sr-research . main contains research changes; upstream-baseline preserves upstream ancestry. Do not squash away upstream history or initialize unrelated Git history.

