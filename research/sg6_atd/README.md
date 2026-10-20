# SG6 + ATD attention: isolated 50K screen

Direct base: `e731483d4ce4e325f3d64dbbc80add342b46f7f1` on
`research/gb-spatial-gate6`. New branch: `research/sg6-atd-attention`.
All changes are confined to `research/sg6_atd/`; the fixed SG6 branch,
original SPAN, shared trainer, metrics and prior outputs are unchanged.

## Structure and reason

SG6 has six local spatial gates and an RGB bypass. The hypothesis is that
content-based long-range grouping can complement these local operations.
This is a hypothesis to test against SG6, not an established PSNR gain.

```text
RGB input -> original SG6 six blocks -> original conv2/concatenation
  -> original conv_cat -> ATD attention adapter -> original upsampler convolution
  -> add original gated RGB image residual -> original PixelShuffle -> RGB output
```

The original SG6 forward is inherited, rather than copied. `conv_cat` is wrapped
by `FusionWithATD`: its `fusion` child is the original conv_cat and its
`attention` child returns `F + TDCA(LN(F)) + ACMSA(Wqkv(LN(F)), categories)`.
There is one residual addition of F. There is no additional FFN, activation,
post-residual normalization, sigmoid gate or zero multiplier to suppress the
new information. An arbitrary extra activation is not needed for shape matching.
The RGB bypass still joins AFTER the upsampler convolution and BEFORE shuffle.

| Interface | Contract |
|---|---|
| Fusion input/output | N,52,H,W float32; identical H,W, device and pixel order |
| Tokens | N,HW,52; row-major flatten, invert same order |
| Normalization | LayerNorm over 52 channels per pixel |
| Dictionary | 128 x 52 trainable parameters; no EMA or per-image buffer |
| TDCA | Q/K dimension 12; dictionary-to-pixel values, category argmax |
| ACMSA | Stable image isolation; global category sorting; groups up to 128, 4 heads x 13 dimensions |
| Boundaries | Reflect-pad sorted token tail, then crop and inverse permutation |
| New initialization | Independent forked seed 50011; retained SG6 state/RNG unchanged |
| Loss connection | Both attention outputs feed final RGB L1; all 14 new parameter tensors update |

TDCA alone is not cross-pixel value exchange. ACMSA supplies the content-grouped
pixel interaction. The dictionary is fixed across images at inference but learned
through training. Discrete sorting/argmax has no derivative; value and projection
paths remain differentiable. No per-image category state persists.

This extracts ATD's TDCA and ACMSA only, not the full ATD network. The exact
upstream source commit and Apache license are in `vendor/NOTICE.md` and
`vendor/LICENSE.txt`. Original reference attention classes are retained for
output/gradient tests. The new glue is specific to SG6.

SG6 deploy parameters: 535,772. New attention: 21,813 (+4.07%).
Combined deploy parameters: 557,585. The shared metadata's cumulative
`extra_deployed_parameters=59377` is relative to original B0 (498,208),
not the direct comparator. `deployed_reference_parameters=535772` and
`max_new_deployed_parameters=53577` explicitly describe the SG6 reference/budget.
No FLOPs number is claimed.

## Execution choice and rejected evidence

The training config uses query/group limits 1,000,000,000, effectively unchunked
on these datasets. A local alternative 16K-query/128-group test had exact
category routes and FP64 attention output, but FP32 attention max error
3.81e-6 became 1.80e-4 at RGB reconstruction. It failed the existing full-output
tolerance; see `rejected-chunk-alternatives.json`. That execution is rejected,
and tolerance was not relaxed to accept it. The 4K alternative was not reached.
The earlier 64K/256 variant passed at 180x320, but was not selected because
that sample does not exercise its large-image query boundary.
The selected unchunked implementation also matches the separately retained
upstream reference attention outputs and gradients in production-sized token
cases. `resources.py` compares the selected execution with its unchunked copy;
that comparison is a configuration wiring check, not independent algorithm
verification. `check_reference.py` provides the independent algorithm check.

Local evidence: CPU/CUDA identity, nonzero-SG6-bypass preservation, six spatial
branches learning, attention updates, optimizer coverage, strict save/load,
resume equality, original-reference attention equivalence and FP32 deployment
fusion checks pass. The actual shared trainer also passed a two-update synthetic accumulation/cache/
partial-log recovery and checkpoint/resume/evaluation-cadence equality test;
see `engineering-workflow.json`. Local CUDA is RTX4060 Laptop / torch2.9.1+cu128 and does
NOT substitute for the required server RTX4090 / torch2.2.2+cu121 tests.
Random-weight local LR180x320 median latency was ~15.49 ms SG6 vs ~22.87 ms
SG6+ATD; this is a cost warning for this machine, not a trained 4090 result.
All resource data and rejected evidence are preserved in JSON files.

## Fixed protocol and SG6 reuse

The archive is read-only:
`/home/featurize/work/span-x4-research/records/FEAT-LG6-SG6-50K-01-1792264356078917269`.
`prepare_reuse.py` verifies archive success, actual source/config, ordered image
SHA256s, environment, complete 1K..50K history, 48K selection and the user's
reported SG6 metrics. It uses the archive's full-precision metrics, not the
rounded table. SG6 weights are hashed only, never loaded or trained.

Candidate: seed10, scratch, 50,000 updates, HR256/LR64, micro16 x accumulation4,
RGB L1, Adam2e-4, original 200K/400K/600K/800K milestones and 1M scheduler
horizon. 50K is only a stopping cap; no shortened scheduler. Every 1K evaluates
all six sets with unchanged Y-channel/uint8/shave4/whole-image/no-ensemble code.
Select ONE checkpoint by best DIV2K_valid PSNR (earliest tie), report every
set from that same weight, and compare directly with SG6's selected weight.
Also provide the fixed-50K secondary comparison. Do not pick each test set's
best checkpoint or treat B0 as the main comparator.

## Server orchestration

Run `launch_reuse.sh GROUP SG6_ARCHIVE` from this new checkout. The launcher
refuses dirty source, reused IDs and busy GPUs; obtains the existing global
research GPU lock; then runs archive audit, upstream reference check,
SG6+ATD CUDA verification, disposable micro16x4 + largest-evaluation-image smoke,
resource timing and real-data preflight. Smoke weights are discarded. Only
then does it nohup the single 50K worker. Launch the launcher itself under
nohup as well, so preflight survives terminal disconnects.

Use a distinct code directory, cache directory and group ID. Existing dataset
paths must be restored rather than editing SG6. `AUTO_RELEASE=1` requests
Featurize instance release after training, comparison and SHA256-verified
persistent backup succeed; failures keep the instance for diagnosis. Another
GPU process suppresses release. `AUTO_RELEASE=0` keeps the instance after success.
No old run or branch is removed, overwritten or resumed.

Artifacts: frozen config/paths/reuse audit, preflight and engineering logs,
`nohup.log`, `status.txt`, 1K checkpoints, `latest.pt`, `best_validation.pt`,
per-image metrics, `comparison-50000.json`, and a timestamped persistent backup
with SHA256 manifest and `BACKUP_COMPLETE`. `backup-path.txt` points to it.
No plotting package or curve requirement is added to the training path.

## Decisions after training

Engineering success is not a PSNR conclusion. Inspect direct deltas vs SG6,
per-image results, resource costs and the shared checkpoint before proceeding.
This is one seed / 50K screening only. If it loses, do not silently extend the
budget or stack another module. Preserve this result and revisit the next
candidate. MambaIRv2-based global modeling remains a separate pending option;
it is not implemented or scheduled by this experiment.
