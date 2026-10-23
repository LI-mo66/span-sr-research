# Source and adaptation record

Official source: https://github.com/hokiyoshi/UCAN

Locked revision: `b812c0adb8ce70d442973db5c87bab7d1e8c3333`.

Architecture: https://github.com/hokiyoshi/UCAN/blob/b812c0adb8ce70d442973db5c87bab7d1e8c3333/basicsr/archs/ucan_arch.py

Retrieved through the GitHub connector on 2026-10-23. The source and Apache-2.0
license are retained under `vendor/`, with SHA256 in `vendor/manifest.json`.
The vendor file is a reference, not an imported architecture/dependency package.

## Mechanisms retained

- SDFL projects raw Q/K/V, runs spatial linear attention and channel interaction,
  concatenates their half-channel results, and projects back to C.
- DFRL receives raw Q/K but computes current V. Its Hedgehog feature maps,
  Fourier mixing, channel temperature, LePE, V, and output projection remain
  receiver-specific. This does **not** cache an identical attention matrix.
- Hedgehog feature maps, signed Fourier coefficient mixing, original flattened
  sequence positions, positive-map normalization, and LePE pixel ordering.
- Channel affinity is 26x26, not HWxHW; spatial evaluation is associated as
  K'V followed by Q'(K'V), without materializing a global quadratic matrix.

## Explicit adaptations (not an exact UCAN model reproduction)

1. Exactly C52/head1, inserted after SG6 blocks 2 and 4. Do not generalize v1
   to arbitrary heads/channels by changing a reshape.
2. One trainable pre-LN per new residual branch, eps=1e-5, over C only.
   No post-LN, BN/GN, window attention, SGFN, LKSA, dropout, or extra gate.
   Source HybridBlock uses a different normalization/residual arrangement;
   that surrounding block is intentionally not copied.
3. The existing final projection is zero-initialized; all interior tensors have
   their normal initialization. No second zero residual scale. First interior
   gradients are zero; subsequent startup is checked explicitly.
4. Position frequencies are registered buffers, initialized on CPU and moved
   with the module. Reference tests move/cast the source's unregistered tensor
   explicitly, without changing reference equations.
5. The source divides Fourier coefficients by their signed sum, without an
   epsilon. v1 used a signed floor1e-4, but a synthetic four-step checkpoint
   still produced context4 correction/input RMS ratios of 7.86--52.14.
   v2 uses coefficient/max(sum(abs(coefficient)),1e-4): each column's absolute
   coefficient sum is <=1. This changes the source equation intentionally,
   avoids signed cancellation amplification, and preserves signed coefficients.
   Both arms use this identical change. A source_signed math path exists only
   for source-equivalence and retrospective diagnostics, not in the fixed
   production config. This bounds Fourier mixing, not the whole network.
6. Explicit per-forward QKState validates B/H/W/C/device/dtype; no detach or
   persistent image cache. No duplicate lepe/dwc registration alias.
7. The SG6 factory is reused. A derived forward copies the protected GB forward
   with two explicit insertions; all original parameter keys and fusion taps
   remain unchanged. Initialization/RNG and nonzero-base branch tests check it.
8. v2 multiplies the projected branch correction by the current input's detached
   per-image RMS (over C/H/W, minimum1e-6). This expresses the normalized branch
   in the local feature's units without normalizing the identity path or adding
   a trainable gate. It deliberately adds no derivative through the RMS factor.
   This is an evidence-driven correction to v1's deep-feature scale mismatch,
   not a claim of improved PSNR. Formula and batch-independence checks cover it.

Core reference tests execute only AST-extracted official classes, compare both
forward and backward in FP64, and share parameters in the source_signed math
path. v2's intentional normalization and RMS writeback differences are checked
against explicit formulas separately. They do not establish whole-v2 equivalence
to UCAN or learned performance. v1 source, failures, and diagnostics are retained
in Git and engineering artifacts rather than overwritten as a successful model.

## Evidence boundaries

- Source fact: implementation above, licensed code, equations and interfaces.
- Design hypothesis: old relation descriptors may help process updated content.
- Historical local record: SeeAll-derived SGLA arms did not exceed SG6 at seed10
  50K; source `../sgla_context/RESULTS-50K.zh.md` in the separate existing
  `span-sg6-sgla` checkout. That experiment is not imported or modified here.
- Synthetic evidence: local CPU/CUDA startup, equivalence, resource and resume
  checks. No real-image PSNR, multi-seed stability, or training gain is claimed.
- Unknown: benefit over SG6, cross-depth semantic compatibility after training,
  deployment-server latency/memory, and complete source/data/environment
  equivalence for reuse of historical SG6 metrics on a future server.
