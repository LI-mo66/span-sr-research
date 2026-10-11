# Experiment gates

1. Finish data/normalization/metric audit and baseline training before candidate implementation. No reuse of incompatible historical baseline metrics.
2. Freeze data hashes, source, seed streams, total updates, optimizer/LR timeline, crop, augmentations, quantization/Y/crop, FP32 precision and no self-ensemble. Fingerprint config/data/code. Different source commits need a manually reviewed equivalence record for baseline reuse, not automatic acceptance.
3. Pre-register single-change hypotheses, candidate location, parameter increment, speed risks and stopping criteria. First increment <=10% of baseline deployed parameters; also report trainable and serialized counts. Establish actual deployed count before candidate admission.
4. Short screen approximately 20 research epochs with unchanged full LR schedule. Screening is not final gain evidence. Failed/ambiguous screens are retained and stopped; do not stack modules to hide failure.
5. Long-run and multiple training seeds (initial plan 10,20,30); use identical budgets per seed. Report independent validation selection plus fixed final update and predeclared last-epoch averaging; separately report monitored benchmark bests. Do not use test scores to choose module hyperparameters.
6. Candidate gate needs gains beyond observed baseline seed variability, per-image evidence, SSIM and efficiency. Bootstrap over images cannot replace training-seed stability. Never promise fixed dB improvement.
7. Compare identical hardware, input sizes, precision, warmups and synchronizations; report params, covered MACs/FLOPs, peak allocated/reserved memory and latency distribution. Report original and pre-fused deployment paths separately after implementing and checking exact fusion.
8. Fuse only candidates with independently verified stable benefits and mechanism complementarity; use same-budget control if parameter growth explains the gain. ×2/×3 require separate training only after ×4 confirmation.
9. Literature phase is deliberately pending. At that phase search published+officially open-source works in 2024-10-11 to 2026-10-11, prioritize 2025-10-11 onward, verify publication/version/license and provide three complementary mechanism cards. No stale generic module collection.

Evidence labels: source fact / paper claim / design hypothesis / synthetic engineering check / real-data experiment / unverified external result. Maintain failed-experiment ledger with ID, protocol, reason and original artifact pointers. Paper value is undecided until stable precision-efficiency and ablation evidence exists.
