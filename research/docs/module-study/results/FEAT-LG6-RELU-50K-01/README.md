# LG6-ReLU: nonfinite-loss interruption

Source: user-provided server console excerpt; no direct server access. Intended protocol seed10, lr2e-4, six affine local gates, zero W / bias0.1 initialization, ReLU, 50k cap. Trainer stopped on FloatingPointError: Nonfinite loss after the logged update21700 and before the next logged 21800. Exact failing update and NaN vs Inf are not available from the excerpt.

Last completed evaluation/save cycle shown: update21000. DIV2K PSNR29.50109530826519, SSIM0.8151251773645914. Last periodic train log: update21700 L1=0.08316622; this alone does not identify the failure cause. Latest checkpoint state must be inspected rather than assumed.

Queue exit_code=1; backup_exit_code=0. Verified backup reported at /home/featurize/work/span-x4-research/records/FEAT-LG6-RELU-50K-01-1792048089650350682; curves/seed-10 contains partial training curves. Backup completion does not imply 50k training completion. Instance was retained because training failed; auto_release did not execute.

Evidence: this concrete candidate/protocol encountered a numerical failure. Unbounded multiplicative gating and changed ReLU statistics are plausible contributing factors, not established causes. We do not know whether forward activation, gradients, parameters or Adam states first became nonfinite; the trainer detects the subsequent loss. Finite loss earlier does not prove earlier gradient/optimizer finiteness. Twenty synthetic engineering steps did not establish real-data long-run stability.

Decision: do not report as completed50k or compare with B0 as equal-budget completed results. Do not blindly resume unchanged deterministic protocol. Preserve raw losses, checkpoint and logs for bounded diagnostic replay if authorized. SiLU is a separate candidate and must be assessed using its own logs; do not infer it failed. Any new stabilization/loss/LR change requires an explicit new version and comparison provenance. Reminder after experiment review: scheme2 image bypass remains pending.
