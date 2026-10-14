# Fixed-checkpoint gate diagnostic, recorded 2026-10-14

Source: user-supplied server output, not independent server access. Full DIV2K validation 100 images, input hashes checked by diagnostic. B0 update48000 PSNR30.15989124662389 and A1 update47000 PSNR30.136644037870514 exactly reproduce recorded metrics. Corrected alpha1 implementation output difference is zero on the implementation check crop. Initial diagnostic had a floating-point subtraction/addition mismatch; corrected script uses original multiplication at alpha1, identity at alpha0, unchanged strict reproduction checks.

Alpha1/0.75/0.5/0.25/0 PSNR: 30.136644 / 19.728270 / 16.116807 / 14.404494 / 13.398178. This demonstrates sensitivity/dependence of this learned checkpoint on its gate, not that the gate improves independently trained baseline accuracy. Weakening after training is unsupported by these observations. This intervention is not independent retraining of a weak residual gate.

Pooled gate mean probabilities F0/F6/B1/T6: 0.026811/0.324843/0.022924/0.625422. Actual gains are 4p. Normalized entropy0.247293 indicates concentrated selection, but does not prove useless expert collapse. F0/B1 never win argmax but retain nonzero soft contributions. Winner fractions F6=17/52 and T6=35/52 suggest possible fixed channel partitioning; pooled statistics cannot establish this, and per-channel, per-image, spatial variability would be required.

All four gated mean absolute feature magnitudes are lower than their ungated counterparts. Larger mean probability does not imply larger effective feature contribution: probability and feature magnitude can be correlated; downstream conv_cat weights also matter. These observations do not identify the cause of the small baseline deficit or prove suppressed shallow features should be restored.

Decision: keep A1-G4 stopped. Do not promote inference weakening or recommend retraining a weak gate solely from this diagnostic. If pursuing this architecture, first measure whether routing is content-dependent and inspect feature-gate correlation/downstream contribution without training. Consider alternatives only with evidence, literature and explicit budget authorization. Baseline reuse remains the documented matched seed10 50k protocol.

Server diagnostic output: /home/featurize/work/span-x4-research/records/GATE-DIAG-50K-FIXED-20261014T150736Z-24586/report.json. No weights changed. No training performed.
