# LG6-SiLU 50k result

Source: user-provided server CPU report from persistent work backup, archived verbatim. No direct server access or independently downloaded weights/plots. Recorded 2026-10-15.

Training commit e1be1bf41998c5e10412221c44b8912d2dd82b2f. Exact effective config is alongside. Seed10, DF2K, original SiLU, six local affine gates, zero gate W / bias0.1, Adam lr2e-4 and original MultiStep, 50k budget/eval1k. Report shows exit_code0, backup_exit_code0, release_requested=true; user reports instance is off. Earlier missing-dataset job is a separate failed launch, not the final training outcome.

50 evaluation records and 50000 logged loss steps; nonfinite logged loss count0. After update1000, maximum logged loss0.05517972819507122 at1030, according to user top10 output. No ReLU-style nonfinite interruption occurred in this run. This is not proof of stability for other seeds or longer budgets.

## Primary: common validation-selected checkpoint rule

B0 update48000; LG6-SiLU update49000. All metrics below use each model's single validation-selected checkpoint, not per-dataset chosen checkpoints and not temporal averages.

|Dataset|B0 PSNR|SiLU PSNR|Delta PSNR|Delta SSIM|
|---|---:|---:|---:|---:|
|DIV2K_valid|30.159891247|30.189747909|+0.029856662|+0.001320465|
|Set5|31.820548063|31.854651378|+0.034103314|+0.001180527|
|Set14|28.367292797|28.403203320|+0.035910523|+0.001696339|
|B100|27.419095383|27.447363287|+0.028267905|+0.001325885|
|Urban100|25.572203489|25.668409764|+0.096206275|+0.003550170|
|Manga109|29.772182273|29.868500910|+0.096318637|+0.002382729|

## Descriptive per-dataset maximum PSNR

|Dataset|B0 peak|SiLU peak|Delta|SiLU update|
|---|---:|---:|---:|---:|
|DIV2K_valid|30.159891247|30.189747909|+0.029856662|49000|
|Set5|31.820548063|31.854651378|+0.034103314|49000|
|Set14|28.367292797|28.414985282|+0.047692484|46000|
|B100|27.419095383|27.450837027|+0.031741644|44000|
|Urban100|25.572203489|25.668409764|+0.096206275|49000|
|Manga109|29.772182273|29.868500910|+0.096318637|49000|

## Interpretation and decision

Positive direction across all six datasets under both recorded selection conventions, with a modest validation gain +0.0298566619647751 dB. This is a promising candidate under this single seed/protocol; no seed robustness or statistical significance established. Gain includes the complete six-block replacement and its changed initial output, so it cannot be uniquely attributed to channel mixing, removing saturation or bias initialization. Deployed parameters514744 vs B0498208 (+16536, about3.32%); real latency/MAC coverage not measured here.

Late validation PSNR is not monotonic: 49000=30.189747909, 50000=30.124618370, drop0.065129539. The chosen best checkpoint remains49000 under the pre-agreed rule; do not label latest.pt the best weight or claim full convergence. Logged interval mean L1 at50000=.026901051. Do not infer ReLU is generally inferior: that version interrupted around217xx, not an equal-budget completed comparison.

Keep LG6-SiLU as a positive 50k candidate. No automatic extension or new training authorized by receipt of these results. As promised, now remind user of scheme2 image-information bypass, to be discussed independently before combinations.

## Server artifacts

Backup: /home/featurize/work/span-x4-research/records/FEAT-LG6-SILU-50K-01-1792052706657616355
Run inside backup: runs/FEAT-LG6-SILU-50K-01-LG6-SiLU-s10
Selected: epoch00049_u49000.pt (user confirms file exists). latest.pt is50000, not selected49000.
Curve: curves/seed-10/LG6-SiLU-overview.png
Full loss/per-image records, checkpoint SHA256 for this candidate and actual plot image remain on server, not uploaded here. The report includes B0 selected checkpoint SHA2569fb64520554220649b979588d78ccd1ad6c6f4c8695e891bdb4af9302d8b6a1f, computed by server baseline audit; it is not the candidate weight hash.
