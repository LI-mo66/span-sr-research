# LG6-SiLU-GB (A+B): completed 50k screening

Recorded 2026-10-16. Source: user-provided server CPU output, archived verbatim. No direct server access, independent weight hash audit or actual plot download. Complete candidate source/environment fields were omitted from this pasted excerpt; planned launch pin was d89bc07b807f5e0b3abbbfbba3d1ff65d436552a and must not be represented as independently reconfirmed here. Reference A selected checkpoint SHA256 is present in the report; candidate hash remains unprovided.

A=LG6-SiLU, B=spatially gated RGB correction, not SiLU alone. User-approved screening tests A+B>A, with no B-only retraining. Both completed 50k, selection by highest DIV2K_valid PSNR, no temporal averaging. A selects49000; A+B selects48000.

## Primary: single validation-selected weight per model

|Dataset|A PSNR|A+B PSNR|Delta PSNR|Delta SSIM|
|---|---:|---:|---:|---:|
|DIV2K_valid|30.189747909|30.234560232|+0.044812323|+0.000390234|
|Set5|31.854651378|31.906342589|+0.051691211|+0.000410233|
|Set14|28.403203320|28.424236508|+0.021033188|+0.000095186|
|B100|27.447363287|27.450453562|+0.003090274|+0.000048143|
|Urban100|25.668409764|25.685699660|+0.017289896|+0.000310546|
|Manga109|29.868500910|29.939266778|+0.070765868|+0.000637542|

## Descriptive per-dataset peak PSNR

|Dataset|B0|A|A+B|A+B minus A|A+B peak update|
|---|---:|---:|---:|---:|---:|
|DIV2K_valid|30.159891247|30.189747909|30.234560232|+0.044812323|48000|
|Set5|31.820548063|31.854651378|31.912555637|+0.057904259|49000|
|Set14|28.367292797|28.414985282|28.424236508|+0.009251227|48000|
|B100|27.419095383|27.450837027|27.454617949|+0.003780922|44000|
|Urban100|25.572203489|25.668409764|25.686956344|+0.018546580|49000|
|Manga109|29.772182273|29.868500910|29.939266778|+0.070765868|48000|

## Observations and decision

All six selected-weight PSNR/SSIM differences are positive; all six per-dataset PSNR-peak differences are positive. Validation gain over A +0.044812323102608076 dB; cumulative validation gain over archived B0 +0.07466898506738312 dB. B100 peak increment only0.0037809223332588 dB, so do not call every dataset a substantial gain or infer significance. Single seed cannot establish seed robustness.

50 evaluation records and50000 logged loss steps; reported nonfinite loss count0. After update1000, highest logged L1 .07323645614087582 at21912; the run recovered and completed, so this is a transient spike, not a ReLU-style interruption. Last interval meanL1 .026558932. Validation48k30.234560232 vs50k30.201261275; cannot conclude full convergence or monotonic improvement. Keep selected48k rather than latest50k for primary result.

Decision: retain A+B as current best candidate under this screening protocol. No new training or extension is triggered by receipt of results. Evidence supports the complete gated-bypass increment on A; it does not separate RGB bypass from spatial gate, demonstrate B-only benefit, establish supra-additive synergy or two independent paper contributions. G may approximate a constant or saturate; verify its content dependence and actual correction magnitude before claiming its mechanism or adding complexity. Inference intervention would be diagnostic, not a retraining ablation.

Parameter counts from implementation/engineering checks: B0498208, A514744, A+B516116. B adds1372 over A (about0.2665%); total extra overB017908. Runtime and real peak-memory increment were not measured in this report. The result is within-seed screening, not a long-training claim or official SPANV2 reproduction.

## Server artifacts

Backup: /home/featurize/work/span-x4-research/records/FEAT-LG6-GB-50K-01-1792082028817331043
Run: runs/FEAT-LG6-GB-50K-01-LG6-SiLU-GB-s10
Selected: epoch00048_u48000.pt, user reports file exists; latest.pt is50k.
Curve: curves/seed-10/LG6-SiLU-GB-overview.png
Full per-image metrics, raw step history, manifests and weights remain in work. Original pasted excerpt does not include full termination status, so record completed50k evaluations and backup path without inventing release-command success status.
