# Released standard SPAN x4 ch48 test, 2026-10-11

Run ID: OFFICIAL-x4-ch48-params-20261011. Actual local evaluation, not from-scratch training. Loaded `params` with all204 keys matched, matching the parameter key explicitly named in the official packaged test log. params_ema was not used for these results.

| Dataset | Images | Local PSNR | Local SSIM | Official packaged log PSNR | Official packaged log SSIM | Local minus log PSNR |
|---|---:|---:|---:|---:|---:|---:|
| Set5 |5|32.204318|0.895331|32.2061|0.8953|-0.001782|
| Set14 |14|28.641507|0.782793|28.6412|0.7828|+0.000307|
| B100 |100|27.600472|0.736847|27.6075|0.7370|-0.007028|
| Urban100 |100|26.125405|0.786282|26.1293|0.7864|-0.003895|
| Manga109 |109|30.556927|0.909233|30.5932|0.9095|-0.036273|

Local pipeline: pinned original arch and metrics; RGB[0,1] input; img_range255 and supplied mean; clamp/round uint8 output; Y-channel; crop4; FP32 full-image batch1; no self-ensemble or tiling. Result directory: results/OFFICIAL-x4-ch48-params-20261011. Contains source/data/checkpoint hashes, env, per-image JSON, console log, and one SR sample per dataset. The checkpoint output range works directly in[0,1], so no inverse normalization should be inserted.

Close agreement does not prove bit-exact reproduction. Original official data paths differ from our user-supplied benchmark data, and original test model wrapper SRModelRAW is not included in the pinned source. Differences in source images/degradation/metric conversion/precision need separate inspection; no causal attribution established. Manga109 gap must remain disclosed. Do not tune test protocol or choose params_ema to maximize these test scores.

Official log reference: weights/official/SPANX4_CH48/test_AttLi_x4_20231123_155330.log (included in downloaded official span.zip). Logs also reference an SRFormer teacher; this is a test configuration and is insufficient evidence for an exact checkpoint training recipe. Training uncertainties are documented in training-method.zh.md.

Accuracy conclusions: published weights can run and the metric path is broadly consistent with packaged logs; from-scratch reproduction remains uncompleted, and no candidate gain or seed stability is measured.
