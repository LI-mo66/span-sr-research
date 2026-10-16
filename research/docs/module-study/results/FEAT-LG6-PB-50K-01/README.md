# Plain RGB bypass control (PB) completed50k

Recorded2026-10-16. Source: user-provided CPU report archived verbatim; no direct server access/independent candidate weight hash audit. Actual commit dd942d4287f2635f9cfe354418bc2ea9a095d3b2, fingerprint9c3edc30825035744ea06e1251afb3ff37e4fe81d0d0a05d86dcd14fe9f4adc4. RTX4090/PyTorch2.2.2+cu121, seed10, lr2e-4 fixed through50k, original MultiStep and common training/evaluation protocol. PB deletes only28 spatial-gate parameters, keeps the same RGB3x3 correction and A main network; deployed516088 vsGB516116.

Training exit0, backup exit0, plotting errorNone, release_requestedtrue.50 evaluations/50000 logged loss steps, nonfinite count0. Highest logged loss after1000 .07085819356143475 at38398, no nonfinite interruption. Latest interval mean L1 .027207352. Candidate selected48k, GB selected48k, A selected49k; latest50k is not selected.

## Per-dataset peak PSNR (descriptive, exact checkpoint identified in JSON)

|Dataset|A|Plain PB|Gated GB|PB minus GB|PB update|GB update|
|---|---:|---:|---:|---:|---:|---:|
|DIV2K_valid|30.189747909|30.223241852|30.234560232|-0.011318379|48000|48000|
|Set5|31.854651378|31.894617418|31.912555637|-0.017938218|48000|49000|
|Set14|28.414985282|28.421564989|28.424236508|-0.002671520|48000|48000|
|B100|27.450837027|27.461447340|27.454617949|+0.006829390|44000|44000|
|Urban100|25.668409764|25.713448858|25.686956344|+0.026492514|48000|49000|
|Manga109|29.868500910|29.936173016|29.939266778|-0.003093762|48000|48000|

## Primary: one validation-selected checkpoint per model

|Dataset|PB minus A PSNR|GB minus PB PSNR|GB minus PB SSIM|
|---|---:|---:|---:|
|DIV2K_valid|+0.033493944|+0.011318379|+0.000027704|
|Set5|+0.039966041|+0.011725170|-0.000256979|
|Set14|+0.018361668|+0.002671520|-0.000144434|
|B100|+0.011804304|-0.008714030|-0.000262020|
|Urban100|+0.045039094|-0.027749198|-0.001012732|
|Manga109|+0.067672106|+0.003093762|-0.000179317|

## Interpretation and decision

PB improves all six selected PSNR/SSIM metrics over A, and all six PSNR peaks over A. Thus direct RGB bypass has positive screening evidence in this setting even without the spatial gate. GB validation peak30.2345602317 vsPB30.2232418524, a small+0.0113183793 advantage. Gate differences are mixed: GB PSNR better on validation, Set5, Set14 and Manga109; PB better on B100 and Urban100. At primary selected checkpoints PB SSIM is higher on five public benchmarks, GB only higher on validation. Neither variant universally dominates.

Previous gate audit found near-constant gain with weak spatial variation; current independent ablation is consistent with substantial usefulness of the bypass and limited/inconsistent additional gate benefit. It does not establish causal percentages, universal gate uselessness, significance, seed robustness or two independent innovations. Initialization and data RNG are matched by engineering checks; source/common trainer audit is documented. No cross-checkpoint or cross-dataset averaging used for selection.

Keep GB as validation-selected primary under the pre-agreed rule, PB as a necessary simpler ablation and useful alternative (especially Urban100), not a failed candidate. Do not cherry-pick GB/PB per test set to claim a single-model result. Avoid further gate-complexity search without new evidence; no new training is triggered by this report.20/50k remain screening budgets, not proof of full convergence. If common training protocol/budget changes later, matched comparisons are required.

## Server locations

Backup: /home/featurize/work/span-x4-research/records/FEAT-LG6-PB-50K-01-1792129615766337782
Run: runs/FEAT-LG6-PB-50K-01-LG6-SiLU-PB-s10
Selected weight: epoch00048_u48000.pt, existence confirmed by user output. B100 peak uses44k; all other PB peaks use48k.
Curve: curves/seed-10/LG6-SiLU-PB-overview.png
Work backup inventory363 files. Candidate hash and raw per-image metrics remain on server and were not independently examined here. A hash is reference-only, not candidate hash.
