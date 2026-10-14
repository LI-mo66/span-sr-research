# Single-seed 20k screening, learning rate 2e-4

B0 and A1-G4 start from scratch with seed 10. Initial learning rate is 0.0002 for both. MultiStep milestones remain 200000, 400000, 600000, 800000, gamma 0.5; therefore the learning rate stays constant throughout this 20000-update screen. Effective batch 64, micro batch 16, DF2K data and evaluation conventions remain unchanged. Evaluate and save a numbered checkpoint every 1000 updates.

Select the highest DIV2K_valid PSNR checkpoint within the update budget, breaking ties by earliest update. Report all benchmark metrics from that same checkpoint. Also retain each dataset peak with its exact checkpoint identifier. Do not average scores across checkpoints. Standard per-image dataset averages remain unchanged.

Default plots are two overview figures, one per run, each showing training L1, validation PSNR and validation SSIM. Detailed plots are optional via --detailed. Raw step losses, evaluation rows and weights are retained.

Use a new group FEAT-MOEFS-G4-LR2E4-20K-01, never resume the old unstable 5e-4 run. This is a controlled protocol change, not evidence that lowering the learning rate fixes instability or that MoE improves PSNR. Original SPAN source and historical configs are preserved.

Server source stays under work; datasets are /home/featurize/datasets; temporary runs and logs are /home/featurize/span-x4-runtime. Before optional instance release, copy results to /home/featurize/work/span-x4-research/records and verify backup hashes. Backup failure prevents automatic release.
