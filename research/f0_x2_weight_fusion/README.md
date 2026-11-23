# F0 x2 deployment-weight fusion, evaluation only

Seven full-image plain+x8 evaluations: original982K, L1 patch96 step4K,
MSE patch96 step7K, equal MSE6/7/8/9/10K average, and three interpolations
(1-alpha)*L1-4K + alpha*MSE-7K for alpha0.25/0.5/0.75.
No optimizer, training data, loss changes, updates, or new architecture.

Every Conv3XC is converted to its equivalent deployment convolution BEFORE
averaging. Averaging the training factors directly is a different operation
because convolution branch compositions contain products. Parameter tensors
are averaged with float64 accumulation and cast to their original dtype.
Buffers must be exactly equal. These are single deployed models, not an
ensemble of several separately evaluated networks. The ordinary D4 x8
test-time ensemble is still applied to each model. Params remain552003.

Input artifacts: original1M source manifest and982K checkpoint, patch96
protocols, step4K L1 and step6-10K MSE checkpoints, final-verification.json.
Only the428 existing evaluation HR/LR pairs are used, from the original
manifest and paths configuration. Their ordered filenames, SHA256 and scale
alignment are audited. No image-cache dependency or cache creation.

Original982K SHA256 and patch96 receipt hashes/fingerprints are checked.
Every source deployment conversion is tested on1x1,17x19,63x65 inputs.
Original982K, L1-4K and MSE-7K are re-evaluated; all12 metric groups must match
their respective recorded values within0.0002dB PSNR and0.000002 SSIM.
The tolerance allows possible uint8-boundary changes from reparameterization.
Actual discrepancies are reported. All deltas use the newly evaluated deployed
982K baseline, so reference and candidates have identical deployment semantics.

ranking uses equal mean PSNR delta over the five x8 benchmark sets. Also
report minimum delta, all-five flag and complete plain/x8 per-image metrics.
This is exploratory selection using benchmark results, not an unbiased
unseen-test generalization claim. No improvement is guaranteed.

Each variant saves deployed.pt, with format F0-x2-deployed-weight-fusion-v1.
It must load into a model built using deploy.fused_copy, not the original
training graph. It has no optimizer/RNG state and is NOT a resumable training
checkpoint. Original artifacts are read-only and hashed again at completion.

Results and deployed weights are saved under persistent work. After successful
evaluation, source-unchanged checks and individual-file fsync/hash checks,
COMPLETE.json is saved and flushed. AUTO_RELEASE=1 permits the worker to
release the instance only after that success and an idle-GPU check. Failures
retain the instance. Launch the shell from/tmp to reduce NFS script-read risks.

verify.py tests arithmetic, endpoints, buffer rejection, actual F0 native to
deployment conversion, x8 shapes and source immutability on local synthetic
weights. Actual trained checkpoint and metric reproduction are server checks.
