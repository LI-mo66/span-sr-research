# Server toolkit engineering evidence

Independent adapters now implemented in modules/candidates.py. CPU and CUDA (RTX4060 Laptop, PyTorch2.9.1+cu128) checks PASS: original parameter initialization unchanged; fusion identity at initial zero logits; batch2 odd13×17 dimensions; finite gradients; every adapter parameter actually changes within3 Adam steps; strict reload and whole-network pre-fusion agreement. New parameter counts measured A11024/B32272/C3218; full three-mechanism engineering fixture adds46514 (not an authorized real training fusion).

CUDA strict deterministic checks use explicit mean/amax global reductions rather than AdaptiveAvg/Max pooling backward paths. Frequency adapter uses an independent fixed low-pass/signed high-pass design, not full AdaIR. Source attribution in candidate-cards.zh.md remains applicable. Original upstream architecture file/README/licenses remain unchanged.

nohup launcher and worker tested via Bash syntax checks; Windows sandbox prevented MSYS initialization, so only those local checks used an approved outside-sandbox invocation. All shutdown calls in tests are mocks. Tests cover success exit0, failure exit7, saved status before shutdown, and refusal of held GPU lock without shutting down the owner. No real machine was powered off in verification. npy cache pixel equality and selection ignoring deliberately contradictory test scores checked with clearly labeled synthetic fixtures; fixture scores are not experiment evidence.

An initial Windows test cleanup failed because a cached mmap still held a temporary file open; test now explicitly clears the mmap cache before cleanup. This did not require changing production image values/augmentation or hiding a model failure.

Remote SSH currently closes the connection; no actual server queue started, no actual remote shutdown verified. Provider-specific shutdown executable must exist; launcher refuses if absent. Worker logs attempted shutdown and failure. AutoDL official guidance https://www.autodl.com/docs/save_money/ uses/usr/bin/shutdown, but the supplied SeetaCloud instance is not assumed empirically verified by that guidance.

No true candidate PSNR, training time on4090, complete FLOPs or stable-gain result is claimed. All first-batch training starts from scratch with matched original weights/data streams; public pretrained checkpoints are not used as unfair short-screen baseline.
