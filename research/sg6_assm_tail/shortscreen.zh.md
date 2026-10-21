# SG6 + ASSM tail RMS / NCHW v3, single-seed 50K

Direct reference: fixed SG6 e731483d4ce4e325f3d64dbbc80add342b46f7f1. Only research/sg6_assm_tail differs from this base. SG6 weights are hashed for reference auditing, never used to initialize the candidate.

## Layout correction and evidence

User reported isolated-check-04: native isolated scan PASS_SMALL_KERNEL, repeated gradient difference 0, but adapted official-equation reference failed (max output difference 0.00014506280422210693). Code inspection found that the candidate omitted the official contiguous NCHW conversion before in_proj. This can choose a different convolution execution path; exact causality on the user's torch 2.2.2 server remains to be confirmed by the built-in preflight.

Version v3 explicitly matches that layout, with no added parameters, norms, activations or optimizer changes. On local torch 2.9.1+cu128 / RTX 4060 Laptop CUDA reference backend, all six captured stages, final output and parameter gradients now match exactly; strict resume and optimizer coverage pass. Local evidence is not a substitute for server-native execution. Original tolerances are retained.

## Registered short-screen protocol

seed10, scratch, exactly 50000 updates, LR2e-4, Adam(0.9,0.999), eps1e-8, RGB L1, FP32, HR256/LR64, micro16 x accumulation4, unchanged original scheduler horizon 1M and milestones 200/400/600/800K. Evaluate/save every1K, Y-channel uint8/shave4 whole-image no ensemble. Select highest DIV2K_valid PSNR within50K, earliest tie, use this ONE checkpoint on every test set. Report fixed50K separately. Single seed is a screen, not convergence or seed-stability evidence.

## One worker and failure behavior

screen_worker.sh holds the shared GPU lock throughout. Audit fixed SG6 source/config/environment, SHA256 and ordered dataset hashes; native kernel and strict full-model engineering check; three-aspect audit; actual trainer integration with interrupted log recovery and exact resume; real training crops micro16 x4 and largest-area validation whole-image resource smoke; then a NEW subprocess trains the candidate from scratch. All engineering weights are discarded. No baseline retraining.

Any failed check stops before training and retains the instance and logs. Successful 50K run produces comparison-50000.json and hash-verified persistent backup. AUTO_RELEASE=1 requests Featurize release only after BACKUP_COMPLETE and no active GPU process. Negative PSNR is still a completed experiment and is backed up/released. Release request is not proof the provider actually shut down; receipt records command status if it returns.

No data mutation, dependency installation or automatic source repair. Existing output IDs are rejected. Existing kernel venv is reused. Source tar and package list are saved with the experiment. The protected SG6 archive remains read-only.
