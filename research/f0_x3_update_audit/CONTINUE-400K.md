# X3 frontcap 310K to 400K

External manager; pinned training tree6e71b81 remains unchanged.
Resume full candidate state. rho=0.0005 for conv_1/block_1, global gradient
clip0.1, RGB L1, FP32, DF2K3450, LR64/HR192, effective batch64.
Original milestones200/400/600/800K and base LR2e-4 remain unchanged.
Update400000 uses1e-4 and scheduler then sets5e-5 in saved400K state.
No step400001 is executed. Run in1K stages, preserve existing quality stops
relative to healthy295K; failure/quality stops retain instance.
No automatic shutdown. Cache can be regenerated, original data must exist.

## Evidence for extension
User supplied310K report: updates300001..310000 continuous, all finite,
model/Adam/scheduler/RNG/SHA passed. Conv1 cap12/10000, block1 cap467/10000.
Max step L1 0.02959805354; max preclip gradient0.20931707323.
310K latest SHA256:62f3cec7577421c9ae68c2daa2e9e35fb46901463d65e9be91ddb0118f15048e.
PSNR fluctuates, no proven sustained growth or future stability guarantee.
Source: user CPU report2026-11-28; attachment01421780-0206-4aef-ab4f-1131cce395aa.
