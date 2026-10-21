# Sources and hypothesis

SeeAll official repository: https://github.com/sxdyyds/SeeAll
Commit: 34175a665a4ff6223ceab5bcd449b04458ed57c6
File: basicsr/archs/seemore16_arch.py
Paper: https://doi.org/10.1016/j.neucom.2026.135274

The implementation follows GateLinearAttentionNoSilu, RoPE and channel-first
LayerNorm. The original reference source is retained in vendor/. No MoDEE or
PDFN is copied into SG6. This is an adaptation, not a reproduction of SeeAll.
The full paper equations were not available during the initial audit.

Fixed adaptation: C52/h1, one residual adapter after SPAB4, zero output
projection. Novel hypothesis: depthwise 3x3 on the 104 projected Q/K channels,
before ELU+1, lets the global matching use explicitly processed neighborhoods.
The depthwise operation starts as identity. V/O and every original attention
operation remain unchanged. No extra activation in Q/K; no new residual scale.
No PSNR gain is claimed without independent real-data experiments.
