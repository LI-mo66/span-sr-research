MambaIRv2: Attentive State Space Restoration, Guo et al., CVPR 2025.
Official repository: https://github.com/csguoh/MambaIR
Locked source commit: 33d7b3460c4665229334cc8de38c7f4c766ed3be
File: basicsr/archs/mambairv2light_arch.py
Full original file SHA256: 0f557d608fe9f2c7f2e95eb3556478d97b527dfc801068f57e8a453afac351bf
License: Apache-2.0, included as LICENSE.txt.

reference.py retains unmodified index_reverse, semantic_neighbor, ASSM and
Selective_Scan definitions. Only imports and a scan-injection test stub differ.
official-light-x4.yml is the original lightweight x4 configuration at that commit.

../adapter.py and ../scan.py are SG6-specific adaptations of the ASSM and
selective scan equations. These are changed files, not a complete reproduction.
Differences are deliberately explicit:
1. C=52 to match SG6, retaining lightweight expand1/state8/tokens64/rank32.
2. Stable category sorting, with BOTH value tokens and readout prompts gathered
   into the same order; inverse permutation restores original pixel coordinates.
   The locked official ASSM sorts values and passes an unsorted prompt to scan.
   We describe this code-level difference, not an author-confirmed upstream bug.
3. Noise-free straight-through hard softmax during training, hard argmax at
   evaluation, rather than unconditional Gumbel sampling. This changes training
   routing and is NOT claimed to reproduce the paper's performance.
4. SG6 wrapper uses a separate channel LayerNorm only on the correction branch,
   a nonzero learnable 0.01 scale, and one outer identity residual.
5. Local window attention and the two FFNs are not included. SG6's original
   local processing is retained. No unused SiLU is advertised: the official
   ASSM defines self.act but does not call it in its forward.
6. Stacked dt parameter initialization is retained without generic reinitializing
   inverse-softplus dt bias, negative A exponent, or positive step sizes.
7. Small pure PyTorch reference is diagnostic-only (<=4096 tokens), never a
   fallback training backend. Production requires a compatible mamba_ssm kernel.

The adapted-reference test patches ONLY the declared routing, category tie and
prompt-order changes and injects the transparent reference scan. Passing it
does not mean equivalence to the untouched official ASSM or native CUDA kernel.
