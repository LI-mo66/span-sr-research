# Third-party attribution

The reference_tab.py extract and the derived implementation in ../tab.py are
based on EquationWalker/CATANet, commit
fd2222f30560d86bfac9b5f07e932c2fec12e6c1:
https://github.com/EquationWalker/CATANet

Paper: Xin Liu, Jie Liu, Jie Tang, Gangshan Wu, CATANet: Efficient Content-Aware
Token Aggregation for Lightweight Image Super-Resolution, CVPR 2025.

The upstream repository distributes LICENSE.txt under Apache License 2.0;
a verbatim copy is included here. No separate upstream NOTICE file was present.

Reference extraction retains TAB and dependencies, removes unrelated classes
and unused BasicSR imports. The adapted implementation replaces einops with
explicit torch layout operations, fixes insufficient padding for short inputs,
uses 4D/chunked attention, inverse gather, explicit buffer updates and standalone
Linear/LayerNorm initialization matching the upstream full-network initializer.
This is an adaptation for SPAN, not a complete reproduction of CATANet.
