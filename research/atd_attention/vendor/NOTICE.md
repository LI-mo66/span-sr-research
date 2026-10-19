Reference: CVL-UESTC/Adaptive-Token-Dictionary, commit c064084c3a33724e850ec03532fc6ddc7d7a0828, basicsr/archs/atd_arch.py, Apache-2.0 (LICENSE.txt).

reference.py extracts ATD_CA, AC_MSA, index_reverse and feature_shuffle with trailing whitespace normalized but no semantic changes; unrelated BasicSR/timm imports omitted. Used only for engineering comparison, not production imports.

adapter.py adapts these two attention branches to SPAN52: fixed C52/QK12/M128/heads4/group128, query/group execution chunks, vectorized inverse permutation; removes window attention and CFFN. Sorting/reflection semantics and learnable temperature preserved. Dictionary is a parameter, no EMA or per-image persistent write. This is not full ATD reproduction. All new glue/tests are independent of original SPAN/shared trainer.
