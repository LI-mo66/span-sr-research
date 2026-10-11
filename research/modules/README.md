# Candidate modules

Candidates are implemented in candidates.py as independent adaptations of SSIU feature-selection, ESC partial-channel context and AdaIR-inspired frequency modulation. Baseline construction remains original SPAN52. Original parameter initialization is preserved with isolated adapter RNG. CPU/CUDA engineering checks pass; no candidate accuracy benefit has been measured. server_queue.py runs the baseline first and uses separate frozen configs/IDs for each single change, seed and fusion. Do not edit basicsr/archs/span_arch.py in place.
