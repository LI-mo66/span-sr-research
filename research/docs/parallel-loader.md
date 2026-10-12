# Parallel PNG loading, 2026-10-12

The first remote baseline exposed poor observability and probable serial PNG
decode overhead. CPU activity alone did not prove the bottleneck; remote
throughput has not yet been measured. The user requested stopping and deleting
that run and restarting with parallel loading.

Opt-in loader_workers uses an ordered thread pool with bounded microbatch
prefetch. OpenCV internal threads are capped at one and PyTorch CPU threads at
four in the new launch configuration. No decoded full-image cache is retained.
Each sample has a private RNG derived from seed and absolute sample index;
selection remains with replacement and crop/flip/transpose distributions are
unchanged. This is a new random stream, not bitwise equivalent to the serial
protocol. Baseline and candidates must use the same loader settings and code.
Resume starts from update times accumulation count and reconstructs pending
samples without relying on thread RNG state. The original serial path remains
available. Threaded npy_mmap is deliberately rejected pending separate checks.

The launch script runs only baseline for 20,000 updates, logs update one and
every ten thereafter, and evaluates every 5,000 updates. ETA is approximate
and does not forecast evaluation overhead. Success or failure invokes shutdown
after writing status; inability to acquire the GPU lock does not shut down.

Validation: check_parallel_loader.py checks actual DIV2K PNG crops for identical
tensors across one/four workers and a resumed microbatch index. Remote speed,
GPU training and actual provider shutdown remain unverified for this revision.
