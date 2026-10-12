"""Ordered threaded PNG decode; sample RNG is independent of worker scheduling."""
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor
from collections import deque


def sample_rng(seed, index):
    token = hashlib.sha256(f'span-loader-v1:{seed}:{index}'.encode()).digest()
    return random.Random(int.from_bytes(token[:8], 'big'))


class PrefetchBatches:
    def __init__(self, pairs, patch, batch, seed, start, end, sample, workers=8, prefetch=2):
        self.pairs, self.patch, self.batch, self.seed = pairs, patch, batch, seed
        self.next_index, self.end, self.sample = start, end, sample
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.pending = deque()
        self.prefetch = prefetch
        self._fill()

    def _one(self, index):
        rng = sample_rng(self.seed, index)
        return self.sample(rng.choice(self.pairs), self.patch, rng=rng)

    def _fill(self):
        while len(self.pending) < self.prefetch and self.next_index < self.end:
            first = self.next_index * self.batch
            self.pending.append([self.pool.submit(self._one, first+i) for i in range(self.batch)])
            self.next_index += 1

    def __iter__(self):
        return self

    def __next__(self):
        if not self.pending:
            raise StopIteration
        futures = self.pending.popleft()
        result = [f.result() for f in futures]
        self._fill()
        return result

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
