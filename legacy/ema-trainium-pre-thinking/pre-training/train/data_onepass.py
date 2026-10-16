"""Dataloader over packed uint32 shards.

The default mode samples from the weighted domain mixture with replacement.
The optional one-pass mode globally shuffles every non-overlapping sequence
block exactly once. Both modes are deterministic from the seed and step, so a
checkpoint resume continues the same stream.

Shapes are fixed for the whole run -- a hard requirement on XLA/Neuron, where a
new shape means a new graph and a fresh multi-minute compilation.
"""

import json
import os

import numpy as np

DTYPE = np.uint32


class DomainShards:
    """Memory-mapped view of one domain's shards, with length-weighted sampling."""

    def __init__(self, shard_dir, names):
        self.arrays, self.lengths = [], []
        for n in names:
            p = os.path.join(shard_dir, n)
            a = np.memmap(p, dtype=DTYPE, mode="r")
            if len(a) > 0:
                self.arrays.append(a)
                self.lengths.append(len(a))
        self.lengths = np.asarray(self.lengths, dtype=np.int64)
        self.total = int(self.lengths.sum())
        # Sampling a shard in proportion to its length, then an offset inside
        # it, is equivalent to sampling uniformly over the concatenation --
        # without ever needing a read that straddles two shards.
        self.p = (self.lengths / self.total) if self.total else None

    def sample(self, rng, seq_len):
        i = int(rng.choice(len(self.arrays), p=self.p))
        a = self.arrays[i]
        hi = len(a) - seq_len - 1
        if hi <= 0:
            return np.asarray(a[: seq_len + 1], dtype=np.int64)
        off = int(rng.integers(0, hi))
        return np.asarray(a[off: off + seq_len + 1], dtype=np.int64)

    def iter_sequential(self, seq_len, max_sequences=None):
        """Deterministic non-overlapping sweep -- used for eval."""
        n = 0
        for a in self.arrays:
            usable = (len(a) - 1) // seq_len
            for j in range(usable):
                if max_sequences is not None and n >= max_sequences:
                    return
                off = j * seq_len
                yield np.asarray(a[off: off + seq_len + 1], dtype=np.int64)
                n += 1

    def block_layout(self, seq_len):
        """Return per-shard and cumulative counts of prediction blocks.

        A block contains ``seq_len + 1`` token IDs but advances by ``seq_len``:
        adjacent blocks share the boundary input token while their prediction
        targets never overlap.
        """
        counts = np.asarray(
            [max((len(a) - 1) // seq_len, 0) for a in self.arrays],
            dtype=np.int64)
        return counts, np.cumsum(counts, dtype=np.int64)

    def block(self, block_index, seq_len, cumulative=None):
        """Fetch one non-overlapping sequence by its domain-local index."""
        if cumulative is None:
            _, cumulative = self.block_layout(seq_len)
        total = int(cumulative[-1]) if len(cumulative) else 0
        if block_index < 0 or block_index >= total:
            raise IndexError(f"block index {block_index} outside [0, {total})")
        shard = int(np.searchsorted(cumulative, block_index, side="right"))
        before = int(cumulative[shard - 1]) if shard else 0
        off = (int(block_index) - before) * seq_len
        a = self.arrays[shard]
        return np.asarray(a[off: off + seq_len + 1], dtype=np.int64)


class Mixture:
    def __init__(self, corpus_dir, split="train"):
        self.corpus_dir = corpus_dir
        self.shard_dir = os.path.join(corpus_dir, "shards")
        with open(os.path.join(corpus_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.domains = list(self.manifest["domains"])
        key = "train_shards" if split == "train" else "val_shards"
        self.data = {}
        for d in self.domains:
            names = self.manifest["domains"][d][key]
            self.data[d] = DomainShards(self.shard_dir, names)
        w = np.asarray([self.manifest["domains"][d]["weight"] for d in self.domains],
                       dtype=np.float64)
        self.weights = w / w.sum()

    def summary(self):
        return {d: dict(tokens=self.data[d].total,
                        weight=float(self.weights[i]),
                        shards=len(self.data[d].arrays))
                for i, d in enumerate(self.domains)}


class MixtureBatcher:
    """Yields fixed-shape (micro_bs, seq_len) int64 input/label pairs.

    In sampled mode each rank draws from its own RNG stream. In one-pass mode
    all ranks construct the same global permutation and take disjoint slices
    indexed by (step, micro-step, rank).
    """

    def __init__(self, mixture, micro_bs, seq_len, seed=0, rank=0, world_size=1,
                 without_replacement=False, grad_accum=1):
        self.mix = mixture
        self.micro_bs = micro_bs
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.without_replacement = without_replacement
        self.grad_accum = grad_accum
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} must be in [0, {world_size})")
        if min(micro_bs, seq_len, world_size, grad_accum) <= 0:
            raise ValueError("micro_bs, seq_len, world_size and grad_accum must be positive")

        self.domain_block_counts = None
        self.total_blocks = None
        self.max_full_steps = None
        self._domain_cumulative = None
        self._shard_cumulative = None
        self._permutation = None
        if self.without_replacement:
            shard_cumulative = []
            domain_counts = []
            for d in self.mix.domains:
                counts, cumulative = self.mix.data[d].block_layout(seq_len)
                shard_cumulative.append(cumulative)
                domain_counts.append(int(counts.sum()))
            self.domain_block_counts = np.asarray(domain_counts, dtype=np.int64)
            self.total_blocks = int(self.domain_block_counts.sum())
            self._domain_cumulative = np.cumsum(
                self.domain_block_counts, dtype=np.int64)
            self._shard_cumulative = shard_cumulative
            samples_per_step = micro_bs * grad_accum * world_size
            self.max_full_steps = self.total_blocks // samples_per_step
            # A single global permutation is intentionally identical on every
            # rank. The index arithmetic below assigns disjoint pieces of it.
            self._permutation = np.random.default_rng(seed).permutation(
                self.total_blocks)

    def _rng(self, step, micro):
        # Independent, reproducible stream per (seed, rank, step, micro-step).
        return np.random.default_rng(
            [self.seed, self.rank, step, micro])

    def batch(self, step, micro=0):
        if self.without_replacement:
            return self._one_pass_batch(step, micro)
        rng = self._rng(step, micro)
        idx = rng.choice(len(self.mix.domains), size=self.micro_bs, p=self.mix.weights)
        rows = np.empty((self.micro_bs, self.seq_len + 1), dtype=np.int64)
        for b, di in enumerate(idx):
            rows[b] = self.mix.data[self.mix.domains[di]].sample(rng, self.seq_len)
        return rows[:, :-1], rows[:, 1:], idx

    def _one_pass_batch(self, step, micro):
        if step < 0 or not 0 <= micro < self.grad_accum:
            raise IndexError(f"invalid step/micro pair ({step}, {micro})")
        global_start = (
            (step * self.grad_accum + micro) * self.world_size * self.micro_bs
            + self.rank * self.micro_bs)
        global_end = global_start + self.micro_bs
        if global_end > self.total_blocks:
            raise StopIteration(
                f"one-pass corpus exhausted at sample {global_start:,}; "
                f"only {self.total_blocks:,} full sequence blocks are available")

        shuffled = self._permutation[global_start:global_end]
        idx = np.searchsorted(self._domain_cumulative, shuffled, side="right")
        rows = np.empty((self.micro_bs, self.seq_len + 1), dtype=np.int64)
        for b, (global_block, di) in enumerate(zip(shuffled, idx)):
            before = int(self._domain_cumulative[di - 1]) if di else 0
            local_block = int(global_block) - before
            domain = self.mix.domains[int(di)]
            rows[b] = self.mix.data[domain].block(
                local_block, self.seq_len, self._shard_cumulative[int(di)])
        return rows[:, :-1], rows[:, 1:], idx


def eval_batches(mixture, domain, micro_bs, seq_len, max_sequences):
    """Fixed-shape batches for one domain; drops the ragged tail so that every
    eval graph has the same shape as every other."""
    buf = []
    for seq in mixture.data[domain].iter_sequential(seq_len, max_sequences):
        buf.append(seq)
        if len(buf) == micro_bs:
            rows = np.stack(buf)
            buf = []
            yield rows[:, :-1], rows[:, 1:]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "corpus"))
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()

    for split in ("train", "val"):
        m = Mixture(a.corpus, split)
        print(f"--- {split} ---")
        for d, s in m.summary().items():
            print(f"  {d:<14} {s['tokens']/1e6:9.1f}M tokens  "
                  f"weight={s['weight']:.4f}  shards={s['shards']}")

    m = Mixture(a.corpus, "train")
    bt = MixtureBatcher(m, a.micro_bs, a.seq_len, seed=0, rank=0,
                        world_size=2)
    counts = np.zeros(len(m.domains))
    for s in range(a.steps):
        x, y, idx = bt.batch(s)
        for i in idx:
            counts[i] += 1
    print(f"\nempirical mix over {a.steps*a.micro_bs} sampled sequences:")
    for i, d in enumerate(m.domains):
        print(f"  {d:<14} {counts[i]/counts.sum():.4f}  (target {m.weights[i]:.4f})")

    x, y, _ = bt.batch(0)
    x2, y2, _ = bt.batch(0)
    print(f"\nshapes: x={x.shape} y={y.shape} dtype={x.dtype}")
    print(f"labels are inputs shifted by one: {np.array_equal(x[:,1:], y[:,:-1])}")
    print(f"deterministic replay: {np.array_equal(x, x2)}")
    r1 = MixtureBatcher(m, a.micro_bs, a.seq_len, seed=0, rank=1,
                        world_size=2).batch(0)[0]
    print(f"rank 0 and rank 1 differ: {not np.array_equal(x, r1)}")
