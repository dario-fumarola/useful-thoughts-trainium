"""Weighted-mixture dataloader over the packed uint32 shards.

Every batch is *sampled* from the domain mixture rather than assembled by
concatenating domains, so each batch reflects the target weights in
expectation. Sampling is fully determined by (seed, rank, step), which makes a
run reproducible and lets resume-from-checkpoint continue the same stream.

Shapes are fixed for the whole run -- a hard requirement on XLA/Neuron, where a
new shape means a new graph and a fresh multi-minute compilation.
"""

import json
import os

import numpy as np

DTYPE = np.uint32


class DomainShards:
    """Memory-mapped view of one domain's shards, with length-weighted sampling."""

    def __init__(self, shard_dir, names, aligned=False):
        # aligned=True: shards hold BLOCKS of seq_len+1 tokens back to back (the
        # slide run's unseen-FineMath corpus); sample whole blocks only.
        self.aligned = aligned
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
        if self.aligned:
            nb = len(a) // (seq_len + 1)
            off = int(rng.integers(0, nb)) * (seq_len + 1)
            return np.asarray(a[off: off + seq_len + 1], dtype=np.int64)
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


class Mixture:
    def __init__(self, corpus_dir, split="train", aligned=False):
        self.corpus_dir = corpus_dir
        self.shard_dir = os.path.join(corpus_dir, "shards")
        with open(os.path.join(corpus_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.domains = list(self.manifest["domains"])
        key = "train_shards" if split == "train" else "val_shards"
        self.data = {}
        for d in self.domains:
            names = self.manifest["domains"][d][key]
            self.data[d] = DomainShards(self.shard_dir, names, aligned=aligned)
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

    Each rank draws from its own RNG stream so the data-parallel ranks never
    see the same sequences.
    """

    def __init__(self, mixture, micro_bs, seq_len, seed=0, rank=0, world_size=1):
        self.mix = mixture
        self.micro_bs = micro_bs
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def _rng(self, step, micro):
        # Independent, reproducible stream per (seed, rank, step, micro-step).
        return np.random.default_rng(
            [self.seed, self.rank, step, micro])

    def batch(self, step, micro=0):
        rng = self._rng(step, micro)
        idx = rng.choice(len(self.mix.domains), size=self.micro_bs, p=self.mix.weights)
        rows = np.empty((self.micro_bs, self.seq_len + 1), dtype=np.int64)
        for b, di in enumerate(idx):
            rows[b] = self.mix.data[self.mix.domains[di]].sample(rng, self.seq_len)
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
    bt = MixtureBatcher(m, a.micro_bs, a.seq_len, seed=0, rank=0)
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
    r1 = MixtureBatcher(m, a.micro_bs, a.seq_len, seed=0, rank=1).batch(0)[0]
    print(f"rank 0 and rank 1 differ: {not np.array_equal(x, r1)}")
