"""Deterministic, resumable source exposure without silent dataset cycling."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import Sequence
from .core import Record
from .artifacts import canonical_hash


def seed_order(records: Sequence[Record], seed: int) -> list[int]:
    ids = [r.id for r in records]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Require nonempty unique record IDs.")
    if any(r.split != "train" for r in records):
        raise ValueError("Only training records may enter the training order.")
    return sorted(range(len(records)),
                  key=lambda i: hashlib.sha256(f"order:{seed}:{ids[i]}".encode()).hexdigest())


@dataclass
class RecordCursor:
    records: Sequence[Record]
    order: list[int]
    cursor: int = 0

    def __post_init__(self):
        if sorted(self.order) != list(range(len(self.records))):
            raise ValueError("Order must be a permutation of the full record pool.")
        if not 0 <= self.cursor <= len(self.order):
            raise ValueError("Cursor outside record pool.")

    @property
    def order_hash(self) -> str:
        return canonical_hash([self.records[i].id for i in self.order])

    def take(self, count: int) -> list[Record]:
        if count < 1 or self.cursor + count > len(self.order):
            raise ValueError("Insufficient fresh records. Do not silently wrap the dataset.")
        indices = self.order[self.cursor:self.cursor + count]
        self.cursor += count
        return [self.records[i] for i in indices]


def distributed_batch_indices(step: int, rank: int, world_size: int,
                              accumulation: int, micro_prefixes: int = 1) -> list[int]:
    """Indices are positions in the SAME seed order, not rank-local shuffle orders."""
    if step < 0 or world_size < 1 or not 0 <= rank < world_size or min(accumulation, micro_prefixes) < 1:
        raise ValueError("Invalid distributed layout.")
    base = step * world_size * accumulation * micro_prefixes
    return [base + (micro * world_size + rank) * micro_prefixes + j
            for micro in range(accumulation) for j in range(micro_prefixes)]
