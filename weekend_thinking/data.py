"""Document-level splitting, deterministic anchors, and strict record I/O."""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from .core import Record


def digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def split_document(document_id: str, salt: str = "weekend-2026-10-16-v1") -> str:
    h = int(hashlib.sha256(f"{salt}:{document_id}".encode()).hexdigest()[:16], 16)
    bucket = h % 100
    return "test" if bucket < 5 else "dev" if bucket < 10 else "train"


def make_record(tokens: list[int], document_id: str, *, domain: str = "unknown",
                prefix_length: int = 512, future_length: int = 16,
                salt: str = "weekend-2026-10-16-v1") -> Record | None:
    """One deterministic random anchor per document, not entropy selected.

    Long context is cropped to the most recent prefix_length tokens. Documents
    shorter than prefix+future are excluded and must be counted by the caller.
    This defines a length-filtered corpus experiment, not all web text.
    """
    if prefix_length < 1 or future_length < 1:
        raise ValueError("Positive lengths required.")
    if any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("Input token IDs must be nonnegative Python integers.")
    if len(tokens) < prefix_length + future_length:
        return None
    h = int(hashlib.sha256(f"anchor:{salt}:{document_id}".encode()).hexdigest()[:16], 16)
    n_choices = len(tokens) - future_length - prefix_length + 1
    anchor = prefix_length + h % n_choices
    rid = f"{document_id}:{anchor}"
    return Record(rid, document_id, split_document(document_id, salt),
                  tuple(int(x) for x in tokens[anchor-prefix_length:anchor]),
                  tuple(int(x) for x in tokens[anchor:anchor+future_length]), domain)


def write_records(path: str | Path, records: Iterable[Record]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    seen, count = set(), 0
    with tmp.open("w") as f:
        for row in records:
            if row.id in seen:
                raise ValueError(f"Duplicate record ID: {row.id}")
            seen.add(row.id)
            f.write(json.dumps(asdict(row), separators=(",", ":")) + "\n")
            count += 1
    tmp.replace(path)
    return count


def load_records(path: str | Path, *, split: str | None = None) -> list[Record]:
    out, seen, doc_splits = [], set(), {}
    with Path(path).open() as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["prefix"], obj["future"] = tuple(obj["prefix"]), tuple(obj["future"])
            row = Record(**obj)
            if row.id in seen:
                raise ValueError(f"Duplicate ID at line {line_number}: {row.id}")
            seen.add(row.id)
            old = doc_splits.setdefault(row.document_id, row.split)
            if old != row.split:
                raise ValueError(f"Document {row.document_id} spans multiple splits.")
            if split is None or row.split == split:
                out.append(row)
    if not out:
        raise ValueError(f"No records for split {split!r} in {path}.")
    return out


def data_index(step: int, micro: int, rank: int, world: int, accumulation: int) -> int:
    """No rank duplication. Same global indices for equal world/accumulation."""
    if min(step, micro, rank) < 0 or world < 1 or accumulation < 1:
        raise ValueError("Invalid distributed index.")
    if rank >= world or micro >= accumulation:
        raise ValueError("Rank/micro out of range.")
    return (step * accumulation + micro) * world + rank


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
