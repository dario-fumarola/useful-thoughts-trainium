"""Verified local checkpoint publication.

Generic filesystem functions are CPU-tested. Multi-rank XLA save/restore remains
instance-side work. Only checkpoints from your own trusted jobs may be loaded.
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import random
from typing import Any
import numpy as np
import torch
from .artifacts import atomic_bytes, atomic_json, sha256_file, utc_now


def capture_cpu_rng() -> dict:
    n = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": {"algorithm": n[0], "keys": torch.tensor(n[1].astype(np.int64)),
                      "position": int(n[2]), "has_gauss": int(n[3]), "cached_gaussian": float(n[4])}}


def restore_cpu_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n["algorithm"], n["keys"].numpy().astype(np.uint32),
                         n["position"], n["has_gauss"], n["cached_gaussian"]))


def save_rank(path: str | Path, state: dict[str, Any]) -> str:
    """State tensors must already be on CPU; an XLA adapter owns device transfer."""
    def check(value):
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            raise ValueError("Transfer every checkpoint tensor to CPU before save_rank.")
        if isinstance(value, dict):
            for v in value.values(): check(v)
        elif isinstance(value, (list, tuple)):
            for v in value: check(v)
    check(state)
    # Use a streaming temporary file in production for large shards. BytesIO is
    # deliberately bounded to CPU qualification jobs here, not a 1.7B save path.
    stream = io.BytesIO()
    torch.save(state, stream)
    atomic_bytes(path, stream.getvalue())
    return sha256_file(path)


def finalize_checkpoint(directory: str | Path, *, world_size: int, step: int,
                        run_id: str, spec_hash: str, metadata: dict | None = None) -> dict:
    """Publish COMPLETE.json only after every expected rank file is durable.

    Call after an actual process barrier. File presence cannot establish a barrier
    or numerical equivalence; the production adapter must supply both.
    """
    directory = Path(directory)
    if world_size < 1 or step < 0 or not run_id or not spec_hash:
        raise ValueError("Invalid checkpoint identity.")
    files = {}
    expected = {f"rank-{rank:05d}.pt" for rank in range(world_size)}
    actual = {p.name for p in directory.glob("rank-*.pt")}
    if expected != actual:
        raise ValueError(f"Shard set mismatch; expected {sorted(expected)}, got {sorted(actual)}")
    for name in sorted(expected):
        path = directory / name
        if path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"Invalid shard: {name}")
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    receipt = {"schema": 1, "run_id": run_id, "spec_hash": spec_hash, "step": step,
               "world_size": world_size, "created_utc": utc_now(), "files": files,
               "metadata": metadata or {}}
    atomic_json(directory / "COMPLETE.json", receipt)
    return receipt


def verify_checkpoint(directory: str | Path, *, expected_spec_hash: str | None = None) -> dict:
    directory = Path(directory)
    receipt = json.loads((directory / "COMPLETE.json").read_text())
    if expected_spec_hash is not None and receipt["spec_hash"] != expected_spec_hash:
        raise ValueError("Checkpoint belongs to a different experiment specification.")
    expected = {f"rank-{i:05d}.pt" for i in range(receipt["world_size"])}
    if set(receipt["files"]) != expected:
        raise ValueError("Invalid complete-shard manifest.")
    for name, item in receipt["files"].items():
        path = directory / name
        if path.is_symlink() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Checkpoint hash/size mismatch: {name}")
    return receipt


def load_cpu_rank(directory: str | Path, rank: int = 0, *, expected_spec_hash: str | None = None) -> dict:
    receipt = verify_checkpoint(directory, expected_spec_hash=expected_spec_hash)
    if not 0 <= rank < receipt["world_size"]:
        raise ValueError("Rank outside checkpoint world.")
    return torch.load(Path(directory) / f"rank-{rank:05d}.pt", map_location="cpu", weights_only=True)
