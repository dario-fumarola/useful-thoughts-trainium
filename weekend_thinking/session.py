"""Executable CPU session with matched initial weights and exact local resumption.

This is an integration reference on a toy model. It must not be used as an
accelerator production loop: its generation recomputes full prefixes.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from .artifacts import atomic_json, canonical_hash
from .checkpoints import capture_cpu_rng, restore_cpu_rng, save_rank, finalize_checkpoint, load_cpu_rank
from .core import Format, Record
from .order import RecordCursor, seed_order
from .reference import TinyCausalLM, UpdateConfig, thought_update, vanilla_update


def toy_records(count: int) -> list[Record]:
    rng = np.random.default_rng(90210)
    return [Record(f"synthetic-{i}", f"synthetic-doc-{i}", "train",
                   tuple(int(x) for x in rng.integers(3, 28, 6)),
                   tuple(int(x) for x in rng.integers(3, 28, 4)), "synthetic_not_research")
            for i in range(count)]


def run_reference(*, out: str | Path, arm: str, steps: int = 6, seed: int = 1234,
                  stop_after: int | None = None, resume: str | Path | None = None) -> dict:
    if arm not in {"thought", "vanilla"} or steps < 1:
        raise ValueError("Choose thought/vanilla and positive steps.")
    end = steps if stop_after is None else stop_after
    if not 1 <= end <= steps:
        raise ValueError("stop_after must be within the fixed total step budget.")
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = TinyCausalLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fmt = Format((29,), (30,), 30, 0)
    cfg = UpdateConfig(group_size=4, max_thought_tokens=8)
    records = toy_records(steps * 2)
    cursor = RecordCursor(records, seed_order(records, seed))
    spec = {"arm": arm, "terminal_steps": steps, "seed": seed, "batch": 2,
            "format": asdict(fmt), "update": asdict(cfg), "order_hash": cursor.order_hash,
            "data_hash": canonical_hash([asdict(r) for r in records]),
            "model": "TinyCausalLM(vocab=31,hidden=24)", "status": "CPU_SYNTHETIC_ONLY"}
    digest = canonical_hash(spec)
    first = 0; metrics = []
    if resume is not None:
        state = load_cpu_rank(resume, expected_spec_hash=digest)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        first = state["step"]; cursor.cursor = state["cursor"]; metrics = state["metrics"]
        if cursor.order_hash != state["order_hash"] or cursor.cursor != first * 2:
            raise ValueError("Resume data position/order mismatch.")
        restore_cpu_rng(state["rng"])
    if end <= first:
        raise ValueError("Requested endpoint is not after the restored checkpoint.")
    for step in range(first, end):
        batch = cursor.take(2)
        fn = thought_update if arm == "thought" else vanilla_update
        measurement = fn(model, optimizer, batch, fmt, cfg)
        metrics.append({"step": step + 1, **measurement})
    directory = Path(out) / f"step-{end:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "rng": capture_cpu_rng(), "cursor": cursor.cursor, "step": end,
             "order_hash": cursor.order_hash, "spec": spec, "metrics": metrics}
    save_rank(directory / "rank-00000.pt", state)
    finalize_checkpoint(directory, world_size=1, step=end, run_id=f"cpu-demo-{arm}-{seed}",
                        spec_hash=digest, metadata={"synthetic": True, "device": "cpu"})
    result = {"status": "CPU_SYNTHETIC_ONLY_NOT_RESEARCH", "arm": arm, "step": end,
              "terminal_steps": steps, "checkpoint": str(directory),
              "spec_hash": digest, "records_consumed": cursor.cursor,
              "order_hash": cursor.order_hash, "metrics": metrics}
    atomic_json(Path(out) / f"summary-{end:06d}.json", result)
    return result
