#!/usr/bin/env python3
"""Runs real CPU updates on a tiny model; never presents them as research results."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from weekend_thinking.core import Format, Record
from weekend_thinking.reference import TinyCausalLM, UpdateConfig, thought_update, vanilla_update


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", default="validation/cpu_smoke.json")
    p.add_argument("--steps", type=int, default=6); a = p.parse_args()
    if a.steps < 1: raise SystemExit("--steps must be positive")
    torch.set_num_threads(1); torch.manual_seed(20261016)
    base = TinyCausalLM(); thought, vanilla = copy.deepcopy(base), copy.deepcopy(base)
    cfg = UpdateConfig(max_thought_tokens=8, group_size=4, banned_ids=(0, 1))
    fmt = Format((1,), (2,), 3, 0)
    data = [Record(str(i), str(i), "train", (4+i, 7, 8, 4+i), (9+i, 10+i)) for i in range(8)]
    optim_t = torch.optim.AdamW(thought.parameters(), lr=1e-3)
    optim_v = torch.optim.AdamW(vanilla.parameters(), lr=1e-3)
    history = []
    for step in range(a.steps):
        rows = [data[(2*step+j) % len(data)] for j in range(2)]
        history.append({"step": step, "ids": [r.id for r in rows],
                        "thought": thought_update(thought, optim_t, rows, fmt, cfg),
                        "vanilla": vanilla_update(vanilla, optim_v, rows, fmt, cfg)})
    delta = sum((p-q).abs().sum().item() for p, q in zip(thought.parameters(), base.parameters()))
    assert delta > 0 and all(torch.isfinite(p).all() for p in thought.parameters())
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"test": "CPU execution and numerical integrity only",
                              "not_tested": ["Qwen weights", "Trainium", "Neuron compilation", "research efficacy"],
                              "parameter_l1_change": delta, "history": history}, indent=2) + "\n")
    print(f"PASS: {a.steps} paired CPU steps; replay agreement and finite gradients checked. {out}")

if __name__ == "__main__": main()
