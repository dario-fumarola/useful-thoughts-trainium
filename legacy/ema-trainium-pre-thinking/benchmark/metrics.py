"""Write ONE file with the final metrics: benchmark/results/METRICS.md

Absolute accuracy per model per task, plus chance floors. Re-run any time;
it picks up whatever score files exist.

  python benchmark/metrics.py
"""
import json, glob, os
from collections import defaultdict

CHANCE = {"arc_challenge":25,"arc_easy":25,"boolq":50,"hellaswag":25,
          "lambada":None,"openbookqa":25,"sciq":25,"winogrande":50}
HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, "results")

runs = {}
for p in sorted(glob.glob(os.path.join(RES, "*.jsonl"))):
    name = os.path.basename(p)[:-6]
    acc, accn, n = defaultdict(list), defaultdict(list), 0
    for l in open(p):
        try: r = json.loads(l)
        except Exception: continue
        acc[r["task"]].append(r["correct"]); accn[r["task"]].append(r["correct_norm"]); n += 1
    if n: runs[name] = (acc, accn, n)

tasks = sorted({t for a, _, _ in runs.values() for t in a})
lines = ["# Benchmark metrics", "",
         "Zero-shot likelihood, 500 examples/task, no fine-tuning.",
         "`acc_norm` = length-normalised (the standard headline for multiple choice).",
         "Incomplete runs are marked; a task with fewer than 500 examples is still in progress.", ""]
lines.append("| task | chance | " + " | ".join(runs) + " |")
lines.append("|---|---|" + "---|" * len(runs))
for t in tasks:
    ch = CHANCE.get(t)
    row = [t, f"{ch}%" if ch else "--"]
    for name, (acc, accn, _) in runs.items():
        v = accn.get(t)
        row.append(f"{100*sum(v)/len(v):.1f}%" + ("" if len(v) >= 500 else f" ({len(v)})") if v else "--")
    lines.append("| " + " | ".join(row) + " |")
row = ["**MACRO**", ""]
for name, (acc, accn, _) in runs.items():
    full = [t for t in accn if len(accn[t]) >= 500]
    tot = sum(sum(accn[t]) for t in full); cnt = sum(len(accn[t]) for t in full)
    row.append(f"**{100*tot/cnt:.1f}%**" + (f" ({len(full)}/8 tasks)" if len(full) < 8 else "") if cnt else "--")
lines.append("| " + " | ".join(row) + " |")
lines += ["", "Files: `<name>.jsonl` = per-example scores (paired across runs).",
          "Deltas + significance: `python benchmark/compare.py --norm --base A.jsonl --cmp B.jsonl`"]
out = os.path.join(RES, "METRICS.md")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n-> {out}")
