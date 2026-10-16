"""Pooled paired McNemar over a PRE-REGISTERED task pool.
  python benchmark/pooled_test.py BASE.jsonl CMP.jsonl [task ...]
Pre-registered 2026-09-12 (before the phase-A final checkpoint existed):
pool = arc_challenge arc_easy boolq sciq winogrande, metric = correct_norm.
Motivation: run 19 (8.99B rung) and run 21 @4200 (8.04B rung) both showed
the thinking checkpoint ahead on ARC-Easy and SciQ; run 21 @4200 pooled over
these five tasks: 108 vs 73 discordant, +1.4 pts, p=0.011 (post hoc)."""
import json, sys
from math import comb
POOL = ("arc_challenge", "arc_easy", "boolq", "sciq", "winogrande")
def load(f):
    return {(r["task"], r["idx"]): r for r in map(json.loads, filter(str.strip, open(f)))}
a, b = load(sys.argv[1]), load(sys.argv[2])
pool = tuple(sys.argv[3:]) or POOL
def test(keys):
    b01 = b10 = n = 0
    for k in keys:
        if k in a and k in b:
            x, y = int(a[k]["correct_norm"]), int(b[k]["correct_norm"]); n += 1
            b01 += (x == 0 and y == 1); b10 += (x == 1 and y == 0)
    m = b01 + b10
    p = min(1.0, 2 * sum(comb(m, i) for i in range(min(b01, b10) + 1)) / 2 ** m) if m else 1.0
    return n, b10, b01, p
for name, ts in (("PRE-REGISTERED POOL " + " ".join(pool), pool), ("all tasks", tuple(sorted({t for t, _ in a})))):
    n, b10, b01, p = test([k for k in a if k[0] in ts])
    print(f"{name}\n  n={n}  base-only-right={b10}  cmp-only-right={b01}  net={b01-b10:+d} ({(b01-b10)/n*100:+.2f} pts)  McNemar p={p:.4f}")
