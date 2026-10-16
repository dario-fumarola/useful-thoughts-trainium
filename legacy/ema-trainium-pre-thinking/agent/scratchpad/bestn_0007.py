"""Selection-breadth axis of G208(c): how does the own-thought ceiling grow with N, the number of
own thoughts the oracle may choose among? Every "oracle-bestG" in this programme is best-of-SIX.

Conversion is domweight_2207.py's, unchanged: per-seq sums / 1024 real tokens, corpus mixture.
Exact best-of-N for N<=6 via order-statistic weights (mean over all C(6,N) subsets of subset max).
"""
import json, math
import numpy as np

R = "/home/ubuntu/ema-trainium/runs/think98_hintrl"
recs = [json.loads(l) for l in open(f"{R}/ceiling/ceil_5320.jsonl")]
T, K, G = 1024, 16, 6
W = {"fineweb": 0.7368, "finemath": 0.1579, "dclm": 0.1053}
DMG = 0.0411

def gains(r):
    """clamped per-sample gains at horizon K; the oracle may choose silence (0)."""
    cc = float(np.array(r["ce_clean"][:K]).sum())
    return np.maximum(np.array([cc - float(np.array(s[:K]).sum()) for s in r["ce_span"]]), 0.0)

# exact E[max of a uniformly random N-subset of the 6] = sum_i w_i x_(i), x ascending
def osw(N):
    w = np.array([math.comb(i - 1, N - 1) for i in range(1, G + 1)], dtype=float)
    return w / math.comb(G, N)
WTS = {N: osw(N) for N in range(1, G + 1)}

def deployed(rs):
    by = {}
    for r in rs:
        if r["H"] > 4.0: by.setdefault((r["set"], r["seq"]), []).append(r)
    out = []
    for _, v in by.items(): out += sorted(v, key=lambda r: -r["H"])[:8]
    return out

GA = {id(r): gains(r) for r in recs}
SORTED = {id(r): np.sort(GA[id(r)]) for r in recs}

def per_seq_mean(rs, arm):
    by = {}
    for r in rs:
        by.setdefault((r["set"], r["seq"]), 0.0)
        by[(r["set"], r["seq"])] += arm(r)
    v = np.array(list(by.values()))
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

def corpus_weighted(selfn, arm):
    d = {}
    for s in sorted(set(r["set"] for r in recs)):
        d[s] = per_seq_mean(selfn([r for r in recs if r["set"] == s]), arm)[0] / T
    return (W["fineweb"] * d["fineweb-edu_val"]
            + W["finemath"] * (d["finemath_val"] + d["finemath_unseen"]) / 2
            + W["dclm"] * d["dclm_val"])

bestN = lambda N: (lambda r: float(SORTED[id(r)] @ WTS[N]))

# ---- B758 collapse check: N=6 equal-weighted, all anchors, nats/seq, vs +16.690 -------------
m, se = per_seq_mean(recs, bestN(6))
print(f"B758 COLLAPSE CHECK: order-statistic N=6, all 1536, equal-weighted = {m:+.3f}+-{se:.3f} "
      f"nats/seq   vs artefact oracle-bestG +16.690+-0.882   diff {m-16.690:+.4f}")
m1, se1 = per_seq_mean(recs, bestN(1))
print(f"       (N=1 same estimator = {m1:+.3f}+-{se1:.3f}; artefact's oracle-1 uses sample 0 ONLY, "
      f"+9.928+-0.722 -- same estimand, different variance, so this is a consistency read not an identity)")

# ---- the curve -------------------------------------------------------------------------------
print(f"\n{'N':>3} {'4x corpus-wtd':>15} {'deployed corpus-wtd':>21} {'4x: x short':>12} {'dep: x short':>13}")
c4, cd = {}, {}
for N in range(1, G + 1):
    c4[N] = corpus_weighted(lambda rs: rs, bestN(N))
    cd[N] = corpus_weighted(deployed, bestN(N))
    print(f"{N:>3} {c4[N]:+15.5f} {cd[N]:+21.5f} {DMG/c4[N]:11.1f}x {DMG/cd[N]:12.1f}x")

# ---- B759 shape: increments per doubling ------------------------------------------------------
print("\nB759 SHAPE -- increments per doubling of N (4x volume, corpus-weighted):")
incs = []
for a, b in ((1, 2), (2, 4), (3, 6)):
    incs.append(c4[b] - c4[a])
    print(f"   N {a} -> {b}: {c4[b]-c4[a]:+.5f}")
print(f"   non-increasing? {'YES (concave)' if all(incs[i] >= incs[i+1] - 1e-12 for i in range(len(incs)-1)) else 'NO'}")

# ---- B760 / B761 ------------------------------------------------------------------------------
print(f"\nB760 N=3 -> N=6 (4x, corpus-weighted): {c4[6]-c4[3]:+.5f} nats/real token  [band +0.0010..+0.0035]")
print(f"B761 best-of-6 / best-of-1 (4x, corpus-weighted): {c4[6]/c4[1]:.3f}  [band 1.4..2.2]")

# ---- B762 extrapolation -----------------------------------------------------------------------
x = np.log(np.arange(1, G + 1)); y = np.array([c4[N] for N in range(1, G + 1)])
slope, icpt = np.polyfit(x, y, 1)
resid = y - (slope * x + icpt)
Nstar = math.exp((DMG - icpt) / slope)
print(f"\nB762 log-linear fit  ceiling(N) = {icpt:+.5f} {slope:+.5f}*ln N   (max |resid| {np.abs(resid).max():.5f})")
print(f"     N needed for the 4x corpus-weighted ceiling to reach the damage {DMG:+.4f}: N* = {Nstar:,.0f}")
print(f"     (OPTIMISTIC by construction: the max of a light-tailed sample grows SLOWER than log N "
      f"eventually, so the true N is larger.)  [band: N* >= 100]")

# ---- B763 hard bound --------------------------------------------------------------------------
hard = lambda r: float(np.array(r["ce_clean"][:K]).sum())
hd = corpus_weighted(deployed, hard); h4 = corpus_weighted(lambda rs: rs, hard)
print(f"\nB763 HARD BOUND (next {K} tokens made FREE by the thought; no writer can beat this):")
print(f"     deployed, corpus-weighted {hd:+.5f} nats/real token = {hd/DMG:.1f}x the damage  [band >= 10x]")
print(f"     4x volume,corpus-weighted {h4:+.5f} = {h4/DMG:.1f}x the damage")
print(f"     best-of-6 reaches {c4[6]/h4*100:.1f}% of the hard bound at 4x volume.")

# ---- B764 concentration -----------------------------------------------------------------------
tot = np.array([float(SORTED[id(r)] @ WTS[6]) for r in recs])
srt = np.sort(tot)[::-1]; n10 = max(1, int(round(0.10 * len(srt))))
print(f"\nB764 CONCENTRATION (4x volume, pooled, best-of-6): top 10% of anchors ({n10}/{len(srt)}) carry "
      f"{srt[:n10].sum()/srt.sum()*100:.1f}% of the benefit  [band >= 50%]")
print(f"     anchors with a positive best-of-6: {(tot>0).mean()*100:.1f}%   median gain {np.median(tot):+.3f} nats")
