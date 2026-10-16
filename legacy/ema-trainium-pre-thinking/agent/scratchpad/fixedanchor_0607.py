# Re-score run 95 rung 190 and the ladder-22800 on ONE common anchor set.
# Mirrors agent/ceiling_analysis.py's help_matrix/per_seq exactly; no new forward pass.
import json, numpy as np

def load(p):
    return [json.loads(l) for l in open(p)]

def key(r): return (r["set"], r["seq"], r["n"])

def helpvec(r, k):
    """gain per draft at horizon k: clean CE minus spliced CE, summed over first k tokens."""
    cc = np.array(r["ce_clean"][:k], dtype=np.float64)
    return np.array([cc.sum() - np.array(s[:k], dtype=np.float64).sum() for s in r["ce_span"]])

def topn(recs, tau, nt):
    """the analysis script's own selection: H > tau, then top-nt per sequence by H"""
    out = []
    byseq = {}
    for r in recs:
        if r["H"] > tau: byseq.setdefault((r["set"], r["seq"]), []).append(r)
    for v in byseq.values():
        v.sort(key=lambda r: -r["H"]); out += v[:nt]
    return out

def per_seq(recs, fn):
    """per-sequence SUM, then mean +- s.e. over sequences -- the script's unit (nats/1024-tok seq)"""
    byseq = {}
    for r in recs: byseq.setdefault((r["set"], r["seq"]), []).append(r)
    vals = np.array([sum(fn(r) for r in v) for v in byseq.values()])
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals)), vals

R95 = load("runs/think95_bestof/ceiling/ceil_190.jsonl")
LAD = load("runs/probes/ceiling/ceil_ladder22800_L6_K16_G6.jsonl")
print("records: r95 %d, ladder %d" % (len(R95), len(LAD)))

s95, slad = set(map(key, R95)), set(map(key, LAD))
print("recorded-anchor overlap: %d of %d (%.1f%%)" % (len(s95 & slad), len(s95), 100*len(s95 & slad)/len(s95)))

# each checkpoint's OWN top-8/seq selection, and their overlap (B810)
t95 = set(map(key, topn(R95, 4.0, 8))); tlad = set(map(key, topn(LAD, 4.0, 8)))
ov = len(t95 & tlad)
print("\nB810 own-selection overlap: %d of %d (%.1f%%) -> %s"
      % (ov, len(tlad), 100*ov/len(tlad), "HIT" if ov/len(tlad) < 0.85 else "MISS"))

# THE FIX: use the LADDER's top-8/seq anchors for BOTH, restricted to those both recorded
common = sorted(tlad & s95)
print("common fixed anchor set (ladder's top-8/seq, also recorded by r95): %d" % len(common))
d95 = {key(r): r for r in R95}; dlad = {key(r): r for r in LAD}
A = [d95[c] for c in common]; B = [dlad[c] for c in common]

for lbl, recs in (("run95@190", A), ("ladder   ", B)):
    bg, bse, _ = per_seq(recs, lambda r: max(helpvec(r,1).max(), 0.0))
    d1, dse, _ = per_seq(recs, lambda r: helpvec(r,1)[0])
    hs = np.mean([1.0 if helpvec(r,1).max() > 0 else 0.0 for r in recs])
    print("  %s  bestG %+.3f+-%.3f | deploy-1 %+.3f+-%.3f | helps %.2f" % (lbl, bg, bse, d1, dse, hs))

# PAIRED by sequence (B811/B812)
_,_,v95 = per_seq(A, lambda r: max(helpvec(r,1).max(), 0.0))
_,_,vlad = per_seq(B, lambda r: max(helpvec(r,1).max(), 0.0))
d = v95 - vlad
md, sd = d.mean(), d.std(ddof=1)/np.sqrt(len(d))
print("\nB811 paired bestG diff on FIXED anchors: %+.3f +- %.3f over %d sequences -> %s"
      % (md, sd, len(d), "HIT" if md >= 2.00 else "MISS LOW"))
print("B812 |t| = %.2f -> %s" % (abs(md/sd), "HIT" if abs(md/sd) >= 3.0 else "MISS LOW"))
print("     (unfixed-anchor difference was +3.245; %.0f%% survives)" % (100*md/3.245))

a1,_,_ = per_seq(A, lambda r: helpvec(r,1)[0]); b1,_,_ = per_seq(B, lambda r: helpvec(r,1)[0])
print("B813 deploy-1 both negative? r95 %+.3f, ladder %+.3f -> %s"
      % (a1, b1, "HIT" if (a1 < 0 and b1 < 0) else "MISS"))
h95 = np.mean([1.0 if helpvec(r,1).max() > 0 else 0.0 for r in A])
hld = np.mean([1.0 if helpvec(r,1).max() > 0 else 0.0 for r in B])
print("B814 helps-share gap %.2f - %.2f = %+.2f -> %s"
      % (h95, hld, h95-hld, "HIT" if h95-hld >= 0.20 else "MISS LOW"))
