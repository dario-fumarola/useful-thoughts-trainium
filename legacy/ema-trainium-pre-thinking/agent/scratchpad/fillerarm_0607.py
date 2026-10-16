"""06:07 tick -- resolves B710-B714 (agent/scratchpad/BANDS_0607.txt, registered before the probe ran).

Own greedy thought vs LENGTH-MATCHED content-free filler at run 98's final checkpoint's own deployment
anchors, in the splicing geometry. Paired anchor by anchor; error unit = the SEQUENCE (G120), cluster
bootstrap 2,000 resamples, same estimator as agent/scratchpad/learnpos_0407.py.
BOTH ABSOLUTE ARMS printed beside every difference (G202).
  .venv/bin/python agent/scratchpad/fillerarm_0607.py [records.jsonl]
"""
import json, sys
import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else "runs/probes/hintceiling/fillerarm_r98_5320.jsonl"
TAU, PER_SEQ = 4.0, 8
recs = [json.loads(l) for l in open(P) if l.strip()]
print(f"{P}\n{len(recs)} records, {len(set((r['set'], r['seq']) for r in recs))} sequences")

# the deployment trigger used by every rung of runs 96-98: tau>4.0, top-8 per sequence
by = {}
for r in recs:
    if r["H"] > TAU:
        by.setdefault((r["set"], r["seq"]), []).append(r)
sel = []
for k, v in by.items():
    sel += sorted(v, key=lambda r: -r["H"])[:PER_SEQ]
print(f"deployment trigger tau>{TAU}, top-{PER_SEQ}/seq: {len(sel)} anchors over {len(by)} sequences")

seqs = np.array([hash((r["set"], r["seq"])) for r in sel])
useq = np.unique(seqs)
rng = np.random.default_rng(20261009)
idx_by = {s: np.where(seqs == s)[0] for s in useq}

def boot(fn, B_=2000):
    out = []
    for _ in range(B_):
        pick = rng.choice(len(useq), len(useq), replace=True)
        ii = np.concatenate([idx_by[useq[p]] for p in pick])
        out.append(fn(ii))
    return float(np.std(out))

cl = np.array([r["ce_clean"] for r in sel])                   # (n, K) no-thought
ow = np.array([r["ce_span"][0] for r in sel])                 # (n, K) own greedy thought
fi = np.array([r["ce_span_filler"][0] for r in sel])          # (n, K) length-matched filler
lens = np.array([r["len"][0] for r in sel])
print(f"thought length written: mean {lens.mean():.2f} tokens (filler is matched anchor by anchor)")

print("\n            OWN vs silence        FILLER vs silence       OWN minus FILLER        own<filler")
rows = {}
for k in (1, 2, 4, 8, 16):
    go = cl[:, :k].sum(1) - ow[:, :k].sum(1)
    gf = cl[:, :k].sum(1) - fi[:, :k].sum(1)
    d = go - gf
    eo, ef, ed = boot(lambda i: go[i].mean()), boot(lambda i: gf[i].mean()), boot(lambda i: d[i].mean())
    rows[k] = (go.mean(), eo, gf.mean(), ef, d.mean(), ed)
    print(f"   k={k:<3} {go.mean():+8.4f} +-{eo:.4f}   {gf.mean():+8.4f} +-{ef:.4f}   "
          f"{d.mean():+8.4f} +-{ed:.4f}   {np.mean(d < 0):.0%}")

print("\n" + "=" * 96)
o1, e1 = rows[1][0], rows[1][1]
print(f"B713  REGISTER CHECK FIRST. own arm at k=1 = {o1:+.4f} +-{e1:.4f}; the 04:07 reading on step 4,750")
print(f"      was -0.1781 +-0.0283. Band: inside [-0.33,-0.03] -> "
      f"{'HIT' if -0.33 <= o1 <= -0.03 else 'MISS -- B710-B714 ARE VOID'}")
d1, de1 = rows[1][4], rows[1][5]
print(f"B710  own-filler k=1 = {d1:+.4f} +-{de1:.4f}; band NEGATIVE and inside [-1.00,-0.02] -> "
      f"{'HIT' if -1.00 <= d1 <= -0.02 else ('MISS LOW' if d1 < -1.00 else 'MISS HIGH')}; point -0.18")
d8, de8 = rows[8][4], rows[8][5]
print(f"B711  own-filler k=8 = {d8:+.4f} +-{de8:.4f}; band NEGATIVE -> "
      f"{'HIT' if d8 < 0 else 'MISS'}; point -0.60")
f1, fe1 = rows[1][2], rows[1][3]
print(f"B712  FILLER absolute level k=1 = {f1:+.4f} +-{fe1:.4f}; band inside [-0.60,+0.10] -> "
      f"{'HIT' if -0.60 <= f1 <= 0.10 else ('MISS LOW' if f1 < -0.60 else 'MISS HIGH')}; point -0.02")
signs = {k: rows[k][4] for k in rows}
allneg = all(v < 0 for v in signs.values())
print(f"B714  own BELOW filler at every k in {{1,2,4,8,16}}: " +
      "  ".join(f"k={k}:{v:+.3f}" for k, v in signs.items()) +
      f"  -> {'HIT' if allneg else 'MISS (CATEGORICAL) -- A POSITIVE SIGN CORRECTS THE PAPER'}")
print("=" * 96)

# how many standard errors, and the share of anchors where the own thought beats its own filler
print(f"\nk=1: own-filler is {abs(d1)/max(de1,1e-9):.1f} se from zero; "
      f"own beats filler at {np.mean((cl[:,:1].sum(1)-ow[:,:1].sum(1)) > (cl[:,:1].sum(1)-fi[:,:1].sum(1))):.0%} of anchors")
print(f"k=8: own-filler is {abs(d8)/max(de8,1e-9):.1f} se from zero")
print(f"\nfor the paper's scale: run 34 @800 own-filler was -0.81 +-0.08 on 465 windows (N2's widest gap);")
print(f"this is the FIRST filler reading on any run after 82, on {len(sel)} anchors / {len(useq)} sequences.")

# length control: the subset where the model wrote exactly L tokens, so filler is identical in length
for Lw in sorted(set(lens.tolist())):
    msk = lens == Lw
    if msk.sum() < 30:
        continue
    go = cl[msk, :1].sum(1) - ow[msk, :1].sum(1); gf = cl[msk, :1].sum(1) - fi[msk, :1].sum(1)
    print(f"  length-{Lw} subset (n={int(msk.sum())}): own {go.mean():+.4f}  filler {gf.mean():+.4f}  diff {(go-gf).mean():+.4f}")
