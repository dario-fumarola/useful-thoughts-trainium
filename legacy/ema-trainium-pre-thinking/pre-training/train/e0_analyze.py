"""E0 analysis (NEXT-STEPS sec. 3, amended 2026-09-26). No model needed: reads probe_e0 npz files.

  python e0_analyze.py runs/e0/e0_*_pure.npz  -> runs/e0/e0_results.csv, e0_curves.png, stdout verdict

Estimators per anchor (valid anchors only), for each mixture cond in {own, shuf, cross, fill}:
  variant a (the identity, exact):  G_i = sum_j [lp_i(j) - lp_clean(j)]        (gamma = 1, k = 64)
                                    C_R = mean over the 32/R disjoint R-subsets of log mean_i exp(G_i)
  variant b (training units):       C_R = sum_j 0.97^j [log mean_i p_i(j) - lp_clean(j)]
Single-thought references: greedy thought, filler length 10 (the paper's floor), true 12-token oracle.
DECISION (primary sampler, variant a):
  ORIGINAL rule (as first written): C_32(own) - G(fill10): NULL if upper 95% CI <= 0.05 on every ckpt
     and C_32 - C_16 < 0.02; POSITIVE if > 0 at > 3 s.e. on >= 2 ckpts; else AMBIGUOUS.
  AMENDED rule (governs): C_32(own) - C_32(ctrl*) > 0 at > 3 s.e. on >= 2 ckpts, where ctrl* is the
     matched-R control (shuf / cross / fill) with the HIGHEST mean on that checkpoint.
"""
import csv, json, math, os, sys
import numpy as np

RS = (1, 2, 4, 8, 16, 32)   # overwritten from the data: powers of 2 <= R_max
GW = 0.97 ** np.arange(64)
rng = np.random.default_rng(0)


def lme(x, axis):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def c_R(G, R):
    """G [n, Rmax] -> per-anchor C_R [n]: mean over disjoint subsets of log-mean-exp."""
    n, Rm = G.shape
    k = Rm // R
    return np.mean([lme(G[:, i * R:(i + 1) * R], 1) for i in range(k)], 0)


def c_R_tok(LPc, LPclean, R):
    """variant b: LPc [n, Rmax, k] per-token logp; LPclean [n, k]."""
    Rm = LPc.shape[1]; k = Rm // R
    vals = []
    for i in range(k):
        mix = lme(LPc[:, i * R:(i + 1) * R, :], 1)              # [n, k]
        vals.append(((mix - LPclean) * GW).sum(-1))
    return np.mean(vals, 0)


def stat(d):
    d = np.asarray(d, dtype=np.float64)
    se = d.std(ddof=1) / math.sqrt(len(d))
    bs = rng.choice(d, size=(10000, len(d))).mean(1) if len(d) > 1 else d
    return dict(mean=float(d.mean()), se=float(se), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), n=int(len(d)))


def main(files):
    global RS
    rows, curves, per_ck = [], {}, {}
    Rmax = int(np.load(files[0])["own"].shape[1])
    RS = tuple(2 ** i for i in range(int(math.log2(Rmax)) + 1))
    RT, RH = RS[-1], RS[-2]                 # top R and the one below (32 and 16 by default)
    for f in files:
        z = np.load(f); meta = json.loads(str(z["meta"]))
        tag = os.path.basename(f).replace(".npz", "")
        if meta["nseq_done"] < meta["nseq"]:
            print(f"# {tag}: PARTIAL {meta['nseq_done']}/{meta['nseq']} sequences")
        v = z["valid"]                                        # [s, N]
        clean = z["clean"][v]                                 # [n, k]
        cond = {c: np.transpose(z[c], (0, 2, 1, 3))[v] for c in ("own", "shuf", "cross", "fill")}
        single = {c: z[c][v] for c in ("greedy", "fill10", "oracle")}
        Ga = {c: (cond[c] - clean[:, None, :]).sum(-1) for c in cond}     # [n, R]
        Sa = {c: (single[c] - clean).sum(-1) for c in single}             # [n]
        Sb = {c: ((single[c] - clean) * GW).sum(-1) for c in single}
        res = {}
        for var in ("a", "b"):
            for R in RS:
                for c in cond:
                    val = c_R(Ga[c], R) if var == "a" else c_R_tok(cond[c], clean, R)
                    res[(var, R, c)] = val
        per_ck[tag] = (res, Sa, Sb, Ga)
        for var in ("a", "b"):
            S1 = Sa if var == "a" else Sb
            for R in RS:
                for c in cond:
                    s_abs = stat(res[(var, R, c)])
                    s_vf = stat(res[(var, R, c)] - S1["fill10"])
                    s_vo = stat(res[(var, R, "own")] - res[(var, R, c)]) if c != "own" else None
                    rows.append(dict(ckpt=tag, sampler=meta["sampler"], register=meta["slot_pos"],
                                     variant=var, R=R, cond=c, C=s_abs["mean"], se=s_abs["se"],
                                     minus_fill10=s_vf["mean"], minus_fill10_se=s_vf["se"],
                                     own_minus_cond=(s_vo["mean"] if s_vo else ""),
                                     own_minus_cond_se=(s_vo["se"] if s_vo else ""),
                                     n=s_abs["n"]))
            for c in S1:
                s_ = stat(S1[c])
                rows.append(dict(ckpt=tag, sampler=meta["sampler"], register=meta["slot_pos"],
                                 variant=var, R=1, cond=f"single_{c}", C=s_["mean"], se=s_["se"],
                                 minus_fill10=float(np.mean(S1[c] - S1["fill10"])),
                                 minus_fill10_se=stat(S1[c] - S1["fill10"])["se"],
                                 own_minus_cond="", own_minus_cond_se="", n=s_["n"]))
        curves[tag] = {c: [float(np.mean(res[("a", R, c)])) for R in RS] for c in cond}
        curves[tag]["fill10"] = float(np.mean(Sa["fill10"])); curves[tag]["greedy"] = float(np.mean(Sa["greedy"]))
        curves[tag]["sd_own"] = float(Ga["own"].std(1, ddof=1).mean())
    out_dir = os.path.dirname(files[0])
    with open(os.path.join(out_dir, "e0_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- verdicts (primary sampler, variant a)
    print("\n=== E0 verdict (variant a = undiscounted window, the identity) ===")
    pos_orig = pos_amend = 0; null_all = True
    for tag, (res, Sa, Sb, Ga) in per_ck.items():
        d_orig = stat(res[("a", RT, "own")] - Sa["fill10"])
        rise = float(np.mean(res[("a", RT, "own")] - res[("a", RH, "own")]))
        ctrl = max(("shuf", "cross", "fill"), key=lambda c: np.mean(res[("a", RT, c)]))
        d_am = stat(res[("a", RT, "own")] - res[("a", RT, ctrl)])
        eq = float(np.mean(Ga["own"]))
        print(f"{tag:28s} E_q[G]={eq:+.2f}  sd_within={curves[tag]['sd_own']:.2f}  "
              f"C32(own)={np.mean(res[('a',RT,'own')]):+.2f}  fill10={np.mean(Sa['fill10']):+.2f}  "
              f"greedy={np.mean(Sa['greedy']):+.2f}  oracle={np.mean(Sa['oracle']):+.2f}")
        print(f"{'':28s} ORIGINAL C32-fill10 {d_orig['mean']:+.3f} +- {d_orig['se']:.3f} "
              f"[{d_orig['lo']:+.3f},{d_orig['hi']:+.3f}]  C32-C16 {rise:+.3f}")
        for c in ("shuf", "cross", "fill"):
            s_ = stat(res[("a", RT, "own")] - res[("a", RT, c)])
            print(f"{'':28s}   own-{c:5s} at R=32: {s_['mean']:+.3f} +- {s_['se']:.3f}"
                  f"   (C32({c})={np.mean(res[('a',RT,c)]):+.2f})")
        print(f"{'':28s} AMENDED own - {ctrl} (strongest control) {d_am['mean']:+.3f} +- {d_am['se']:.3f}")
        if d_orig["mean"] > 3 * d_orig["se"]:
            pos_orig += 1
        if not (d_orig["hi"] <= 0.05 and rise < 0.02):
            null_all = False
        if d_am["mean"] > 3 * d_am["se"]:
            pos_amend += 1
    orig = "POSITIVE" if pos_orig >= 2 else ("NULL" if null_all else "AMBIGUOUS")
    amend = "POSITIVE" if pos_amend >= 2 else "NOT POSITIVE"
    print(f"\nORIGINAL rule: {orig} ({pos_orig} ckpts > 3 s.e.)   AMENDED rule: {amend} ({pos_amend} ckpts)")
    print("E2 is licensed only by the AMENDED rule.")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 3.6), squeeze=False)
        for ax, (tag, cv) in zip(axs[0], curves.items()):
            for c, col in (("own", "C0"), ("shuf", "C1"), ("cross", "C2"), ("fill", "C7")):
                ax.plot(RS, cv[c], "o-", color=col, label=c)
            ax.axhline(cv["fill10"], color="k", ls="--", lw=1, label="filler (1 pass)")
            ax.axhline(cv["greedy"], color="C3", ls=":", lw=1, label="greedy (1 pass)")
            ax.set_xscale("log", base=2); ax.set_title(tag, fontsize=9); ax.set_xlabel("R")
        axs[0][0].set_ylabel("C_R (nats / 64-token window)"); axs[0][0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "e0_curves.png"), dpi=130)
    except Exception as e:
        print("plot failed:", e)


if __name__ == "__main__":
    main(sys.argv[1:])
