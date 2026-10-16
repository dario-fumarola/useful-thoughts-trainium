"""Zero-update probe: how much is a PERFECT hint worth as a function of WHERE
in the future it comes from?  oracle offset o: content = x[n+1+o : n+1+o+L].
o = k (=64) is the trainer's oracle ("the L tokens beyond the scored window");
o = 0 is the immediate continuation. Scores the reader of a checkpoint on
held-out text with the training-faithful splice (think_par.score_buffer).
  USE_TORCH_XLA=0 python probe_oracle_offset.py <ckpt.pt> <out.json> [n_per_domain] [self16]

self16 (4th arg, default 0 = off, old behaviour byte-for-byte): adds the
SELF-FORECAST condition of hypothesis queue #5 -- the model's own greedy
prediction of the window, spliced as the thought. It is computed
TEACHER-FORCED (argmax at clean row n+j, i.e. conditioned on the TRUE prefix
x[:n+1+j]), which is NOT available at inference and is therefore an UPPER
BOUND on any free-running self-forecast a writer could be trained toward:
free-running greedy compounds its own errors and so overlaps the true window
less, and overlap with the true window is what the reader pays for.
The point of the bound: if even self16 fails to clear pause_ctl (the filler
floor, SUPERVISOR.md S1), no writer trained to forecast can ever help, and
queue #5 closes on one cheap forward instead of a run. If it clears, the
expensive free-running version is worth building.
`ovl16` reports its mean content-token overlap with the true window, the same
statistic recall_r24_3800.json reports (4.0/16) for retrieval thoughts.

prec (5th arg, default 0 = off): the PRECISION LADDER. `mix<K>` is the true
o0 hint with only K of its LEN tokens kept and the rest replaced by tokens
drawn from this sequence's own earlier context -- register-correct,
content-wrong, which is exactly what an imperfect forecast looks like. It
answers the number the whole programme turns on: HOW ACCURATE must a
12-token forecast be before it is worth more than silence? The break-even K
is where the mix curve crosses pause_ctl. If break-even sits near 12/12, no
writer can ever pay for itself and the ceiling is a property of the reader,
not of our training; if it sits near 6/12, self16 (which already achieves
~5.7/12) is one improvement away from working.

abst (6th arg, default 0 = off, old behaviour byte-for-byte): the TWO
PRECONDITION PROBES for the ABSTENTION idea (hypothesis queue #7). The ladder
of 2026-09-18 says two things that point the same way: filler in the slot
costs ~0 (pause_ctl -0.735 vs silence -0.689), and a CONFIDENT WRONG token
costs MORE than noise (self16 is right 5.30/12 yet scores 1.2 nats below the
mix rung at that accuracy). So: let the writer commit only where it is
confident and emit the filler token everywhere else.
  pmix<K>: like mix<K>, but the LEN-K wrong positions are filled with the
    PAUSE token instead of context noise. If pmix break-even is BELOW mix's
    ~4/12, abstention buys something and the idea is alive. CAVEAT the probe
    exists to settle: interleaved pause is OUT OF DISTRIBUTION for a reader
    trained on a `[SPACE]*N + [T_CLOSE]` prefix BLOCK, so a bad pmix means
    "needs a mixed-fill reader", NOT "abstention is dead".
  selfconf<C>: the teacher-forced self-forecast keeping only its C MOST
    CONFIDENT positions and pausing the rest; `prec@C` is the fraction of
    those kept positions that are positionally CORRECT. This is the number
    that decides everything: if the model's own confidence does not select
    correct positions (prec@C near self16's 5.30/12 = 0.44), no abstention
    scheme can work regardless of what the reader does with the pauses.
"""
import contextlib, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_init import build_model, untie_lm_head
from data import Mixture, eval_batches
import think_par as tp
ck, out = sys.argv[1], sys.argv[2]
NPER = int(sys.argv[3]) if len(sys.argv) > 3 else 12
SELF16 = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
PREC = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
ABST = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False
# 7th arg (default off, keeps every older invocation bit-identical): the rung run 32
# ACTUALLY trains on. `pmix<K>` above drops positions with rng.choice, i.e. UNIFORMLY AT
# RANDOM, so its surviving true tokens include content words. `--abstain-mode freq` keeps
# a true token only where it is a top-300 unigram -- precisely the tokens the reader can
# already predict -- so pmix6 (+0.77) was never the right reference for run 32's target.
# pmixfreq measures that target, and pmixfreqrand is its COUNT-MATCHED random control:
# same number of kept positions per anchor, chosen at random instead of by frequency.
# The pair isolates WHICH tokens survive from HOW MANY survive.
FREQ = bool(int(sys.argv[7])) if len(sys.argv) > 7 else False
# 8th arg (default off, keeps every older invocation bit-identical): WHICH selector should
# run 33 use? `freq` keeps the true token where it is a frequent unigram, and pmixfreq -
# pmixfreqrand measured that at -0.22/-0.63/-1.19 nats (m=4/7/12): frequency is a measurably
# BAD selector, because the tokens it keeps are the ones the reader already predicts.
# Queue #14 proposes selecting by the model's own CONFIDENCE instead -- but confidence is a
# BETTER predictor of predictability than frequency is, so the same objection may apply with
# MORE force, and that is a measurement, not an opinion. So score both directions:
#   pmixconf<m>   keep the true token at the model's MOST confident positions
#   pmixanti<m>   keep the true token at its LEAST confident positions
# both COUNT-MATCHED per anchor to the freq mask, so all four selectors (freq / random /
# conf / anti) differ ONLY in WHICH positions survive, never in how many. Requires FREQ=1,
# which supplies both the per-anchor count and the shared pausectl<m> floor.
CONF = bool(int(sys.argv[8])) if len(sys.argv) > 8 else False
if CONF and not FREQ:
    sys.exit("CONF (arg 8) needs FREQ (arg 7) on: the count-match and the pausectl<m> floor come from it.")
MIXK = [2, 4, 6, 8, 10, 12]
CONFC = [2, 3, 4, 6, 8]   # how many positions the writer would commit to
T, K, L, GAMMA = 1024, 64, 16, 0.97
LEN = 12   # tokens before </T>: the typical trained length; every trained thought ends with </T>
SPACE = 220
model, _, _ = build_model(); untie_lm_head(model)
sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items() if not k.startswith("mix.")}
model.load_state_dict(sd, strict=False); model = model.float().eval()
layout = tp.ParLayout(T, K, K, L, 64)
pos = tp.build_positions(layout); mask = tp.build_full_mask(layout)
wk = torch.from_numpy((GAMMA ** np.arange(K)).astype(np.float32))
rows_win = torch.from_numpy(layout.win_rows().reshape(-1)); rows_clean = torch.from_numpy(tp.clean_rows_for_windows(layout))
OFFS = [0, 8, 16, 24, 32, 48, 64]
# silence_ctl = a thought of only </T> (empty). pause_ctl = the SPACE-filled filler that
# think_train.py --pause-len 10 trains on and that run 26 showed costs ~nothing: it is the
# FLOOR any written thought must beat, so the whole ladder is on one scale, one checkpoint.
PAUSE_LEN = 10
conds = [f"o{o}" for o in OFFS] + ["shuf0", "shuf64", "silence_ctl", "pause_ctl"]
if SELF16: conds += ["self16"]
if PREC: conds += [f"mix{MK}" for MK in MIXK]   # NB: NOT `K` -- K is the window length
# G24: every loop variable below is PREFIXED (MK/PK/PC). Bare K, L, LEN, pos, mask,
# rng, span are module-scope names the layout was built with; rebinding one is the
# bug that only shows up on the SECOND sequence.
if ABST: conds += [f"pmix{PK}" for PK in MIXK] + [f"selfconf{PC}" for PC in CONFC] \
                 + [f"selfanti{PC}" for PC in CONFC]
# selfanti<C> added 2026-09-19 14:25. selfconf<C> is the ACHIEVABLE value of a
# confidence-selected writer: its own argmax at the C positions it is most sure of. The
# count-matched ceiling probe (arg 8) says the CEILING is far higher at the positions it is
# LEAST sure of -- but those are exactly the positions where its argmax is most often wrong,
# and a confident wrong token costs more than a blank slot. selfanti<C> is the same
# construction at the C LEAST confident positions, so selfconf<C> vs selfanti<C> is the
# ceiling comparison repeated with the WRITER'S REAL ACCURACY priced in. Consumes no rng, so
# every older column is unchanged.
# CAVEAT, measured not assumed (2026-09-19 08:07): the pmixfreqrand draw below consumes the
# SHARED module-scope `rng`, so with FREQ=1 the shuf0/shuf64 columns get a DIFFERENT (still
# valid) random draw from sequence 2 onward than a FREQ=0 run would. Verified: seq 1 is
# byte-identical (shuf0 +1.26, shuf64 -1.41 both ways, FREQ consumes nothing before it) and
# seq 2 diverges (shuf0 +2.06 -> +2.22). FREQ=0 is bit-identical throughout, since nothing
# here touches rng then. The deterministic columns (o*, silence_ctl, pause_ctl, pausectl<m>)
# are unaffected either way. Do NOT compare shuf0/shuf64 across a FREQ=1 and a FREQ=0 file.
FQM = [4, 7, 12]   # the trainer's SSL hint is the thought's own trained length (~7), NOT 12
if FREQ: conds += [f"{p}{FM}" for FM in FQM for p in ("pmixfreq", "pmixfreqrand", "pausectl")]
if CONF: conds += [f"{p}{FM}" for FM in FQM for p in ("pmixconf", "pmixanti")]
# the SAME file and the SAME cut the trainer uses, so this is run 32's target, not a proxy
FQ_KEEP = (np.load("/home/ubuntu/ema-trainium/pre-training/train/"
                   "stop_tokens_top300.npy")[:300] if FREQ else None)
fq_cov = []   # per-anchor share of the next-LEN tokens the mask KEEPS
fq_nk = {}    # {m: per-anchor count kept by the freq mask}, refilled each sequence (CONF)
prec_at = {PC: [] for PC in CONFC}   # precision of the C most confident positions
prec_an = {PC: [] for PC in CONFC}   # precision of the C LEAST confident positions
ovl = []   # self16 SET overlap with the true window (token appears anywhere in it)
posacc = []   # self16 POSITIONAL accuracy: token j equals the true token at j.
# The two differ and the difference is the whole diagnosis: the mix<K> ladder keeps K
# tokens IN THEIR TRUE POSITIONS, so only `pos` is comparable to a break-even K.
res = {c: [] for c in conds}
rng = np.random.default_rng(7)
mix = Mixture("/home/ubuntu/ema-trainium/data/corpus", "val")
t0 = time.time(); nseq = 0
for dom in mix.domains:
    for x, y in eval_batches(mix, dom, 1, T, NPER):
        xr = np.asarray(x[0], dtype=np.int64); nseq += 1
        vw, vo = tp.validity(layout, xr)
        ok = vo > 0
        if ok.sum() == 0: continue
        tgt = torch.from_numpy(np.stack([xr[int(n)+1:int(n)+1+K] for n in layout.anchors]).reshape(-1).astype(np.int64))
        def score(content):
            buf = tp.build_buffer(layout, xr, content, SPACE)
            lw, lc = tp.score_buffer(model.model, model.lm_head, buf, pos, mask, layout, wk, rows_win, rows_clean, tgt, contextlib.nullcontext)
            return (lw - lc)[ok]
        for o in OFFS:
            c = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
            for a, n in enumerate(layout.anchors): c[a, :LEN] = xr[int(n)+1+o:int(n)+1+o+LEN]
            res[f"o{o}"].extend(score(c).tolist())
            if o in (0, 64):
                cs = c.copy()
                for a in range(layout.N): cs[a, :LEN] = rng.permutation(c[a, :LEN])
                res[f"shuf{o}"].extend(score(cs).tolist())
        # control: a thought of only </T> (empty) -- pure insertion cost of the block
        res["silence_ctl"].extend(score(np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)).tolist())
        # control: run 26's content-free filler, the floor a written thought must clear
        cp = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
        cp[:, :min(PAUSE_LEN, L - 1)] = SPACE
        res["pause_ctl"].extend(score(cp).tolist())
        # run 32's ACTUAL supervision target, and its count-matched random control.
        # G24: every name here is fq-prefixed; `span`/`rng`/`LEN` are module scope.
        if FREQ:
            for FM in FQM:
                cfq = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
                crq = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
                cpq = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
                cpq[:, :FM] = SPACE   # filler floor at THIS length, not the fixed 10
                for fqa, fqn in enumerate(layout.anchors):
                    fqn = int(fqn)
                    fqspan = xr[fqn + 1:fqn + 1 + FM].copy()
                    fqkeep = np.isin(fqspan, FQ_KEEP)
                    if FM == LEN: fq_cov.append(float(fqkeep.mean()))
                    cfq[fqa, :FM] = np.where(fqkeep, fqspan, SPACE)
                    fqrand = fqspan.copy()
                    fqnk = int(fqkeep.sum())
                    fq_nk.setdefault(FM, np.zeros(layout.N, dtype=np.int64))[fqa] = fqnk
                    if fqnk < FM:
                        fqrand[rng.choice(FM, FM - fqnk, replace=False)] = SPACE
                    crq[fqa, :FM] = fqrand
                res[f"pmixfreq{FM}"].extend(score(cfq).tolist())
                res[f"pmixfreqrand{FM}"].extend(score(crq).tolist())
                res[f"pausectl{FM}"].extend(score(cpq).tolist())
        # queue #5: the model's OWN greedy forecast of the window as the thought.
        # Teacher-forced argmax on the clean rows of a silence buffer -- those rows
        # never attend to a thought block, so this is the no-thought next-token
        # prediction of x[n+1+j] given the true x[:n+1+j].
        # precision ladder: the true hint with only K of LEN tokens kept, the rest
        # replaced by context tokens (right register, wrong content).
        if PREC:
            for MK in MIXK:
                cm = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
                for a, na in enumerate(layout.anchors):
                    na = int(na)
                    span = xr[na + 1:na + 1 + LEN].copy()
                    if MK < LEN:
                        drop = rng.choice(LEN, LEN - MK, replace=False)
                        span[drop] = xr[rng.integers(0, na, size=LEN - MK)]
                    cm[a, :LEN] = span
                res[f"mix{MK}"].extend(score(cm).tolist())
        if ABST:
            # pmix<K>: true tokens in true positions, PAUSE token elsewhere.
            for PK in MIXK:
                cq = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
                for a, na in enumerate(layout.anchors):
                    na = int(na)
                    qspan = xr[na + 1:na + 1 + LEN].copy()
                    if PK < LEN:
                        qdrop = rng.choice(LEN, LEN - PK, replace=False)
                        qspan[qdrop] = SPACE
                    cq[a, :LEN] = qspan
                res[f"pmix{PK}"].extend(score(cq).tolist())
        if SELF16 or ABST or CONF:
            rows = (layout.anchors[:, None] + np.arange(LEN)[None, :]).reshape(-1)
            with torch.no_grad():
                bt = torch.from_numpy(
                    tp.build_buffer(layout, xr,
                                    np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64),
                                    SPACE)[None])
                h = model.model(input_ids=bt, position_ids=pos,
                                attention_mask=mask, use_cache=False)[0]
                lg = model.lm_head(h[0, torch.from_numpy(rows)])
                pred = lg.argmax(-1).view(layout.N, LEN).numpy().astype(np.int64)
                # p(argmax) without materialising a softmax over the vocab
                conf = (lg.max(-1).values - lg.logsumexp(-1)).exp() \
                          .view(layout.N, LEN).numpy()
        if ABST:
            # selfconf<C>: commit to the C most confident positions, pause the rest.
            for PC in CONFC:
                cc = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
                cc[:, :LEN] = SPACE
                for a, na in enumerate(layout.anchors):
                    keep = np.argsort(-conf[a])[:PC]
                    cc[a, keep] = pred[a, keep]
                    if ok[a]:
                        tspan = xr[int(na) + 1:int(na) + 1 + LEN]
                        prec_at[PC].append(float((pred[a, keep] == tspan[keep]).mean()))
                res[f"selfconf{PC}"].extend(score(cc).tolist())
            # selfanti<C>: the same commitment at the C LEAST confident positions.
            for PC in CONFC:
                ca = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
                ca[:, :LEN] = SPACE
                for a, na in enumerate(layout.anchors):
                    anti = np.argsort(-conf[a])[-PC:]
                    ca[a, anti] = pred[a, anti]
                    if ok[a]:
                        aspan = xr[int(na) + 1:int(na) + 1 + LEN]
                        prec_an[PC].append(float((pred[a, anti] == aspan[anti]).mean()))
                res[f"selfanti{PC}"].extend(score(ca).tolist())
        if SELF16:
            cf = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
            cf[:, :LEN] = pred
            res["self16"].extend(score(cf).tolist())
            for a, n in enumerate(layout.anchors):
                if ok[a]:
                    span = xr[int(n) + 1:int(n) + 1 + LEN]
                    ovl.append(float(sum(1 for t in pred[a] if t in set(span.tolist()))))
                    posacc.append(float((pred[a] == span).sum()))
        if CONF:
            # G24: every name here is cf-prefixed. No rng use, so all columns above are
            # bit-identical to a CONF=0 run on the same checkpoint -- pmixfreq<m> and
            # pausectl<m> reproducing their published values IS this probe's alignment check.
            for CM in FQM:
                ccf = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
                can = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
                ccf[:, :CM] = SPACE; can[:, :CM] = SPACE
                for cfa, cfn in enumerate(layout.anchors):
                    cfn = int(cfn)
                    cfspan = xr[cfn + 1:cfn + 1 + CM]
                    cfk = int(fq_nk[CM][cfa])            # count-matched to the freq mask
                    if cfk > 0:
                        cford = np.argsort(-conf[cfa, :CM])
                        cftop, cfbot = cford[:cfk], cford[-cfk:]
                        ccf[cfa, cftop] = cfspan[cftop]
                        can[cfa, cfbot] = cfspan[cfbot]
                res[f"pmixconf{CM}"].extend(score(ccf).tolist())
                res[f"pmixanti{CM}"].extend(score(can).tolist())
        print(f"{dom} seq {nseq} ({time.time()-t0:.0f}s): " + " ".join(f"{c}={np.mean(res[c]):+.2f}" for c in conds), flush=True)
summary = {c: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1)/np.sqrt(len(v))), win=float(np.mean(np.array(v) > 0)), n=len(v)) for c, v in res.items()}
if SELF16 and ovl:
    summary["ovl16"] = dict(mean=float(np.mean(ovl)), se=float(np.std(ovl, ddof=1) / np.sqrt(len(ovl))), win=float(np.mean(np.array(ovl) > 0)), n=len(ovl), of=LEN)
    summary["pos16"] = dict(mean=float(np.mean(posacc)), se=float(np.std(posacc, ddof=1) / np.sqrt(len(posacc))), win=float(np.mean(np.array(posacc) > 0)), n=len(posacc), of=LEN)
_out = dict(ckpt=ck, self16=SELF16, prec=PREC, abst=ABST, summary=summary)
if FREQ:   # per-anchor, in one shared anchor order, so every pair below is PAIRED (G26)
    _out["freq"] = dict(cov_mean=float(np.mean(fq_cov)), cov_n=len(fq_cov),
                        per_anchor={c: res[c] for c in
                                    [f"{p}{FM}" for FM in FQM
                                     for p in ("pmixfreq", "pmixfreqrand", "pausectl")]
                                    + ([f"{p}{FM}" for FM in FQM
                                        for p in ("pmixconf", "pmixanti")] if CONF else [])
                                    + ["pause_ctl", "o0"]})
json.dump(_out, open(out, "w"), indent=1)
print("=== oracle gain by offset (nats over silence, discounted k=64 window) ===")
for c in conds: print(f"  {c:<12} {summary[c]['mean']:+7.2f} ± {summary[c]['se']:.2f}   win {summary[c]['win']:.2f}   n={summary[c]['n']}")
if "ovl16" in summary:
    print(f"  self16 SET overlap with the true window:  {summary['ovl16']['mean']:.2f} / {LEN}")
    print(f"  self16 POSITIONAL accuracy (comparable to mix<K>): {summary['pos16']['mean']:.2f} / {LEN}")
    print(f"  SUCCESS BAR: self16 - pause_ctl = {summary['self16']['mean'] - summary['pause_ctl']['mean']:+.2f} nats "
          f"(must be > 0 for queue #5 to survive; this is an UPPER BOUND)")
if FREQ:
    def _pr(lbl, d):
        se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
        print(f"    {lbl:<38} {d.mean():+7.3f} +- {se:.3f}   t={d.mean()/se:+6.2f}")
    print(f"  RUN 32'S ACTUAL TARGET (top-300 mask; keeps {np.mean(fq_cov):.3f} of the "
          f"next-{LEN} tokens, n={len(fq_cov)} anchors).")
    print("  Every clause below is PAIRED and MATCHED IN LENGTH. m=7 is the length the writer")
    print("  actually emits, so IT is run 32's operating point; m=12 is the ladder's rung.")
    for FM in FQM:
        _f = np.array(res[f"pmixfreq{FM}"]); _r = np.array(res[f"pmixfreqrand{FM}"])
        _p = np.array(res[f"pausectl{FM}"])
        print(f"  --- m={FM} ---")
        _pr("pmixfreq - pause(m)   (the CEILING)", _f - _p)
        _pr("pmixfreq - pmixfreqrand (selection)", _f - _r)
        _pr("pmixfreqrand - pause(m) (count only)", _r - _p)
    if CONF:
        print("  RUN 33'S SELECTOR CHOICE, all COUNT-MATCHED to the freq mask and PAIRED.")
        print("  The question: with the SAME number of true tokens kept, does selecting them by")
        print("  the model's own confidence beat selecting them by unigram frequency -- or does")
        print("  it lose to selecting the positions the model is LEAST sure of?")
        for CM in FQM:
            _f = np.array(res[f"pmixfreq{CM}"]); _r = np.array(res[f"pmixfreqrand{CM}"])
            _p = np.array(res[f"pausectl{CM}"]); _c = np.array(res[f"pmixconf{CM}"])
            _a = np.array(res[f"pmixanti{CM}"])
            print(f"  --- m={CM} (ceiling over the blank slot of the same length) ---")
            _pr("pmixconf - pause(m)", _c - _p)
            _pr("pmixanti - pause(m)", _a - _p)
            _pr("pmixfreq - pause(m)  [published]", _f - _p)
            _pr("pmixconf - pmixfreq  (RUN 33 GATE)", _c - _f)
            _pr("pmixanti - pmixfreq", _a - _f)
            _pr("pmixanti - pmixconf", _a - _c)
            _pr("pmixconf - pmixfreqrand", _c - _r)
        print("    READ: the RUN 33 GATE is `pmixconf - pmixfreq` at m=7. Queue #14 asserts the"
              "\n    model's confidence is a better abstention selector than unigram frequency;"
              "\n    that clause > 0 by 2 se is the assertion, and <= 0 falsifies #14 BEFORE a run"
              "\n    is spent on it. `pmixanti` is the competing hypothesis: if the value lives in"
              "\n    the tokens the reader CANNOT already predict, anti beats both and run 33"
              "\n    should select on LOW confidence -- the opposite of what the queue says.")
    print("    READ: clause 1 is what a PERFECT run-32 writer could buy over a blank slot of the"
          "\n    SAME length -- if it is <= 0 at m=7 the run's supervision target is itself below"
          "\n    the filler floor AT ITS OPERATING LENGTH and no training can clear it. Clause 2"
          "\n    isolates WHICH tokens survive from HOW MANY: it is the cost of using unigram"
          "\n    frequency as the abstention selector, i.e. the size of the run-33 prize.")
if ABST:
    for PC in CONFC:
        summary[f"prec@{PC}"] = dict(mean=float(np.mean(prec_at[PC])),
                                     se=float(np.std(prec_at[PC], ddof=1) / np.sqrt(len(prec_at[PC]))),
                                     win=float(np.mean(np.array(prec_at[PC]) > 0)), n=len(prec_at[PC]), of=PC)
    for PC in CONFC:
        summary[f"precanti@{PC}"] = dict(mean=float(np.mean(prec_an[PC])),
                                     se=float(np.std(prec_an[PC], ddof=1) / np.sqrt(len(prec_an[PC]))),
                                     win=float(np.mean(np.array(prec_an[PC]) > 0)), n=len(prec_an[PC]), of=PC)
if PREC:
    floor = summary["pause_ctl"]["mean"]
    be = next((MK for MK in MIXK if summary[f"mix{MK}"]["mean"] > floor), None)
    print(f"  PRECISION LADDER (filler floor {floor:+.2f}): break-even at "
          + (f"{be}/{LEN} correct tokens" if be else f"NOT REACHED even at {LEN}/{LEN}"))
if ABST:
    afloor = summary["pause_ctl"]["mean"]
    pbe = next((PK for PK in MIXK if summary[f"pmix{PK}"]["mean"] > afloor), None)
    print(f"  ABSTENTION LADDER pmix (pause-filled, floor {afloor:+.2f}): break-even at "
          + (f"{pbe}/{LEN}" if pbe else f"NOT REACHED even at {LEN}/{LEN}")
          + "   -- compare with the context-noise mix break-even ~4/12")
    for PK in MIXK: print(f"    pmix{PK:<3} {summary[f'pmix{PK}']['mean']:+7.2f} +- {summary[f'pmix{PK}']['se']:.2f}")
    print("  SELF-CONFIDENCE SELECTION (the number that decides abstention):")
    for PC in CONFC:
        print(f"    selfconf{PC:<3} gain {summary[f'selfconf{PC}']['mean']:+7.2f} +- {summary[f'selfconf{PC}']['se']:.2f}"
              f"   prec@{PC} {summary[f'prec@{PC}']['mean']:.3f}   (self16 baseline accuracy 5.30/12 = 0.442)")
    print("  THE SAME COMMITMENT AT THE LEAST CONFIDENT POSITIONS (ceiling priced at real accuracy):")
    _af = summary["pause_ctl"]["mean"]
    for PC in CONFC:
        _c = summary[f"selfconf{PC}"]["mean"]; _a = summary[f"selfanti{PC}"]["mean"]
        print(f"    selfanti{PC:<3} gain {_a:+7.2f} +- {summary[f'selfanti{PC}']['se']:.2f}"
              f"   precanti@{PC} {summary[f'precanti@{PC}']['mean']:.3f}"
              f"   | anti-conf {_a - _c:+6.2f}   anti-floor {_a - _af:+6.2f}")
    print("    READ: the count-matched CEILING probe (arg 8) prices the TRUE token at each"
          "\n    selected position. These rungs price the token the writer would ACTUALLY emit."
          "\n    If pmixanti beats pmixconf but selfanti LOSES to selfconf, the information does"
          "\n    live in the low-confidence positions and the writer still cannot deliver it --"
          "\n    and run 33 must not chase the ceiling. `anti-floor` > 0 is the bar that matters.")
