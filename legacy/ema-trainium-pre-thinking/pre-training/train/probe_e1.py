"""E1 gate probe (NEXT-STEPS.md sec. 4; run 40's pre-registered G1/G2), zero-update, CPU.

Held-out synthetic addition (synth.eval_set: disjoint seed stream, 14 problems per
sequence, anchor on the ' ' after ' =', window starts at the answer's first digit).
Every anchor is scored in FIVE modes on the same weights, same anchors, paired:
    clean   no thought (the clean rows of the same buffer)
    filler  10 content-free tokens + </T>      (the within-weights floor)
    tf      the TRUE scratchpad (reversed answer + </T>) teacher-forced in the slot
            -> G1: tf - filler window gain >= 2 nats, > 3 s.e.  (the reader can use it)
    greedy  the model's own greedy thought      -> G2 (think mode)
    sample  one unselected temperature-1 thought (the E1 training sampler)
Answer EXACT MATCH per mode = every answer digit and the closing newline is the argmax
given the true prefix; this is identical to greedy decoding the answer (greedy decode
reproduces the truth iff every teacher-forced argmax is right).
G2: EM(greedy) >= EM(filler) + 10 points with McNemar p < 0.01, AND own(greedy) - filler
window gain > 0 at > 3 s.e.  UNINFORMATIVE if EM(filler) >= 0.85.

  USE_TORCH_XLA=0 python probe_e1.py <ckpt> <out.json> [--nseq 143] [--slot-pos aligned]
"""
import argparse, contextlib, json, math, os, sys, time
from types import SimpleNamespace
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_init import build_model, untie_lm_head
from think_gen import ScatterCache
import thinking as th
import think_par as tp
import synth as sy

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--nseq", type=int, default=143)
ap.add_argument("--slot-pos", choices=("built", "aligned"), default="aligned")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--width", type=int, default=0, help="synth fixed width (run 44: 8)")
ap.add_argument("--task", choices=("add", "chase"), default="add")
ap.add_argument("--hops", type=int, default=5, help="chase hop count (run 45 = 5)")
ap.add_argument("--hops-max", type=int, default=0,
                help="chase: >hops draws hops ~ U{hops..hops_max} per problem (run 46)")
ap.add_argument("--distract", type=int, default=1, help="chase distractor pairs")
ap.add_argument("--L", type=int, default=16, help="slot length; MUST match the run")
ap.add_argument("--stride", type=int, default=64, help="anchor stride == window k; MUST match the run")
ap.add_argument("--first-anchor", type=int, default=64,
                help="position of the first anchor; MUST match the run (run 46 = 96). "
                     "Default 64 reproduces runs 40-45 exactly.")
# SAMPLER FLAGS (added 2026-09-28 headless tick, for run 54). Defaults are EXACTLY the values that
# were hardcoded in `gargs` before this patch, so every probe run before today reproduces bit-for-bit.
# They exist because the `sample` arm must be drawn from the SAME policy the run trains (run 54 sets
# --repeat-ban 1 --freq-pen 0.3 --temp 1.15 --min-think 6), and think_par special-cases the greedy
# speculation row (l.209/l.249: `if r == spec_row: argmax`), so these reach the SAMPLED rows only.
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--freq-pen", type=float, default=0.0)
ap.add_argument("--repeat-ban", type=int, default=0)
ap.add_argument("--min-think", type=int, default=2)
ap.add_argument("--top-k-thought", type=int, default=512)
ap.add_argument("--thought-vocab", choices=("all", "digits"), default="all",
                help="run 56: restrict thoughts to digits + space (+ </T>), as the run trains")
ap.add_argument("--fill-len", type=int, default=10,
                help="content-free filler length (run 45 = 10). Set to the mean scratchpad "
                     "length for a length-matched floor.")
a = ap.parse_args()

T, K, L, SPACE = 1024, a.stride, a.L, 220
torch.set_grad_enabled(False)
model, cfg, _ = build_model(); untie_lm_head(model)
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v
      for k, v in sd.items() if not k.startswith("mix.")}
model.load_state_dict(sd, strict=False); model = model.float().eval()
trunk, head = model.model, model.lm_head
lay = tp.ParLayout(T, K, K, L, a.first_anchor); N, S = lay.N, lay.S
pos = tp.build_positions(lay, a.slot_pos); mask = tp.build_full_mask(lay)
win_local = torch.from_numpy(lay.win_rows().reshape(-1) - T)
clean_rows = torch.from_numpy(tp.clean_rows_for_windows(lay))
cache_pre = ScatterCache(cfg.num_hidden_layers, S)
cache_b = ScatterCache(cfg.num_hidden_layers, S)
gargs = SimpleNamespace(top_k_thought=a.top_k_thought, temp=a.temp, freq_pen=a.freq_pen,
                        min_think=a.min_think, repeat_ban=a.repeat_ban,
                        thought_allow=(np.array(list(range(15, 25)) + [220], dtype=np.int64)
                                       if a.thought_vocab == "digits" else None))
gen = tp.ParGenerator(trunk, head, lay, cfg.num_hidden_layers, 2, torch.device("cpu"),
                      contextlib.nullcontext, lambda: None, topk_transfer=0,
                      slot_pos=a.slot_pos)
gw = (0.97 ** np.arange(K)).astype(np.float64)


def lp_am(lg, tgt):
    lsm = torch.log_softmax(lg.float(), -1)
    return (lsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).numpy(),
            (lg.argmax(-1) == tgt).numpy())


def prefill(xr, tgt):
    cache_pre.set_positions(torch.arange(T))
    h = trunk(input_ids=torch.from_numpy(xr[None]), position_ids=pos[:, :T],
              attention_mask=mask[:, :, :T, :], past_key_values=cache_pre, use_cache=True)[0]
    lp, am = lp_am(head(h[0, clean_rows]), tgt.reshape(-1))
    return lp.reshape(N, K), am.reshape(N, K)


def score_tail(xr, contents, tgt):
    LP, AM = [], []
    for c0 in range(0, len(contents), a.batch):
        chunk = contents[c0:c0 + a.batch]; B = len(chunk)
        bufs = np.stack([tp.build_buffer(lay, xr, c, SPACE)[T:] for c in chunk])
        if cache_b.layers[0].keys is None or cache_b.layers[0].keys.shape[0] != B:
            for l_ in cache_b.layers:
                l_.keys = None; l_.values = None; l_.is_initialized = False
        cache_b.broadcast_from(cache_pre, B); cache_b.set_positions(torch.arange(T, S))
        h = trunk(input_ids=torch.from_numpy(bufs), position_ids=pos[:, T:].expand(B, -1),
                  attention_mask=mask[:, :, T:, :].expand(B, -1, -1, -1),
                  past_key_values=cache_b, use_cache=True)[0]
        lp, am = lp_am(head(h[:, win_local]), tgt.reshape(1, -1).expand(B, -1))
        LP.append(lp.reshape(B, N, K)); AM.append(am.reshape(B, N, K))
    return np.concatenate(LP), np.concatenate(AM)


def mcnemar(x_ok, y_ok):
    b = int((x_ok & ~y_ok).sum()); c = int((~x_ok & y_ok).sum()); n = b + c
    if n == 0:
        return dict(b=b, c=c, p=1.0)
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return dict(b=b, c=c, p=p)


fill = np.full((N, L), th.T_CLOSE, dtype=np.int64); fill[:, :a.fill_len] = SPACE
modes = ("clean", "filler", "tf", "greedy", "sample", "fwd")
R = {m: dict(em=[], ans_lp=[], gain=[]) for m in modes}
scr = dict(ex_g=[], ex_s=[], dig_g=[], dig_s=[], len_g=[])
ex_dump = []
rng = np.random.default_rng(a.seed)
t0 = time.time()
_ekw = (dict(width=a.width, T=T, first=a.first_anchor, stride=a.stride, n_anchor=N, L=L)
        if a.task == "add" else
        dict(T=T, first=a.first_anchor, stride=a.stride, n_anchor=N, L=L,
             hops=a.hops, hops_max=a.hops_max, n_distract=a.distract))
hops_per_anchor = []
for i, sq in enumerate(sy.eval_set(a.nseq, task=a.task, **_ekw)):
    xr = sq["x"][:T]
    tgt = torch.from_numpy(np.stack([xr[int(n) + 1:int(n) + 1 + K] for n in lay.anchors]))
    lpc, amc = prefill(xr, tgt)
    z, _ = gen.generate(torch.from_numpy(xr[None]), rng, gargs, 1, th.EOD)
    # fwd = the answer in FORWARD order in the slot: a READ-HEALTH check (the
    # literal next tokens; the vanilla init gains ~+11 nats from it, run 43 @500
    # gained 0 -- training had switched the slot off).  Not a gate.
    fwd = np.full((N, L), th.T_CLOSE, dtype=np.int64)
    for an in range(N):
        fwd[an, :len(sq["ans"][an])] = sq["ans"][an]
    lps, ams = score_tail(xr, [fill, sq["scr"], z[1], z[0], fwd], tgt)
    per = dict(clean=(lpc, amc), filler=(lps[0], ams[0]), tf=(lps[1], ams[1]),
               greedy=(lps[2], ams[2]), sample=(lps[3], ams[3]), fwd=(lps[4], ams[4]))
    for an in range(N):
        D = len(sq["ans"][an]) + 1                     # answer tokens + newline
        for m in modes:
            lp, am = per[m]
            R[m]["em"].append(bool(am[an, :D].all()))
            R[m]["ans_lp"].append(float(lp[an, :D].sum()))
            R[m]["gain"].append(float(((lp[an] - lpc[an]) * gw).sum()))
        for tag, zr in (("g", z[1]), ("s", z[0])):
            d = int(sq["slen"][an]); hit = zr[an] == sq["scr"][an]
            scr[f"ex_{tag}"].append(bool(hit[:d + 1].all()))
            scr[f"dig_{tag}"].append(float(hit[:d].mean()))
            m_ = 0                                   # correct chain values, as a prefix
            while m_ < d and hit[m_]:
                m_ += 1
            scr.setdefault(f"pv_{tag}", []).append((m_ + 1) // 3)
            # run 56: the same count after dropping LEADING SPACE tokens of the thought
            # (the writer learned to open with ' ', which the strict count reads as 0)
            zz = zr[an][np.argmax(zr[an] != 220):] if (zr[an] != 220).any() else zr[an]
            hit2 = zz[:d] == sq["scr"][an][:len(zz[:d])]
            m2 = 0
            while m2 < min(d, len(hit2)) and hit2[m2]:
                m2 += 1
            scr.setdefault(f"pvls_{tag}", []).append((m2 + 1) // 3)
        scr["len_g"].append(int(tp.stop_pos(z[1][an:an + 1], L)[0]))
        hops_per_anchor.append(int(sq["hops"][an]) if "hops" in sq else a.hops)
    if len(ex_dump) < 8:
        ex_dump.append(dict(problem=str(sq["ops"][0]), answer=str(sq["ans"][0]),
                            scratch_true=str(sq["scr"][0].tolist()),
                            greedy=z[1][0].tolist(), sample=z[0][0].tolist()))
    if i % 10 == 0 or i == a.nseq - 1:
        print(f"seq {i+1}/{a.nseq} ({time.time()-t0:.0f}s) EM " +
              " ".join(f"{m}={np.mean(R[m]['em']):.3f}" for m in modes) +
              f" | scr ex_g={np.mean(scr['ex_g']):.3f} | tf-fill gain="
              f"{np.mean(np.array(R['tf']['gain']) - np.array(R['filler']['gain'])):+.2f}",
              flush=True)


def paired(x, y):
    d = np.asarray(x) - np.asarray(y)
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(len(d))), n=len(d))


summ = {m: dict(em=float(np.mean(R[m]["em"])), ans_lp=float(np.mean(R[m]["ans_lp"])),
                gain=float(np.mean(R[m]["gain"]))) for m in modes}
emf = np.array(R["filler"]["em"]); emg = np.array(R["greedy"]["em"])
g1 = paired(R["tf"]["gain"], R["filler"]["gain"])
g2_gain = paired(R["greedy"]["gain"], R["filler"]["gain"])
g2_gain_s = paired(R["sample"]["gain"], R["filler"]["gain"])
mc = mcnemar(emg, emf)
# Per-depth breakdown (run 46's ladder): the harvest is expected to survive only
# where one pass cannot do the chain. Empty/degenerate for a fixed-hops run.
hp = np.array(hops_per_anchor)
emc = np.array(R["clean"]["em"])
gg = np.array(R["greedy"]["gain"]); gf = np.array(R["filler"]["gain"])
by_hops = {}
for h in sorted(set(hp.tolist())):
    sel = hp == h
    if sel.sum() < 2:
        continue
    d = gg[sel] - gf[sel]
    by_hops[str(h)] = dict(n=int(sel.sum()), em_clean=float(emc[sel].mean()),
                           em_filler=float(emf[sel].mean()), em_greedy=float(emg[sel].mean()),
                           em_greedy_minus_filler=float(emg[sel].mean() - emf[sel].mean()),
                           gain_greedy_minus_filler=float(d.mean()),
                           pv_greedy=float(np.mean(np.array(scr["pv_g"])[sel])),
                           se=float(d.std(ddof=1) / math.sqrt(len(d))))
verdict = dict(
    G1_tf_minus_filler=g1,
    G1_pass=bool(g1["mean"] >= 2.0 and g1["mean"] > 3 * g1["se"]),
    G2_em_greedy_minus_filler=float(emg.mean() - emf.mean()), G2_mcnemar=mc,
    G2_gain_greedy_minus_filler=g2_gain, G2_gain_sample_minus_filler=g2_gain_s,
    G2_pass=bool(emg.mean() - emf.mean() >= 0.10 and mc["p"] < 0.01
                 and g2_gain["mean"] > 3 * g2_gain["se"]),
    UNINFORMATIVE_filler_em_ge_085=bool(emf.mean() >= 0.85),
    by_hops=by_hops,
    em_greedy_vs_clean=mcnemar(emg, np.array(R["clean"]["em"])),
    scratch=dict(ex_greedy=float(np.mean(scr["ex_g"])), ex_sample=float(np.mean(scr["ex_s"])),
                 digit_greedy=float(np.mean(scr["dig_g"])), digit_sample=float(np.mean(scr["dig_s"])),
                 len_greedy=float(np.mean(scr["len_g"])),
                 pv_greedy=float(np.mean(scr["pv_g"])), pv_sample=float(np.mean(scr["pv_s"]))))
json.dump(dict(ckpt=a.ckpt, slot_pos=a.slot_pos, n_anchors=len(emf),
               layout=dict(L=L, stride=a.stride, k=K, N=N, fill_len=a.fill_len,
                           hops=a.hops, hops_max=a.hops_max, distract=a.distract),
               hops_per_anchor=hops_per_anchor, summary=summ,
               verdict=verdict, per_anchor=R, scratch_per_anchor=scr, examples=ex_dump),
          open(a.out, "w"), indent=1)
print(json.dumps(dict(summary=summ, verdict=verdict), indent=1))
