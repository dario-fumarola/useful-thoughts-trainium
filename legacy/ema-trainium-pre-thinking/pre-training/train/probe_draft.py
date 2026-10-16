"""How much of a thought can the model get RIGHT before it starts to hurt?

A perfect ordered 12-token hint is worth +11 nats; the model's own 10-token
thought is worth -1.2. If the damage comes from the WRONG tail of a long guess,
a SHORT draft should be better, and the value of truth should be visible at
every length. For m in {2,4,6,8,12} we score three contents at each anchor:
  own    = the model's own greedy thought truncated to m tokens
  oracle = the true next m tokens (upper bound at that length)
  filler = m content-free tokens (the floor: extra positions, no content)
  USE_TORCH_XLA=0 python probe_draft.py <ckpt.pt> <out.json> [n_per_domain] [--slot-pos built|aligned]"""
import contextlib, json, os, sys, time
from types import SimpleNamespace
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_init import build_model, untie_lm_head
from data import Mixture, eval_batches
import thinking as th
import think_par as tp
ck, out = sys.argv[1], sys.argv[2]
NPER = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 8

# --slot-pos {built|aligned}: which position convention the CHECKPOINT WAS
# TRAINED WITH (`--slot-pos` in its run's config.sh).  Runs 13-29 are "built"
# (thought token i at n+2+i); run 30 onward is "aligned" (token i at n+1+i).
# Default "built" reproduces every probe run before 2026-09-18 22:07 exactly.
# Probing an aligned-trained checkpoint with built positions measures it one
# position out of its own register and understates it -- see POSTMORTEM G29.
# Both the scoring pass and the generator are switched together.
def _slot_pos_arg(argv):
    sp = argv[argv.index("--slot-pos") + 1] if "--slot-pos" in argv else "built"
    assert sp in ("built", "aligned"), sp
    return sp
SLOT_POS = _slot_pos_arg(sys.argv)
T, K, L, SPACE = 1024, 64, 16, 220
MS = [2, 4, 6, 8, 12]
model, cfg, _ = build_model(); untie_lm_head(model)
sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items() if not k.startswith("mix.")}
model.load_state_dict(sd, strict=False); model = model.float().eval()
layout = tp.ParLayout(T, K, K, L, 64)
pos = tp.build_positions(layout, SLOT_POS); mask = tp.build_full_mask(layout)
wk = torch.from_numpy((0.97 ** np.arange(K)).astype(np.float32))
rows_win = torch.from_numpy(layout.win_rows().reshape(-1)); rows_clean = torch.from_numpy(tp.clean_rows_for_windows(layout))
gen = tp.ParGenerator(model.model, model.lm_head, layout, cfg.num_hidden_layers, 2,
                      torch.device("cpu"), contextlib.nullcontext, lambda: None, topk_transfer=0,
                      slot_pos=SLOT_POS)
gargs = SimpleNamespace(top_k_thought=512, temp=1.0, freq_pen=1.0, min_think=8)
rng = np.random.default_rng(3)
conds = [f"{k}{m}" for m in MS for k in ("own", "oracle", "filler")]
res = {c: [] for c in conds}; agree = {m: [] for m in MS}
mix = Mixture("/home/ubuntu/ema-trainium/data/corpus", "val"); t0 = time.time(); nseq = 0
with torch.no_grad():
    for dom in mix.domains:
        for x, y in eval_batches(mix, dom, 1, T, NPER):
            xr = np.asarray(x[0], dtype=np.int64); nseq += 1
            vw, _ = tp.validity(layout, xr); ok = vw > 0
            if ok.sum() == 0: continue
            z, _ = gen.generate(torch.from_numpy(xr[None]), rng, gargs, 1, th.EOD)
            g = z[1]                                     # greedy row
            tgt = torch.from_numpy(np.stack([xr[int(n)+1:int(n)+1+K] for n in layout.anchors]).reshape(-1))
            def score(c):
                buf = tp.build_buffer(layout, xr, c, SPACE)
                lw, lc = tp.score_buffer(model.model, model.lm_head, buf, pos, mask, layout, wk, rows_win, rows_clean, tgt, contextlib.nullcontext)
                return (lw - lc)[ok]
            for m in MS:
                C = {k: np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64) for k in ("own", "oracle", "filler")}
                for a, n in enumerate(layout.anchors):
                    n = int(n)
                    C["own"][a, :m] = g[a, :m]
                    C["oracle"][a, :m] = xr[n+1:n+1+m]
                    C["filler"][a, :m] = SPACE
                for k in ("own", "oracle", "filler"): res[f"{k}{m}"].extend(score(C[k]).tolist())
                agree[m].extend([float(np.mean(g[a, :m] == xr[int(layout.anchors[a])+1:int(layout.anchors[a])+1+m])) for a in np.where(ok)[0]])
            print(f"{dom} seq {nseq} ({time.time()-t0:.0f}s) n={len(res['own2'])}: "
                  + " ".join(f"{c}={np.mean(res[c]):+.2f}" for c in conds), flush=True)
S = {c: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1)/np.sqrt(len(v))), n=len(v)) for c, v in res.items()}
S.update({f"agree{m}": dict(mean=float(np.mean(agree[m]))) for m in MS})
# per_anchor dump (2026-09-19 06:07 tick): `res[c]` and `agree[m]` are already
# per-anchor lists in a fixed anchor order shared by EVERY condition, so dumping
# them lets own-filler be compared PAIRED (POSTMORTEM G26) and lets the
# anchor-level heterogeneity of a thought's value be read at all. Adds nothing
# to the computation; the `summary` block is byte-for-byte what it always was.
json.dump(dict(ckpt=ck, slot_pos=SLOT_POS, summary=S,
               per_anchor={c: [float(v) for v in res[c]] for c in conds},
               per_anchor_agree={f"agree{m}": [float(v) for v in agree[m]] for m in MS}),
          open(out, "w"), indent=1)
print("=== draft-length probe (nats over silence; own = model's greedy, truncated) ===")
print(f"{'m':>3} {'own':>16} {'filler':>16} {'oracle':>16}  own-filler   token agreement")
for m in MS:
    o, f_, r = S[f"own{m}"], S[f"filler{m}"], S[f"oracle{m}"]
    print(f"{m:>3} {o['mean']:+8.2f}±{o['se']:.2f} {f_['mean']:+8.2f}±{f_['se']:.2f} {r['mean']:+8.2f}±{r['se']:.2f}  {o['mean']-f_['mean']:+8.2f}   {S[f'agree{m}']['mean']:.3f}")
