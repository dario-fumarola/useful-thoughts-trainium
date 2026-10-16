"""THE 'DOES THINKING BUY ANYTHING' METRIC, at inference, on held-out text.

For each held-out sequence and each anchor (every 64 tokens), the checkpoint
writes its OWN thought -- one temperature-1 sample (as in training) and one
greedy -- spliced exactly as in training; we score the next k=64 real tokens
with and without it. Nothing is selected, nothing is an oracle: this is what
a user of the model would get from letting it think.

  USE_TORCH_XLA=0 python probe_thinkmode.py <ckpt.pt> <out.json> [n_per_domain]

Reports per condition: discounted gain (training units, gamma .97) and the
flat change in per-token cross-entropy over the window (nats/token; negative
= thinking helps).
"""
import contextlib, json, os, sys, time
from types import SimpleNamespace
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_init import build_model, untie_lm_head
from data import Mixture, eval_batches
import thinking as th
import think_par as tp
ck, out = sys.argv[1], sys.argv[2]
NPER = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 12

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
# --pause N: CONTROL condition. Instead of letting the checkpoint write its own
# thought, splice the same content-free filler that `think_train.py --pause-len N`
# trains on ([SPACE]*N + [T_CLOSE]*(L-N)) into every slot. Same weights, same
# reader, same splice geometry -- the only thing removed is what the thought SAYS.
# Default 0 keeps the original own-thought behaviour exactly.
PAUSE = 0
if "--pause" in sys.argv:
    PAUSE = int(sys.argv[sys.argv.index("--pause") + 1])

# --bridge: baseline the thought against an explicit SILENCE buffer instead of the
# clean rows. 2026-09-29 10:07 headless tick: `BRIDGE` was READ at l.85/101/131 but
# never ASSIGNED, so EVERY invocation of this probe died with `NameError: BRIDGE`
# (the committed version at HEAD is broken too -- the last good thinkmode probes are
# the r33/r34 ones from 2026-09-20). Default False reproduces the pre-bridge
# behaviour exactly: the baseline stays `lc`, the clean rows, as it was for every
# published thinkmode number. The probe's own mask is built WITHOUT a bridge
# (`tp.build_full_mask(layout)`), which is what fixes the default at False.
BRIDGE = "--bridge" in sys.argv
T, K, L = 1024, 64, 16
SPACE = 220
model, cfg, _ = build_model(); untie_lm_head(model)
sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v
      for k, v in sd.items() if not k.startswith("mix.")}
model.load_state_dict(sd, strict=False); model = model.float().eval()
layout = tp.ParLayout(T, K, K, L, 64)
pos = tp.build_positions(layout, SLOT_POS); mask = tp.build_full_mask(layout)
w_disc = torch.from_numpy((0.97 ** np.arange(K)).astype(np.float32))
w_flat = torch.from_numpy(np.full(K, 1.0 / K, dtype=np.float32))
rows_win = torch.from_numpy(layout.win_rows().reshape(-1))
rows_clean = torch.from_numpy(tp.clean_rows_for_windows(layout))
nosync = lambda: None
gen = tp.ParGenerator(model.model, model.lm_head, layout, cfg.num_hidden_layers,
                      2, torch.device("cpu"), contextlib.nullcontext, nosync,
                      topk_transfer=0, slot_pos=SLOT_POS)
gargs = SimpleNamespace(top_k_thought=512, temp=1.0, freq_pen=1.0, min_think=8)
rng = np.random.default_rng(11)
res = {k: [] for k in ("disc_sample", "disc_greedy", "flat_sample", "flat_greedy",
                       "len_sample", "len_greedy", "disc_filler", "flat_filler")}
examples = []
mix = Mixture("/home/ubuntu/ema-trainium/data/corpus", "val")
t0 = time.time(); nseq = 0
with torch.no_grad():
    for dom in mix.domains:
        for x, y in eval_batches(mix, dom, 1, T, NPER):
            xr = np.asarray(x[0], dtype=np.int64); nseq += 1
            vw, _ = tp.validity(layout, xr); ok = vw > 0
            if ok.sum() == 0:
                continue
            if PAUSE > 0:
                z = np.full((2, layout.N, L), th.T_CLOSE, dtype=np.int64)
                z[:, :, :min(PAUSE, L - 1)] = SPACE
            else:
                z, _ = gen.generate(torch.from_numpy(xr[None]), rng, gargs, 1, th.EOD)
            tgt = torch.from_numpy(np.stack([xr[int(n)+1:int(n)+1+K]
                                             for n in layout.anchors]).reshape(-1))
            base = {}
            if BRIDGE:
                sil = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
                bufs = tp.build_buffer(layout, xr, sil, SPACE)
                for tag, wkx in (("disc", w_disc), ("flat", w_flat)):
                    lws, _ = tp.score_buffer(model.model, model.lm_head, bufs, pos, mask,
                                             layout, wkx, rows_win, rows_clean, tgt,
                                             contextlib.nullcontext)
                    base[tag] = lws
            fill = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64); fill[:, :10] = SPACE
            for name, row in (("sample", 0), ("greedy", 1), ("filler", -1)):
                c = fill.copy() if row == -1 else z[row].copy()
                buf = tp.build_buffer(layout, xr, c, SPACE)
                for tag, wk in (("disc", w_disc), ("flat", w_flat)):
                    lw, lc = tp.score_buffer(model.model, model.lm_head, buf, pos,
                                             mask, layout, wk, rows_win, rows_clean,
                                             tgt, contextlib.nullcontext)
                    d = (lw - (base[tag] if BRIDGE else lc))[ok]
                    res[f"{tag}_{name}"].extend((d if tag == "disc" else -d).tolist())
                if name != "filler":
                    res[f"len_{name}"].extend(tp.stop_pos(c, L)[ok].tolist())
            if len(examples) < 6:
                a = int(np.where(ok)[0][len(examples) % max(ok.sum(), 1)])
                n = int(layout.anchors[a])
                examples.append(dict(domain=dom, ctx=xr[max(0, n-40):n+1].tolist(),
                                     sample=z[0][a].tolist(), greedy=z[1][a].tolist(),
                                     window=xr[n+1:n+1+K].tolist()))
            print(f"{dom} seq {nseq} ({time.time()-t0:.0f}s): "
                  + " ".join(f"{k}={np.mean(v):+.3f}" for k, v in res.items()), flush=True)
S = {k: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1)/np.sqrt(len(v))),
             frac_pos=float(np.mean(np.array(v) > 0)), n=len(v)) for k, v in res.items()}
# per_anchor: the raw per-anchor lists, in traversal order. The traversal is
# deterministic (fixed domains, fixed eval_batches, fixed NPER) and the rng only
# draws thought tokens, so element i is the SAME anchor across two invocations of
# this script on the same ckpt+NPER -- which makes own-vs-`--pause` a PAIRED
# comparison. Without these lists the difference can only be combined unpaired,
# which is what forced the conservative +-0.148 on run 27 (G26). Additive key:
# nothing reads it yet, every existing key is unchanged.
json.dump(dict(ckpt=ck, pause=PAUSE, summary=S, per_anchor=res, examples=examples),
          open(out, "w"), indent=1)
print(f"=== think-mode at inference ({'PAUSE filler len %d' % PAUSE if PAUSE else 'own thought, no selection'}) ===")
print(f"  discounted gain  sample {S['disc_sample']['mean']:+.3f} ± {S['disc_sample']['se']:.3f}  "
      f"greedy {S['disc_greedy']['mean']:+.3f} ± {S['disc_greedy']['se']:.3f}   (nats/window, >0 = helps)")
print(f"  window CE change sample {S['flat_sample']['mean']:+.4f}  greedy {S['flat_greedy']['mean']:+.4f}   (nats/token, <0 = helps)")
print(f"  thought length   sample {S['len_sample']['mean']:.1f}  greedy {S['len_greedy']['mean']:.1f}")
print(f"  filler floor     {S['disc_filler']['mean']:+.3f} ± {S['disc_filler']['se']:.3f}   "
      f"own greedy MINUS filler {S['disc_greedy']['mean'] - S['disc_filler']['mean']:+.3f}"
      f"   (bridge={BRIDGE})")
