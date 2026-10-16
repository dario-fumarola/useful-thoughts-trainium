"""Is the 11%-vs-44% gap EXPOSURE BIAS, or a slot trained not to forecast?

Established 2026-09-18: the writer's free-running greedy thought agrees with the
true continuation on 4% (run 25 @900) to 11% (run 28) of tokens, while the same
model's teacher-forced next-token argmax is right ~44%.  `think_par.py` takes a
raw argmax on the speculation row (no top-k, no freq-pen, no temperature), so the
gap is NOT decoding -- it is what argmax does inside the <T> slot.  Two causes
remain and they imply opposite next runs:

  exposure bias      the slot is fine; free-running drifts because it conditions
                     on its own wrong tokens.  Fix: scheduled sampling.
  a broken slot      the slot itself no longer forecasts, however good its prefix.
                     Fix: nothing about exposure bias will help.

This probe splits them with one forward pass per sequence.  Three accuracies on
the SAME checkpoint and the SAME anchors:

  ntp      clean rows n+i predicting x[n+1+i]           -- the ~0.44 reference,
                                                           re-measured here so it
                                                           is not borrowed
  tfslot   the <T> block filled with the TRUE continuation, row (a,i) predicting
           z_i = x[n+1+i]                               -- teacher-forced INSIDE
                                                           the slot
  own      the model's own free-running greedy thought  -- the 0.04-0.11 number

  tfslot ~ ntp    => the slot forecasts fine; the loss is exposure bias.
  tfslot ~ own    => the slot itself is the problem.

  USE_TORCH_XLA=0 OMP_NUM_THREADS=24 python probe_tfslot.py <ckpt.pt> <out.json> [n_per_domain] [slot_pos]

slot_pos (4th arg, default "built" = the behaviour of every probe run before
2026-09-18 22:07) MUST match the convention the checkpoint was TRAINED with --
`--slot-pos` in the run's config.sh.  Runs 13-29 are "built"; run 30 onward is
"aligned".  Probing an aligned-trained checkpoint with "built" positions (and a
"built" generator) measures it one position out of its own register and
understates `own`; that is the exact quantity `--slot-pos aligned` changes, so
the mismatch silently tests nothing.  Both the scoring pass and the generator
are switched together, as think_par.build_positions' docstring requires.
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
NPER = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SLOT_POS = (sys.argv[sys.argv.index("--slot-pos") + 1] if "--slot-pos" in sys.argv
            else sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("-")
            else "built")
assert SLOT_POS in ("built", "aligned"), SLOT_POS
# --nar (default OFF, so every probe run before 2026-09-19 reproduces byte for byte):
# add a fourth column measuring a NON-AUTOREGRESSIVE / parallel slot.  The block is
# filled with content-free filler instead of the true continuation, so block row
# (a,i) -- which sits at the same position and sees the same context as always --
# predicts z_i = x[n+1+i] WITHOUT ever seeing z_0..z_{i-1}, true or own.  It is the
# upper bound on any parallel decoding of this slot, and it is the condition in
# which error propagation is structurally impossible.
#   nar ~ tfslot  => self-conditioning is the whole loss; a parallel slot recovers it.
#   nar ~ own     => position i genuinely needs z_<i; parallel decoding is dead too.
# Alignment check, free: at i=0 the block row is <T> with an empty prefix in ALL
# THREE conditions, so nar[0] must equal tfslot[0] == own[0] exactly.
NAR = "--nar" in sys.argv
COLS = ("ntp", "tfslot", "own") + (("nar",) if NAR else ())
T, K, L, SPACE = 1024, 64, 16, 220
MS = [1, 2, 4, 8, 12, 16]

model, cfg, _ = build_model(); untie_lm_head(model)
sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v
      for k, v in sd.items() if not k.startswith("mix.")}
model.load_state_dict(sd, strict=False); model = model.float().eval()

layout = tp.ParLayout(T, K, K, L, 64)
pos = tp.build_positions(layout, SLOT_POS); mask = tp.build_full_mask(layout)
# row (a, i) of the block predicts z_i: <T> at t=0 predicts z_0, z_i predicts z_{i+1}
rows_blk = torch.from_numpy(layout.block_rows()[:, :L].reshape(-1).copy())
gen = tp.ParGenerator(model.model, model.lm_head, layout, cfg.num_hidden_layers, 2,
                      torch.device("cpu"), contextlib.nullcontext, lambda: None, topk_transfer=0,
                      slot_pos=SLOT_POS)
gargs = SimpleNamespace(top_k_thought=512, temp=1.0, freq_pen=1.0, min_think=8)
rng = np.random.default_rng(3)

hit = {k: np.zeros(L) for k in COLS}
cnt = np.zeros(L)
# 2026-09-19 06:07 tick: run 32 supervises the writer to ABSTAIN (emit filler)
# where the true token is not forecastable, whose pre-registered failure mode is
# "collapse to all-filler" = run 26, which scores -0.74 and buys nothing. These
# two counters make that visible directly instead of inferring it from own ~ 0.
# They only READ the already-generated greedy row; the `own` column is untouched.
own_sp = np.zeros(L)
own_st = np.zeros(L)
mix = Mixture("/home/ubuntu/ema-trainium/data/corpus", "val"); t0 = time.time(); nseq = 0
with torch.no_grad():
    for dom in mix.domains:
        for x, _y in eval_batches(mix, dom, 1, T, NPER):
            xr = np.asarray(x[0], dtype=np.int64); nseq += 1
            vw, _ = tp.validity(layout, xr); ok = np.where(vw > 0)[0]
            if ok.size == 0: continue
            true = np.stack([xr[int(n)+1:int(n)+1+L] for n in layout.anchors])   # [N, L]
            g, _ = gen.generate(torch.from_numpy(xr[None]), rng, gargs, 1, th.EOD)
            own = g[1]                                                          # greedy row [N, L]

            buf = tp.build_buffer(layout, xr, true, SPACE)                      # TRUE content in the slot
            bt = torch.from_numpy(buf[None])
            h = model.model(input_ids=bt, position_ids=pos, attention_mask=mask,
                            use_cache=False)[0]
            am_blk = model.lm_head(h[0, rows_blk]).float().argmax(-1).numpy().reshape(layout.N, L)
            rows_cln = torch.from_numpy(
                (layout.anchors[:, None] + np.arange(L)[None, :]).reshape(-1).copy())
            am_cln = model.lm_head(h[0, rows_cln]).float().argmax(-1).numpy().reshape(layout.N, L)

            if NAR:
                fill = np.full((layout.N, L), SPACE, dtype=np.int64)
                bn = torch.from_numpy(tp.build_buffer(layout, xr, fill, SPACE)[None])
                hn = model.model(input_ids=bn, position_ids=pos, attention_mask=mask,
                                 use_cache=False)[0]
                am_nar = model.lm_head(hn[0, rows_blk]).float().argmax(-1).numpy(
                         ).reshape(layout.N, L)
                hit["nar"] += (am_nar[ok] == true[ok]).sum(0)

            hit["ntp"]    += (am_cln[ok] == true[ok]).sum(0)
            hit["tfslot"] += (am_blk[ok] == true[ok]).sum(0)
            hit["own"]    += (own[ok]    == true[ok]).sum(0)
            own_sp        += (own[ok]    == SPACE).sum(0)
            own_st        += (own[ok]    == th.T_CLOSE).sum(0)
            cnt           += ok.size
            print(f"{dom} seq {nseq} ({time.time()-t0:.0f}s) n={int(cnt[0])}: "
                  + " ".join(f"{k}@1={hit[k][0]/cnt[0]:.3f} {k}@12={hit[k][:12].sum()/cnt[:12].sum():.3f}"
                             for k in COLS), flush=True)

S = {k: {f"acc{m}": float(hit[k][:m].sum() / cnt[:m].sum()) for m in MS} for k in hit}
for k in hit:
    S[k]["per_pos"] = (hit[k] / cnt).tolist()
S["n_anchors"] = int(cnt[0])
S["own_space_rate"] = (own_sp / cnt).tolist()
S["own_stop_rate"] = (own_st / cnt).tolist()
json.dump(dict(ckpt=ck, slot_pos=SLOT_POS, nar=NAR, summary=S), open(out, "w"), indent=1)
print(f"=== teacher-forced-in-slot probe (slot_pos={SLOT_POS}) ===")
print(f"{'prefix m':>9} " + " ".join(f"{k:>9}" for k in COLS))
for m in MS:
    print(f"{m:>9} " + " ".join(f"{S[k][f'acc{m}']:>9.3f}" for k in COLS))
print(f"n = {S['n_anchors']} anchors")
print("own filler rate per position: "
      + " ".join(f"{v:.2f}" for v in S["own_space_rate"]))
print("own </T> rate per position:   "
      + " ".join(f"{v:.2f}" for v in S["own_stop_rate"]))
