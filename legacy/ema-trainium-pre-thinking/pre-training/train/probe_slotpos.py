"""Is the thought slot handicapped by an OFF-BY-ONE in its positions?

build_positions puts <T> at n+1 and thought token i at n+2+i.  So a thought that
IS the true continuation carries x[n+1+i] at position n+2+i: every token one step
further from the context than the text it stands for.  Inside the slot the model
must then predict x[n+1+t] from a row sitting at n+1+t rather than n+t -- a
geometry it never saw in pretraining, which would depress in-slot accuracy
(measured: 0.15-0.33 vs 0.44 on clean text) without any exposure bias at all.

This scores the SAME buffer twice, changing only the position vector:
  as-built   block (a,t) -> n+1+t          (what every run has trained with)
  shifted    block (a,t) -> n+t for t>=1   (thought token i at the position of
                                            the real token it copies)
and reports in-slot next-token accuracy and the reader's gain under each.
  USE_TORCH_XLA=0 OMP_NUM_THREADS=24 python probe_slotpos.py <ckpt> <out.json> [n_per_domain]"""
import contextlib, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_init import build_model, untie_lm_head
from data import Mixture, eval_batches
import think_par as tp
ck, out = sys.argv[1], sys.argv[2]
NPER = int(sys.argv[3]) if len(sys.argv) > 3 else 8
T, K, L, SPACE = 1024, 64, 16, 220
model, cfg, _ = build_model(); untie_lm_head(model)
sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items() if not k.startswith("mix.")}
model.load_state_dict(sd, strict=False); model = model.float().eval()
layout = tp.ParLayout(T, K, K, L, 64)
mask = tp.build_full_mask(layout)
pos_built = tp.build_positions(layout)
pos_shift = pos_built.clone()
for a in range(layout.N):                      # thought token i -> position of x[n+1+i]
    n = float(layout.anchors[a]); b0 = layout.block_row(a, 0)
    pos_shift[0, b0 + 1:b0 + layout.blk] = torch.from_numpy(
        (n + 1.0 + np.arange(layout.blk - 1)).astype(np.float32))
wk = torch.from_numpy((0.97 ** np.arange(K)).astype(np.float32))
rows_win = torch.from_numpy(layout.win_rows().reshape(-1)); rows_clean = torch.from_numpy(tp.clean_rows_for_windows(layout))
blk = layout.block_rows()[:, :L]
acc = {k: [] for k in ("built", "shifted")}; gain = {k: [] for k in ("built", "shifted")}; ntp = []
mix = Mixture("/home/ubuntu/ema-trainium/data/corpus", "val"); t0 = time.time(); nseq = 0
with torch.no_grad():
    for dom in mix.domains:
        for x, y in eval_batches(mix, dom, 1, T, NPER):
            xr = np.asarray(x[0], dtype=np.int64); nseq += 1
            vw, _ = tp.validity(layout, xr); ok = vw > 0
            if ok.sum() == 0: continue
            content = np.stack([xr[int(n)+1:int(n)+1+L] for n in layout.anchors]).astype(np.int64)
            buf = tp.build_buffer(layout, xr, content, SPACE)
            tgt_win = torch.from_numpy(np.stack([xr[int(n)+1:int(n)+1+K] for n in layout.anchors]).reshape(-1))
            bt = torch.from_numpy(buf[None])
            for name, P in (("built", pos_built), ("shifted", pos_shift)):
                h = model.model(input_ids=bt, position_ids=P, attention_mask=mask, use_cache=False)[0]
                lg = model.lm_head(h[0, torch.from_numpy(blk.reshape(-1))]).argmax(-1).numpy().reshape(layout.N, L)
                tg = content                                   # row (a,t) must predict content[t]
                acc[name].extend(np.mean(lg[ok, :L-1] == tg[ok, :L-1], axis=1).tolist())
                lw, lc = tp.score_buffer(model.model, model.lm_head, buf, P, mask, layout, wk, rows_win, rows_clean, tgt_win, contextlib.nullcontext)
                gain[name].extend((lw - lc)[ok].tolist())
            hc = model.model(input_ids=torch.from_numpy(xr[None]), use_cache=False)[0]
            pc = model.lm_head(hc[0]).argmax(-1).numpy()
            ntp.extend([float(np.mean(pc[int(n):int(n)+L-1] == xr[int(n)+1:int(n)+L])) for n in layout.anchors[ok]])
            print(f"{dom} seq {nseq} ({time.time()-t0:.0f}s) n={len(acc['built'])}: "
                  f"acc built {np.mean(acc['built']):.3f} shifted {np.mean(acc['shifted']):.3f} ntp {np.mean(ntp):.3f} | "
                  f"gain built {np.mean(gain['built']):+.2f} shifted {np.mean(gain['shifted']):+.2f}", flush=True)
S = {f"{k}_{n}": dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1)/np.sqrt(len(v))), n=len(v))
     for n, d in (("acc", acc), ("gain", gain)) for k, v in d.items()}
S["ntp_acc"] = dict(mean=float(np.mean(ntp)), se=float(np.std(ntp, ddof=1)/np.sqrt(len(ntp))))
json.dump(dict(ckpt=ck, summary=S), open(out, "w"), indent=1)
print("=== in-slot accuracy with the TRUE continuation teacher-forced ===")
print(f"  as-built positions (n+1+t) : acc {S['built_acc']['mean']:.3f} ± {S['built_acc']['se']:.3f}   reader gain {S['built_gain']['mean']:+.2f}")
print(f"  shifted positions  (n+t)   : acc {S['shifted_acc']['mean']:.3f} ± {S['shifted_acc']['se']:.3f}   reader gain {S['shifted_gain']['mean']:+.2f}")
print(f"  clean-text next-token acc  : {S['ntp_acc']['mean']:.3f}   (the ceiling both should match)")
