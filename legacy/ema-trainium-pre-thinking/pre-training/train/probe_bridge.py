"""How much is there to recover, and who recovers it?

Under --bridge B the window rows lose the last B context tokens. This probe puts
five contents in the slot on the SAME anchors and scores the window each time,
all against silence under the same mask:
  cost_of_hiding : unmasked silence minus masked silence -- what the mask took away
  oracle         : the true hidden span (a perfect carry)     -- the ceiling
  own            : the model's own greedy thought              -- what it does
  filler         : content-free tokens                         -- the floor
  copy_frac      : token overlap between own thought and the hidden span
  USE_TORCH_XLA=0 python probe_bridge.py <ckpt> <out.json> [n_per_domain] [bridge]"""
import contextlib, json, os, sys, time
from types import SimpleNamespace
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_init import build_model, untie_lm_head
from data import Mixture, eval_batches
import thinking as th
import think_par as tp
ck, out = sys.argv[1], sys.argv[2]
NPER = int(sys.argv[3]) if len(sys.argv) > 3 else 6
BRIDGES = [int(b) for b in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["32", "128"])]
T, K, L, SPACE = 1024, 64, 16, 220
model, cfg, _ = build_model(); untie_lm_head(model)
sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items() if not k.startswith("mix.")}
model.load_state_dict(sd, strict=False); model = model.float().eval()
layout = tp.ParLayout(T, K, K, L, 64)
pos = tp.build_positions(layout)
wk = torch.from_numpy((0.97 ** np.arange(K)).astype(np.float32))
rows_win = torch.from_numpy(layout.win_rows().reshape(-1)); rows_clean = torch.from_numpy(tp.clean_rows_for_windows(layout))
masks = {b: tp.build_full_mask(layout, b) for b in [0] + BRIDGES}
gen = tp.ParGenerator(model.model, model.lm_head, layout, cfg.num_hidden_layers, 2,
                      torch.device("cpu"), contextlib.nullcontext, lambda: None, topk_transfer=0)
gargs = SimpleNamespace(top_k_thought=512, temp=1.0, freq_pen=1.0, min_think=8)
rng = np.random.default_rng(23)
res = {b: {k: [] for k in ("cost", "oracle", "own", "filler", "copy")} for b in BRIDGES}
mix = Mixture("/home/ubuntu/ema-trainium/data/corpus", "val"); t0 = time.time(); nseq = 0
with torch.no_grad():
    for dom in mix.domains:
        for x, y in eval_batches(mix, dom, 1, T, NPER):
            xr = np.asarray(x[0], dtype=np.int64); nseq += 1
            vw, _ = tp.validity(layout, xr); ok = vw > 0
            if ok.sum() == 0: continue
            z, _ = gen.generate(torch.from_numpy(xr[None]), rng, gargs, 1, th.EOD)
            own = z[1]
            tgt = torch.from_numpy(np.stack([xr[int(n)+1:int(n)+1+K] for n in layout.anchors]).reshape(-1))
            sil = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
            fill = sil.copy(); fill[:, :10] = SPACE
            def lp(content, mask):
                buf = tp.build_buffer(layout, xr, content, SPACE)
                lw, _ = tp.score_buffer(model.model, model.lm_head, buf, pos, mask, layout, wk,
                                        rows_win, rows_clean, tgt, contextlib.nullcontext)
                return lw
            lp_unmasked = lp(sil, masks[0])
            for b in BRIDGES:
                base = lp(sil, masks[b])
                orc = np.stack([xr[int(n)-L:int(n)] for n in layout.anchors]).astype(np.int64)
                res[b]["cost"].extend((lp_unmasked - base)[ok].tolist())
                res[b]["oracle"].extend((lp(orc, masks[b]) - base)[ok].tolist())
                res[b]["own"].extend((lp(own, masks[b]) - base)[ok].tolist())
                res[b]["filler"].extend((lp(fill, masks[b]) - base)[ok].tolist())
                for a in np.where(ok)[0]:
                    n = int(layout.anchors[a]); s0 = max(1, int(tp.stop_pos(own, L)[a]))
                    hid = set(xr[n-b:n].tolist())
                    res[b]["copy"].append(float(np.mean([t in hid for t in own[a, :s0]])))
            print(f"{dom} seq {nseq} ({time.time()-t0:.0f}s) n={len(res[BRIDGES[0]]['own'])}: " +
                  " | ".join(f"B={b} cost {np.mean(res[b]['cost']):+.2f} oracle {np.mean(res[b]['oracle']):+.2f} "
                             f"own {np.mean(res[b]['own']):+.2f} filler {np.mean(res[b]['filler']):+.2f} copy {np.mean(res[b]['copy']):.2f}"
                             for b in BRIDGES), flush=True)
S = {f"b{b}_{k}": dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1)/np.sqrt(len(v))), n=len(v))
     for b in BRIDGES for k, v in res[b].items()}
json.dump(dict(ckpt=ck, summary=S), open(out, "w"), indent=1)
print("=== the bridge, decomposed (nats vs silence under the same mask) ===")
for b in BRIDGES:
    print(f"  hiding {b:>3} tokens costs {S[f'b{b}_cost']['mean']:+6.2f}"
          f" | a perfect carry recovers {S[f'b{b}_oracle']['mean']:+6.2f}"
          f" | its own thought {S[f'b{b}_own']['mean']:+6.2f} ± {S[f'b{b}_own']['se']:.2f}"
          f" | filler {S[f'b{b}_filler']['mean']:+6.2f}"
          f" | own tokens drawn from the hidden span {S[f'b{b}_copy']['mean']:.2f}")
