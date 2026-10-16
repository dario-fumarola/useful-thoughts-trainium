"""Why does a perfect carry score -21? Terminator, or geometry?
Six contents in the slot, same anchors, all scored against silence under the same mask:
  hidden_raw   the 16 hidden tokens, no stop token        (what probe_bridge used)
  hidden_term  12 hidden tokens + </T>                    (terminated, like a real thought)
  hidden_tail  the LAST 12 hidden tokens + </T>           (ends where the context ends)
  future_term  the 12 TRUE next tokens + </T>             (a forecast, the old oracle)
  own          the model's own greedy thought
  filler       10 filler tokens + </T>
  USE_TORCH_XLA=0 python probe_bridge2.py <ckpt> <out.json> [n] [bridge]"""
import contextlib, json, os, sys, time
from types import SimpleNamespace
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_init import build_model, untie_lm_head
from data import Mixture, eval_batches
import thinking as th
import think_par as tp
ck, out = sys.argv[1], sys.argv[2]
NPER = int(sys.argv[3]) if len(sys.argv) > 3 else 4
B = int(sys.argv[4]) if len(sys.argv) > 4 else 32
T, K, L, SPACE = 1024, 64, 16, 220
model, cfg, _ = build_model(); untie_lm_head(model)
sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items() if not k.startswith("mix.")}
model.load_state_dict(sd, strict=False); model = model.float().eval()
layout = tp.ParLayout(T, K, K, L, 64)
pos = tp.build_positions(layout); mask = tp.build_full_mask(layout, B)
wk = torch.from_numpy((0.97 ** np.arange(K)).astype(np.float32))
rw = torch.from_numpy(layout.win_rows().reshape(-1)); rc = torch.from_numpy(tp.clean_rows_for_windows(layout))
gen = tp.ParGenerator(model.model, model.lm_head, layout, cfg.num_hidden_layers, 2,
                      torch.device("cpu"), contextlib.nullcontext, lambda: None, topk_transfer=0)
ga = SimpleNamespace(top_k_thought=512, temp=1.0, freq_pen=1.0, min_think=8)
rng = np.random.default_rng(31)
conds = ["hidden_raw", "hidden_term", "hidden_tail", "future_term", "own", "filler"]
res = {c: [] for c in conds}
mix = Mixture("/home/ubuntu/ema-trainium/data/corpus", "val"); t0 = time.time(); nseq = 0
with torch.no_grad():
    for dom in mix.domains:
        for x, y in eval_batches(mix, dom, 1, T, NPER):
            xr = np.asarray(x[0], dtype=np.int64); nseq += 1
            vw, _ = tp.validity(layout, xr); ok = vw > 0
            if ok.sum() == 0: continue
            z, _ = gen.generate(torch.from_numpy(xr[None]), rng, ga, 1, th.EOD)
            tgt = torch.from_numpy(np.stack([xr[int(n)+1:int(n)+1+K] for n in layout.anchors]).reshape(-1))
            def mk(fill_fn, ln=12):
                c = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
                for a, n in enumerate(layout.anchors):
                    v = np.asarray(fill_fn(int(n)), dtype=np.int64)[:ln]
                    c[a, :len(v)] = v
                return c
            C = {"hidden_raw": np.stack([xr[int(n)-L:int(n)] for n in layout.anchors]).astype(np.int64),
                 "hidden_term": mk(lambda n: xr[n-B:n-B+12]),
                 "hidden_tail": mk(lambda n: xr[n-12:n]),
                 "future_term": mk(lambda n: xr[n+1:n+13]),
                 "own": z[1],
                 "filler": mk(lambda n: np.full(10, SPACE, dtype=np.int64), 10)}
            def lp(c):
                buf = tp.build_buffer(layout, xr, c, SPACE)
                lw, _ = tp.score_buffer(model.model, model.lm_head, buf, pos, mask, layout, wk, rw, rc, tgt, contextlib.nullcontext)
                return lw
            base = lp(np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64))
            for c in conds: res[c].extend((lp(C[c]) - base)[ok].tolist())
            print(f"{dom} seq {nseq} ({time.time()-t0:.0f}s) n={len(res['own'])}: "
                  + " ".join(f"{c}={np.mean(res[c]):+.2f}" for c in conds), flush=True)
S = {c: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1)/np.sqrt(len(v))), n=len(v)) for c, v in res.items()}
json.dump(dict(ckpt=ck, bridge=B, summary=S), open(out, "w"), indent=1)
print(f"=== what the slot accepts, bridge={B} (nats vs silence under the same mask) ===")
for c in conds: print(f"  {c:<12} {S[c]['mean']:+7.2f} ± {S[c]['se']:.2f}")
