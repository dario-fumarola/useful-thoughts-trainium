"""Is a RECALL thought worth anything to the reader?
recall  = the L-token span of distant context (ending >= 32 tokens before the
          anchor) sharing the most content tokens with the upcoming window.
          Learnable (it is in context) and potentially useful (moves distant
          evidence next to the prediction) -- thinking as working memory.
randctx = a random distant context span (same type, no relevance): control.
o16     = the true span 16..28 tokens ahead (the in-window sketch reference).
  USE_TORCH_XLA=0 python probe_recall.py <ckpt.pt> <out.json> [n_per_domain]"""
import contextlib, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_init import build_model, untie_lm_head
from data import Mixture, eval_batches
import think_par as tp
ck, out = sys.argv[1], sys.argv[2]
NPER = int(sys.argv[3]) if len(sys.argv) > 3 else 12
T, K, L, LEN, GAP = 1024, 64, 16, 12, 32
SPACE = 220
STOP = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stop_tokens_top300.npy"))
isstop = np.zeros(160000, dtype=bool); isstop[STOP] = True
model, _, _ = build_model(); untie_lm_head(model)
sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items() if not k.startswith("mix.")}
model.load_state_dict(sd, strict=False); model = model.float().eval()
layout = tp.ParLayout(T, K, K, L, 64)
pos = tp.build_positions(layout); mask = tp.build_full_mask(layout)
wk = torch.from_numpy((0.97 ** np.arange(K)).astype(np.float32))
rows_win = torch.from_numpy(layout.win_rows().reshape(-1)); rows_clean = torch.from_numpy(tp.clean_rows_for_windows(layout))
def recall_span(xr, n):
    W = np.unique(xr[n+1:n+1+K]); W = W[~isstop[W]]
    hi = n - GAP - LEN + 1
    if hi <= 0 or len(W) == 0: return None, 0
    m = np.isin(xr[:n - GAP], W) & ~isstop[xr[:n - GAP]]
    sc = np.convolve(m.astype(np.int32), np.ones(LEN, dtype=np.int32), "valid")[:hi]
    i = int(len(sc) - 1 - np.argmax(sc[::-1]))          # best, ties -> latest
    return i, int(sc[i])
conds = ["recall", "randctx", "o16", "empty"]; res = {c: [] for c in conds}; scores = []
rng = np.random.default_rng(5); mix = Mixture("/home/ubuntu/ema-trainium/data/corpus", "val"); t0 = time.time(); nseq = 0
with torch.no_grad():
    for dom in mix.domains:
        for x, y in eval_batches(mix, dom, 1, T, NPER):
            xr = np.asarray(x[0], dtype=np.int64); nseq += 1
            vw, _ = tp.validity(layout, xr)
            spans = [recall_span(xr, int(n)) for n in layout.anchors]
            ok = (vw > 0) & np.array([s[0] is not None and s[1] >= 2 for s in spans])
            if ok.sum() == 0: continue
            # an EOD inside the recalled span would cross a document: skip those anchors
            for a, (i, s) in enumerate(spans):
                if ok[a] and (xr[i:i+LEN] == tp.EOD).any(): ok[a] = False
            if ok.sum() == 0: continue
            tgt = torch.from_numpy(np.stack([xr[int(n)+1:int(n)+1+K] for n in layout.anchors]).reshape(-1))
            def score(c):
                buf = tp.build_buffer(layout, xr, c, SPACE)
                lw, lc = tp.score_buffer(model.model, model.lm_head, buf, pos, mask, layout, wk, rows_win, rows_clean, tgt, contextlib.nullcontext)
                return (lw - lc)[ok]
            C = {c: np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64) for c in conds}
            for a, n in enumerate(layout.anchors):
                n = int(n); i, s = spans[a]
                if i is not None: C["recall"][a, :LEN] = xr[i:i+LEN]
                j = int(rng.integers(0, max(1, n - GAP - LEN))); C["randctx"][a, :LEN] = xr[j:j+LEN]
                C["o16"][a, :LEN] = xr[n+1+16:n+1+16+LEN]
            for c in conds: res[c].extend(score(C[c]).tolist())
            scores.extend([spans[a][1] for a in np.where(ok)[0]])
            print(f"{dom} seq {nseq} ({time.time()-t0:.0f}s) n={len(res['recall'])}: " + " ".join(f"{c}={np.mean(res[c]):+.2f}" for c in conds), flush=True)
S = {c: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1)/np.sqrt(len(v))), win=float(np.mean(np.array(v) > 0)), n=len(v)) for c, v in res.items()}
json.dump(dict(ckpt=ck, summary=S, mean_overlap=float(np.mean(scores))), open(out, "w"), indent=1)
print("=== recall probe (nats over silence, discounted window) ===")
for c in conds: print(f"  {c:<8} {S[c]['mean']:+6.2f} ± {S[c]['se']:.2f}  win {S[c]['win']:.2f}  n={S[c]['n']}")
print(f"  mean content-token overlap of the recalled span: {np.mean(scores):.2f}")
