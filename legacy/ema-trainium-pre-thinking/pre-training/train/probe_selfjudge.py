"""Can the model tell its own good thought from its own bad one?

If it can, a keep/drop head turns a negative mean into a positive one for free:
about a fifth of unselected thoughts already beat silence, and an empty slot costs
nothing.  We take the model's OWN features -- the hidden state at the anchor, the
hidden state at the end of its thought, and the thought's own log-probability --
and fit a logistic probe to predict sign(gain) on held-out anchors, scoring AUC by
5-fold cross-validation.  No training of the model, no device.

  USE_TORCH_XLA=0 OMP_NUM_THREADS=20 python probe_selfjudge.py <ckpt> <out.json> [n_per_domain]"""
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
T, K, L, SPACE = 1024, 64, 16, 220
model, cfg, _ = build_model(); untie_lm_head(model)
sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items() if not k.startswith("mix.")}
model.load_state_dict(sd, strict=False); model = model.float().eval()
layout = tp.ParLayout(T, K, K, L, 64)
pos = tp.build_positions(layout); mask = tp.build_full_mask(layout)
wk = torch.from_numpy((0.97 ** np.arange(K)).astype(np.float32))
rows_win = torch.from_numpy(layout.win_rows().reshape(-1)); rows_clean = torch.from_numpy(tp.clean_rows_for_windows(layout))
gen = tp.ParGenerator(model.model, model.lm_head, layout, cfg.num_hidden_layers, 2,
                      torch.device("cpu"), contextlib.nullcontext, lambda: None, topk_transfer=0)
gargs = SimpleNamespace(top_k_thought=512, temp=1.0, freq_pen=1.0, min_think=8)
rng = np.random.default_rng(17)
X, Y, ent = [], [], []
mix = Mixture("/home/ubuntu/ema-trainium/data/corpus", "val"); t0 = time.time(); nseq = 0
blk = layout.block_rows()
with torch.no_grad():
    for dom in mix.domains:
        for x, y in eval_batches(mix, dom, 1, T, NPER):
            xr = np.asarray(x[0], dtype=np.int64); nseq += 1
            vw, _ = tp.validity(layout, xr); ok = np.where(vw > 0)[0]
            if len(ok) == 0: continue
            z, _ = gen.generate(torch.from_numpy(xr[None]), rng, gargs, 1, th.EOD)
            c = z[0]                                        # one unselected sample
            buf = tp.build_buffer(layout, xr, c, SPACE)
            tgt = torch.from_numpy(np.stack([xr[int(n)+1:int(n)+1+K] for n in layout.anchors]).reshape(-1))
            bt = torch.from_numpy(buf[None])
            h = model.model(input_ids=bt, position_ids=pos, attention_mask=mask, use_cache=False)[0][0]
            lw, lc = tp.score_buffer(model.model, model.lm_head, buf, pos, mask, layout, wk,
                                     rows_win, rows_clean, tgt, contextlib.nullcontext)
            gain = lw - lc
            stop = tp.stop_pos(c, L)
            hc = model.model(input_ids=torch.from_numpy(xr[None]), use_cache=False)[0][0]
            lgc = torch.log_softmax(model.lm_head(hc[layout.anchors]).float(), -1)
            H = -(lgc.exp() * lgc).sum(-1).numpy()          # the anchor's own predictive entropy
            for a in ok:
                n = int(layout.anchors[a]); s0 = max(1, int(stop[a]))
                f_anchor = h[n].numpy()                      # state where the thought was decided
                f_end = h[layout.block_row(a, min(s0, L))].numpy()   # state at the thought's end
                X.append(np.concatenate([f_anchor, f_end, [H[a], s0]]))
                Y.append(1.0 if gain[a] > 0 else 0.0); ent.append(H[a])
            print(f"{dom} seq {nseq} ({time.time()-t0:.0f}s) n={len(Y)} pos_rate={np.mean(Y):.3f}", flush=True)
X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y); ent = np.asarray(ent)
X = (X - X.mean(0)) / (X.std(0) + 1e-6)
def fit(Xtr, ytr, iters=300, lr=0.25, l2=1e-2):
    w = np.zeros(Xtr.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = Xtr.T @ (p - ytr) / len(ytr) + l2 * w
        w -= lr * g; b -= lr * (p - ytr).mean()
    return w, b
def auc(sc, lab):
    o = np.argsort(sc); r = np.empty(len(sc)); r[o] = np.arange(1, len(sc) + 1)
    n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
idx = np.arange(len(Y)); rng.shuffle(idx); folds = np.array_split(idx, 5)
aucs, aucs_ent = [], []
for i in range(5):
    te = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
    w, b = fit(X[tr], Y[tr]); aucs.append(auc(X[te] @ w + b, Y[te]))
    aucs_ent.append(auc(ent[te], Y[te]))
S = dict(n=len(Y), pos_rate=float(Y.mean()), auc_probe=float(np.mean(aucs)), auc_probe_se=float(np.std(aucs, ddof=1) / np.sqrt(5)),
         auc_entropy_only=float(np.mean(aucs_ent)),
         mean_gain_all=None)
json.dump(dict(ckpt=ck, summary=S), open(out, "w"), indent=1)
print("=== can the model judge its own thought? ===")
print(f"  anchors {S['n']}, fraction that beat silence {S['pos_rate']:.3f}")
print(f"  AUC, logistic probe on its own states : {S['auc_probe']:.3f} +- {S['auc_probe_se']:.3f}")
print(f"  AUC, anchor entropy alone             : {S['auc_entropy_only']:.3f}")
