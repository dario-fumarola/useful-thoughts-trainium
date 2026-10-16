"""E0 (NEXT-STEPS.md sec. 3, as amended 2026-09-26): the COMPUTATION TERM, zero-update.

For every held-out anchor, draw R_MAX unselected thoughts z_i ~ q(.|c) and store the
per-token window log-probs under FOUR mixtures of the same size R, so every estimator
    C_R(cond) = log (1/R) sum_i exp(G_i)            (IWAE; -> log(m/p0) as R -> inf)
can be compared at MATCHED R:
    own    the model's own sampled thoughts z_i
    shuf   each z_i with its content tokens permuted       (order-borne part)
    cross  thoughts sampled for a DIFFERENT held-out sequence, same anchor slot
           (context relevance)
    fill   content-free filler of length |z_i|             (length-only diversity)
plus the single-thought references: greedy thought, filler of length 10, and the true
12-token continuation (reader-health oracle).  Why matched R: a single filler is ONE
pass, C_R(own) is an R-way mixture; log-mean-exp of noisy gains rises by ~sigma^2/2 with
no computation at all (runs 25/31: within-anchor sd 1.22/1.68 nats), so C_R(own) vs a
single filler cannot be read as "computation".  The amended decision rule is
C_R(own) - max(C_R(shuf), C_R(cross), C_R(fill)) > 0 at > 3 s.e.  (e0_analyze.py).

Everything needed for any estimator is saved raw (per-token logp, [seq, R, N, k]), so the
analysis can be redone without the model.  Traversal = probe_thinkmode.py's (data/corpus
val, NPER per domain, same order), so anchors pair with every earlier thinkmode file.

Speed: the clean prefix is identical across conditions, so it is prefilled ONCE per
sequence into a ScatterCache; each condition then runs only the N*(L+1+k) block+window
rows, B conditions per forward.  Verified against tp.score_buffer at start-up.

  USE_TORCH_XLA=0 python probe_e0.py <ckpt> <out.npz> --slot-pos {built|aligned}
      --min-think M [--nper 12] [--R 32] [--sampler pure|train] [--batch 8]
"""
import argparse, contextlib, json, os, sys, time
from types import SimpleNamespace
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_init import build_model, untie_lm_head
from data import Mixture, eval_batches
from think_gen import ScatterCache
import thinking as th
import think_par as tp

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--slot-pos", choices=("built", "aligned"), required=True)
ap.add_argument("--min-think", type=int, required=True)
ap.add_argument("--nper", type=int, default=12)
ap.add_argument("--R", type=int, default=32)
ap.add_argument("--sampler", choices=("pure", "train"), default="pure")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seed", type=int, default=20260926)
ap.add_argument("--max-seq", type=int, default=0, help="debug: stop after this many")
a = ap.parse_args()

T, K, L, SPACE = 1024, 64, 16, 220
torch.set_grad_enabled(False)
model, cfg, _ = build_model(); untie_lm_head(model)
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v
      for k, v in sd.items() if not k.startswith("mix.")}
missing = model.load_state_dict(sd, strict=False)
model = model.float().eval()
trunk, head = model.model, model.lm_head
lay = tp.ParLayout(T, K, K, L, 64)
N, S = lay.N, lay.S
pos = tp.build_positions(lay, a.slot_pos)                 # [1, S] float
mask = tp.build_full_mask(lay)                            # [1, 1, S, S]
win_local = torch.from_numpy(lay.win_rows().reshape(-1) - T)   # rows inside the tail
clean_rows = torch.from_numpy(tp.clean_rows_for_windows(lay))
pos_tail = pos[:, T:]
mask_tail = mask[:, :, T:, :]
mask_pre = mask[:, :, :T, :]

if a.sampler == "pure":      # the model's actual distribution: the identity holds for it
    gargs = SimpleNamespace(top_k_thought=0, temp=1.0, freq_pen=0.0,
                            min_think=a.min_think, repeat_ban=0)
else:                        # the training sampler of runs 18-39
    gargs = SimpleNamespace(top_k_thought=512, temp=1.0, freq_pen=1.0,
                            min_think=a.min_think, repeat_ban=1)
R = a.R
gen = tp.ParGenerator(trunk, head, lay, cfg.num_hidden_layers, R + 1,
                      torch.device("cpu"), contextlib.nullcontext, lambda: None,
                      topk_transfer=0, slot_pos=a.slot_pos)
cache_pre = ScatterCache(cfg.num_hidden_layers, S)
cache_b = ScatterCache(cfg.num_hidden_layers, S)
ar_pre = torch.arange(T)
ar_tail = torch.arange(T, S)


def prefill(xr):
    """clean rows once: returns window-target logp under the CLEAN rows [N, k]."""
    cache_pre.set_positions(ar_pre)
    h = trunk(input_ids=torch.from_numpy(xr[None]), position_ids=pos[:, :T],
              attention_mask=mask_pre, past_key_values=cache_pre, use_cache=True)[0]
    tgt = torch.from_numpy(np.stack([xr[int(n) + 1:int(n) + 1 + K] for n in lay.anchors]))
    lg = head(h[0, clean_rows]).float()
    lp = torch.log_softmax(lg, -1).gather(1, tgt.reshape(-1, 1)).view(N, K)
    return lp.numpy(), tgt


def score_tail(xr, contents, tgt):
    """contents: list of [N, L] int arrays. Returns [len(contents), N, k] window logp."""
    out = []
    for c0 in range(0, len(contents), a.batch):
        chunk = contents[c0:c0 + a.batch]; B = len(chunk)
        bufs = np.stack([tp.build_buffer(lay, xr, c, SPACE)[T:] for c in chunk])
        if cache_b.layers[0].keys is None or cache_b.layers[0].keys.shape[0] != B:
            for l_ in cache_b.layers:
                l_.keys = None; l_.values = None; l_.is_initialized = False
        cache_b.broadcast_from(cache_pre, B)
        cache_b.set_positions(ar_tail)
        h = trunk(input_ids=torch.from_numpy(bufs), position_ids=pos_tail.expand(B, -1),
                  attention_mask=mask_tail.expand(B, -1, -1, -1),
                  past_key_values=cache_b, use_cache=True)[0]            # [B, S-T, H]
        lg = head(h[:, win_local]).float()                               # [B, N*k, V]
        lp = torch.log_softmax(lg, -1).gather(
            2, tgt.reshape(1, -1, 1).expand(B, -1, 1)).view(B, N, K)
        out.append(lp.numpy())
    return np.concatenate(out, 0)


def filler(m):
    c = np.full(L, th.T_CLOSE, dtype=np.int64); c[:min(m, L)] = SPACE
    return c


# ---------------------------------------------------------------- sequences
mix = Mixture("/home/ubuntu/ema-trainium/data/corpus", "val")
seqs = []
for dom in mix.domains:
    for x, y in eval_batches(mix, dom, 1, T, a.nper):
        seqs.append((dom, np.asarray(x[0], dtype=np.int64)))
if a.max_seq:
    seqs = seqs[:a.max_seq]
nS = len(seqs)
print(f"{nS} sequences, N={N}, R={R}, sampler={a.sampler}, slot={a.slot_pos}, "
      f"min_think={a.min_think}, load missing={len(missing.missing_keys)}", flush=True)

# ---- self-check: tail scoring == full-buffer scoring (probe_thinkmode's scorer)
xr0 = seqs[0][1]
lpc, tgt0 = prefill(xr0)
c_chk = np.stack([filler(10)] * N)
mine = score_tail(xr0, [c_chk], tgt0)[0]
w_ones = torch.ones(K)
ref_w, ref_c = tp.score_buffer(trunk, head, tp.build_buffer(lay, xr0, c_chk, SPACE), pos,
                               mask, lay, w_ones, torch.from_numpy(lay.win_rows().reshape(-1)),
                               clean_rows, tgt0.reshape(-1), contextlib.nullcontext)
err_w = float(np.abs(mine.sum(-1) - ref_w).max()); err_c = float(np.abs(lpc.sum(-1) - ref_c).max())
print(f"self-check vs score_buffer: max |diff| window {err_w:.2e}  clean {err_c:.2e}", flush=True)
assert err_w < 5e-3 and err_c < 5e-3, "tail scorer disagrees with the reference scorer"

# ---- phase 1: generate every thought (cross needs another sequence's samples)
rng = np.random.default_rng(a.seed)
Z = np.zeros((nS, R + 1, N, L), dtype=np.int64)          # row R = greedy
VW = np.zeros((nS, N), dtype=bool)
t0 = time.time()
for s, (dom, xr) in enumerate(seqs):
    vw, _ = tp.validity(lay, xr); VW[s] = vw > 0
    z, _ = gen.generate(torch.from_numpy(xr[None]), rng, gargs, R, th.EOD)
    Z[s] = z
    if s % 6 == 0:
        print(f"gen {s+1}/{nS} ({time.time()-t0:.0f}s)", flush=True)

# ---- phase 2: score
keys = ("own", "shuf", "cross", "fill")
LP = {k_: np.zeros((nS, R, N, K), dtype=np.float32) for k_ in keys}
LP_clean = np.zeros((nS, N, K), dtype=np.float32)
LP_greedy = np.zeros((nS, N, K), dtype=np.float32)
LP_fill10 = np.zeros((nS, N, K), dtype=np.float32)
LP_oracle = np.zeros((nS, N, K), dtype=np.float32)
LENS = np.zeros((nS, R, N), dtype=np.int16)
for s, (dom, xr) in enumerate(seqs):
    ts = time.time()
    lpc, tgt = prefill(xr); LP_clean[s] = lpc
    own = [Z[s, r] for r in range(R)]
    shuf = []
    for r in range(R):
        c = own[r].copy(); s0 = tp.stop_pos(c, L)
        for an in range(N):
            if s0[an] > 1:
                c[an, :s0[an]] = rng.permutation(c[an, :s0[an]])
        shuf.append(c)
    cross = [Z[(s + 1) % nS, r] for r in range(R)]
    lens = np.stack([tp.stop_pos(own[r], L) for r in range(R)])     # [R, N]
    LENS[s] = lens
    ulen = sorted(set(int(v) for v in lens.reshape(-1)) | {10})
    orc = np.full((N, L), th.T_CLOSE, dtype=np.int64)
    for an, n in enumerate(lay.anchors):
        orc[an, :12] = xr[int(n) + 1:int(n) + 13]
    extra = [Z[s, R], orc] + [np.stack([filler(m)] * N) for m in ulen]
    res = score_tail(xr, own + shuf + cross + extra, tgt)
    LP["own"][s], LP["shuf"][s], LP["cross"][s] = res[:R], res[R:2 * R], res[2 * R:3 * R]
    LP_greedy[s], LP_oracle[s] = res[3 * R], res[3 * R + 1]
    byl = {m: res[3 * R + 2 + i] for i, m in enumerate(ulen)}
    LP_fill10[s] = byl[10]
    for r in range(R):
        for an in range(N):
            LP["fill"][s, r, an] = byl[int(lens[r, an])][an]
    g = lambda lp: ((lp - lpc).sum(-1))[..., VW[s]]
    print(f"seq {s+1}/{nS} {dom} {time.time()-ts:.0f}s | undiscounted G: own "
          f"{g(LP['own'][s]).mean():+.2f} fill10 {g(LP_fill10[s]).mean():+.2f} "
          f"greedy {g(LP_greedy[s]).mean():+.2f} oracle {g(LP_oracle[s]).mean():+.2f}",
          flush=True)
    if (s + 1) % 6 == 0 or s == nS - 1:       # checkpoint the partial result
        np.savez_compressed(a.out, clean=LP_clean[:s + 1], own=LP["own"][:s + 1],
                            shuf=LP["shuf"][:s + 1], cross=LP["cross"][:s + 1],
                            fill=LP["fill"][:s + 1], greedy=LP_greedy[:s + 1],
                            fill10=LP_fill10[:s + 1], oracle=LP_oracle[:s + 1],
                            lens=LENS[:s + 1], valid=VW[:s + 1], Z=Z[:s + 1],
                            meta=json.dumps(dict(vars(a), nseq_done=s + 1, nseq=nS,
                                                 domains=[d for d, _ in seqs])))
print("done", a.out, f"{(time.time()-t0)/3600:.2f}h")
