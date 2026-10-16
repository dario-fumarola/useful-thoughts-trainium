"""Held-out check for run 61 (the moving span). At span offset o the thought stands for x[n+o .. n+o+14], sits at
its own positions (aligned, pos_shift = o-1), the reader is denied max(0,-o) tokens before the anchor, the writer sees
up to x[n]. Four contents in the slot, same anchors, all scored against SILENCE UNDER THE SAME MASK (discounted
64-token log-lik gain, gamma 0.97, like training):
  own       the model's own greedy thought (closed in its last slot, as in training)
  shuf      another anchor's own thought (same sequence, rolled by one): what content-free "a thought" is worth
  span      the true span + </T> (the ceiling the writer is trained toward)
  filler    10 SPACE + </T>
  echo      THE L-1 TOKENS IMMEDIATELY BEFORE THE SPAN, VERBATIM + </T>  <-- the only addition in this file
Verdict: own - filler, paired, > 0 at > 3 s.e.  Data: the unseen-FineMath VAL blocks (never trained on).
  USE_TORCH_XLA=0 python probe_span.py <ckpt> <out.json> <o> [n_seq]
READER_CKPT=<ckpt> (env, run 70 --frozen-reader): the notes are WRITTEN by <ckpt> and SCORED by READER_CKPT.

WHY THIS FILE EXISTS (2026-10-02 08:07 tick, G117). Run 71's headline is own_minus_shuf ~ +4.5 at o=+1
against an untrained +0.95: the writer "says something specific about a span it has not seen". But the
step-1180 dumps show the note is a near-verbatim COPY OF THE TEXT IMMEDIATELY BEFORE THE WINDOW, which
is what --imit-w 1.0 teaches. `shuf` is the note for the anchor 64 tokens EARLIER, so it echoes text
further away -- and therefore PURE LOCALITY, with zero forecasting, already predicts own_minus_shuf > 0.
`echo` is that null hypothesis made into a control: the last L-1 tokens the writer can see, pasted in as
the note. If echo ~ own, the headline is locality and must be withdrawn. If own beats echo, the writer
is doing something a copy cannot.
probe_span.py IS NOT EDITED -- two live automations (the gate enforcer and the transfer curve) are
calling it right now, and G114(a) forbids changing an instrument under a running process."""
import contextlib, json, os, sys, time
from types import SimpleNamespace
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_init import build_model, untie_lm_head
import thinking as th
import think_par as tp
ck, out, O = sys.argv[1], sys.argv[2], int(sys.argv[3])
NSEQ = int(sys.argv[4]) if len(sys.argv) > 4 else 64
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "32")))
T, K, L, SPACE = 1024, 64, 16, 220
BR = max(0, -O)
model, cfg, _ = build_model(); untie_lm_head(model)
sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items() if not k.startswith("mix.")}
model.load_state_dict(sd, strict=False); model = model.float().eval()
rmodel = model
if os.environ.get("READER_CKPT"):
    rmodel, _, _ = build_model(); untie_lm_head(rmodel)
    sd = torch.load(os.environ["READER_CKPT"], map_location="cpu", weights_only=False); sd = sd.get("model", sd)
    sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items() if not k.startswith("mix.")}
    rmodel.load_state_dict(sd, strict=False); rmodel = rmodel.float().eval()
layout = tp.ParLayout(T, K, K, L, 64)
pos = tp.build_positions(layout, "aligned", 0, O - 1); mask = tp.build_full_mask(layout, BR, 0)
wk = torch.from_numpy((0.97 ** np.arange(K)).astype(np.float32))
rw = torch.from_numpy(layout.win_rows().reshape(-1)); rc = torch.from_numpy(tp.clean_rows_for_windows(layout))
gen = tp.ParGenerator(model.model, model.lm_head, layout, cfg.num_hidden_layers, 2,
                      torch.device("cpu"), contextlib.nullcontext, lambda: None, topk_transfer=0,
                      slot_pos="aligned")
gen.set_writer_delay(0, O - 1)
ga = SimpleNamespace(top_k_thought=512, temp=1.0, freq_pen=0.0, min_think=3, force_close=1, code_size=0)
rng = np.random.default_rng(61)
conds = ["own", "shuf", "span", "filler", "echo"]
res = {c: [] for c in conds}
n_eod_echo = 0   # how often the echo window straddles a document boundary
res_clean = []   # full-context (clean rows) log-lik minus silence under the mask = what the blinding costs
VAL = "/home/ubuntu/ema-trainium/data/corpus_fm_unseen22800/shards/finemath_unseen_val_000.bin"
blocks = np.memmap(VAL, dtype=np.uint32, mode="r").reshape(-1, T + 1)
t0 = time.time()
with torch.no_grad():
    for s in range(min(NSEQ, len(blocks))):
        xr = np.asarray(blocks[s, :T], dtype=np.int64)
        vw, _ = tp.validity(layout, xr)
        spans = np.stack([xr[int(n) + O:int(n) + O + L] for n in layout.anchors])
        ok = (vw > 0) & ~(spans == th.EOD).any(-1)
        if ok.sum() == 0:
            continue
        z, _ = gen.generate(torch.from_numpy(xr[None]), rng, ga, 1, th.EOD)
        own = np.asarray(z[1], dtype=np.int64)
        tgt = torch.from_numpy(np.stack([xr[int(n) + 1:int(n) + 1 + K] for n in layout.anchors]).reshape(-1))
        span = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64); span[:, :L - 1] = spans[:, :L - 1]
        fill = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64); fill[:, :10] = SPACE
        # echo: the L-1 tokens ending just BEFORE the span, verbatim. The span is xr[n+O .. n+O+L-1],
        # so these are xr[n+O-(L-1) .. n+O-1] -- all visible to the writer whenever O <= 1.
        echo = np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64)
        echo[:, :L - 1] = np.stack([xr[int(n) + O - (L - 1):int(n) + O] for n in layout.anchors])
        n_eod_echo += int((echo[:, :L - 1] == th.EOD).any(-1)[ok].sum())
        C = {"own": own, "shuf": np.roll(own, 1, axis=0), "span": span, "filler": fill, "echo": echo}

        def lp(c):
            buf = tp.build_buffer(layout, xr, c, SPACE)
            lw, _ = tp.score_buffer(rmodel.model, rmodel.lm_head, buf, pos, mask, layout, wk, rw, rc, tgt,
                                    contextlib.nullcontext)
            return np.asarray(lw)
        _buf0 = tp.build_buffer(layout, xr, np.full((layout.N, L), tp.T_CLOSE, dtype=np.int64), SPACE)
        _lw0, _lc0 = tp.score_buffer(rmodel.model, rmodel.lm_head, _buf0, pos, mask, layout, wk, rw, rc, tgt,
                                     contextlib.nullcontext)
        base = np.asarray(_lw0)
        res_clean.extend((np.asarray(_lc0) - base)[ok].tolist())
        for c in conds:
            res[c].extend((lp(C[c]) - base)[ok].tolist())
        print(f"o={O} seq {s} ({time.time()-t0:.0f}s) n={len(res['own'])}: "
              + " ".join(f"{c}={np.mean(res[c]):+.2f}" for c in conds), flush=True)
S = {c: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(len(v))), n=len(v)) for c, v in res.items()}
d = np.asarray(res["own"]) - np.asarray(res["filler"])
ds = np.asarray(res["own"]) - np.asarray(res["shuf"])
de = np.asarray(res["own"]) - np.asarray(res["echo"])
es = np.asarray(res["echo"]) - np.asarray(res["shuf"])
ef = np.asarray(res["echo"]) - np.asarray(res["filler"])
V = dict(own_minus_filler=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))),
         own_minus_shuf=float(ds.mean()), se_shuf=float(ds.std(ddof=1) / np.sqrt(len(ds))),
         own_minus_echo=float(de.mean()), se_echo=float(de.std(ddof=1) / np.sqrt(len(de))),
         echo_minus_shuf=float(es.mean()), se_echo_shuf=float(es.std(ddof=1) / np.sqrt(len(es))),
         echo_minus_filler=float(ef.mean()), se_echo_filler=float(ef.std(ddof=1) / np.sqrt(len(ef))),
         n_eod_echo=int(n_eod_echo))
V["verdict"] = "PASS" if V["own_minus_filler"] > 3 * V["se"] else "FAIL"
S["clean"] = dict(mean=float(np.mean(res_clean)), se=float(np.std(res_clean, ddof=1) / np.sqrt(len(res_clean))),
                  n=len(res_clean))
json.dump(dict(ckpt=ck, reader=os.environ.get("READER_CKPT", ck), offset=O, bridge=BR, summary=S, paired=V),
          open(out, "w"), indent=1)
print(f"=== span offset o={O}, reader denied {BR} (nats vs silence under the same mask) ===")
for c in conds + ["clean"]:
    print(f"  {c:<8} {S[c]['mean']:+7.2f} ± {S[c]['se']:.2f}  (n={S[c]['n']})")
print(f"  own-filler {V['own_minus_filler']:+.2f} ± {V['se']:.2f}   own-shuf {V['own_minus_shuf']:+.2f} ± "
      f"{V['se_shuf']:.2f}   -> {V['verdict']}")
print(f"  THE LOCALITY TEST: own-echo {V['own_minus_echo']:+.2f} ± {V['se_echo']:.2f}"
      f"   echo-shuf {V['echo_minus_shuf']:+.2f} ± {V['se_echo_shuf']:.2f}"
      f"   echo-filler {V['echo_minus_filler']:+.2f} ± {V['se_echo_filler']:.2f}"
      f"   (echo windows crossing EOD: {V['n_eod_echo']}/{S['own']['n']})")
