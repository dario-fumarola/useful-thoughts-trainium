"""CEILING PROBE (owner, 2026-10-06): how much could a CHEATING gate buy, as a function of the thinking configuration,
and how much of that do label-free signals recover? CPU, zero-update, per-position records for offline analysis.

For a checkpoint, on held-out sequences, at the top-N highest-entropy positions (H > TAU_MIN) the model writes G thoughts
(<= L tokens, Gumbel, temp 1, logit floor 12 -- exactly the training sampler). Each thought is scored by the SAME model on
the next K TRUE tokens (a causal branch after the thought: target row j re-reads x[n+j] with the thought in context and
predicts x[n+j+1]); the clean rows of the same buffer give the no-thought loss on the same span. Per anchor we record:
  H, n, domain, seq, clean CE on x[n+1..n+K], clean top-1 and its prob at n,
  per thought g: CE on the span with the thought, mean CE of the thought's own tokens (its likelihood), length,
                 whether token 0 is SPACE (the run-92+ silence decision), top-1 and its prob at the first target row.
Analysis (agent/ceiling_analysis.py) sweeps horizon k <= K, best-of-G, tau, positions-per-sequence, and the natural gates.
  USE_TORCH_XLA=0 python probe_ceiling.py <ckpt> <out.jsonl> [n_seq_per_set=12] [G=6] [N=32] [L=32] [K=16] [tau_min=2.0]
"""
import contextlib, json, os, sys, time, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "24")))
for name in ["torch_xla", "torch_xla.amp", "torch_xla.core", "torch_xla.core.xla_model", "torch_xla.runtime",
             "torch_xla.distributed", "torch_xla.distributed.xla_backend", "evaluate"]:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["torch_xla"].sync = lambda: None
sys.modules["evaluate"].evaluate = None
sys.modules["evaluate"].format_eval = None
sys.modules["torch_xla.runtime"].global_ordinal = lambda: 0
sys.modules["torch_xla.runtime"].world_size = lambda: 1
from model_init import build_model, untie_lm_head
import thinking as th
import think_entropy as te

ck, out = sys.argv[1], sys.argv[2]
NSEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 12
G = int(sys.argv[4]) if len(sys.argv) > 4 else 6
N = int(sys.argv[5]) if len(sys.argv) > 5 else 32
L = int(sys.argv[6]) if len(sys.argv) > 6 else 32
K = int(sys.argv[7]) if len(sys.argv) > 7 else 16
TAU_MIN = float(sys.argv[8]) if len(sys.argv) > 8 else 2.0
T, MINPOS = 1024, 32
T_OPEN, T_CLOSE, EOD, NEG, GATE = th.T_OPEN, th.T_CLOSE, th.EOD, th.NEG, 220

model, cfg, _ = build_model(); untie_lm_head(model)
sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items() if not k.startswith("mix.")}
model.load_state_dict(sd, strict=False); model = model.float().eval()
# gate heads (think_oracle.py checkpoints): the NATURAL gates trained against the oracle. Absent -> fields stay None.
HEADS = None
if "gate_pre.weight" in sd and "gate_post.weight" in sd:
    HEADS = dict(pre_w=sd["gate_pre.weight"].float()[0], pre_b=float(sd["gate_pre.bias"].float()[0]),
                 post_w=sd["gate_post.weight"].float()[0], post_b=float(sd["gate_post.bias"].float()[0]))
    print("GATE HEADS found in checkpoint: gate_pre / gate_post evaluated", flush=True)
dev = torch.device("cpu"); nc = contextlib.nullcontext
class PGen(te.ETGen):
    """plain-text positions for the sampler: <T> at n+1, Z[t] at n+2+t (ETGen shares <T>'s position with Z[0])."""
    def set_anchors(self, anchors):
        super().set_anchors(anchors)
        anc = anchors.astype(np.float32)
        for t in range(self.L):
            self.pos_dec[t].copy_(torch.from_numpy((anc + 1.0 + t)[None, :]))
gen = PGen(model.model, model.lm_head, T, N, L, cfg.num_hidden_layers, dev, nc, lambda: None)
args = types.SimpleNamespace(temp=1.0, logit_floor=12.0)

PB = (L + 2) + K                      # rows per anchor block: <T>, L thought slots, </T>, K branch rows (re-read x[n], then x[n+1..])
S = T + N * PB


def span_buffer(x, anchors, Z, lens):
    """clean sequence + per anchor [<T>, Z(L), </T>, K branch rows x[n..n+K-1]] in INSERTED-TEXT geometry with a RE-READ:
    the thought sits at positions n+1..n+k+1 (k tokens + </T>), then the branch re-reads x[n] at n+k+2 and continues
    x[n+1].. at n+k+3.., each branch row predicting the next true token. The re-read matches every trained model's
    geometry for the first target (runs 83-94 always re-read x[n] after the thought; the </T> row itself was never
    trained to predict text, and the base model's prior after </think> is a newline: scoring the </T> row read
    -19 nats/thought). The clean rows of the same buffer give the no-thought loss on the same span.
    (A first version re-used position ids n+1.. for both the thought and the continuation -- RoPE collisions.)"""
    buf = np.full(S, T_CLOSE, dtype=np.int64); pos = np.zeros(S, dtype=np.float32)
    m = np.full((S, S), NEG, dtype=np.float32); tgt = np.zeros(S, dtype=np.int64)
    buf[:T] = x; pos[:T] = np.arange(T); m[:T, :T] = np.triu(np.full((T, T), NEG, dtype=np.float32), k=1)
    tgt[:T - 1] = x[1:]
    for a in range(N):
        n = int(anchors[a]); B = T + a * PB; k = max(int(lens[a]), 1)
        rows = np.arange(B, B + L + 1)                                   # <T> + L slots
        buf[B] = T_OPEN; buf[B + 1:B + L + 1] = Z[a]
        pos[rows] = n + 1.0 + np.arange(L + 1, dtype=np.float32)        # <T> at n+1, Z[j] at n+2+j (plain inserted text)
        m[rows, :n + 1] = 0.0
        m[rows[:, None], rows[None, :]] = np.triu(np.full((L + 1, L + 1), NEG, dtype=np.float32), k=1)
        tgt[B:B + k] = Z[a, :k]; tgt[B + k] = T_CLOSE                  # thought rows predict the next thought token
        rc = B + L + 1; buf[rc] = T_CLOSE; pos[rc] = n + 2.0 + k
        m[rc, :n + 1] = 0.0; m[rc, B:B + k + 1] = 0.0; m[rc, rc] = 0.0
        branch = [rc]
        for j in range(K):
            rt = rc + 1 + j; buf[rt] = x[n + j]; pos[rt] = n + 3.0 + k + j; tgt[rt] = x[n + j + 1]
            m[rt, :n + 1] = 0.0; m[rt, B:B + k + 1] = 0.0
            for r_ in branch: m[rt, r_] = 0.0
            m[rt, rt] = 0.0; branch.append(rt)
    return buf, pos, m, tgt


def forward(buf, pos, m, tgt):
    with torch.no_grad():
        h = model.model(input_ids=torch.from_numpy(buf[None]), position_ids=torch.from_numpy(pos[None]),
                        attention_mask=torch.from_numpy(m[None, None]), use_cache=False)[0]
        lg = model.lm_head(h[0]); lp = torch.log_softmax(lg.float(), -1)
        ce = -lp.gather(1, torch.from_numpy(tgt)[:, None])[:, 0].numpy()
        top = lp.argmax(-1).numpy(); ptop = lp.max(-1).values.exp().numpy()
        ent = -(lp.exp() * lp).sum(-1).numpy()   # added 2026-10-08 04:07 tick, G173: the p_top1
        post_p = None                            # artefact control needs the full post-thought entropy
        if HEADS is not None:
            post_p = torch.sigmoid(h[0].float() @ HEADS["post_w"] + HEADS["post_b"]).numpy()
    return ce, top, ptop, post_p, ent


sets = {"finemath_unseen": np.memmap("/home/ubuntu/ema-trainium/data/corpus_fm_unseen22800/shards/finemath_unseen_val_000.bin",
                                     dtype=np.uint32, mode="r").reshape(-1, T + 1)[:NSEQ].astype(np.int64)}
for dom in ("fineweb-edu", "finemath", "dclm"):
    a = np.memmap(f"/home/ubuntu/ema-trainium/data/corpus/shards/{dom}_val_000.bin", dtype=np.uint32, mode="r")
    sets[dom + "_val"] = np.stack([a[i * (T + 1):(i + 1) * (T + 1)] for i in range(NSEQ)]).astype(np.int64)
torch.manual_seed(int(os.environ.get("PROBE_SEED", "7")))
fo = open(out, "w"); t0 = time.time(); n_rec = 0
with torch.no_grad():
    for name, blk in sets.items():
        for s in range(len(blk)):
            x = blk[s, :T].copy(); x[(x == T_OPEN) | (x == T_CLOSE)] = 220
            h0 = gen.prefill(torch.from_numpy(x[None])); H = gen.entropy(h0)
            pre_p = torch.sigmoid(h0[0].float() @ HEADS["pre_w"] + HEADS["pre_b"]).numpy() if HEADS is not None else None
            cand = np.zeros(T, bool); cand[MINPOS:T - 1 - K] = True; cand &= H > TAU_MIN
            for j in range(K + 1):                                       # no EOD inside the scored span
                cand[:T - j] &= x[j:] != EOD
            idx = np.where(cand)[0]
            if len(idx) > N:
                idx = idx[np.argsort(-H[idx])[:N]]
            idx = np.sort(idx); nv = len(idx)
            if nv == 0:
                continue
            anchors = np.full(N, MINPOS, dtype=np.int64); anchors[:nv] = idx
            gen.set_anchors(anchors)
            recs = [dict(set=name, seq=s, n=int(anchors[a]), H=float(H[anchors[a]]), ce_clean=None, clean_top1=None,
                         clean_p=None, clean_ent=None, ce_span=[], ll_thought=[], len=[], space=[], top1=[], p_top1=[], ent=[], first=[],
                         gate_pre=(float(pre_p[anchors[a]]) if pre_p is not None else None), gate_post=[],
                         next_ids=[int(t) for t in x[anchors[a] + 1:anchors[a] + 1 + K]], prev_ids=[int(t) for t in x[max(0, anchors[a] - 63):anchors[a] + 1]],
                         thought_ids=[]) for a in range(nv)]
            for g in range(G):
                Z, lens = gen.decode(args)
                buf, pos, m, tgt = span_buffer(x, anchors, Z, lens)
                ce, top, ptop, post_p, ent = forward(buf, pos, m, tgt)
                for a in range(nv):
                    n = int(anchors[a]); B = T + a * PB; rc = B + L + 1; k = max(int(lens[a]), 1)
                    r = recs[a]
                    if g == 0:
                        r["ce_clean"] = [float(v) for v in ce[n:n + K]]
                        r["clean_top1"] = int(top[n]); r["clean_p"] = float(ptop[n]); r["clean_ent"] = float(ent[n])
                    r["ce_span"].append([float(v) for v in ce[rc + 1:rc + 1 + K]])
                    r["ll_thought"].append(float(ce[B:B + k + 1].mean()))
                    r["len"].append(int(lens[a])); r["space"].append(bool(Z[a, 0] == GATE))
                    r["top1"].append(int(top[rc + 1])); r["p_top1"].append(float(ptop[rc + 1])); r["ent"].append(float(ent[rc + 1])); r["first"].append(int(Z[a, 0]))
                    r["gate_post"].append(float(post_p[rc]) if post_p is not None else None)
                    r["thought_ids"].append([int(t) for t in Z[a, :k]])
            for r in recs:
                fo.write(json.dumps(r) + "\n"); n_rec += 1
            fo.flush()
            print(f"{name} seq {s} ({time.time() - t0:.0f}s) anchors {nv} recs {n_rec}", flush=True)
fo.close()
print(f"done: {n_rec} anchor records -> {out} in {time.time() - t0:.0f}s", flush=True)
