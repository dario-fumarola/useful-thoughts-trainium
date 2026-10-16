"""ENTROPY-TRIGGERED THINKING, kept only when it pays (Emanuele's design, 2026-10-03).

Continued pre-training of the 9B vanilla rung on the SAME 3.02B unseen tokens, in the SAME one-pass order, that the
vanilla run consumed between 9B and 12B -- so every vanilla ladder rung is an exact equal-data control.

At every position n the model predicts x[n+1]. Where its uncertainty (entropy of that prediction) exceeds --tau nats,
at most --n-think positions per sequence (the most uncertain ones) get a thought: the model writes up to --L tokens of
its own text after x[n]. The thought is checked every --chunk tokens by re-predicting the SAME true token x[n+1] with
the thought in context: stop and KEEP as soon as the loss on x[n+1] has fallen by --margin nats; stop and DISCARD as
soon as a check shows no further improvement; discard at the budget. A discarded thought leaves no trace. The loss is
the ordinary next-token loss on the real tokens, plus, on KEPT thoughts only, (a) the loss of producing the thought
(the model learns to write what worked) and (b) the loss on x[n+1] with the thought in context (it learns to use it).

Every check is done post hoc on one scoring forward over all --L/--chunk prefixes, which gives exactly the sequential
decisions. A shuffled copy of each thought is scored the same way every --ctrl-every micro-batches as the chance
control. Gradients are all-reduced. Sampling is on-device Gumbel-max (torch.topk is broken on this runtime).
Invoked as `think_train.py --etrig 1 ...`.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_xla
import torch_xla.amp
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

import train as vt
from data_onepass import Mixture as MixOP, MixtureBatcher as BatcherOP
from model_init import build_model, untie_lm_head, MODEL_ID
import thinking as th
import think_par as tp
from think_gen import ScatterCache

T_OPEN, T_CLOSE, EOD, NEG = th.T_OPEN, th.T_CLOSE, th.EOD, th.NEG


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--etrig", type=int, default=1)
    p.add_argument("--corpus", required=True)
    p.add_argument("--init-from", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--resume-model-only", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=46128)
    p.add_argument("--max-hours", type=float, default=None)
    # the one-pass order: first unseen block of the vanilla run at step 22800 = 256 * 22800
    p.add_argument("--block-offset", type=int, default=256 * 22800)
    p.add_argument("--perm-seed", type=int, default=1234)
    p.add_argument("--initial-tokens-seen", type=int, default=8988917760)
    # the thinking rule
    p.add_argument("--tau", type=float, default=4.0, help="think where next-token entropy > tau nats")
    p.add_argument("--n-think", type=int, default=16, help="at most this many thoughts per sequence (top entropies)")
    p.add_argument("--min-pos", type=int, default=32)
    p.add_argument("--L", type=int, default=64, help="max thought tokens")
    p.add_argument("--chunk", type=int, default=8, help="check every this many thought tokens")
    p.add_argument("--margin", type=float, default=0.5, help="keep when the loss on x[n+1] falls by this much")
    p.add_argument("--cont-metric", choices=("loss", "entropy"), default="loss",
                   help="what must keep improving between checks to continue")
    # CONTENT TEST (run 85, 2026-10-03). Runs 83/84 collapsed to digit-string thoughts: a run of digits before a numeric
    # position primes the reader toward digits and passes any loss margin (1.0, then 2.0) by FORMAT alone; imitation
    # then spreads the format everywhere. With contrast-margin > 0 a thought is kept only if, at the same prefix, it
    # also beats ITS OWN SHUFFLE by this many nats (shuffled digits are still digits -> 0 contrast -> not kept; a real
    # explanation loses its meaning when shuffled -> large contrast). Costs the shuffled scoring pass every micro.
    p.add_argument("--contrast-margin", type=float, default=0.0)
    # CONTROL (run 88, 2026-10-04): 1 = no thinking at all -- plain next-token training on the same data order and
    # batch, so the base-damage seen in runs 86/87 can be attributed to the thinking losses or to the batch (65k vs
    # the vanilla run's 262k tokens/step at the same learning rate).
    p.add_argument("--no-think", type=int, default=0)
    p.add_argument("--ctrl-every", type=int, default=4, help="score a shuffled copy every k micro-batches")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--logit-floor", type=float, default=12.0)
    p.add_argument("--imit-w", type=float, default=1.0)
    p.add_argument("--reader-w", type=float, default=1.0)
    p.add_argument("--reader-all", type=int, default=0)   # reader loss on EVERY valid thought, not only the kept ones
    p.add_argument("--ntp-w", type=float, default=1.0)
    # LOSS SCALING (bug found at step 47 of the first launch, 2026-10-03 19:0x). "own": the kept-thought terms are each
    # averaged over their own few tokens, so one kept thought weighs as much as the whole 1024-token NTP loss -- measured
    # grad norms on one micro-batch: NTP 3.3, reader 124, imitation 27 (38x), and with clip 1.0 the NTP update was scaled
    # to 0.008 of itself: the base could not learn. "tokens": every kept-thought token and kept target is one more token
    # in the per-token average (divided by T, weights --imit-w/--reader-w per token).
    p.add_argument("--thought-norm", choices=("own", "tokens"), default="own")
    # optimisation (vanilla run: constant 3e-5, wd 0.1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr-horizon", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--param-dtype", choices=("bf16", "fp32"), default="fp32")
    p.add_argument("--stock-adamw", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--loss-chunks", type=int, default=8)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--ckpt-every", type=int, default=760)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--metrics-file", default=None)
    p.add_argument("--dump-every", type=int, default=25)
    p.add_argument("--dump-file", default=None)
    args, unknown = p.parse_known_args()
    args.zero1 = False
    args.unknown = unknown
    assert args.L % args.chunk == 0
    return args


# ---------------------------------------------------------------------------------------------------------------
# geometry for the two spliced buffers (anchors vary per micro-batch; shapes do not)
# ---------------------------------------------------------------------------------------------------------------
class Geo:
    def __init__(self, T, N, L, chunk):
        self.T, self.N, self.L, self.chunk = T, N, L, chunk
        self.nc = L // chunk                       # checks per thought
        self.PA = (L + 1) + self.nc + self.nc      # score rows per anchor: block, closers, targets
        self.S_score = T + N * self.PA
        self.PG = (L + 2) + 1                      # grad rows per anchor: block [<T>, L tokens, </T>] + target
        self.S_grad = T + N * self.PG

    # ---- scoring buffer: all prefixes of every thought at once
    def score_buffer(self, x, anchors, Z, lens=None):
        T, N, L, nc, ch = self.T, self.N, self.L, self.nc, self.chunk
        if lens is None:
            lens = np.full(N, L, dtype=np.int64)
        S = self.S_score
        buf = np.full(S, T_CLOSE, dtype=np.int64)
        pos = np.zeros(S, dtype=np.float32)
        m = np.full((S, S), NEG, dtype=np.float32)
        buf[:T] = x
        pos[:T] = np.arange(T)
        m[:T, :T] = np.triu(np.full((T, T), NEG, dtype=np.float32), k=1)
        tgt = np.zeros(S, dtype=np.int64)
        tgt[:T - 1] = x[1:]
        slotpos = np.concatenate([[0.0], np.arange(L, dtype=np.float32)])          # <T> and z_i: n+1+max(i,0)
        for a in range(N):
            n = int(anchors[a])
            B = T + a * self.PA
            blk = slice(B, B + L + 1)
            buf[B] = T_OPEN
            buf[B + 1:B + L + 1] = Z[a]
            pos[blk] = n + 1.0 + slotpos
            rows = np.arange(B, B + L + 1)
            m[rows, :n + 1] = 0.0
            m[rows[:, None], rows[None, :]] = np.triu(np.full((L + 1, L + 1), NEG, dtype=np.float32), k=1)
            for c in range(nc):
                k = min(ch * (c + 1), max(int(lens[a]), 1))   # prefix, capped at the thought's own length
                rc = B + L + 1 + c                 # closer after prefix k
                buf[rc] = T_CLOSE
                pos[rc] = n + 1.0 + k
                m[rc, :n + 1] = 0.0
                m[rc, B:B + k + 1] = 0.0
                m[rc, rc] = 0.0
                rt = B + L + 1 + nc + c            # target row: token x[n] at position n predicts x[n+1]
                buf[rt] = x[n]
                pos[rt] = n
                tgt[rt] = x[n + 1]
                m[rt, :n] = 0.0
                m[rt, B:B + k + 1] = 0.0
                m[rt, rc] = 0.0
                m[rt, rt] = 0.0
        return buf, pos, m, tgt

    def target_rows(self):
        a = np.arange(self.N)[:, None]
        c = np.arange(self.nc)[None, :]
        return self.T + a * self.PA + (self.L + 1) + self.nc + c      # [N, nc]

    # ---- gradient buffer: the kept prefix of each thought, closed, plus its target row
    def grad_buffer(self, x, y, anchors, Z, keep_len, read_len=None):
        T, N, L = self.T, self.N, self.L
        S = self.S_grad
        buf = np.full(S, T_CLOSE, dtype=np.int64)
        pos = np.zeros(S, dtype=np.float32)
        m = np.full((S, S), NEG, dtype=np.float32)
        tgt = np.zeros(S, dtype=np.int64)
        w_imit = np.zeros(S, dtype=np.float32)
        w_read = np.zeros(S, dtype=np.float32)
        if read_len is None:
            read_len = keep_len
        buf[:T] = x
        pos[:T] = np.arange(T)
        tgt[:T] = y
        m[:T, :T] = np.triu(np.full((T, T), NEG, dtype=np.float32), k=1)
        slotpos = np.concatenate([[0.0], np.arange(L + 1, dtype=np.float32)])      # <T>, z_0..z_{L-1}, </T>
        for a in range(N):
            n = int(anchors[a]); La = int(keep_len[a]); Lr = int(read_len[a])
            Lb = max(La, Lr)                       # tokens actually written into the thought block
            G = T + a * self.PG
            rows = np.arange(G, G + L + 2)
            buf[G] = T_OPEN
            pos[rows] = n + 1.0 + slotpos
            m[rows, :n + 1] = 0.0
            if Lb > 0:
                buf[G + 1:G + 1 + Lb] = Z[a, :Lb]
                live = Lb + 2                      # <T>, Lb tokens, </T>
                r_live = rows[:live]
                m[r_live[:, None], r_live[None, :]] = np.triu(np.full((live, live), NEG, dtype=np.float32), k=1)
                m[rows[live:], rows[live:]] = 0.0  # padding rows attend clean + self; zero weight
                if La > 0:
                    tgt[G:G + La + 1] = buf[G + 1:G + La + 2]
                    w_imit[G:G + La + 1] = 1.0
            else:
                m[rows, rows] = 0.0
            rt = G + L + 2
            buf[rt] = x[n]
            pos[rt] = n
            tgt[rt] = x[n + 1]
            m[rt, :n] = 0.0
            m[rt, rt] = 0.0
            if Lr > 0:
                m[rt, G:G + Lb + 2] = 0.0
                w_read[rt] = 1.0
        return buf, pos, m, tgt, w_imit, w_read


# ---------------------------------------------------------------------------------------------------------------
# generator: prefill once, entropy at every position, then L decode launches for N thoughts in parallel
# ---------------------------------------------------------------------------------------------------------------
class ETGen:
    def __init__(self, trunk, lm_head, T, N, L, n_layers, device, autocast_ctx, sync_fn):
        self.trunk, self.lm_head = trunk, lm_head
        self.T, self.N, self.L, self.device = T, N, L, device
        self.autocast_ctx, self.sync = autocast_ctx, sync_fn
        self.C = T + L * N
        self.pre = ScatterCache(n_layers, self.C)
        self.dec = ScatterCache(n_layers, self.C)
        self.pos_base = torch.arange(T, dtype=torch.long, device=device)
        self.m_pre = tp.build_prefill_mask(SimpleNamespace(T=T, C=self.C)).to(torch.bfloat16).to(device)
        self.pos_pre = torch.arange(T, dtype=torch.float32, device=device)[None, :]
        self.m_dec = [torch.zeros((1, 1, N, self.C), dtype=torch.bfloat16, device=device) for _ in range(L)]
        self.pos_dec = [torch.zeros((1, N), dtype=torch.float32, device=device) for _ in range(L)]
        self.col_t = [torch.from_numpy(T + t * N + np.arange(N)).to(device) for t in range(L)]
        self.open_toks = torch.full((1, N), T_OPEN, dtype=torch.long, device=device)
        V = lm_head.weight.shape[0]
        ban = torch.zeros(V, dtype=torch.float32)
        ban[EOD:] = -1e9                           # plain text only: no EOD, no <T> inside a thought
        self.ban = ban.to(device)
        ban2 = ban.clone()
        ban2[T_CLOSE] = 0.0                        # ... but the model MAY end its thought with </T> once a chunk is written
        self.ban_stop = ban2.to(device)
        self.min_stop = 1

    def set_anchors(self, anchors):
        N, C, T = self.N, self.C, self.T
        base = np.full((N, C), NEG, dtype=np.float32)
        for a in range(N):
            base[a, :int(anchors[a]) + 1] = 0.0
        own = T + np.arange(self.L)[None, :] * N + np.arange(N)[:, None]     # [N, L] own cache columns
        anc = anchors.astype(np.float32)
        for t in range(self.L):
            m = base.copy()
            m[np.arange(N)[:, None], own[:, :t + 1]] = 0.0
            self.m_dec[t].copy_(torch.from_numpy(m[None, None]).to(torch.bfloat16))
            self.pos_dec[t].copy_(torch.from_numpy((anc + 1.0 + (0.0 if t == 0 else t - 1.0))[None, :]))

    def _eval(self):
        """transformers drops past_key_values under gradient checkpointing in train() mode (think_gen lesson,
        2026-09-02); Qwen3 has no dropout, so flipping to eval() for generation changes nothing else."""
        was = self.trunk.training
        self.trunk.eval()
        return was

    def prefill(self, x_dev):
        was = self._eval()
        try:
            with torch.no_grad():
                self.pre.set_positions(self.pos_base)
                with self.autocast_ctx():
                    h0 = self.trunk(input_ids=x_dev, position_ids=self.pos_pre, attention_mask=self.m_pre,
                                    past_key_values=self.pre, use_cache=True)[0]
        finally:
            if was:
                self.trunk.train()
        return h0

    def entropy(self, h0, chunk=128):
        """next-token entropy at every position; each chunk is executed on its own (a fused 16-chunk graph exceeded the
        compiler's instruction limit; a fused 4-chunk one allocated 311 MB of logits at once and OOM'd next to Adam)"""
        out = []
        with torch.no_grad():
            for i in range(0, self.T, chunk):
                with self.autocast_ctx():
                    lg = self.lm_head(h0[0, i:i + chunk])
                lp = torch.log_softmax(lg.float(), -1)
                out.append((-(lp.exp() * lp).sum(-1)).cpu().numpy())
                self.sync()
        return np.concatenate(out)

    def decode(self, args):
        was = self._eval()
        try:
            return self._decode(args)
        finally:
            if was:
                self.trunk.train()

    def _decode(self, args):
        """Returns (z [N, L], lens [N]): lens = tokens written before the model's own </T> (L if it never stopped).
        Positions after a stop hold </T>. Without a stop action (first launch, 2026-10-03) the writer, trained by
        imitation on ~8-11-token kept prefixes + </T>, was forced to continue for 64 tokens and produced garbage tails."""
        N, L = self.N, self.L
        z = np.zeros((N, L), dtype=np.int64)
        stopped = np.zeros(N, dtype=bool)
        lens = np.full(N, L, dtype=np.int64)
        with torch.no_grad():
            self.dec.broadcast_from(self.pre, 1)
            self.sync()
            for t in range(L):
                tok_t = self.open_toks if t == 0 else torch.from_numpy(z[None, :, t - 1].copy()).to(self.device)
                self.dec.set_positions(self.col_t[t])
                with self.autocast_ctx():
                    ht = self.trunk(input_ids=tok_t, position_ids=self.pos_dec[t], attention_mask=self.m_dec[t],
                                    past_key_values=self.dec, use_cache=True)[0]
                    lg = self.lm_head(ht)
                lg = lg.float() + (self.ban_stop if t >= self.min_stop else self.ban)
                if args.logit_floor > 0:
                    mx = lg.max(-1, keepdim=True).values
                    lg = torch.where(lg < mx - args.logit_floor, torch.full_like(lg, -1e9), lg)
                lg = lg / args.temp
                u = torch.rand(lg.shape, device=self.device).clamp(1e-10, 1.0 - 1e-6)
                samp = torch.argmax(lg - torch.log(-torch.log(u)), -1)
                tok = samp.cpu().numpy()[0]
                tok = np.where(stopped, T_CLOSE, tok)
                newly = (~stopped) & (tok == T_CLOSE)
                lens[newly] = t
                stopped |= newly
                z[:, t] = tok
                self.sync()
        return z, lens


def decide(base, l, ch, margin, La=None):
    """base: clean loss on x[n+1]; l: [nc] loss with prefix min(ch*(c+1), La). Sequential rule, applied post hoc:
    keep at the first check whose fall reaches the margin (kept length = that prefix, capped at the thought's own
    length La); discard at the first check with no further fall, or at the thought's end / the budget.
    Returns (kept length, index of the last check performed)."""
    L_ = len(l) * ch
    La = L_ if La is None else max(int(La), 1)
    nc_eff = min(len(l), (La + ch - 1) // ch)
    prev = base
    for c in range(nc_eff):
        if base - l[c] >= margin:
            return min(ch * (c + 1), La), c
        if l[c] >= prev:
            return 0, c
        prev = l[c]
    return 0, nc_eff - 1


def decide_contrast(base, l, ls, ch, margin, cmargin, La=None):
    """Like decide(), plus the content test: keep at the first check where base - l[c] >= margin AND ls[c] - l[c] >=
    cmargin (ls = loss with the thought's own tokens shuffled, same prefix). Stop/discard when the loss stops falling."""
    L_ = len(l) * ch
    La = L_ if La is None else max(int(La), 1)
    nc_eff = min(len(l), (La + ch - 1) // ch)
    prev = base
    for c in range(nc_eff):
        if base - l[c] >= margin and ls[c] - l[c] >= cmargin:
            return min(ch * (c + 1), La), c
        if l[c] >= prev:
            return 0, c
        prev = l[c]
    return 0, nc_eff - 1


def rep_share(z, n):
    """share of tokens equal to the most common token in the first n thought tokens (1.0 = constant)"""
    if n <= 0:
        return 0.0
    return float(np.unique(z[:n], return_counts=True)[1].max() / n)


def main():
    args = parse()
    torch.manual_seed(args.seed)
    device = torch_xla.device()
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank, world = xr.global_ordinal(), xr.world_size()
    xm.set_rng_state(int(args.seed) * 1000 + int(rank), device=device)
    T, N, L, ch = args.seq_len, args.n_think, args.L, args.chunk
    geo = Geo(T, N, L, ch)
    vt.log("=" * 72)
    vt.log(f"ENTROPY-TRIGGERED THINKING | world={world} T={T} tau={args.tau} n_think={N} L={L} chunk={ch} "
           f"margin={args.margin} cont={args.cont_metric} | S_score={geo.S_score} S_grad={geo.S_grad} "
           f"| one-pass offset block {args.block_offset:,} perm seed {args.perm_seed}")
    if args.unknown:
        vt.log(f"ignored flags: {' '.join(args.unknown)}")
    vt.log("=" * 72)

    _pdt = torch.float32 if args.param_dtype == "fp32" else torch.bfloat16
    backbone, cfg, _ = build_model(MODEL_ID, seq_len=T, seed=args.seed, dtype=_pdt)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        untie_lm_head(backbone)
        sd = ck["model"]
        if any(kk.startswith("backbone.") for kk in sd):
            sd = {kk[len("backbone."):]: v for kk, v in sd.items() if kk.startswith("backbone.")}
        backbone.load_state_dict(sd, strict=True)
        vt.log(f"backbone <- {args.init_from}")
        del ck, sd
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": True, "preserve_rng_state": False})
    model = backbone.to(device)
    model.train()
    for p_ in model.parameters():
        if p_.requires_grad:
            p_.grad = torch.zeros_like(p_)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    trunk, lm_head = backbone.model, backbone.lm_head
    lr_tensor = torch.tensor(args.lr, dtype=torch.float32, device=device)
    lr_host = torch.empty((), dtype=torch.float32)
    opt, _, _ = vt.build_optimizer(model, args, lr_tensor)
    autocast_ctx = lambda: torch_xla.amp.autocast(device, dtype=torch.bfloat16)

    # ---- the one-pass data order of the vanilla run, from its first unseen block
    mix = MixOP(args.corpus, "train")
    bat = BatcherOP(mix, 1, T, seed=args.perm_seed, rank=0, world_size=1, without_replacement=True, grad_accum=1)
    perm, dcum = bat._permutation, bat._domain_cumulative
    shard_cum = bat._shard_cumulative
    total_blocks = bat.total_blocks
    vt.log(f"one-pass: {total_blocks:,} blocks; offset {args.block_offset:,}; "
           f"{(total_blocks - args.block_offset) // (world * args.grad_accum):,} full steps available")

    def one_pass_row(gidx):
        gb = int(perm[gidx])
        di = int(np.searchsorted(dcum, gb, side="right"))
        before = int(dcum[di - 1]) if di else 0
        dom = mix.domains[di]
        return mix.data[dom].block(gb - before, T, cumulative=shard_cum[di]), di

    gen = ETGen(trunk, lm_head, T, N, L, cfg.num_hidden_layers, device, autocast_ctx, torch_xla.sync)
    trows_np = geo.target_rows().reshape(-1)
    trows_d = torch.from_numpy(trows_np).to(device)
    n1_dev = torch.tensor(float(T), device=device)
    tok = None
    if vt.is_master() and args.dump_file:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)

    start_step, tokens_seen = 0, args.initial_tokens_seen
    if args.resume:
        start_step, tokens_seen = vt.load_checkpoint(model, opt, args, device, rank, world)
    metrics_f = open(args.metrics_file, "a") if (args.metrics_file and vt.is_master()) else None
    dump_f = open(args.dump_file, "a") if (args.dump_file and vt.is_master()) else None

    def score_forward(buf, pos, m, tgt):
        """no-grad: per-row CE over the spliced buffer; returns ce [S] (float64 np) and target-row entropies [N*nc]"""
        with torch.no_grad():
            bt = torch.from_numpy(buf[None]).to(device)
            pt = torch.from_numpy(pos[None]).to(device)
            mt = torch.from_numpy(m[None, None]).to(torch.bfloat16).to(device)
            tt = torch.from_numpy(tgt).to(device)
            with autocast_ctx():
                h = trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
            ces = []
            Hs = None
            step_ = (h.shape[1] + 7) // 8
            for i in range(0, h.shape[1], step_):
                with autocast_ctx():
                    lg = lm_head(h[0, i:i + step_])
                lp = torch.log_softmax(lg.float(), -1)
                ces.append(-lp.gather(1, tt[i:i + step_, None])[:, 0])
            ce = torch.cat(ces)
            with autocast_ctx():
                lgt = lm_head(h[0, trows_d])
            lpt = torch.log_softmax(lgt.float(), -1)
            Ht = -(lpt.exp() * lpt).sum(-1)
            ce_np, Ht_np = ce.cpu().numpy().astype(np.float64), Ht.cpu().numpy().astype(np.float64)
        torch_xla.sync()
        return ce_np, Ht_np

    t_start = time.time()
    micro_counter = 0
    for step in range(start_step, args.max_steps):
        lr = vt.lr_at(step, args)
        lr_host.fill_(lr)
        lr_tensor.copy_(lr_host)
        t0 = time.time()
        M = dict(contrast_kept=0.0, digit_share=0.0, len_gen=0.0, open_share=0.0, n_micro=0.0, ntp1=0.0, n_cand=0.0, n_valid=0.0, n_kept=0.0, H_anc=0.0, base_anc=0.0,
                 drop_full=0.0, drop_best=0.0, drop_kept=0.0, len_kept=0.0, rep_kept=0.0, imit_ce=0.0,
                 read_ce=0.0, n_ctrl=0.0, ctrl_kept=0.0, ctrl_drop_full=0.0, n_read=0.0, n_rce=0.0, n_ice=0.0, n_dom=np.zeros(3))
        for micro in range(args.grad_accum):
            gidx = args.block_offset + (step * args.grad_accum + micro) * world + rank
            row, di = one_pass_row(gidx)
            x, y = row[:T].copy(), row[1:T + 1].copy()
            for arr in (x, y):
                arr[(arr == T_OPEN) | (arr == T_CLOSE)] = 220
            x_dev = torch.from_numpy(x[None]).to(device)
            rng = np.random.default_rng([args.seed, rank, step, micro, 23])
            if args.no_think:
                y_dev = torch.from_numpy(y[None]).to(device)
                with autocast_ctx():
                    h = trunk(input_ids=x_dev, use_cache=False)[0]
                ces = []
                step_ = (T + args.loss_chunks - 1) // args.loss_chunks
                for i in range(0, T, step_):
                    with autocast_ctx():
                        lg = lm_head(h[0, i:i + step_])
                    ces.append(torch.nn.functional.cross_entropy(lg.float(), y_dev[0, i:i + step_], reduction="none"))
                ce_all = torch.cat(ces)
                (ce_all.mean() * args.ntp_w / args.grad_accum).backward()
                M["n_micro"] += 1.0
                M["ntp1"] += float(ce_all.mean().cpu())
                M["n_dom"][di] += 1
                torch_xla.sync()
                continue

            # ---- trigger
            h0 = gen.prefill(x_dev)
            H = gen.entropy(h0)
            del h0
            cand = np.zeros(T, dtype=bool)
            cand[args.min_pos:T - 1] = True
            cand &= (H > args.tau)
            cand[:-1] &= (x[1:] != EOD)
            cand &= (x != EOD)
            idx = np.where(cand)[0]
            n_cand = len(idx)
            if n_cand > N:
                idx = idx[np.argsort(-H[idx])[:N]]
            valid = np.zeros(N, dtype=bool)
            anchors = np.full(N, args.min_pos, dtype=np.int64)
            idx = np.sort(idx)
            anchors[:len(idx)] = idx
            valid[:len(idx)] = True

            # ---- think
            gen.set_anchors(anchors)
            Z, lens = gen.decode(args)                                      # [N, L], [N]

            # ---- check every chunk (post hoc), decide keep / discard
            buf, pos, m, tgt = geo.score_buffer(x, anchors, Z, lens)
            ce, Ht = score_forward(buf, pos, m, tgt)
            lt = ce[trows_np].reshape(N, geo.nc)                            # loss on x[n+1] after each prefix
            Ht = Ht.reshape(N, geo.nc)
            base = ce[anchors]                                              # clean loss on x[n+1] at the anchor
            Hbase = H[anchors]
            lts = None
            if args.contrast_margin > 0 or (args.ctrl_every > 0 and micro_counter % args.ctrl_every == 0):
                Zs = Z.copy()
                for a in range(N):
                    la = max(int(lens[a]), 1)
                    Zs[a, :la] = rng.permutation(Z[a, :la])
                bufs, poss, ms, tgts = geo.score_buffer(x, anchors, Zs, lens)
                ces, _ = score_forward(bufs, poss, ms, tgts)
                lts = ces[trows_np].reshape(N, geo.nc)
            keep_len = np.zeros(N, dtype=np.int64)
            for a in range(N):
                if not valid[a]:
                    continue
                if args.contrast_margin > 0:
                    keep_len[a], _ = decide_contrast(base[a], lt[a], lts[a], ch, args.margin, args.contrast_margin, lens[a])
                elif args.cont_metric == "loss":
                    keep_len[a], _ = decide(base[a], lt[a], ch, args.margin, lens[a])
                else:
                    # continue while the entropy of the prediction keeps falling; keep on the loss margin
                    prev = Hbase[a]
                    nce = min(geo.nc, (max(int(lens[a]), 1) + ch - 1) // ch)
                    for c in range(nce):
                        if base[a] - lt[a, c] >= args.margin:
                            keep_len[a] = min(ch * (c + 1), max(int(lens[a]), 1)); break
                        if Ht[a, c] >= prev:
                            break
                        prev = Ht[a, c]
            kept = valid & (keep_len > 0)

            # ---- chance control: how often the SHUFFLED thought would pass the plain loss margin
            if lts is not None:
                ck_ = np.array([decide(base[a], lts[a], ch, args.margin, lens[a])[0] > 0 if valid[a] else False
                                for a in range(N)])
                M["n_ctrl"] += float(valid.sum())
                M["ctrl_kept"] += float(ck_.sum())
                M["ctrl_drop_full"] += float((base - lts[:, -1])[valid].sum())
            micro_counter += 1

            # ---- the gradient pass: clean NTP everywhere + kept thoughts (imitation) + kept targets (reader)
            read_len = keep_len
            if args.reader_all:
                # the reader is trained on EVERY valid thought at its FULL generated length -- exactly what the
                # label-free deploy probe conditions on. keep_len then only drives imitation and the diagnostics.
                read_len = np.where(valid, np.clip(lens.astype(np.int64), 1, geo.L), 0).astype(np.int64)
            M["n_read"] += float((read_len > 0).sum())
            gbuf, gpos, gm, gtgt, w_imit, w_read = geo.grad_buffer(x, y, anchors, Z, keep_len, read_len)
            w1 = np.zeros(geo.S_grad, dtype=np.float32); w1[:T] = args.ntp_w
            if args.thought_norm == "tokens":
                n2 = n3 = float(T)
            else:
                n2 = max(float(w_read.sum()), 1.0)
                n3 = max(float(w_imit.sum()), 1.0)
            bt = torch.from_numpy(gbuf[None]).to(device)
            pt = torch.from_numpy(gpos[None]).to(device)
            mt = torch.from_numpy(gm[None, None]).to(torch.bfloat16).to(device)
            tt = torch.from_numpy(gtgt[None]).to(device)
            with autocast_ctx():
                h = trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
            loss, ce_det = tp.weighted_ce3(
                h, lm_head, tt,
                torch.from_numpy(w1[None]).to(device),
                torch.from_numpy((args.reader_w * w_read)[None]).to(device),
                torch.from_numpy((args.imit_w * w_imit)[None]).to(device),
                n1_dev, torch.tensor(n2, device=device), torch.tensor(n3, device=device),
                args.loss_chunks, autocast_ctx)
            (loss / args.grad_accum).backward()
            ce_g = ce_det.cpu().numpy()
            torch_xla.sync()

            # ---- stats for this micro
            M["n_micro"] += 1.0
            M["ntp1"] += float(ce_g[:T].mean())
            M["n_cand"] += float(n_cand)
            M["n_valid"] += float(valid.sum())
            M["n_kept"] += float(kept.sum())
            M["n_dom"][di] += 1
            if valid.any():
                M["len_gen"] += float(lens[valid].sum())
                # share of generated thought tokens that are single digits (Qwen ids 15..24): the digit-string attractor
                _dg = [((Z[a, :max(int(lens[a]), 1)] >= 15) & (Z[a, :max(int(lens[a]), 1)] <= 24)).mean() for a in np.where(valid)[0]]
                M["digit_share"] += float(np.mean(_dg)) * float(valid.sum())
                ops = [tuple(Z[a, :3].tolist()) for a in np.where(valid)[0]]
                M["open_share"] += float(max(ops.count(o) for o in set(ops)) / len(ops)) if len(ops) > 1 else 0.0
                M["H_anc"] += float(Hbase[valid].sum())
                M["base_anc"] += float(base[valid].sum())
                M["drop_full"] += float((base - lt[:, -1])[valid].sum())
                M["drop_best"] += float((base[:, None] - lt).max(1)[valid].sum())
            if kept.any():
                kl = keep_len[kept]
                lk = np.array([lt[a, keep_len[a] // ch - 1] for a in np.where(kept)[0]])
                M["drop_kept"] += float((base[kept] - lk).sum())
                if lts is not None:
                    lsk = np.array([lts[a, keep_len[a] // ch - 1] for a in np.where(kept)[0]])
                    M["contrast_kept"] += float((lsk - lk).sum())
                M["len_kept"] += float(kl.sum())
                M["rep_kept"] += float(sum(rep_share(Z[a], keep_len[a]) for a in np.where(kept)[0]))
            # per-token CE of the imitation / reader rows, each averaged over the micros that ACTUALLY
            # HAD such rows. Until 2026-10-04 both lived inside `if kept.any()` while being divided by
            # n_micro, so under --reader-all they reported true_CE x P(micro has a kept thought) -- i.e.
            # keep_rate in disguise (G129).
            if w_imit.sum() > 0:
                M["imit_ce"] += float((ce_g * w_imit).sum() / float(w_imit.sum()))   # per-token CE, independent of the norm
                M["n_ice"] += 1.0
            if w_read.sum() > 0:
                M["read_ce"] += float((ce_g * w_read).sum() / float(w_read.sum()))
                M["n_rce"] += 1.0
            if dump_f and tok is not None and step % args.dump_every == 0 and micro == 0 and valid.any():
                recs = []
                for a in np.where(valid)[0][:6]:
                    n = int(anchors[a])
                    recs.append(dict(anchor=n, H=float(Hbase[a]), base=float(base[a]), gen_len=int(lens[a]),
                                     losses=[round(float(v), 3) for v in lt[a]], kept_len=int(keep_len[a]),
                                     context=tok.decode(x[max(0, n - 50):n + 1].tolist()),
                                     next_token=tok.decode([int(x[n + 1])]),
                                     thought=tok.decode(Z[a, :max(int(lens[a]), 1)].tolist())))
                dump_f.write(json.dumps(dict(step=step, rank=rank, anchors=recs)) + "\n")
                dump_f.flush()

        if world > 1:
            tp.sync_gradients(params, world, xm, torch_xla.sync)
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=False)
        torch_xla.sync()
        if step == start_step + 1:
            with torch.no_grad():
                _cs = float(sum(p_.double().sum() for p_ in params[:4]).cpu())
            print(f"[replica-check] rank {rank} step {step} sync_grads=1 checksum {_cs:.10e}", flush=True)

        # ---- reduce stats over ranks
        keys = [k for k in M if k != "n_dom"]
        vec = torch.tensor([M[k] for k in keys] + list(M["n_dom"]), dtype=torch.float32, device=device)
        if world > 1:
            vec = xm.all_reduce(xm.REDUCE_SUM, vec)
        v = vec.cpu().numpy()
        S_ = dict(zip(keys, v[:len(keys)])); dom = v[len(keys):]
        nm = max(S_["n_micro"], 1.0); nv = max(S_["n_valid"], 1.0); nk = max(S_["n_kept"], 1.0)
        nc_ = max(S_["n_ctrl"], 1.0)
        tokens_seen += args.grad_accum * T * world
        dt = time.time() - t0
        if step % args.log_every == 0:
            rec = dict(step=step, ntp1=S_["ntp1"] / nm, len_gen=S_["len_gen"] / nv, open_share=S_["open_share"] / nm,
                       digit_share=S_["digit_share"] / nv, contrast_kept=S_["contrast_kept"] / nk,
                       n_cand=S_["n_cand"] / nm, n_valid=S_["n_valid"] / nm,
                       n_kept=S_["n_kept"] / nm, keep_rate=S_["n_kept"] / nv, H_anc=S_["H_anc"] / nv,
                       base_anc=S_["base_anc"] / nv, drop_full=S_["drop_full"] / nv, drop_best=S_["drop_best"] / nv,
                       drop_kept=S_["drop_kept"] / nk, len_kept=S_["len_kept"] / nk, rep_kept=S_["rep_kept"] / nk,
                       imit_ce=S_["imit_ce"] / max(S_["n_ice"], 1.0), read_ce=S_["read_ce"] / max(S_["n_rce"], 1.0),
                       n_ice=S_["n_ice"], n_rce=S_["n_rce"],
                       # 2026-10-07 10:07: raw n_micro too, so every `/nm` mean above is auditable for
                       # "never computed" vs "measured zero" (see runs/digest.py COUNTS).
                       n_micro=S_["n_micro"],
                       ctrl_rate=S_["ctrl_kept"] / nc_, ctrl_drop_full=S_["ctrl_drop_full"] / nc_,
                       n_ctrl=S_["n_ctrl"], n_read=S_["n_read"] / nm, dom_mix=[int(d) for d in dom], lr=lr, gnorm=float(gnorm.cpu()),
                       dt=dt, tokens_seen=tokens_seen, world=world)
            rec = {k: (float(v_) if isinstance(v_, (np.floating, np.integer)) else v_) for k, v_ in rec.items()}
            vt.log(f"step {step:6d}/{args.max_steps} ntp1 {rec['ntp1']:.3f} | cand {rec['n_cand']:.0f} valid "
                   f"{rec['n_valid']:.1f} kept {rec['n_kept']:.2f} ({100*rec['keep_rate']:.0f}%) | H {rec['H_anc']:.2f} "
                   f"base {rec['base_anc']:.2f} drop full {rec['drop_full']:+.2f} best {rec['drop_best']:+.2f} kept "
                   f"{rec['drop_kept']:+.2f} (vs shuf {rec['contrast_kept']:+.2f}) len {rec['len_kept']:.1f}/{rec['len_gen']:.1f} rep {rec['rep_kept']:.2f} "
                   f"open {rec['open_share']:.2f} dig {rec['digit_share']:.2f} | ctrl {100*rec['ctrl_rate']:.0f}% "
                   f"{rec['ctrl_drop_full']:+.2f} | imCE {rec['imit_ce']:.2f} rdCE {rec['read_ce']:.2f} | gn {rec['gnorm']:.2f} "
                   f"{dt:.1f}s")
            if metrics_f:
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
        if (step + 1) % args.ckpt_every == 0:
            vt.save_checkpoint(model, opt, step + 1, args, tokens_seen, rank, world)
        if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
            vt.log(f"time limit {args.max_hours}h reached at step {step}")
            if (step + 1) % args.ckpt_every != 0:
                vt.save_checkpoint(model, opt, step + 1, args, tokens_seen, rank, world)
            break
    vt.log(f"done: {tokens_seen/1e9:.3f}B total tokens, {(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
