"""HINDSIGHT -> FORESIGHT: post-hoc explanations turned into thoughts (Emanuele's idea, 2026-10-02).

Every step does two passes over the same text, at fixed-stride note positions n (stride 64, window k=64):

  PASS 1 -- HINDSIGHT (the teacher). The writer sees the context AND the next H tokens x[n+1 .. n+H] and writes a
  post-hoc note <T> z </T>, placed AFTER what it read (positions n+1+H+i). G notes per position, sampled.
  Each note is scored by a FROZEN copy of the init (the reader) in the FORESIGHT geometry -- the note sits at
  positions n+1+i, the reader sees x[..n-1] + the note and predicts x[n+1 .. n+64]; the note's hidden states are
  recomputed from the context alone, so the future reaches the reader ONLY through the written words.
     usefulness U = discounted 64-token log-lik with the note - with silence      (the paper's reward)
     price      P = log q_teacher(z | x, future) - log p_student(z | x)            (future info smuggled in)
     reward     R = U - beta * P            (beta = 1: the ELBO on log p(future | context) through the note)
  GRPO over the G notes of each position; REINFORCE on the top-k and bottom-k by advantage (one grad pass).

  PASS 2 -- FORESIGHT (the student = the same weights, seeing only x[..n]). It IMITATES the best hindsight notes
  (top --imit-top by R, only notes that beat silence), and runs its own GRPO (Gs samples) on the paper's reward
  (no price: it cannot see the future). Clean next-token loss keeps the LM healthy.

Gradients are ALL-REDUCED across ranks and clipped (the think_train.py bug of 2026-10-02 is not repeated here).
Invoked as `think_train.py --hindsight 1 ...` so every launcher/monitor keeps working.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch_xla
import torch_xla.amp
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

import train as vt
from data import Mixture, MixtureBatcher
from model_init import build_model, untie_lm_head, MODEL_ID
import thinking as th
import think_par as tp

SPACE = 220
T_OPEN, T_CLOSE, EOD, NEG = th.T_OPEN, th.T_CLOSE, th.EOD, th.NEG


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--hindsight", type=int, default=1)
    p.add_argument("--corpus", required=True)
    p.add_argument("--init-from", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--resume-model-only", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--first-anchor", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--aligned-blocks", type=int, default=1)
    # hindsight / foresight
    p.add_argument("--hs-H", type=int, default=64, help="future tokens the teacher sees")
    p.add_argument("--hs-G", type=int, default=64, help="teacher notes per position")
    p.add_argument("--hs-beta", type=float, default=1.0, help="price weight (1 = ELBO)")
    # KL ANNEALING: at init the price (a 15-token note the student cannot predict costs ~30 nats) swamps the
    # usefulness signal (a few nats) and GRPO would only learn "write less" -- the classic posterior collapse.
    # The price ramps 0 -> hs-beta linearly over the first --hs-beta-ramp steps (0 = no ramp).
    p.add_argument("--hs-beta-ramp", type=int, default=300)
    # CONTRASTIVE REWARD (run 80, 2026-10-03). Under the frozen reader the own-vs-silence reward is maximised by a
    # context-free constant ('2222...' scores ~+3 vs silence): runs 79 and 76R both collapsed onto it. With 1, every
    # note's usefulness is measured AGAINST the same rollout's note for the previous position placed here:
    #   C[r,a] = U(z[r,a] at a) - U(z[r,a-1] at a)
    # so a constant (or any position-independent note) scores exactly 0. Teacher reward = C - beta*P, student GRPO
    # on C, imitation only of teacher notes with C > 0 AND U > 0. Costs a second reader pass over every note.
    p.add_argument("--hs-contrast", type=int, default=0)
    # IMITATION TARGET RULE (run 82). Run 81 collapsed at step 44 although its RL rewards were contrastive: the
    # imitation filter "C > 0 AND U > 0" kept notes that beat SILENCE, which near-constants do reliably (U ~ +3,
    # C ~ 0 +- noise) and real explanations often do not -- so the student was taught digit strings. c_margin:
    # imitate a teacher note only if it is clearly specific (C > --imit-cmin) and not repetitive.
    p.add_argument("--imit-rule", choices=("u_and_c", "c_margin"), default="u_and_c")
    p.add_argument("--imit-cmin", type=float, default=2.0)
    p.add_argument("--hs-tgrad", type=int, default=10, help="teacher REINFORCE on top-k AND bottom-k by advantage")
    p.add_argument("--st-G", type=int, default=16, help="student notes per position (its own GRPO)")
    p.add_argument("--st-tgrad", type=int, default=8, help="student REINFORCE on top-k AND bottom-k by advantage")
    p.add_argument("--imit-top", type=int, default=4, help="teacher notes the student imitates per position")
    p.add_argument("--gen-chunk", type=int, default=16, help="sampled rows per generation call (+1 greedy row)")
    p.add_argument("--mb-blocks", type=int, default=20, help="blocks per position in a multi-block pass")
    p.add_argument("--score-batch", type=int, default=8)
    # store the frozen reader in bf16: under autocast every matmul is bf16 anyway; frees 1.2 GB per core
    # (run 76's first smoke failed to LOAD the batched scoring graph: NRT_RESOURCE)
    p.add_argument("--reader-bf16", type=int, default=0)
    p.add_argument("--t-rl-w", type=float, default=1.0)
    p.add_argument("--s-rl-w", type=float, default=1.0)
    p.add_argument("--imit-w", type=float, default=1.0)
    p.add_argument("--ntp-w", type=float, default=1.0)
    p.add_argument("--neg-scale", type=float, default=0.5)
    p.add_argument("--adv-clip", type=float, default=5.0)
    p.add_argument("--rl-start", type=int, default=0)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--top-k-thought", type=int, default=512, help="unused since 2026-10-03 (device topk is broken)")
    p.add_argument("--logit-floor", type=float, default=12.0,
                   help="sampling keeps tokens within this many nats of the row max (device-side top-k substitute)")
    p.add_argument("--min-think", type=int, default=3)
    # optimisation (names match train.py helpers)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min-lr", type=float, default=5e-6)
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
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--metrics-file", default=None)
    p.add_argument("--dump-every", type=int, default=25)
    p.add_argument("--dump-file", default=None)
    args, unknown = p.parse_known_args()
    args.zero1 = False
    args.unknown = unknown
    return args


# ---------------------------------------------------------------------------------------------------------------
# multi-block geometry: [clean T | per position a, M blocks of <T> z_0..z_{L-1}], no window rows
# ---------------------------------------------------------------------------------------------------------------
class MultiBlock:
    def __init__(self, layout, M):
        self.lo, self.M = layout, M
        self.T, self.N, self.L, self.blk = layout.T, layout.N, layout.L, layout.blk
        self.S = self.T + self.N * M * self.blk

    def base(self, a, m):
        return self.T + (a * self.M + m) * self.blk

    def mask(self, vis):
        """block rows see clean[: n+1+vis] and their own block causally; clean rows causal."""
        T, S, blk = self.T, self.S, self.blk
        m = np.full((S, S), NEG, dtype=np.float32)
        m[:T, :T] = np.triu(np.full((T, T), NEG, dtype=np.float32), k=1)
        tri = np.triu(np.full((blk, blk), NEG, dtype=np.float32), k=1)
        for a in range(self.N):
            n = int(self.lo.anchors[a])
            for mm in range(self.M):
                b = self.base(a, mm)
                m[b:b + blk, :n + 1 + vis] = 0.0
                m[b:b + blk, b:b + blk] = tri
        return torch.from_numpy(m[None, None])

    def positions(self, shift):
        """aligned: <T> at n+1+shift, z_i at n+1+shift+i (think_par.build_positions 'aligned')."""
        p = np.zeros(self.S, dtype=np.float32)
        p[:self.T] = np.arange(self.T, dtype=np.float32)
        anc = self.lo.anchors.astype(np.float32)
        blk_pos = np.concatenate([[0.0], np.arange(self.L, dtype=np.float32)])        # [blk]
        pb = (anc[:, None, None] + 1.0 + shift) + blk_pos[None, None, :]               # [N,1,blk]
        p[self.T:] = np.broadcast_to(pb, (self.N, self.M, self.blk)).reshape(-1)
        return torch.from_numpy(p[None, :])

    def buffer(self, x_row, Z):
        """Z [M, N, L] -> int64 [S]."""
        buf = np.empty(self.S, dtype=np.int64)
        buf[:self.T] = x_row
        b = buf[self.T:].reshape(self.N, self.M, self.blk)
        b[:, :, 0] = T_OPEN
        b[:, :, 1:] = Z.transpose(1, 0, 2)
        return buf

    def targets(self, y_row, Z):
        tgt = np.zeros(self.S, dtype=np.int64)
        tgt[:self.T] = y_row
        t = tgt[self.T:].reshape(self.N, self.M, self.blk)
        t[:, :, :self.L] = Z.transpose(1, 0, 2)
        return tgt

    def row_weights(self, Wt):
        """Wt [M, N, L] per-token weights -> float32 [S] (block row t carries token t's weight)."""
        w = np.zeros(self.S, dtype=np.float32)
        ww = w[self.T:].reshape(self.N, self.M, self.blk)
        ww[:, :, :self.L] = Wt.transpose(1, 0, 2)
        return w

    def block_rows(self):
        """[M, N, L] row indices of the rows that predict z_0..z_{L-1}."""
        a = np.arange(self.N)[None, :, None]
        m = np.arange(self.M)[:, None, None]
        t = np.arange(self.L)[None, None, :]
        return self.T + (a * self.M + m) * self.blk + t


def token_mask(Z, L):
    """[R,N,L] bool: tokens that were SAMPLED (content + a sampled </T>); a </T> forced into the last slot and the
    padding after a stop are excluded. These are the tokens REINFORCE / imitation / log-probs act on."""
    hit = Z == T_CLOSE
    first = np.where(hit.any(-1), hit.argmax(-1), L)                      # [R,N]
    ar = np.arange(L)[None, None, :]
    upto = np.where(first < L - 1, first, L - 2)                          # forced close at L-1 is not sampled
    return ar <= upto[..., None]


# ---------------------------------------------------------------------------------------------------------------
# sampling with log-probs (top-K transfer, vectorised)
# ---------------------------------------------------------------------------------------------------------------
def sample_lp(vals, idx, t, z, lp, stopped, rng, args, spec_row, L):
    R, N, K = vals.shape
    if t == L - 1:                                   # every note is TERMINATED in its last slot (G53)
        z[:, :, t] = T_CLOSE
        lp[:, :, t] = 0.0
        stopped[:] = True
        return
    v = vals.astype(np.float64)
    ban = idx >= EOD
    if t >= args.min_think:
        ban &= idx != T_CLOSE
    v = np.where(ban, -np.inf, v)
    tk = args.top_k_thought
    if 0 < tk < K:
        kth = np.partition(v, -tk, axis=-1)[..., -tk][..., None]
        v = np.where(v < kth, -np.inf, v)
    v = v / args.temp
    v = v - v.max(-1, keepdims=True)
    pr = np.exp(v)
    pr /= pr.sum(-1, keepdims=True)
    u = rng.random((R, N, 1))
    j = np.minimum((np.cumsum(pr, -1) < u).sum(-1), K - 1)
    if spec_row is not None and spec_row < R:
        j[spec_row] = pr[spec_row].argmax(-1)
    tok = np.take_along_axis(idx, j[..., None], -1)[..., 0]
    lpt = np.log(np.take_along_axis(pr, j[..., None], -1)[..., 0] + 1e-30)
    tok = np.where(stopped, T_CLOSE, tok)
    lpt = np.where(stopped, 0.0, lpt)
    z[:, :, t] = tok
    lp[:, :, t] = lpt
    stopped |= tok == T_CLOSE


class Gen(tp.ParGenerator):
    """ParGenerator with per-token sampling log-probs and the vectorised sampler. One instance serves both views
    (set_writer_delay(-H) = hindsight, set_writer_delay(0) = foresight): same shapes, no recompilation."""

    def generate_lp(self, x_dev, rng, args, spec_row):
        was = self.trunk.training
        if was:
            self.trunk.eval()
        try:
            return self._gen_lp(x_dev, rng, args, spec_row)
        finally:
            if was:
                self.trunk.train()

    def _gen_lp(self, x_dev, rng, args, spec_row):
        """DEVICE-SIDE sampling (2026-10-03). torch.topk on this Neuron stack returns the right VALUES with WRONG
        INDICES for every multi-row input (runs/probes/gentest: 0/64 rows correct at K=8..528; argmax 70/70
        correct), which made every sampled note of run 76R multilingual garbage. So: Gumbel-max on device --
        bans + a relative logit floor (drop tokens > --logit-floor nats below the row max; plays the role of
        top-k) + temperature, then argmax(logits + Gumbel). log q = log_softmax gathered at the sample. Only the
        [R,N] tokens and log-probs leave the device."""
        lo, L, N, R = self.layout, self.layout.L, self.layout.N, self.R
        z = np.zeros((R, N, L), dtype=np.int64)
        lp = np.zeros((R, N, L), dtype=np.float64)
        stopped = np.zeros((R, N), dtype=bool)
        if getattr(self, "_ban_pre", None) is None:
            V = self.lm_head.weight.shape[0]
            b = torch.zeros(V, dtype=torch.float32)
            b[EOD:] = -1e9
            self._ban_pre = b.to(self.device)
            b2 = b.clone()
            b2[T_CLOSE] = 0.0
            self._ban_post = b2.to(self.device)
        floor = float(getattr(args, "logit_floor", 12.0))
        with torch.no_grad():
            self.pre.set_positions(self.pos_base)
            with self.autocast_ctx():
                self.trunk(input_ids=x_dev, position_ids=self.pos_pre, attention_mask=self.m_pre,
                           past_key_values=self.pre, use_cache=True)
            self.dec.broadcast_from(self.pre, R)
            self.sync()
            for t in range(L):
                tok_t = self.open_toks if t == 0 else torch.from_numpy(
                    np.ascontiguousarray(z[:, :, t - 1])).to(self.device)
                self.dec.set_positions(self.col_t[t])
                with self.autocast_ctx():
                    ht = self.trunk(input_ids=tok_t, position_ids=self.pos_dec[t], attention_mask=self.m_dec[t],
                                    past_key_values=self.dec, use_cache=True)[0]
                    lgt = self.lm_head(ht)
                lg = lgt.float() + (self._ban_post if t >= args.min_think else self._ban_pre)
                if floor > 0:
                    mx = lg.max(-1, keepdim=True).values
                    lg = torch.where(lg < mx - floor, torch.full_like(lg, -1e9), lg)
                lg = lg / args.temp
                u = torch.rand(lg.shape, device=self.device).clamp(1e-10, 1.0 - 1e-6)
                samp = torch.argmax(lg - torch.log(-torch.log(u)), -1)
                grd = torch.argmax(lg, -1)
                lsm = torch.log_softmax(lg, -1)
                lp_s = lsm.gather(-1, samp[..., None])[..., 0]
                lp_g = lsm.gather(-1, grd[..., None])[..., 0]
                samp, grd = samp.cpu().numpy(), grd.cpu().numpy()
                lp_s, lp_g = lp_s.cpu().numpy(), lp_g.cpu().numpy()
                self.sync()
                if t == L - 1:                       # every note is TERMINATED in its last slot (G53)
                    z[:, :, t] = T_CLOSE
                    lp[:, :, t] = 0.0
                    stopped[:] = True
                    continue
                tok = samp.astype(np.int64)
                lpt = lp_s.astype(np.float64)
                if spec_row is not None and spec_row < R:
                    tok[spec_row] = grd[spec_row]
                    lpt[spec_row] = lp_g[spec_row]
                tok = np.where(stopped, T_CLOSE, tok)
                lpt = np.where(stopped, 0.0, lpt)
                z[:, :, t] = tok
                lp[:, :, t] = lpt
                stopped |= tok == T_CLOSE
        return z, lp


def _rss_gb():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1048576
    except Exception:
        pass
    return -1.0


def memlog(tag, rank):
    """HS_MEMLOG=1: rank 0 prints its host RSS at each phase (run 76 died of host OOM at ~30 GB/rank)."""
    if os.environ.get("HS_MEMLOG") and rank == 0:
        print(f"[mem] {time.strftime('%H:%M:%S')} {tag} rss={_rss_gb():.2f}GB", flush=True)


class Ctx:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def reader_scores_fn(c, x_row, notes, tgt_win_d):
    """notes [R, N, L] -> discounted window log-lik [R, N] under the FROZEN reader, foresight geometry
    (note at n+1+i, reader sees x[..n-1] + the note). Always runs full batches of c.B (one compiled graph)."""
    R, N, L, k, B = notes.shape[0], c.N, c.L, c.k, c.B
    out = np.zeros((R, N), dtype=np.float64)
    sil = np.full((N, L), T_CLOSE, dtype=np.int64)
    for s0 in range(0, R, B):
        chunk = [notes[r] for r in range(s0, min(s0 + B, R))]
        nb = len(chunk)
        chunk += [sil] * (B - nb)
        bufs = np.stack([tp.build_buffer(c.layout, x_row, cc, SPACE) for cc in chunk])
        with torch.no_grad():
            bt = torch.from_numpy(bufs).to(c.device)
            with c.autocast():
                h = c.r_trunk(input_ids=bt, position_ids=c.pos_score, attention_mask=c.mask_score,
                              use_cache=False)[0]
            res = []
            for b in range(B):
                with c.autocast():
                    lg = c.r_head(h[b, c.rows_win_d])
                res.append((torch.log_softmax(lg.float(), -1).gather(1, tgt_win_d[:, None])
                            .view(N, k) * c.wk_t).sum(-1))
            res = torch.stack(res).cpu().numpy()
        c.sync()
        out[s0:s0 + nb] = res[:nb]
    return out


def student_logp_fn(c, x_row, y_row, Z):
    """Z [R, N, L] -> per-token log p_student [R, N, L] (foresight view, trained weights, no grad)."""
    MB, N, L = c.MB, c.N, c.L
    R, M = Z.shape[0], MB.M
    out = np.zeros((R, N, L), dtype=np.float64)
    pad = np.full((N, L), T_CLOSE, dtype=np.int64)
    for s0 in range(0, R, M):
        zz = [Z[r] for r in range(s0, min(s0 + M, R))]
        nb = len(zz)
        zz = np.stack(zz + [pad] * (M - nb))
        buf = MB.buffer(x_row, zz)
        tgt = MB.targets(y_row, zz)
        with torch.no_grad():
            bt = torch.from_numpy(buf[None]).to(c.device)
            tg = torch.from_numpy(tgt[c.mb_rows.reshape(-1)]).to(c.device)
            with c.autocast():
                h = c.trunk(input_ids=bt, position_ids=c.pos_S, attention_mask=c.mask_S, use_cache=False)[0]
            hb = h[0, c.mb_rows_d]
            parts = []
            st_ = (hb.shape[0] + 3) // 4
            for i in range(0, hb.shape[0], st_):
                with c.autocast():
                    lg = c.lm_head(hb[i:i + st_])
                parts.append(torch.log_softmax(lg.float(), -1).gather(1, tg[i:i + st_, None])[:, 0])
            lpv = torch.cat(parts).cpu().numpy().reshape(M, N, L)
        c.sync()
        out[s0:s0 + nb] = lpv[:nb]
    return out


def main():
    args = parse()
    torch.manual_seed(args.seed)
    device = torch_xla.device()
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank, world = xr.global_ordinal(), xr.world_size()
    # per-rank device RNG for the Gumbel sampler (same seed on every rank would draw identical noise)
    xm.set_rng_state(int(args.seed) * 1000 + int(rank), device=device)
    T, L, k, H = args.seq_len, args.L, args.k, args.hs_H
    assert H <= k, "the teacher may see at most the scored window"
    layout = tp.ParLayout(T, args.stride, k, L, args.first_anchor)
    N = layout.N
    MB = MultiBlock(layout, args.mb_blocks)
    assert args.imit_top + 2 * args.st_tgrad <= args.mb_blocks and 2 * args.hs_tgrad <= args.mb_blocks
    assert args.hs_G % args.gen_chunk == 0 and args.st_G % args.gen_chunk == 0
    assert 2 * args.st_tgrad <= args.st_G and 2 * args.hs_tgrad <= args.hs_G
    vt.log("=" * 72)
    vt.log(f"HINDSIGHT->FORESIGHT | world={world} T={T} N={N} k={k} L={L} H={H} G={args.hs_G} "
           f"beta={args.hs_beta} Gs={args.st_G} imit_top={args.imit_top} tgrad={args.hs_tgrad} "
           f"S_score={layout.S} S_multi={MB.S}")
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
    memlog("init: model loaded on host", 0 if vt.is_master() else 1)
    reader_bb = copy.deepcopy(backbone)              # the frozen reader = the init, before any update
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": True, "preserve_rng_state": False})
    model = backbone.to(device)
    model.train()
    for p_ in model.parameters():
        if p_.requires_grad:
            p_.grad = torch.zeros_like(p_)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    if args.reader_bf16:
        reader_bb = reader_bb.to(torch.bfloat16)
    reader_bb = reader_bb.to(device).eval()
    for p_ in reader_bb.parameters():
        p_.requires_grad_(False)
    r_trunk, r_head = reader_bb.model, reader_bb.lm_head
    import gc
    gc.collect()
    memlog("init: model + frozen reader on device", rank)
    trunk, lm_head = backbone.model, backbone.lm_head

    lr_tensor = torch.tensor(args.lr, dtype=torch.float32, device=device)
    lr_host = torch.empty((), dtype=torch.float32)
    opt, _, _ = vt.build_optimizer(model, args, lr_tensor)
    autocast_ctx = lambda: torch_xla.amp.autocast(device, dtype=torch.bfloat16)

    # ---- constants on device
    wk_np = (args.gamma ** np.arange(k)).astype(np.float32)
    wk_t = torch.from_numpy(wk_np).to(device)
    B = args.score_batch
    pos_score = tp.build_positions(layout, "aligned").to(device).expand(B, -1).contiguous()
    mask_score = tp.build_full_mask(layout, 0).to(torch.bfloat16).to(device)
    rows_win_d = torch.from_numpy(layout.win_rows().reshape(-1)).to(device)
    mask_T = MB.mask(H).to(torch.bfloat16).to(device)      # hindsight view (sees x[..n+H])
    mask_S = MB.mask(0).to(torch.bfloat16).to(device)      # foresight view (sees x[..n])
    pos_T = MB.positions(H).to(device)                      # post-hoc: the note sits after what it read
    pos_S = MB.positions(0).to(device)
    mb_rows = MB.block_rows()                               # [M, N, L]
    mb_rows_d = torch.from_numpy(mb_rows.reshape(-1)).to(device)
    zeros_S = torch.zeros((1, MB.S), dtype=torch.float32, device=device)
    one_d = torch.tensor(1.0, device=device)

    mixdata = Mixture(args.corpus, "train", aligned=bool(args.aligned_blocks))
    batcher = MixtureBatcher(mixdata, 1, T, seed=args.seed + 7, rank=rank, world_size=world)
    C = args.gen_chunk
    gen = Gen(trunk, lm_head, layout, cfg.num_hidden_layers, C + 1, device, autocast_ctx, torch_xla.sync,
              topk_transfer=1, slot_pos="aligned")
    tok = None
    if vt.is_master() and args.dump_file:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)

    start_step, tokens_seen = 0, 0
    if args.resume:
        start_step, tokens_seen = vt.load_checkpoint(model, opt, args, device, rank, world)
    metrics_f = open(args.metrics_file, "a") if (args.metrics_file and vt.is_master()) else None
    dump_f = open(args.dump_file, "a") if (args.dump_file and vt.is_master()) else None

    ctx = Ctx(device=device, layout=layout, MB=MB, autocast=autocast_ctx, B=B, k=k, L=L, N=N, wk_t=wk_t,
              r_trunk=r_trunk, r_head=r_head, trunk=trunk, lm_head=lm_head, pos_score=pos_score,
              mask_score=mask_score, rows_win_d=rows_win_d, pos_S=pos_S, mask_S=mask_S, mb_rows=mb_rows,
              mb_rows_d=mb_rows_d, sync=torch_xla.sync)
    reader_scores = lambda x_row, notes, tgt_win_d: reader_scores_fn(ctx, x_row, notes, tgt_win_d)
    student_logp = lambda x_row, y_row, Z: student_logp_fn(ctx, x_row, y_row, Z)

    def grad_pass(x_row, y_row, Z, w_ntp, w_imit, w_rl, n_imit, n_rl, view):
        """One multi-block forward/backward. Z [M,N,L]; w_* float32 [S] row weights."""
        buf = MB.buffer(x_row, Z)
        tgt = MB.targets(y_row, Z)
        bt = torch.from_numpy(buf[None]).to(device)
        tt = torch.from_numpy(tgt[None]).to(device)
        m_, p_ = (mask_T, pos_T) if view == "T" else (mask_S, pos_S)
        with autocast_ctx():
            h = trunk(input_ids=bt, position_ids=p_, attention_mask=m_, use_cache=False)[0]
        w1 = torch.from_numpy(w_ntp[None]).to(device) if w_ntp is not None else zeros_S
        w2 = torch.from_numpy(w_imit[None]).to(device) if w_imit is not None else zeros_S
        w3 = torch.from_numpy(w_rl[None]).to(device) if w_rl is not None else zeros_S
        n1 = torch.tensor(float(T), device=device)
        n2 = torch.tensor(max(n_imit, 1.0), device=device)
        n3 = torch.tensor(max(n_rl, 1.0), device=device)
        loss, ce = tp.weighted_ce3(h, lm_head, tt, w1, w2, w3, n1, n2, n3, args.loss_chunks, autocast_ctx)
        (loss / args.grad_accum).backward()
        ce_np = ce.cpu().numpy()
        torch_xla.sync()
        return ce_np

    def grpo(Rw, valid):
        """Rw [G, N] -> advantages [G, N] (per position, normalised over its G samples; 0 at invalid positions)."""
        mu = Rw.mean(0, keepdims=True)
        sd = Rw.std(0, keepdims=True)
        A = np.where(sd > 1e-6, (Rw - mu) / (sd + 1e-6), 0.0)
        A = np.clip(A, -args.adv_clip, args.adv_clip) * valid[None, :]
        return np.where(A < 0, A * args.neg_scale, A)

    t_start = time.time()
    for step in range(start_step, args.max_steps):
        lr = vt.lr_at(step, args)
        lr_host.fill_(lr)
        lr_tensor.copy_(lr_host)
        rl_on = 1.0 if step >= args.rl_start else 0.0
        beta = args.hs_beta * (min(1.0, step / args.hs_beta_ramp) if args.hs_beta_ramp > 0 else 1.0)
        t0 = time.time()
        M = dict(C_t=0.0, Cmax_t=0.0, C_s=0.0, C_g=0.0, rep_t=0.0, rep_g=0.0, ntp1=0.0, U_t=0.0, Umax_t=0.0, P_t=0.0, R_t=0.0, pos_t=0.0, len_t=0.0, U_top=0.0, n_top=0.0,
                 U_s=0.0, pos_s=0.0, U_g=0.0, U_f=0.0, len_s=0.0, len_g=0.0, imit_ce=0.0, lps_top=0.0,
                 n_valid=0.0, n_micro=0.0)
        for micro in range(args.grad_accum):
            x, y, _ = batcher.batch(step, micro)
            for arr in (x, y):
                arr[(arr == T_OPEN) | (arr == T_CLOSE)] = SPACE
            xr_, yr_ = x[0], y[0]
            rng = np.random.default_rng([args.seed, rank, step, micro, 17])
            vw, _ = tp.validity(layout, xr_)
            valid = (vw > 0).astype(np.float64)
            x_dev = torch.from_numpy(x.astype(np.int64)).to(device)
            tgt_win_d = torch.from_numpy(np.stack([xr_[int(n) + 1:int(n) + 1 + k] for n in layout.anchors])
                                         .reshape(-1).astype(np.int64)).to(device)

            memlog(f"step{step} start", rank)
            # ---- PASS 1: hindsight notes (G per position)
            gen.set_writer_delay(-H, 0)
            ZT, LQ = [], []
            for c in range(args.hs_G // C):
                z, lp = gen.generate_lp(x_dev, rng, args, spec_row=C)
                ZT.append(z[:C])
                LQ.append(lp[:C])
            ZT = np.concatenate(ZT)                                  # [G, N, L]
            LQ = np.concatenate(LQ)
            # ---- foresight samples (Gs) + greedy, same generator, foresight view
            gen.set_writer_delay(0, 0)
            ZS = []
            for c in range(args.st_G // C):
                z, _ = gen.generate_lp(x_dev, rng, args, spec_row=C)
                ZS.append(z[:C])
                if c == 0:
                    zgreedy = z[C:C + 1]
            zs = np.concatenate(ZS + [zgreedy])                      # [Gs+1, N, L], last row greedy
            memlog(f"step{step} after generation", rank)
            Gs = args.st_G
            filler = np.full((1, N, L), T_CLOSE, dtype=np.int64)
            filler[0, :, :10] = SPACE
            silence = np.full((1, N, L), T_CLOSE, dtype=np.int64)

            # ---- score everything with the frozen reader in ONE batched sweep
            allnotes = np.concatenate([silence, filler, zs, ZT])     # [2 + Gs+1 + G, N, L]
            lw = reader_scores(xr_, allnotes, tgt_win_d)
            base = lw[0]
            U_fill = lw[1] - base
            U_s_all = lw[2:2 + Gs + 1] - base[None]
            U_s, U_g = U_s_all[:Gs], U_s_all[Gs]
            U_t = lw[2 + Gs + 1:] - base[None]                      # [G, N]
            if args.hs_contrast:
                rolled = np.roll(allnotes[2:], 1, axis=1)            # rollout r's note for a-1, placed at a
                UX = reader_scores(xr_, rolled, tgt_win_d) - base[None]
                C_s_all = U_s_all - UX[:Gs + 1]
                C_s, C_g = C_s_all[:Gs], C_s_all[Gs]
                C_t = U_t - UX[Gs + 1:]
            else:
                C_s, C_g, C_t = U_s, U_g, U_t
            memlog(f"step{step} after reader scoring", rank)

            # ---- price: log q_teacher - log p_student over the sampled tokens
            tmT = token_mask(ZT, L)
            lps_tok = student_logp(xr_, yr_, ZT)
            memlog(f"step{step} after student logp", rank)
            lq = (LQ * tmT).sum(-1)
            lps = (lps_tok * tmT).sum(-1)
            P = lq - lps                                            # [G, N]
            R_t = C_t - beta * P

            # ---- teacher GRPO, REINFORCE on top-k and bottom-k by advantage (hindsight view)
            A_t = grpo(R_t, valid)
            order = np.argsort(-A_t, axis=0)                        # [G, N] best first
            sel = np.concatenate([order[:args.hs_tgrad], order[-args.hs_tgrad:]])    # [2k, N]
            Mb = args.mb_blocks
            Zg = np.full((Mb, N, L), T_CLOSE, dtype=np.int64)
            Wg = np.zeros((Mb, N, L), dtype=np.float32)
            for i in range(sel.shape[0]):
                Zg[i] = ZT[sel[i], np.arange(N)]
                Wg[i] = (A_t[sel[i], np.arange(N)][:, None] * tmT[sel[i], np.arange(N)]).astype(np.float32)
            Wg *= args.t_rl_w * rl_on
            n_rl_t = float((np.abs(Wg) > 0).sum())
            grad_pass(xr_, yr_, Zg, None, None, MB.row_weights(Wg), 0.0, n_rl_t, "T")
            memlog(f"step{step} after teacher grad", rank)

            # ---- foresight: imitate the top teacher notes (that beat silence) + own GRPO on the paper reward
            tmS = token_mask(zs[:Gs], L)
            A_s = grpo(C_s, valid)
            ord_s = np.argsort(-A_s, axis=0)
            sel_s = np.concatenate([ord_s[:args.st_tgrad], ord_s[-args.st_tgrad:]])  # [2ks, N]
            bestR = np.argsort(-R_t, axis=0)[:args.imit_top]       # [I, N]
            Zs = np.full((Mb, N, L), T_CLOSE, dtype=np.int64)
            Wi = np.zeros((Mb, N, L), dtype=np.float32)
            Wr = np.zeros((Mb, N, L), dtype=np.float32)
            I = args.imit_top
            for i in range(I):
                Zs[i] = ZT[bestR[i], np.arange(N)]
                if args.imit_rule == "c_margin":
                    zi = ZT[bestR[i], np.arange(N)]
                    tmi = tmT[bestR[i], np.arange(N)]
                    norep = np.array([(np.unique(zi[a_][tmi[a_]], return_counts=True)[1].max() / max(tmi[a_].sum(), 1)
                                       < 0.7) if tmi[a_].any() else False for a_ in range(N)])
                    ok = (C_t[bestR[i], np.arange(N)] > args.imit_cmin) & norep & (valid > 0)
                else:
                    ok = (U_t[bestR[i], np.arange(N)] > 0) & (C_t[bestR[i], np.arange(N)] > 0) & (valid > 0)
                Wi[i] = (tmT[bestR[i], np.arange(N)] * ok[:, None]).astype(np.float32)
            ar_ = np.arange(N)
            for i in range(sel_s.shape[0]):
                Zs[I + i] = zs[sel_s[i], ar_]
                Wr[I + i] = (A_s[sel_s[i], ar_][:, None] * tmS[sel_s[i], ar_]).astype(np.float32)
            Wi *= args.imit_w
            Wr *= args.s_rl_w * rl_on
            n_im = float((Wi > 0).sum())
            n_rl_s = float((np.abs(Wr) > 0).sum())
            w_ntp = np.zeros(MB.S, dtype=np.float32)
            w_ntp[:T] = args.ntp_w
            ce = grad_pass(xr_, yr_, Zs, w_ntp, MB.row_weights(Wi), MB.row_weights(Wr), n_im, n_rl_s, "S")
            memlog(f"step{step} after student grad", rank)

            # ---- stats (this rank)
            v = valid > 0
            nv = max(v.sum(), 1)
            M["n_valid"] += float(v.sum())
            M["n_micro"] += 1.0
            M["ntp1"] += float(ce[:T].mean())
            M["U_t"] += float(U_t[:, v].mean()) if v.any() else 0.0
            M["Umax_t"] += float(U_t[:, v].max(0).mean()) if v.any() else 0.0
            M["P_t"] += float(P[:, v].mean()) if v.any() else 0.0
            M["R_t"] += float(R_t[:, v].mean()) if v.any() else 0.0
            M["pos_t"] += float((U_t[:, v] > 0).mean()) if v.any() else 0.0
            M["len_t"] += float(tmT[:, v].sum(-1).mean()) if v.any() else 0.0
            M["C_t"] += float(C_t[:, v].mean()) if v.any() else 0.0
            M["Cmax_t"] += float(C_t[:, v].max(0).mean()) if v.any() else 0.0
            M["C_s"] += float(C_s[:, v].mean()) if v.any() else 0.0
            M["C_g"] += float(C_g[v].mean()) if v.any() else 0.0

            def _rep(Z):  # share of notes whose most common token fills >= 70% of their sampled tokens
                tm_ = token_mask(Z, L)
                out = []
                for r_ in range(Z.shape[0]):
                    for a_ in np.where(v)[0]:
                        tk = Z[r_, a_][tm_[r_, a_]]
                        if len(tk):
                            out.append(np.unique(tk, return_counts=True)[1].max() / len(tk) >= 0.7)
                return float(np.mean(out)) if out else 0.0
            M["rep_t"] += _rep(ZT[:8])
            M["rep_g"] += _rep(zs[Gs:Gs + 1])
            topU = U_t[bestR[0], np.arange(N)]
            M["U_top"] += float(topU[v].mean()) if v.any() else 0.0
            M["U_s"] += float(U_s[:, v].mean()) if v.any() else 0.0
            M["pos_s"] += float((U_s[:, v] > 0).mean()) if v.any() else 0.0
            M["U_g"] += float(U_g[v].mean()) if v.any() else 0.0
            M["U_f"] += float(U_fill[v].mean()) if v.any() else 0.0
            M["len_s"] += float(tmS[:, v].sum(-1).mean()) if v.any() else 0.0
            M["len_g"] += float(token_mask(zs[Gs:Gs + 1], L)[0, v].sum(-1).mean()) if v.any() else 0.0
            imr = MB.row_weights(Wi) > 0
            M["imit_ce"] += float(ce[imr].mean()) if imr.any() else 0.0
            tm0 = tmT[bestR[0], np.arange(N)]
            lp0 = lps_tok[bestR[0], np.arange(N)]
            M["lps_top"] += float((lp0 * tm0)[v].sum() / max(tm0[v].sum(), 1)) if v.any() else 0.0

            if dump_f and tok is not None and step % args.dump_every == 0 and micro == 0 and v.any():
                a0 = int(np.where(v)[0][len(np.where(v)[0]) // 2])
                n0 = int(layout.anchors[a0])
                b0 = int(bestR[0, a0])
                g0 = tmS[0, a0].sum()
                dump_f.write(json.dumps(dict(
                    step=step, anchor=n0,
                    context=tok.decode(xr_[max(0, n0 - 60):n0 + 1].tolist()),
                    window=tok.decode(xr_[n0 + 1:n0 + 1 + k].tolist()),
                    teacher_best=tok.decode(ZT[b0, a0][tmT[b0, a0]].tolist()),
                    teacher_best_U=float(U_t[b0, a0]), teacher_best_P=float(P[b0, a0]),
                    teacher_rand=tok.decode(ZT[0, a0][tmT[0, a0]].tolist()), teacher_rand_U=float(U_t[0, a0]),
                    student_greedy=tok.decode(zs[Gs, a0][token_mask(zs[Gs:Gs + 1], L)[0, a0]].tolist()),
                    student_greedy_U=float(U_g[a0]), filler_U=float(U_fill[a0]),
                )) + "\n")
                dump_f.flush()

        if world > 1:
            tp.sync_gradients(params, world, xm, torch_xla.sync)
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=False)
        torch_xla.sync()
        memlog(f"step{step} after optimizer", rank)
        if step == start_step + 1:
            with torch.no_grad():
                _cs = float(sum(p_.double().sum() for p_ in params[:4]).cpu())
            print(f"[replica-check] rank {rank} step {step} sync_grads=1 checksum {_cs:.10e}", flush=True)

        # ---- metrics: mean over micros, then over ranks
        keys = sorted(M.keys())
        vec = torch.tensor([M[kk] for kk in keys], dtype=torch.float32, device=device)
        if world > 1:
            vec = xm.all_reduce(xm.REDUCE_SUM, vec)
        vals = vec.cpu().numpy()
        S_ = dict(zip(keys, vals))
        nm = max(S_["n_micro"], 1.0)
        tokens_seen += args.grad_accum * T * world
        dt = time.time() - t0
        if step % args.log_every == 0:
            rec = {kk: float(S_[kk] / nm) for kk in keys if kk not in ("n_micro",)}
            rec.update(step=step, dt=dt, lr=lr, beta=beta, tokens_seen=tokens_seen, gnorm=float(gnorm.cpu()),
                       own_minus_filler=rec["U_g"] - rec["U_f"], world=world)
            vt.log(f"step {step:5d}/{args.max_steps} ntp1 {rec['ntp1']:.3f} | T: U {rec['U_t']:+.2f} "
                   f"Umax {rec['Umax_t']:+.2f} P {rec['P_t']:.1f} R {rec['R_t']:+.2f} U>0 {rec['pos_t']:.2f} "
                   f"len {rec['len_t']:.1f} top {rec['U_top']:+.2f} | S: U {rec['U_s']:+.2f} greedy "
                   f"{rec['U_g']:+.2f} filler {rec['U_f']:+.2f} own-fill {rec['own_minus_filler']:+.2f} "
                   f"len {rec['len_s']:.1f}/{rec['len_g']:.1f} imCE {rec['imit_ce']:.2f} "
                   f"lpTop {rec['lps_top']:.2f} | C: T {rec['C_t']:+.2f} Tmax {rec['Cmax_t']:+.2f} S {rec['C_s']:+.2f} "
                   f"greedy {rec['C_g']:+.2f} | rep T {rec['rep_t']:.2f} G {rec['rep_g']:.2f} | "
                   f"gn {rec['gnorm']:.2f} {dt:.1f}s")
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
    vt.log(f"done: {tokens_seen/1e6:.1f}M real tokens, {(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
