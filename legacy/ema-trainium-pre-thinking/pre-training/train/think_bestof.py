"""MASSIVE-GROUP BEST-OF-G thinking, no oracle at eval (Emanuele, 2026-10-06 15:30: "a massive group G in the GRPO (128 or 256),
score each thought and only keep the best one; we measure the advantage there, and as training goes on the model implicitly
learns what to prefer as a thought").

At the --n-think most uncertain positions (entropy > --tau) the LIVE model writes --G thoughts (<= --L tokens, Gumbel, temp 1,
logit floor 12, in rounds of --rounds-of rows sharing one prefill). Every thought is scored by the LIVE model on the next --K
TRUE tokens in PLAIN INSERTED-TEXT geometry (byte-identical to a plain sequence "x[:n+1] <T> thought </T> x[n] x[n+1..]",
verified): gain = sum over the span of (clean CE - CE with the thought). Per spot the BEST thought is kept.
  BASE:   clean next-token loss on all T tokens (--ntp-w).
  WRITER: REINFORCE on the best thought's tokens with its group-normalised advantage A* = (gain* - mean)/std, clipped at
          --adv-clip, weight --rl-w (positive only: the model imitates what it selected), plus the k3 KL-to-init on those rows
          (--kl-ref, frozen bf16 copy) against collapse (runs 83-86: imitation of self-selected thoughts collapsed the writer in
          ~150 steps; the KL is the standard counterweight and D2 stops the run if it fails).
  READER: CE on the span rows with the best thought in context, only if its gain > 0 (--reader-w): the reader learns to use the
          thoughts the writer is learning to prefer.
  GATES:  gate_pre on h[n] (label: share of the G that help) and gate_post on h[</T>] of the best AND of one random other thought
          (label: gain > 0), BCE at --gate-w: the natural gates, evaluated by probe_ceiling.py; they never feed writer or reader.
Deployment (the probe) samples ONE thought: the bet is that best-of-G selection pulls the single-sample distribution toward what
the group's best looked like. Invoked as `think_train.py --bestof 1 ...`. Gradients all-reduced. Fixed shapes throughout.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

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
import think_entropy as te
from think_oracle import OGeo

T_OPEN, T_CLOSE, EOD, NEG = th.T_OPEN, th.T_CLOSE, th.EOD, th.NEG


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--bestof", type=int, default=1)
    p.add_argument("--corpus", required=True)
    p.add_argument("--init-from", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--resume-model-only", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=11532)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--block-offset", type=int, default=256 * 22800)
    p.add_argument("--perm-seed", type=int, default=1234)
    p.add_argument("--initial-tokens-seen", type=int, default=8988917760)
    p.add_argument("--tau", type=float, default=4.0)
    p.add_argument("--n-think", type=int, default=8)
    p.add_argument("--min-pos", type=int, default=32)
    p.add_argument("--L", type=int, default=16, help="max thought length")
    p.add_argument("--K", type=int, default=4, help="horizon: true tokens the thought is scored on")
    p.add_argument("--G", type=int, default=128, help="thoughts per spot")
    p.add_argument("--rounds-of", type=int, default=1, help="UNUSED since the flat-row generator (kept for config compatibility)")
    p.add_argument("--score-group", type=int, default=32, help="thoughts per scoring forward")
    p.add_argument("--dec-chunks", type=int, default=2, help="row chunks for the decode-time lm_head + sampling (memory)")
    p.add_argument("--rl-w", type=float, default=1.0)
    # 2026-10-07 06:07 tick. "best" = run 95 and every earlier best-of-G run: REINFORCE on the ARGMAX thought with a
    # POSITIVE-ONLY normalised advantage, i.e. the gradient of E[max over G].  "mean" = the gradient of E[gain] over the
    # UNSELECTED policy: two i.i.d. rows of the same group get a SIGNED advantage (g - mu)/sd.  Run 95 measured E[max]
    # rising while E[gain] FELL (z -13 over 1,187 post-warmup steps) and the independent probe ceiling stayed flat over
    # six rungs, so the thing worth optimising is the mean.  DEFAULT IS "best": this file is re-read by run_think.sh on a
    # keepalive relaunch, and a default change would silently alter a live run's objective mid-flight.
    p.add_argument("--rl-obj", choices=("best", "mean"), default="best")
    p.add_argument("--adv-clip", type=float, default=3.0)
    p.add_argument("--reader-w", type=float, default=1.0)
    p.add_argument("--gate-w", type=float, default=1.0)
    p.add_argument("--ntp-w", type=float, default=1.0)
    p.add_argument("--freeze-embed", type=int, default=0, help="freeze the input embeddings (no grad, no Adam state: frees ~1.9 GB/core)")
    p.add_argument("--freeze-head", type=int, default=0, help="freeze the (untied) lm_head instead: same ~1.9 GB/core, no checkpointing caveat")
    p.add_argument("--kl-ref", type=float, default=0.04)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--logit-floor", type=float, default=12.0)
    p.add_argument("--margin", type=float, default=1.0, help="logged keep rate: best gain >= margin (diagnostic only)")
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
    p.add_argument("--ckpt-every", type=int, default=190)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--metrics-file", default=None)
    p.add_argument("--dump-every", type=int, default=25)
    p.add_argument("--dump-file", default=None)
    a, unknown = p.parse_known_args()
    a.unknown = unknown
    a.zero1 = False            # train.build_optimizer / save_checkpoint read it (think_grpo.py:113)
    return a


class BGen(te.ETGen):
    """ETGen over N*G FLAT rows (row a*G+g = sample g of spot a): ONE shared prefill, ONE decode round of L launches for the
    whole group. The first version broadcast the prefill cache R times and ran G/R rounds: 16 rounds x 16 steps = 256 device
    launches per micro at ~1 s each (245 s/micro on smoke 4). Launch count is what costs; rows are cheap. KV cost is the
    same (C = T + L*N*G columns, allocated twice: prefill + decode). PLAIN-TEXT positions: <T> at n+1, Z[t] at n+2+t."""

    def set_anchors(self, anchors):
        super().set_anchors(anchors)
        anc = anchors.astype(np.float32)
        for t in range(self.L):
            self.pos_dec[t].copy_(torch.from_numpy((anc + 1.0 + t)[None, :]))

    def _decode(self, args):
        """ETGen._decode with the lm_head + Gumbel sampling done in --dec-chunks row chunks, each synced to the host: the fp32 logits
        of all N*G rows at once ([768, 151936] = 466 MB, plus the noise) were the allocation that OOM'd launches 2-4 at step 1."""
        N, L = self.N, self.L
        z = np.zeros((N, L), dtype=np.int64)
        stopped = np.zeros(N, dtype=bool)
        lens = np.full(N, L, dtype=np.int64)
        nch = max(1, int(getattr(args, "dec_chunks", 2)))
        bounds = [(i * N // nch, (i + 1) * N // nch) for i in range(nch)]
        with torch.no_grad():
            self.dec.broadcast_from(self.pre, 1)
            self.sync()
            for t in range(L):
                tok_t = self.open_toks if t == 0 else torch.from_numpy(z[None, :, t - 1].copy()).to(self.device)
                self.dec.set_positions(self.col_t[t])
                with self.autocast_ctx():
                    ht = self.trunk(input_ids=tok_t, position_ids=self.pos_dec[t], attention_mask=self.m_dec[t],
                                    past_key_values=self.dec, use_cache=True)[0]
                tok = np.zeros(N, dtype=np.int64)
                for c0, c1 in bounds:
                    with self.autocast_ctx():
                        lg = self.lm_head(ht[:, c0:c1])
                    lg = lg.float() + (self.ban_stop if t >= self.min_stop else self.ban)
                    if args.logit_floor > 0:
                        mx = lg.max(-1, keepdim=True).values
                        lg = torch.where(lg < mx - args.logit_floor, torch.full_like(lg, -1e9), lg)
                    lg = lg / args.temp
                    u = torch.rand(lg.shape, device=self.device).clamp(1e-10, 1.0 - 1e-6)
                    tok[c0:c1] = torch.argmax(lg - torch.log(-torch.log(u)), -1).cpu().numpy()[0]
                    self.sync()
                tok = np.where(stopped, T_CLOSE, tok)
                newly = (~stopped) & (tok == T_CLOSE)
                lens[newly] = t
                stopped |= newly
                z[:, t] = tok
        return z, lens

    def decode_group(self, args, N, G):
        z, l = self.decode(args)                                            # [N*G, L], [N*G]
        return z.reshape(N, G, self.L).transpose(1, 0, 2).copy(), l.reshape(N, G).T.copy()   # [G, N, L], [G, N]


def main():
    args = parse()
    torch.manual_seed(args.seed)
    device = torch_xla.device()
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank, world = xr.global_ordinal(), xr.world_size()
    xm.set_rng_state(int(args.seed) * 1000 + int(rank), device=device)
    T, N, L, K, G = args.seq_len, args.n_think, args.L, args.K, args.G
    SG, R = args.score_group, args.rounds_of
    assert G % SG == 0
    sgeo = OGeo(T, N, SG, L, K)                                              # scoring: SG thoughts per spot per forward
    ggeo = OGeo(T, N, 2, L, K)                                               # grad: the best + one random other per spot
    vt.log("=" * 72)
    vt.log(f"BEST-OF-G THINKING | world={world} T={T} GA={args.grad_accum} tau={args.tau} spots={N} G={G} rounds_of={R} "
           f"score_group={SG} L={L} K={K} rl_obj={args.rl_obj} rl_w={args.rl_w} reader_w={args.reader_w} gate_w={args.gate_w} kl_ref={args.kl_ref} "
           f"| S_score={sgeo.S} S_grad={ggeo.S}")
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
        missing, unexpected = backbone.load_state_dict(sd, strict=False)
        vt.log(f"backbone <- {args.init_from} (missing {len(missing)}, unexpected {len(unexpected)})")
        del ck, sd
    reader_bb = copy.deepcopy(backbone) if args.kl_ref > 0 else None
    Hd = backbone.config.hidden_size
    backbone.gate_pre = torch.nn.Linear(Hd, 1); backbone.gate_post = torch.nn.Linear(Hd, 1)
    with torch.no_grad():
        for hd in (backbone.gate_pre, backbone.gate_post):
            hd.weight.zero_(); hd.bias.fill_(-0.8)
    if args.freeze_embed:
        backbone.model.embed_tokens.weight.requires_grad_(False)
        backbone.enable_input_require_grads()      # reentrant checkpointing needs an input that requires grad, or the trunk gets NO gradient
        vt.log("input embeddings FROZEN (memory: no grad / Adam state for the 151936 x 1024 table)")
    if args.freeze_head:
        backbone.lm_head.weight.requires_grad_(False)
        vt.log("lm_head FROZEN (memory: no grad / Adam state for the 151936 x 1024 output table)")
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True, "preserve_rng_state": False})
    model = backbone.to(device); model.train()
    for p_ in model.parameters():
        if p_.requires_grad:
            p_.grad = torch.zeros_like(p_)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    head_params = list(backbone.gate_pre.parameters()) + list(backbone.gate_post.parameters())
    head_ids = {id(p_) for p_ in head_params}
    trunk_params = [p_ for p_ in params if id(p_) not in head_ids]
    trunk, lm_head = backbone.model, backbone.lm_head
    if reader_bb is not None:
        reader_bb = reader_bb.to(torch.bfloat16).to(device).eval()
        for p_ in reader_bb.parameters():
            p_.requires_grad_(False)
        r_trunk, r_lm = reader_bb.model, reader_bb.lm_head
        vt.log("KL-TO-INIT reference: frozen bf16 copy of the init")
    lr_tensor = torch.tensor(args.lr, dtype=torch.float32, device=device)
    lr_host = torch.empty((), dtype=torch.float32)
    opt, _, _ = vt.build_optimizer(model, args, lr_tensor)
    autocast_ctx = lambda: torch_xla.amp.autocast(device, dtype=torch.bfloat16)

    mix = MixOP(args.corpus, "train")
    bat = BatcherOP(mix, 1, T, seed=args.perm_seed, rank=0, world_size=1, without_replacement=True, grad_accum=1)
    perm, dcum, shard_cum = bat._permutation, bat._domain_cumulative, bat._shard_cumulative

    def one_pass_row(gidx):
        gb = int(perm[gidx]); di = int(np.searchsorted(dcum, gb, side="right"))
        before = int(dcum[di - 1]) if di else 0
        return mix.data[mix.domains[di]].block(gb - before, T, cumulative=shard_cum[di]), di

    gen = BGen(trunk, lm_head, T, N * G, L, cfg.num_hidden_layers, device, autocast_ctx, torch_xla.sync)   # N*G flat rows
    n1_dev = torch.tensor(float(T), device=device)
    s_br = sgeo.BR                                                            # [N, SG, K] branch rows of the score buffer
    g_th = torch.from_numpy(ggeo.TH.reshape(-1)).to(device)                  # grad buffer thought rows (fixed)
    g_rc = torch.from_numpy(ggeo.RC.reshape(-1)).to(device)                  # grad buffer closer rows (fixed)
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
        """detached CE over a scoring buffer"""
        with torch.no_grad():
            bt = torch.from_numpy(buf[None]).to(device); pt = torch.from_numpy(pos[None]).to(device)
            mt = torch.from_numpy(m[None, None]).to(torch.bfloat16).to(device); tt = torch.from_numpy(tgt).to(device)
            with autocast_ctx():
                h = trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
            ces = []
            step_ = (h.shape[1] + 15) // 16
            for i in range(0, h.shape[1], step_):
                with autocast_ctx():
                    lg = lm_head(h[0, i:i + step_])
                ces.append(-torch.log_softmax(lg.float(), -1).gather(1, tt[i:i + step_, None])[:, 0])
            ce = torch.cat(ces).cpu().numpy().astype(np.float64)
        torch_xla.sync()
        return ce

    t_start = time.time()
    for step in range(start_step, args.max_steps):
        lr = vt.lr_at(step, args); lr_host.fill_(lr); lr_tensor.copy_(lr_host)
        t0 = time.time()
        M = dict(n_micro=0.0, ntp1=0.0, n_cand=0.0, n_valid=0.0, H_anc=0.0, gain_mean=0.0, gain_best=0.0, gain1_best=0.0,
                 adv_best=0.0, adv_samp=0.0, adv_samp_abs=0.0, gain_samp=0.0, helps=0.0, best_helps=0.0, n_kept=0.0, r_std=0.0, read_ce=0.0, n_rce=0.0, gate_pre_bce=0.0,
                 gate_post_bce=0.0, gate_post_acc=0.0, len_gen=0.0, len_best=0.0, rep_kept=0.0, digit_share=0.0, open_share=0.0,
                 open_best=0.0, kl_ref=0.0, n_klr=0.0, t_gen=0.0, t_score=0.0, n_dom=np.zeros(3))
        for micro in range(args.grad_accum):
            gidx = args.block_offset + (step * args.grad_accum + micro) * world + rank
            row, di = one_pass_row(gidx)
            x, y = row[:T].copy(), row[1:T + 1].copy()
            for arr in (x, y):
                arr[(arr == T_OPEN) | (arr == T_CLOSE)] = 220
            x_dev = torch.from_numpy(x[None]).to(device)
            rng = np.random.default_rng([args.seed, rank, step, micro, 31])

            tg0 = time.time()
            h0 = gen.prefill(x_dev); H = gen.entropy(h0, chunk=64); del h0
            cand = np.zeros(T, dtype=bool); cand[args.min_pos:T - 1 - K] = True
            cand &= H > args.tau
            for j in range(K + 1):
                cand[:T - j] &= x[j:] != EOD
            idx = np.where(cand)[0]; n_cand = len(idx)
            if n_cand > N:
                idx = idx[np.argsort(-H[idx])[:N]]
            idx = np.sort(idx)
            valid = np.zeros(N, dtype=bool); anchors = np.full(N, args.min_pos, dtype=np.int64)
            anchors[:len(idx)] = idx; valid[:len(idx)] = True
            gen.set_anchors(np.repeat(anchors, G))                              # row a*G+g -> spot a
            Z, lens = gen.decode_group(args, N, G)                           # [G, N, L], [G, N]
            t_gen = time.time() - tg0

            # ---- score all G thoughts, SG per forward (clean span CE comes from the same buffer's clean rows)
            ts0 = time.time()
            gain = np.zeros((G, N)); gain1 = np.zeros((G, N)); ce_clean = None
            an = anchors[:, None] + np.arange(K)[None, :]                        # [N, K]
            for g0 in range(0, G, SG):
                sl = slice(g0, g0 + SG)
                buf, pos, m, tgt, _ = sgeo.buffer(x, anchors, Z[sl], lens[sl])
                ce = score_forward(buf, pos, m, tgt)
                if ce_clean is None:
                    ce_clean = ce[an]                                            # [N, K]
                ce_br = ce[s_br]                                                 # [N, SG, K]
                gain[sl] = (ce_clean[:, None, :] - ce_br).sum(-1).T              # [SG, N]
                gain1[sl] = (ce_clean[:, None, 0] - ce_br[:, :, 0]).T
            t_score = time.time() - ts0
            mu = gain.mean(0); sd = gain.std(0)
            best = gain.argmax(0)                                                # [N]
            ar = np.arange(N)
            g_best = gain[best, ar]
            A_best = np.where(sd > 1e-6, (g_best - mu) / (sd + 1e-6), 0.0)
            A_best = np.clip(A_best, 0.0, args.adv_clip) * valid
            other = (best + rng.integers(1, G, size=N)) % G                      # a random different thought per spot
            Zsel = np.stack([Z[best, ar], Z[other, ar]])                          # [2, N, L]
            lsel = np.stack([lens[best, ar], lens[other, ar]])                    # [2, N]
            kept_best = (g_best > 0) & valid                                      # reader uses the best only if it helps
            # ---- MEAN objective: the two gradient rows are group rows 0 and 1, which are i.i.d. draws from the policy
            # (the whole group is), so no extra rng draw is needed and the sample is UNBIASED -- unlike `other`, which is
            # uniform over the G-1 NON-best thoughts and is therefore biased low by construction.
            A_samp = np.zeros((2, N), dtype=np.float64)
            if args.rl_obj == "mean":
                A_samp = np.where(sd[None, :] > 1e-6, (gain[:2] - mu[None, :]) / (sd[None, :] + 1e-6), 0.0)
                A_samp = np.clip(A_samp, -args.adv_clip, args.adv_clip) * valid[None, :]
                Zsel = np.stack([Z[0], Z[1]]); lsel = np.stack([lens[0], lens[1]])

            # ---- grad buffer: best + other per spot
            buf, pos, m, tgt, thm = ggeo.buffer(x, anchors, Zsel, lsel)
            bt = torch.from_numpy(buf[None]).to(device); pt = torch.from_numpy(pos[None]).to(device)
            mt = torch.from_numpy(m[None, None]).to(torch.bfloat16).to(device); tt = torch.from_numpy(tgt[None]).to(device)
            with autocast_ctx():
                h = trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
            w1 = np.zeros(ggeo.S, dtype=np.float32); w1[:T] = args.ntp_w
            w2 = np.zeros(ggeo.S, dtype=np.float32)
            w2[ggeo.BR[:, 0, :].reshape(-1)] = np.repeat(kept_best.astype(np.float32) * args.reader_w, K)
            w3 = np.zeros(ggeo.S, dtype=np.float32)
            w_tok = np.zeros((N, 2), dtype=np.float32)
            if args.rl_obj == "mean":
                # w = +A because weighted_ce3 minimises w*CE = -w*log p, so maximising A*log p needs weight +A.
                # A is SIGNED here: a below-average thought gets a negative weight and is pushed DOWN.
                w_tok[:, 0] = A_samp[0] * args.rl_w; w_tok[:, 1] = A_samp[1] * args.rl_w
            else:
                w_tok[:, 0] = A_best * args.rl_w
            w3[ggeo.TH.reshape(-1)] = (w_tok[:, :, None] * thm).reshape(-1)
            kl_ref_v = None
            if reader_bb is not None and args.kl_ref > 0:
                with torch.no_grad():
                    with autocast_ctx():
                        hr = r_trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
                        lg_r = r_lm(hr[0, g_th]); lg_l = lm_head(h.detach()[0, g_th])
                    ce_r = -torch.log_softmax(lg_r.float(), -1).gather(1, tt[0, g_th, None])[:, 0]
                    ce_l = -torch.log_softmax(lg_l.float(), -1).gather(1, tt[0, g_th, None])[:, 0]
                    dlt = (ce_l - ce_r).clamp(-5.0, 5.0); ratio = dlt.exp(); k3 = ratio - dlt - 1.0
                    bm = np.zeros((N, 2, L + 1), dtype=np.float32)
                    if args.rl_obj == "mean":
                        bm[:, 0, :] = thm[:, 0, :] * valid[:, None]                # KL on EVERY row that gets gradient
                        bm[:, 1, :] = thm[:, 1, :] * valid[:, None]
                    else:
                        bm[:, 0, :] = thm[:, 0, :] * valid[:, None]                # KL on the best's rows
                    mask = bm.reshape(-1)
                    w_kl = (args.kl_ref * (ratio - 1.0)).cpu().numpy() * mask
                    k3h = k3.cpu().numpy() * mask
                    kl_ref_v = float(k3h.sum() / max(mask.sum(), 1.0))
                    del hr, lg_r, lg_l
                w3[ggeo.TH.reshape(-1)] = w3[ggeo.TH.reshape(-1)] + w_kl.astype(np.float32)
            loss, ce_det = tp.weighted_ce3(h, lm_head, tt, torch.from_numpy(w1[None]).to(device),
                                           torch.from_numpy(w2[None]).to(device), torch.from_numpy(w3[None]).to(device),
                                           n1_dev, n1_dev, n1_dev, args.loss_chunks, autocast_ctx)
            anc_d = torch.from_numpy(anchors).to(device)
            lg_pre = backbone.gate_pre(h[0, anc_d].float())[:, 0]
            lg_post = backbone.gate_post(h[0, g_rc].float())[:, 0]                 # [N*2]: best, other per spot
            helps_share = ((gain > 0).mean(0)).astype(np.float32)
            y_pre = torch.from_numpy(helps_share).to(device)
            y_post_np = np.stack([(g_best > 0), (gain[other, ar] > 0)], 1).reshape(-1).astype(np.float32)
            y_post = torch.from_numpy(y_post_np).to(device)
            m_pre = torch.from_numpy(valid.astype(np.float32)).to(device)
            m_post = torch.from_numpy(np.repeat(valid.astype(np.float32), 2)).to(device)
            bce_pre = (F.binary_cross_entropy_with_logits(lg_pre, y_pre, reduction="none") * m_pre).sum() / m_pre.sum().clamp(min=1.0)
            bce_post = (F.binary_cross_entropy_with_logits(lg_post, y_post, reduction="none") * m_post).sum() / m_post.sum().clamp(min=1.0)
            loss = loss + args.gate_w * (bce_pre + bce_post)
            (loss / args.grad_accum).backward()
            bce_pre_v = float(bce_pre.detach().cpu()); bce_post_v = float(bce_post.detach().cpu())
            acc_post = float((((lg_post.detach() > 0).float() == y_post).float() * m_post).sum().cpu() / max(float(m_post.sum().cpu()), 1.0))
            ce_g = ce_det.cpu().numpy()
            torch_xla.sync()

            # ---- stats
            v = valid; nv = float(v.sum())
            M["n_micro"] += 1.0; M["ntp1"] += float(ce_g[:T].mean()); M["n_cand"] += float(n_cand)
            M["n_valid"] += nv; M["n_dom"][di] += 1; M["t_gen"] += t_gen; M["t_score"] += t_score
            M["gate_pre_bce"] += bce_pre_v; M["gate_post_bce"] += bce_post_v; M["gate_post_acc"] += acc_post
            if kl_ref_v is not None:
                M["kl_ref"] += kl_ref_v; M["n_klr"] += 1.0
            if v.any():
                M["H_anc"] += float(H[anchors][v].sum())
                M["gain_mean"] += float(mu[v].sum()); M["gain_best"] += float(g_best[v].sum())
                M["gain1_best"] += float(gain1[best, ar][v].sum()); M["adv_best"] += float(A_best[v].sum())
                M["adv_samp"] += float(A_samp[:, v].mean(0).sum()); M["adv_samp_abs"] += float(np.abs(A_samp[:, v]).mean(0).sum())
                M["gain_samp"] += float(gain[:2][:, v].mean(0).sum())
                M["helps"] += float(helps_share[v].sum()); M["best_helps"] += float((g_best[v] > 0).sum())
                M["n_kept"] += float((g_best[v] >= args.margin).sum()); M["r_std"] += float(sd[v].sum())
                M["len_gen"] += float(lens[:, v].mean(0).sum()); M["len_best"] += float(lens[best, ar][v].sum())
                rr = w2 > 0
                if rr.any():
                    M["read_ce"] += float(ce_g[rr].mean()); M["n_rce"] += 1.0
                M["rep_kept"] += float(np.mean([te.rep_share(Z[best[a], a], max(int(lens[best[a], a]), 1)) for a in np.where(v)[0]])) * nv
                dg = [((Z[0, a, :max(int(lens[0, a]), 1)] >= 15) & (Z[0, a, :max(int(lens[0, a]), 1)] <= 24)).mean() for a in np.where(v)[0]]
                M["digit_share"] += float(np.mean(dg)) * nv
                ops = [tuple(Z[0, a, :3].tolist()) for a in np.where(v)[0]]
                M["open_share"] += float(max(ops.count(o) for o in set(ops)) / len(ops)) if len(ops) > 1 else 0.0
                opb = [tuple(Z[best[a], a, :3].tolist()) for a in np.where(v)[0]]
                M["open_best"] += float(max(opb.count(o) for o in set(opb)) / len(opb)) if len(opb) > 1 else 0.0
            if dump_f and tok is not None and step % args.dump_every == 0 and micro == 0 and v.any():
                a0 = int(np.where(v)[0][len(np.where(v)[0]) // 2]); n0 = int(anchors[a0]); gb = int(best[a0]); gw = int(gain[:, a0].argmin())
                dump_f.write(json.dumps(dict(step=step, anchor=n0, H=float(H[n0]), context=tok.decode(x[max(0, n0 - 50):n0 + 1].tolist()),
                    span=tok.decode(x[n0 + 1:n0 + 1 + K].tolist()), best=tok.decode(Z[gb, a0, :max(int(lens[gb, a0]), 1)].tolist()),
                    best_gain=float(gain[gb, a0]), best_gain1=float(gain1[gb, a0]), adv_best=float(A_best[a0]),
                    worst=tok.decode(Z[gw, a0, :max(int(lens[gw, a0]), 1)].tolist()), worst_gain=float(gain[gw, a0]),
                    group_mean=float(mu[a0]), group_std=float(sd[a0]), helps_share=float(helps_share[a0]),
                    sample0=tok.decode(Z[0, a0, :max(int(lens[0, a0]), 1)].tolist()), sample0_gain=float(gain[0, a0]))) + "\n")
                dump_f.flush()

        if world > 1:
            tp.sync_gradients(params, world, xm, torch_xla.sync)
        # clip the trunk and the gate heads SEPARATELY: the zero-initialised heads see raw hidden states (norm ~ tens) and their
        # weight gradients alone drove the GLOBAL norm to ~4.4 on smoke 5 (run 94 without heads: 0.25), so a single clip at 1.0
        # was scaling every trunk gradient, NTP included, by ~1/4 -- an effective learning-rate cut the heads should not buy.
        gnorm = torch.nn.utils.clip_grad_norm_(trunk_params, args.grad_clip)
        gnorm_h = torch.nn.utils.clip_grad_norm_(head_params, args.grad_clip)
        opt.step(); opt.zero_grad(set_to_none=False); torch_xla.sync()
        if step == start_step + 1:
            with torch.no_grad():
                _cs = float(sum(p_.double().sum() for p_ in params[:4]).cpu())
            print(f"[replica-check] rank {rank} step {step} sync_grads=1 checksum {_cs:.10e}", flush=True)
        keys = [k for k in M if k != "n_dom"]
        vec = torch.tensor([M[k] for k in keys] + list(M["n_dom"]), dtype=torch.float32, device=device)
        if world > 1:
            vec = xm.all_reduce(xm.REDUCE_SUM, vec)
        vv = vec.cpu().numpy(); S_ = dict(zip(keys, vv[:len(keys)])); dom = vv[len(keys):]
        nm = max(S_["n_micro"], 1.0); nv_ = max(S_["n_valid"], 1.0)
        tokens_seen += args.grad_accum * T * world
        dt = time.time() - t0
        if step % args.log_every == 0:
            rec = dict(step=step, ntp1=S_["ntp1"] / nm, n_cand=S_["n_cand"] / nm, n_valid=S_["n_valid"] / nm, H_anc=S_["H_anc"] / nv_,
                       gain_mean=S_["gain_mean"] / nv_, gain_best=S_["gain_best"] / nv_, gain1_best=S_["gain1_best"] / nv_,
                       adv_best=S_["adv_best"] / nv_, adv_samp=S_["adv_samp"] / nv_,
                       adv_samp_abs=S_["adv_samp_abs"] / nv_, gain_samp=S_["gain_samp"] / nv_, helps=S_["helps"] / nv_, best_helps=S_["best_helps"] / nv_,
                       keep_rate=S_["n_kept"] / nv_, r_std=S_["r_std"] / nv_, read_ce=S_["read_ce"] / max(S_["n_rce"], 1.0),
                       gate_pre_bce=S_["gate_pre_bce"] / nm, gate_post_bce=S_["gate_post_bce"] / nm, gate_post_acc=S_["gate_post_acc"] / nm,
                       len_gen=S_["len_gen"] / nv_, len_best=S_["len_best"] / nv_, rep_kept=S_["rep_kept"] / nv_,
                       digit_share=S_["digit_share"] / nv_, open_share=S_["open_share"] / nm, open_best=S_["open_best"] / nm,
                       kl_ref=S_["kl_ref"] / max(S_["n_klr"], 1.0), drop_full=S_["gain_mean"] / nv_,
                       # 2026-10-07 10:07: RAW denominators (sums over this log window), logged so a
                       # report can tell "the mean is 0.000" from "the mean was never computed".  Every
                       # average above is `S_["x"] / max(S_["n_x"], 1.0)`, which silently turns an empty
                       # window into a measured 0.0 -- run 94 logged read_ce == fool_kl == 0.000 on all
                       # 3,554 rows with n_rce == 0 and every report called it "reader CE 0.000".  Purely
                       # additive: no value any reader already consumes changes.
                       n_rce=S_["n_rce"], n_klr=S_["n_klr"], n_micro=S_["n_micro"],
                       t_gen=S_["t_gen"] / nm, t_score=S_["t_score"] / nm,
                       dom_mix=[int(d) for d in dom], lr=lr, gnorm=float(gnorm.cpu()), gnorm_h=float(gnorm_h.cpu()), dt=dt, tokens_seen=tokens_seen, world=world)
            rec = {k: (float(v_) if isinstance(v_, (np.floating, np.integer)) else v_) for k, v_ in rec.items()}
            vt.log(f"step {step:6d}/{args.max_steps} ntp1 {rec['ntp1']:.3f} | cand {rec['n_cand']:.0f} spots {rec['n_valid']:.1f} H {rec['H_anc']:.2f} "
                   f"| group mean {rec['gain_mean']:+.2f} best {rec['gain_best']:+.2f} (k1 {rec['gain1_best']:+.2f}) "
                   f"A* {rec['adv_best']:.2f} samp {rec['gain_samp']:+.2f} As {rec['adv_samp']:+.2f}/{rec['adv_samp_abs']:.2f} "
                   f"helps {100*rec['helps']:.0f}% best>0 {100*rec['best_helps']:.0f}% sd {rec['r_std']:.2f} | rdCE {rec['read_ce']:.2f} "
                   f"| gate pre {rec['gate_pre_bce']:.3f} post {rec['gate_post_bce']:.3f} acc {rec['gate_post_acc']:.2f} "
                   f"| len {rec['len_gen']:.1f}/{rec['len_best']:.1f} open {rec['open_share']:.2f}/{rec['open_best']:.2f} dig {rec['digit_share']:.2f} "
                   f"klref {rec['kl_ref']:.3f} | gen {rec['t_gen']:.0f}s score {rec['t_score']:.0f}s | gn {rec['gnorm']:.2f} {dt:.1f}s")
            if metrics_f:
                metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
        if (step + 1) % args.ckpt_every == 0:
            vt.save_checkpoint(model, opt, step + 1, args, tokens_seen, rank, world)
        if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
            vt.log(f"time limit {args.max_hours}h reached at step {step}")
            if (step + 1) % args.ckpt_every != 0:
                vt.save_checkpoint(model, opt, step + 1, args, tokens_seen, rank, world)
            break
    vt.log("done")


if __name__ == "__main__":
    main()
