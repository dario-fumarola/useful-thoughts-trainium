"""ORACLE-GATED thinking over a SPAN -- "train with the cheat code" (Emanuele + owner, 2026-10-06).

The label is free in pretraining, so the oracle is used DURING TRAINING: at each of the --n-think most uncertain positions
(entropy > --tau) the LIVE model writes --G thoughts (<= --L tokens, Gumbel, temp 1, logit floor 12). Each thought is
scored by the LIVE model on the next --K TRUE tokens in INSERTED-TEXT geometry (the thought sits at positions n+1..n+k+1,
the branch re-reads x[n] at n+k+2 and continues x[n+1..] after it; the clean rows of the same buffer give the no-thought
loss on the same span). gain[g, a] = sum over the span of (clean CE - CE with the thought).
  BASE:   clean next-token loss on all T tokens (--ntp-w).
  READER: CE of the true span tokens with the thought in context, ONLY for thoughts whose gain > 0 (the oracle gate):
          the reader learns to exploit helpful thoughts and never sees a harmful one (--reader-w).
  WRITER: REINFORCE on the thought tokens with the group-normalised span gain as advantage (negatives x --neg-scale,
          clipped at --adv-clip, weight --rl-w), plus the k3 KL-to-init on the sampled tokens (--kl-ref, frozen bf16 copy
          of the init) folded into the per-row weight exactly as think_grpo.py does.
  GATES:  two linear heads trained by BCE against the oracle's decision, the NATURAL gates a deployed model can run:
          gate_pre  on the clean hidden state at n       -> P(a sampled thought here helps)  (label: share of the G that help)
          gate_post on the hidden state of the </T> row  -> P(this thought helps)            (label: gain > 0)
          (--gate-w). They are extra parameters (backbone.gate_pre / gate_post), saved in the checkpoint, read by
          probe_ceiling.py, and they never feed the writer or reader.
EVALUATION: probe_ceiling.py + agent/ceiling_analysis.py on held-out text report the CEILING (oracle gate at eval, the cheat)
and the NATURAL gates (heads, agreement, confidence, ...) for every horizon <= K. The ceiling minus the best natural gate is
the gap this line exists to close.
Invoked as `think_train.py --oracle 1 ...`. Gradients are all-reduced. All row sets have FIXED shapes (one XLA graph).
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

T_OPEN, T_CLOSE, EOD, NEG = th.T_OPEN, th.T_CLOSE, th.EOD, th.NEG


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", type=int, default=1)
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
    p.add_argument("--L", type=int, default=8, help="max thought length")
    p.add_argument("--K", type=int, default=16, help="horizon: true tokens the thought is scored on")
    p.add_argument("--G", type=int, default=4, help="thoughts per spot")
    p.add_argument("--rl-w", type=float, default=1.0)
    p.add_argument("--neg-scale", type=float, default=0.5)
    p.add_argument("--adv-clip", type=float, default=3.0)
    p.add_argument("--reader-w", type=float, default=1.0)
    p.add_argument("--gate-w", type=float, default=1.0)
    p.add_argument("--ntp-w", type=float, default=1.0)
    p.add_argument("--freeze-embed", type=int, default=0, help="freeze the input embeddings (no grad, no Adam state: frees ~1.9 GB/core)")
    p.add_argument("--freeze-head", type=int, default=0, help="freeze the (untied) lm_head instead: same ~1.9 GB/core, no checkpointing caveat")
    p.add_argument("--kl-ref", type=float, default=0.04)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--logit-floor", type=float, default=12.0)
    p.add_argument("--margin", type=float, default=1.0, help="logged keep rate: gain >= margin (diagnostic only)")
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


class OGeo:
    """One buffer per micro: clean sequence + N*G blocks [<T>, L slots, </T>, K branch rows] (inserted geometry, re-read)."""

    def __init__(self, T, N, G, L, K):
        self.T, self.N, self.G, self.L, self.K = T, N, G, L, K
        self.PB = (L + 2) + K
        self.S = T + N * G * self.PB
        a = np.arange(N)[:, None]; g = np.arange(G)[None, :]
        self.B = T + (a * G + g) * self.PB                                   # [N, G] block starts
        self.RC = self.B + L + 1                                             # [N, G] closer rows
        self.BR = self.RC[:, :, None] + 1 + np.arange(K)[None, None, :]      # [N, G, K] branch rows
        self.TH = self.B[:, :, None] + np.arange(L + 1)[None, None, :]       # [N, G, L+1] <T> + L slots

    def buffer(self, x, anchors, Z, lens):
        """x [T], anchors [N], Z [G, N, L], lens [G, N] -> buf, pos, mask, tgt, thought-row mask [N, G, L+1]"""
        T, N, G, L, K, S = self.T, self.N, self.G, self.L, self.K, self.S
        buf = np.full(S, T_CLOSE, dtype=np.int64); pos = np.zeros(S, dtype=np.float32)
        m = np.full((S, S), NEG, dtype=np.float32); tgt = np.zeros(S, dtype=np.int64)
        buf[:T] = x; pos[:T] = np.arange(T); m[:T, :T] = np.triu(np.full((T, T), NEG, dtype=np.float32), k=1)
        tgt[:T - 1] = x[1:]
        thm = np.zeros((N, G, L + 1), dtype=np.float32)
        tri = np.triu(np.full((L + 1, L + 1), NEG, dtype=np.float32), k=1)
        slot = np.arange(L + 1, dtype=np.float32)                       # <T> at n+1, Z[j] at n+2+j (plain inserted text)
        for a in range(N):
            n = int(anchors[a])
            for g in range(G):
                B = int(self.B[a, g]); k = max(int(lens[g, a]), 1)
                rows = np.arange(B, B + L + 1)
                buf[B] = T_OPEN; buf[B + 1:B + L + 1] = Z[g, a]
                pos[rows] = n + 1.0 + slot
                m[rows, :n + 1] = 0.0; m[rows[:, None], rows[None, :]] = tri
                tgt[B:B + k] = Z[g, a, :k]; tgt[B + k] = T_CLOSE; thm[a, g, :k + 1] = 1.0
                rc = B + L + 1; buf[rc] = T_CLOSE; pos[rc] = n + 2.0 + k
                m[rc, :n + 1] = 0.0; m[rc, B:B + k + 1] = 0.0; m[rc, rc] = 0.0
                for j in range(K):
                    rt = rc + 1 + j; buf[rt] = x[n + j]; pos[rt] = n + 3.0 + k + j; tgt[rt] = x[n + j + 1]
                    m[rt, :n + 1] = 0.0; m[rt, B:B + k + 1] = 0.0; m[rt, rc:rt + 1] = 0.0
        return buf, pos, m, tgt, thm


class OGen(te.ETGen):
    """ETGen with G sequential decodes sharing one prefill, and PLAIN-TEXT positions: <T> at n+1, Z[t] at n+2+t
    (ETGen puts Z[0] on the same position as <T>, a quirk of the re-read geometry of runs 83-94)."""

    def set_anchors(self, anchors):
        super().set_anchors(anchors)
        anc = anchors.astype(np.float32)
        for t in range(self.L):
            self.pos_dec[t].copy_(torch.from_numpy((anc + 1.0 + t)[None, :]))

    def decode_group(self, args, G):
        Z, lens = [], []
        for _ in range(G):
            z, l = self.decode(args); Z.append(z); lens.append(l)
        return np.stack(Z), np.stack(lens)                                   # [G, N, L], [G, N]


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
    geo = OGeo(T, N, G, L, K)
    vt.log("=" * 72)
    vt.log(f"ORACLE-GATED SPAN THINKING | world={world} T={T} GA={args.grad_accum} tau={args.tau} spots={N} G={G} L={L} K={K} "
           f"rl_w={args.rl_w} reader_w={args.reader_w} gate_w={args.gate_w} kl_ref={args.kl_ref} | S={geo.S}")
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
    reader_bb = copy.deepcopy(backbone) if args.kl_ref > 0 else None       # the init, before any update
    Hd = backbone.config.hidden_size
    backbone.gate_pre = torch.nn.Linear(Hd, 1); backbone.gate_post = torch.nn.Linear(Hd, 1)
    with torch.no_grad():
        for hd in (backbone.gate_pre, backbone.gate_post):
            hd.weight.zero_(); hd.bias.fill_(-0.8)                           # starts at P(help) ~ 0.31, the measured share
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

    gen = OGen(trunk, lm_head, T, N, L, cfg.num_hidden_layers, device, autocast_ctx, torch_xla.sync)
    n1_dev = torch.tensor(float(T), device=device)
    th_rows = torch.from_numpy(geo.TH.reshape(-1)).to(device)               # all thought rows, fixed
    rc_rows = torch.from_numpy(geo.RC.reshape(-1)).to(device)               # closer rows, fixed
    tok = None
    if vt.is_master() and args.dump_file:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
    start_step, tokens_seen = 0, args.initial_tokens_seen
    if args.resume:
        start_step, tokens_seen = vt.load_checkpoint(model, opt, args, device, rank, world)
    metrics_f = open(args.metrics_file, "a") if (args.metrics_file and vt.is_master()) else None
    dump_f = open(args.dump_file, "a") if (args.dump_file and vt.is_master()) else None

    t_start = time.time()
    for step in range(start_step, args.max_steps):
        lr = vt.lr_at(step, args); lr_host.fill_(lr); lr_tensor.copy_(lr_host)
        t0 = time.time()
        M = dict(n_micro=0.0, ntp1=0.0, n_cand=0.0, n_valid=0.0, n_th=0.0, H_anc=0.0, gain1=0.0, gainK=0.0, gainK_best=0.0,
                 helps1=0.0, helpsK=0.0, n_kept=0.0, r_std=0.0, read_ce=0.0, n_rce=0.0, gate_pre_bce=0.0, gate_post_bce=0.0,
                 gate_post_acc=0.0, len_gen=0.0, rep_kept=0.0, digit_share=0.0, open_share=0.0, kl_ref=0.0, n_klr=0.0,
                 n_dom=np.zeros(3))
        for micro in range(args.grad_accum):
            gidx = args.block_offset + (step * args.grad_accum + micro) * world + rank
            row, di = one_pass_row(gidx)
            x, y = row[:T].copy(), row[1:T + 1].copy()
            for arr in (x, y):
                arr[(arr == T_OPEN) | (arr == T_CLOSE)] = 220
            x_dev = torch.from_numpy(x[None]).to(device)

            h0 = gen.prefill(x_dev); H = gen.entropy(h0, chunk=64); del h0
            cand = np.zeros(T, dtype=bool); cand[args.min_pos:T - 1 - K] = True
            cand &= H > args.tau
            for j in range(K + 1):
                cand[:T - j] &= x[j:] != EOD                                   # no EOD inside the scored span
            idx = np.where(cand)[0]; n_cand = len(idx)
            if n_cand > N:
                idx = idx[np.argsort(-H[idx])[:N]]
            idx = np.sort(idx)
            valid = np.zeros(N, dtype=bool); anchors = np.full(N, args.min_pos, dtype=np.int64)
            anchors[:len(idx)] = idx; valid[:len(idx)] = True
            gen.set_anchors(anchors)
            Z, lens = gen.decode_group(args, G)                               # [G, N, L], [G, N]

            buf, pos, m, tgt, thm = geo.buffer(x, anchors, Z, lens)
            bt = torch.from_numpy(buf[None]).to(device); pt = torch.from_numpy(pos[None]).to(device)
            mt = torch.from_numpy(m[None, None]).to(torch.bfloat16).to(device); tt = torch.from_numpy(tgt[None]).to(device)
            with autocast_ctx():
                h = trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
            # ---- detached CE over the whole buffer (the labels), chunked lm_head over h.detach()
            with torch.no_grad():
                hd_ = h.detach()[0]; ces = []
                step_ = (geo.S + 15) // 16
                for i in range(0, geo.S, step_):
                    with autocast_ctx():
                        lg = lm_head(hd_[i:i + step_])
                    ces.append(-torch.log_softmax(lg.float(), -1).gather(1, tt[0, i:i + step_, None])[:, 0])
                ce_det = torch.cat(ces).cpu().numpy().astype(np.float64)
            torch_xla.sync()
            an = anchors[:, None] + np.arange(K)[None, :]                        # [N, K] clean rows of the span
            ce_clean = ce_det[an]                                                # [N, K]
            ce_br = ce_det[geo.BR]                                               # [N, G, K]
            gainK = (ce_clean[:, None, :] - ce_br).sum(-1)                       # [N, G] span gain
            gain1 = ce_clean[:, None, 0] - ce_br[:, :, 0]                        # [N, G] first-token gain
            # ---- advantages (group-normalised span gain), REINFORCE weights on thought rows
            mu = gainK.mean(1, keepdims=True); sd = gainK.std(1, keepdims=True)
            A = np.where(sd > 1e-6, (gainK - mu) / (sd + 1e-6), 0.0)
            A = np.clip(A, -args.adv_clip, args.adv_clip) * valid[:, None]
            w_tok = np.where(A < 0, A * args.neg_scale, A) * args.rl_w            # [N, G]
            w3 = np.zeros(geo.S, dtype=np.float32)
            w3[geo.TH.reshape(-1)] = (w_tok[:, :, None] * thm).reshape(-1)
            # ---- reader: branch rows of oracle-kept thoughts (gain > 0)
            kept = (gainK > 0) & valid[:, None]                                  # [N, G]
            w2 = np.zeros(geo.S, dtype=np.float32)
            w2[geo.BR.reshape(-1)] = (kept[:, :, None].astype(np.float32) * args.reader_w).repeat(K, axis=2).reshape(-1)
            w1 = np.zeros(geo.S, dtype=np.float32); w1[:T] = args.ntp_w
            # ---- KL-to-init on the thought rows (fixed row set, masked by thm)
            kl_ref_v = None
            if reader_bb is not None and args.kl_ref > 0:
                with torch.no_grad():
                    with autocast_ctx():
                        hr = r_trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
                        lg_r = r_lm(hr[0, th_rows]); lg_l = lm_head(h.detach()[0, th_rows])
                    ce_r = -torch.log_softmax(lg_r.float(), -1).gather(1, tt[0, th_rows, None])[:, 0]
                    ce_l = -torch.log_softmax(lg_l.float(), -1).gather(1, tt[0, th_rows, None])[:, 0]
                    dlt = (ce_l - ce_r).clamp(-5.0, 5.0); ratio = dlt.exp(); k3 = ratio - dlt - 1.0
                    mask = thm.reshape(-1) * np.repeat(valid.astype(np.float32), G * (L + 1))
                    w_kl = (args.kl_ref * (ratio - 1.0)).cpu().numpy() * mask
                    k3h = k3.cpu().numpy() * mask
                    kl_ref_v = float(k3h.sum() / max(mask.sum(), 1.0))
                    del hr, lg_r, lg_l
                w3[geo.TH.reshape(-1)] = w3[geo.TH.reshape(-1)] + w_kl.astype(np.float32)
            loss, _ = tp.weighted_ce3(h, lm_head, tt, torch.from_numpy(w1[None]).to(device),
                                      torch.from_numpy(w2[None]).to(device), torch.from_numpy(w3[None]).to(device),
                                      n1_dev, n1_dev, n1_dev, args.loss_chunks, autocast_ctx)
            # ---- gate heads: BCE against the oracle (fixed shapes: N pre logits, N*G post logits, host masks)
            anc_d = torch.from_numpy(anchors).to(device)
            lg_pre = backbone.gate_pre(h[0, anc_d].float())[:, 0]                 # [N]
            lg_post = backbone.gate_post(h[0, rc_rows].float())[:, 0]            # [N*G]
            y_pre = torch.from_numpy(kept.mean(1).astype(np.float32)).to(device)
            y_post = torch.from_numpy(kept.reshape(-1).astype(np.float32)).to(device)
            m_pre = torch.from_numpy(valid.astype(np.float32)).to(device)
            m_post = torch.from_numpy(np.repeat(valid.astype(np.float32), G)).to(device)
            bce_pre = (F.binary_cross_entropy_with_logits(lg_pre, y_pre, reduction="none") * m_pre).sum() / m_pre.sum().clamp(min=1.0)
            bce_post = (F.binary_cross_entropy_with_logits(lg_post, y_post, reduction="none") * m_post).sum() / m_post.sum().clamp(min=1.0)
            loss = loss + args.gate_w * (bce_pre + bce_post)
            (loss / args.grad_accum).backward()
            bce_pre_v = float(bce_pre.detach().cpu()); bce_post_v = float(bce_post.detach().cpu())
            acc_post = float((((lg_post.detach() > 0).float() == y_post).float() * m_post).sum().cpu() / max(float(m_post.sum().cpu()), 1.0))
            torch_xla.sync()

            # ---- stats
            v = valid; nv = float(v.sum())
            M["n_micro"] += 1.0; M["ntp1"] += float(ce_det[:T].mean()); M["n_cand"] += float(n_cand)
            M["n_valid"] += nv; M["n_dom"][di] += 1
            M["gate_pre_bce"] += bce_pre_v; M["gate_post_bce"] += bce_post_v; M["gate_post_acc"] += acc_post
            if kl_ref_v is not None:
                M["kl_ref"] += kl_ref_v; M["n_klr"] += 1.0
            if v.any():
                M["H_anc"] += float(H[anchors][v].sum())
                M["gain1"] += float(gain1[v].mean(1).sum()); M["gainK"] += float(gainK[v].mean(1).sum())
                M["gainK_best"] += float(gainK[v].max(1).sum())
                M["helps1"] += float((gain1[v] > 0).mean(1).sum()); M["helpsK"] += float((gainK[v] > 0).mean(1).sum())
                M["n_kept"] += float((gainK[v] >= args.margin).mean(1).sum())
                M["r_std"] += float(gainK[v].std(1).sum()); M["len_gen"] += float(lens[:, v].mean(0).sum())
                rr = w2 > 0
                if rr.any():
                    M["read_ce"] += float(ce_det[rr].mean()); M["n_rce"] += 1.0
                if kept.any():
                    M["rep_kept"] += float(np.mean([te.rep_share(Z[g, a], max(int(lens[g, a]), 1)) for a, g in zip(*np.where(kept))])) * nv
                dg = [((Z[0, a, :max(int(lens[0, a]), 1)] >= 15) & (Z[0, a, :max(int(lens[0, a]), 1)] <= 24)).mean() for a in np.where(v)[0]]
                M["digit_share"] += float(np.mean(dg)) * nv
                ops = [tuple(Z[0, a, :3].tolist()) for a in np.where(v)[0]]
                M["open_share"] += float(max(ops.count(o) for o in set(ops)) / len(ops)) if len(ops) > 1 else 0.0
            if dump_f and tok is not None and step % args.dump_every == 0 and micro == 0 and v.any():
                a0 = int(np.where(v)[0][len(np.where(v)[0]) // 2]); n0 = int(anchors[a0])
                gb, gw = int(np.argmax(gainK[a0])), int(np.argmin(gainK[a0]))
                dump_f.write(json.dumps(dict(step=step, anchor=n0, H=float(H[n0]), context=tok.decode(x[max(0, n0 - 50):n0 + 1].tolist()),
                    span=tok.decode(x[n0 + 1:n0 + 1 + K].tolist()), best=tok.decode(Z[gb, a0, :max(int(lens[gb, a0]), 1)].tolist()),
                    best_gainK=float(gainK[a0, gb]), best_gain1=float(gain1[a0, gb]),
                    worst=tok.decode(Z[gw, a0, :max(int(lens[gw, a0]), 1)].tolist()), worst_gainK=float(gainK[a0, gw]),
                    group_gainK_mean=float(gainK[a0].mean()), kept=int(kept[a0].sum()))) + "\n")
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
                       gain1=S_["gain1"] / nv_, gainK=S_["gainK"] / nv_, gainK_best=S_["gainK_best"] / nv_,
                       helps1=S_["helps1"] / nv_, helpsK=S_["helpsK"] / nv_, keep_rate=S_["n_kept"] / nv_, r_std=S_["r_std"] / nv_,
                       read_ce=S_["read_ce"] / max(S_["n_rce"], 1.0), gate_pre_bce=S_["gate_pre_bce"] / nm,
                       gate_post_bce=S_["gate_post_bce"] / nm, gate_post_acc=S_["gate_post_acc"] / nm,
                       len_gen=S_["len_gen"] / nv_, rep_kept=S_["rep_kept"] / nv_, digit_share=S_["digit_share"] / nv_,
                       open_share=S_["open_share"] / nm, kl_ref=S_["kl_ref"] / max(S_["n_klr"], 1.0),
                       drop_full=S_["gain1"] / nv_, dom_mix=[int(d) for d in dom], lr=lr, gnorm=float(gnorm.cpu()), gnorm_h=float(gnorm_h.cpu()), dt=dt,
                       tokens_seen=tokens_seen, world=world)
            rec = {k: (float(v_) if isinstance(v_, (np.floating, np.integer)) else v_) for k, v_ in rec.items()}
            vt.log(f"step {step:6d}/{args.max_steps} ntp1 {rec['ntp1']:.3f} | cand {rec['n_cand']:.0f} spots {rec['n_valid']:.1f} H {rec['H_anc']:.2f} "
                   f"| gain1 {rec['gain1']:+.2f} gainK {rec['gainK']:+.2f} best {rec['gainK_best']:+.2f} helpsK {100*rec['helpsK']:.0f}% "
                   f"r_sd {rec['r_std']:.2f} | rdCE {rec['read_ce']:.2f} | gate pre {rec['gate_pre_bce']:.3f} post {rec['gate_post_bce']:.3f} "
                   f"acc {rec['gate_post_acc']:.2f} | len {rec['len_gen']:.1f} open {rec['open_share']:.2f} dig {rec['digit_share']:.2f} "
                   f"klref {rec['kl_ref']:.3f} | gn {rec['gnorm']:.2f} {dt:.1f}s")
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
