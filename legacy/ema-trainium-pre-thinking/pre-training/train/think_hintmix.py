"""50/50 HINT MIXTURE (Emanuele, 2026-10-06 15:10 / go 2026-10-07 10:05): "a loss composed by standard NTP + an RL loss that tries to
predict the next token; 50% of the data is given as example and it is a paraphrase of the next token (but it never uses it); the other
50% is the hole we want to fill."

What __thoughts.txt said the data wants (2026-10-07): a good thought is a SHORT draft with one right content word; surface does not
matter; the READER is the lever -- it must learn to use a right draft and ignore a wrong one, and to RESUME the text after a thought
over the whole continuation (run 95 trained it on 4 tokens and 16-token spans were catastrophic). So:
  STREAM micros (the hole; GA of them per step, the vanilla batch):  the live model writes --G short thoughts (<= --L-own tokens) at
     the --n-think most uncertain positions; each is scored on the next --K TRUE tokens in plain inserted-text geometry by the live
     model (one forward with grad; the labels come from its detached CE). Losses: NTP on all T tokens (--ntp-w); REINFORCE on every
     thought with the group-normalised span gain (both signs, negatives x --neg-scale, --rl-w) + k3 KL-to-init (--kl-ref, frozen bf16
     copy); READER CE on the K span rows of EVERY thought (--reader-w-own) -- right and wrong drafts alike, true tokens as targets;
     gate heads (BCE vs gain > 0).
  HINT micros (the example; --hint-micros per step, drawn round-robin from the pool data/hints/*.jsonl = gold paraphrases written by
     Qwen2.5-3B-Instruct at the init model's own trigger positions on blocks from the END of the one-pass order, leakage-filtered):
     the hint is inserted as the thought. Losses: WRITER imitation CE on the hint tokens (--imit-w); READER CE on the K span rows with
     the hint (--reader-w-hint); gate heads. No NTP (the stream micros are the vanilla batch).
  BASE geometry/plumbing from think_bestof.py / think_oracle.py: OGeo buffers, BGen flat generator, --freeze-head, separate clipping.
Invoked as `think_train.py --hintmix 1 ...`. Fixed shapes. Gradients all-reduced.
"""
import argparse
import copy
import glob
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
from think_bestof import BGen

T_OPEN, T_CLOSE, EOD, NEG = th.T_OPEN, th.T_CLOSE, th.EOD, th.NEG


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--hintmix", type=int, default=1)
    p.add_argument("--corpus", required=True)
    p.add_argument("--init-from", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--resume-model-only", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--grad-accum", type=int, default=4, help="STREAM micros per step (the vanilla batch)")
    p.add_argument("--hint-micros", type=int, default=1, help="HINT micros per step (no NTP)")
    p.add_argument("--hint-pool", default="/home/ubuntu/ema-trainium/data/hints/pool_qwen3b_part*.jsonl")
    p.add_argument("--max-steps", type=int, default=11532)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--block-offset", type=int, default=256 * 22800)
    p.add_argument("--perm-seed", type=int, default=1234)
    p.add_argument("--initial-tokens-seen", type=int, default=8988917760)
    p.add_argument("--tau", type=float, default=4.0)
    p.add_argument("--n-think", type=int, default=6)
    p.add_argument("--min-pos", type=int, default=32)
    p.add_argument("--L", type=int, default=8, help="buffer slot length = max hint length (hints truncated)")
    p.add_argument("--L-own", type=int, default=6, help="max length of the model's own thoughts")
    p.add_argument("--K", type=int, default=16, help="horizon: true tokens the reader is trained on / the thought is scored on")
    p.add_argument("--G", type=int, default=8, help="own thoughts per spot")
    p.add_argument("--dec-chunks", type=int, default=1)
    p.add_argument("--rl-w", type=float, default=1.0)
    p.add_argument("--neg-scale", type=float, default=0.5)
    p.add_argument("--adv-clip", type=float, default=3.0)
    p.add_argument("--reader-w-own", type=float, default=0.5)
    p.add_argument("--reader-w-hint", type=float, default=1.0)
    p.add_argument("--imit-w", type=float, default=1.0)
    p.add_argument("--gate-w", type=float, default=1.0)
    p.add_argument("--ntp-w", type=float, default=1.0)
    p.add_argument("--kl-ref", type=float, default=0.1)
    p.add_argument("--hint-rl", type=int, default=0, help="DEFAULT OFF (run 98): in hint micros also write G own thoughts at the hint spots and REINFORCE them against the GOLD HINT's gain")
    p.add_argument("--hint-rl-w", type=float, default=1.0, help="weight of the be-as-useful-as-the-hint REINFORCE term")
    p.add_argument("--freeze-head", type=int, default=1)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--logit-floor", type=float, default=12.0)
    p.add_argument("--margin", type=float, default=1.0)
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
    a.zero1 = False
    return a


def load_pool(pattern, L):
    """block -> list of (n, hint_ids[:L]); sorted blocks"""
    pool = {}
    for f in sorted(glob.glob(pattern)):
        for l in open(f):
            d = json.loads(l)
            ids = [t for t in d["hint_ids"] if t not in (T_OPEN, T_CLOSE, EOD)][:L]
            if ids:
                pool.setdefault(int(d["block"]), []).append((int(d["n"]), ids))
    blocks = sorted(pool)
    return pool, blocks


def main():
    args = parse()
    torch.manual_seed(args.seed)
    device = torch_xla.device()
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank, world = xr.global_ordinal(), xr.world_size()
    xm.set_rng_state(int(args.seed) * 1000 + int(rank), device=device)
    T, N, L, Lo, K, G = args.seq_len, args.n_think, args.L, args.L_own, args.K, args.G
    assert Lo <= L
    geo = OGeo(T, N, G, L, K)                    # stream buffer: N spots x G own thoughts
    MH = 1 + (G if args.hint_rl else 0)
    hgeo = OGeo(T, N, MH, L, K)                  # hint buffer: N spots x (1 hint [+ G own thoughts when --hint-rl])
    pool, pool_blocks = load_pool(args.hint_pool, L)
    vt.log("=" * 72)
    vt.log(f"HINT MIXTURE | world={world} T={T} stream GA={args.grad_accum} hint micros={args.hint_micros} | tau={args.tau} spots={N} "
           f"G={G} L_own={Lo} L={L} K={K} | rl_w={args.rl_w} reader_own={args.reader_w_own} reader_hint={args.reader_w_hint} "
           f"imit_w={args.imit_w} kl_ref={args.kl_ref} gate_w={args.gate_w} | S_stream={geo.S} S_hint={hgeo.S} | pool {sum(len(v) for v in pool.values())} hints / {len(pool_blocks)} blocks")
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
    if args.freeze_head:
        backbone.lm_head.weight.requires_grad_(False)
        vt.log("lm_head FROZEN")
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

    gen = BGen(trunk, lm_head, T, N * G, Lo, cfg.num_hidden_layers, device, autocast_ctx, torch_xla.sync)
    n1_dev = torch.tensor(float(T), device=device)
    s_th = torch.from_numpy(geo.TH.reshape(-1)).to(device); s_rc = torch.from_numpy(geo.RC.reshape(-1)).to(device)
    h_rc = torch.from_numpy(hgeo.RC.reshape(-1)).to(device)
    h_th = torch.from_numpy(hgeo.TH.reshape(-1)).to(device)
    tok = None
    if vt.is_master() and args.dump_file:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
    start_step, tokens_seen = 0, args.initial_tokens_seen
    if args.resume:
        start_step, tokens_seen = vt.load_checkpoint(model, opt, args, device, rank, world)
    metrics_f = open(args.metrics_file, "a") if (args.metrics_file and vt.is_master()) else None
    dump_f = open(args.dump_file, "a") if (args.dump_file and vt.is_master()) else None
    n_micro_total = args.grad_accum + args.hint_micros

    def detached_ce(h, tt, S):
        with torch.no_grad():
            hd_ = h.detach()[0]; ces = []; step_ = (S + 15) // 16
            for i in range(0, S, step_):
                with autocast_ctx():
                    lg = lm_head(hd_[i:i + step_])
                ces.append(-torch.log_softmax(lg.float(), -1).gather(1, tt[0, i:i + step_, None])[:, 0])
            ce = torch.cat(ces).cpu().numpy().astype(np.float64)
        torch_xla.sync()
        return ce

    def gate_loss(h, anchors, valid, rc_rows, y_pre_np, y_post_np, M_):
        anc_d = torch.from_numpy(anchors).to(device)
        lg_pre = backbone.gate_pre(h[0, anc_d].float())[:, 0]
        lg_post = backbone.gate_post(h[0, rc_rows].float())[:, 0]
        y_pre = torch.from_numpy(y_pre_np.astype(np.float32)).to(device); y_post = torch.from_numpy(y_post_np.astype(np.float32)).to(device)
        m_pre = torch.from_numpy(valid.astype(np.float32)).to(device); m_post = torch.from_numpy(np.repeat(valid.astype(np.float32), M_)).to(device)
        bce_pre = (F.binary_cross_entropy_with_logits(lg_pre, y_pre, reduction="none") * m_pre).sum() / m_pre.sum().clamp(min=1.0)
        bce_post = (F.binary_cross_entropy_with_logits(lg_post, y_post, reduction="none") * m_post).sum() / m_post.sum().clamp(min=1.0)
        acc = ((((lg_post.detach() > 0).float() == y_post).float() * m_post).sum() / m_post.sum().clamp(min=1.0))
        return bce_pre, bce_post, acc

    t_start = time.time()
    for step in range(start_step, args.max_steps):
        lr = vt.lr_at(step, args); lr_host.fill_(lr); lr_tensor.copy_(lr_host)
        t0 = time.time()
        M = dict(n_micro=0.0, ntp1=0.0, n_cand=0.0, n_valid=0.0, H_anc=0.0, gain_mean=0.0, gain_best=0.0, gain1_mean=0.0, helps=0.0,
                 best_helps=0.0, r_std=0.0, read_ce_own=0.0, n_rco=0.0, kl_ref=0.0, n_klr=0.0, len_gen=0.0, rep_kept=0.0, digit_share=0.0,
                 open_share=0.0, gate_pre_bce=0.0, gate_post_bce=0.0, gate_post_acc=0.0, n_gate=0.0,
                 n_hint=0.0, n_hvalid=0.0, hint_gain=0.0, hint_gain1=0.0, hint_helps=0.0, hint_helps1=0.0, imit_ce=0.0, hint_read_ce=0.0,
                 hint_len=0.0, hrl_adv=0.0, hrl_beats=0.0, hrl_own_gain=0.0, n_hrl=0.0, n_dom=np.zeros(3))
        for micro in range(n_micro_total):
            is_hint = micro >= args.grad_accum
            if not is_hint:
                gidx = args.block_offset + (step * args.grad_accum + micro) * world + rank
            else:
                hi = (step * args.hint_micros + (micro - args.grad_accum)) * world + rank
                gidx = pool_blocks[hi % len(pool_blocks)]
            row, di = one_pass_row(gidx)
            x, y = row[:T].copy(), row[1:T + 1].copy()
            for arr in (x, y):
                arr[(arr == T_OPEN) | (arr == T_CLOSE)] = 220

            if not is_hint:
                # ---------------- STREAM micro: own thoughts
                x_dev = torch.from_numpy(x[None]).to(device)
                h0 = gen.prefill(x_dev); H = gen.entropy(h0, chunk=64); del h0
                cand = np.zeros(T, dtype=bool); cand[args.min_pos:T - 1 - K] = True; cand &= H > args.tau
                for j in range(K + 1):
                    cand[:T - j] &= x[j:] != EOD
                idx = np.where(cand)[0]; n_cand = len(idx)
                if n_cand > N:
                    idx = idx[np.argsort(-H[idx])[:N]]
                idx = np.sort(idx)
                valid = np.zeros(N, dtype=bool); anchors = np.full(N, args.min_pos, dtype=np.int64)
                anchors[:len(idx)] = idx; valid[:len(idx)] = True
                gen.set_anchors(np.repeat(anchors, G))
                Zo, lens = gen.decode_group(args, N, G)                       # [G, N, Lo], [G, N]
                Z = np.full((G, N, L), T_CLOSE, dtype=np.int64); Z[:, :, :Lo] = Zo
                buf, pos, m, tgt, thm = geo.buffer(x, anchors, Z, lens)
                bt = torch.from_numpy(buf[None]).to(device); pt = torch.from_numpy(pos[None]).to(device)
                mt = torch.from_numpy(m[None, None]).to(torch.bfloat16).to(device); tt = torch.from_numpy(tgt[None]).to(device)
                with autocast_ctx():
                    h = trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
                ce_det = detached_ce(h, tt, geo.S)
                an = anchors[:, None] + np.arange(K)[None, :]
                ce_clean = ce_det[an]; ce_br = ce_det[geo.BR]                    # [N, K], [N, G, K]
                gain = (ce_clean[:, None, :] - ce_br).sum(-1)                   # [N, G]
                gain1 = ce_clean[:, None, 0] - ce_br[:, :, 0]
                mu = gain.mean(1, keepdims=True); sd = gain.std(1, keepdims=True)
                A = np.where(sd > 1e-6, (gain - mu) / (sd + 1e-6), 0.0)
                A = np.clip(A, -args.adv_clip, args.adv_clip) * valid[:, None]
                w_tok = np.where(A < 0, A * args.neg_scale, A) * args.rl_w       # [N, G]
                w1 = np.zeros(geo.S, dtype=np.float32); w1[:T] = args.ntp_w
                w2 = np.zeros(geo.S, dtype=np.float32)
                w2[geo.BR.reshape(-1)] = np.repeat((valid[:, None] * np.ones((N, G))).reshape(-1) * args.reader_w_own, K).astype(np.float32)
                w3 = np.zeros(geo.S, dtype=np.float32); w3[geo.TH.reshape(-1)] = (w_tok[:, :, None] * thm).reshape(-1)
                kl_ref_v = None
                if reader_bb is not None and args.kl_ref > 0:
                    with torch.no_grad():
                        with autocast_ctx():
                            hr = r_trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
                            lg_r = r_lm(hr[0, s_th]); lg_l = lm_head(h.detach()[0, s_th])
                        ce_r = -torch.log_softmax(lg_r.float(), -1).gather(1, tt[0, s_th, None])[:, 0]
                        ce_l = -torch.log_softmax(lg_l.float(), -1).gather(1, tt[0, s_th, None])[:, 0]
                        dlt = (ce_l - ce_r).clamp(-5.0, 5.0); ratio = dlt.exp(); k3 = ratio - dlt - 1.0
                        mask = (thm * valid[:, None, None]).reshape(-1).astype(np.float32)
                        w_kl = (args.kl_ref * (ratio - 1.0)).cpu().numpy() * mask
                        kl_ref_v = float((k3.cpu().numpy() * mask).sum() / max(mask.sum(), 1.0))
                        del hr, lg_r, lg_l
                    w3[geo.TH.reshape(-1)] = w3[geo.TH.reshape(-1)] + w_kl.astype(np.float32)
                loss, _ = tp.weighted_ce3(h, lm_head, tt, torch.from_numpy(w1[None]).to(device), torch.from_numpy(w2[None]).to(device),
                                          torch.from_numpy(w3[None]).to(device), n1_dev, n1_dev, n1_dev, args.loss_chunks, autocast_ctx)
                bce_pre, bce_post, acc = gate_loss(h, anchors, valid, s_rc, (gain > 0).mean(1), (gain > 0).reshape(-1), G)
                loss = loss + args.gate_w * (bce_pre + bce_post)
                (loss / n_micro_total).backward()
                bce_pre_v, bce_post_v, acc_v = float(bce_pre.detach().cpu()), float(bce_post.detach().cpu()), float(acc.cpu())
                torch_xla.sync()
                v = valid; nv = float(v.sum())
                M["n_micro"] += 1.0; M["ntp1"] += float(ce_det[:T].mean()); M["n_cand"] += float(n_cand); M["n_valid"] += nv; M["n_dom"][di] += 1
                M["gate_pre_bce"] += bce_pre_v; M["gate_post_bce"] += bce_post_v; M["gate_post_acc"] += acc_v; M["n_gate"] += 1.0
                if kl_ref_v is not None:
                    M["kl_ref"] += kl_ref_v; M["n_klr"] += 1.0
                if v.any():
                    M["H_anc"] += float(H[anchors][v].sum()); M["gain_mean"] += float(gain[v].mean(1).sum()); M["gain_best"] += float(gain[v].max(1).sum())
                    M["gain1_mean"] += float(gain1[v].mean(1).sum()); M["helps"] += float((gain[v] > 0).mean(1).sum()); M["best_helps"] += float((gain[v].max(1) > 0).sum())
                    M["r_std"] += float(gain[v].std(1).sum()); M["len_gen"] += float(lens[:, v].mean(0).sum())
                    M["read_ce_own"] += float(ce_br[v].mean()); M["n_rco"] += 1.0
                    # 2026-10-09 12:07 tick, G210: ALL THREE COLLAPSE COLUMNS BELOW READ Zo[0] -- SAMPLE 0 OF
                    # G=8 -- so they watch 12.5% of the thoughts this run generates. The same is true of
                    # think_grpo.py:469/471, think_grpo_gate.py:580/582, think_oracle.py:370/372 and
                    # think_bestof.py:426/428; only think_entropy.py (one thought per anchor) watches all of
                    # them, which makes etrig_monitor.sh's old "on ALL generated thoughts" comment false in
                    # every trainer that has a G dimension, including the one it named. Consequences:
                    #   rep_kept   -- its name says "kept" and THIS FAMILY KEEPS NOTHING (see the keep_rate
                    #                 note below); it is the repetition share of sample 0. etrig_monitor.sh
                    #                 also SKIPS the D2 rep_kept clause whenever r_std is logged, which this
                    #                 trainer does, so the axis was never gated on for any of run 98's steps.
                    #   digit_share-- a MEAN over thoughts, so sample 0 is an UNBIASED estimator with 8-fold
                    #                 variance. Nothing to correct beyond the variance.
                    #   open_share -- a MODAL share, and a modal share over m items cannot fall below 1/m.
                    #                 Reading 1 of G raises its floor from 1/(G*nv) to 1/nv = 0.167 here, and
                    #                 run 98 sat on that floor for all 1,403 steps (0.1710 vs 0.1673), as did
                    #                 runs 90 and 95. It still CATCHES collapse (run 89 hit 7.9x its floor),
                    #                 but its low value is the statistic's minimum, not a measurement.
                    # A SUCCESSOR IN THIS FAMILY SHOULD COMPUTE ALL THREE OVER ALL G SAMPLES. Left as-is here
                    # because runs 96-98 are terminal and these columns are their record.
                    M["rep_kept"] += float(np.mean([te.rep_share(Zo[0, a], max(int(lens[0, a]), 1)) for a in np.where(v)[0]])) * nv
                    dg = [((Zo[0, a, :max(int(lens[0, a]), 1)] >= 15) & (Zo[0, a, :max(int(lens[0, a]), 1)] <= 24)).mean() for a in np.where(v)[0]]
                    M["digit_share"] += float(np.mean(dg)) * nv
                    ops = [tuple(Zo[0, a, :3].tolist()) for a in np.where(v)[0]]
                    M["open_share"] += float(max(ops.count(o) for o in set(ops)) / len(ops)) if len(ops) > 1 else 0.0
                if dump_f and tok is not None and step % args.dump_every == 0 and micro == 0 and v.any():
                    a0 = int(np.where(v)[0][len(np.where(v)[0]) // 2]); n0 = int(anchors[a0]); gb = int(gain[a0].argmax()); gw = int(gain[a0].argmin())
                    dump_f.write(json.dumps(dict(kind="own", step=step, anchor=n0, H=float(H[n0]), context=tok.decode(x[max(0, n0 - 50):n0 + 1].tolist()),
                        span=tok.decode(x[n0 + 1:n0 + 1 + K].tolist()), best=tok.decode(Zo[gb, a0, :max(int(lens[gb, a0]), 1)].tolist()), best_gain=float(gain[a0, gb]),
                        best_gain1=float(gain1[a0, gb]), worst=tok.decode(Zo[gw, a0, :max(int(lens[gw, a0]), 1)].tolist()), worst_gain=float(gain[a0, gw]),
                        group_mean=float(gain[a0].mean()), helps_share=float((gain[a0] > 0).mean()), sample0=tok.decode(Zo[0, a0, :max(int(lens[0, a0]), 1)].tolist()),
                        sample0_gain=float(gain[a0, 0]))) + "\n"); dump_f.flush()
            else:
                # ---------------- HINT micro: gold paraphrase as the thought (+ G own thoughts at the same spots when --hint-rl)
                hints = sorted(pool[gidx], key=lambda t: t[0])[:N]
                valid = np.zeros(N, dtype=bool); anchors = np.full(N, args.min_pos, dtype=np.int64)
                Zh = np.full((MH, N, L), T_CLOSE, dtype=np.int64); lens_h = np.ones((MH, N), dtype=np.int64)
                for a, (n_, ids) in enumerate(hints):
                    if n_ + K + 1 >= T: continue
                    anchors[a] = n_; valid[a] = True; Zh[0, a, :len(ids)] = ids; lens_h[0, a] = len(ids)
                if args.hint_rl:
                    x_dev = torch.from_numpy(x[None]).to(device)
                    h0 = gen.prefill(x_dev); del h0
                    gen.set_anchors(np.repeat(anchors, G))
                    Zo_h, lens_o = gen.decode_group(args, N, G)                    # [G, N, Lo], [G, N]
                    Zh[1:, :, :Lo] = Zo_h; lens_h[1:] = lens_o
                buf, pos, m, tgt, thm = hgeo.buffer(x, anchors, Zh, lens_h)
                bt = torch.from_numpy(buf[None]).to(device); pt = torch.from_numpy(pos[None]).to(device)
                mt = torch.from_numpy(m[None, None]).to(torch.bfloat16).to(device); tt = torch.from_numpy(tgt[None]).to(device)
                with autocast_ctx():
                    h = trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
                ce_det = detached_ce(h, tt, hgeo.S)
                an = anchors[:, None] + np.arange(K)[None, :]
                ce_clean = ce_det[an]; ce_br = ce_det[hgeo.BR]                   # [N, K], [N, MH, K]
                gain_all = (ce_clean[:, None, :] - ce_br).sum(-1)                 # [N, MH]
                gain = gain_all[:, 0]; gain1 = (ce_clean[:, 0] - ce_br[:, 0, 0])
                w1 = np.zeros(hgeo.S, dtype=np.float32)                           # no NTP on pool blocks
                w_read = np.zeros((N, MH), dtype=np.float32); w_read[:, 0] = valid * args.reader_w_hint
                if args.hint_rl:
                    w_read[:, 1:] = valid[:, None] * args.reader_w_own
                w2 = np.zeros(hgeo.S, dtype=np.float32); w2[hgeo.BR.reshape(-1)] = np.repeat(w_read.reshape(-1), K)
                w_tok = np.zeros((N, MH), dtype=np.float32); w_tok[:, 0] = valid * args.imit_w
                if args.hint_rl:
                    # be as useful as the hint: advantage of each own thought = (its gain - the HINT's gain) / group sd
                    go = gain_all[:, 1:]; sd = go.std(1, keepdims=True)
                    A = np.where(sd > 1e-6, (go - gain[:, None]) / (sd + 1e-6), 0.0)
                    A = np.clip(A, -args.adv_clip, args.adv_clip) * valid[:, None]
                    w_tok[:, 1:] = np.where(A < 0, A * args.neg_scale, A) * args.hint_rl_w
                w3 = np.zeros(hgeo.S, dtype=np.float32); w3[hgeo.TH.reshape(-1)] = (w_tok[:, :, None] * thm).reshape(-1)
                if args.hint_rl and reader_bb is not None and args.kl_ref > 0:
                    with torch.no_grad():
                        with autocast_ctx():
                            hr = r_trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
                            lg_r = r_lm(hr[0, h_th]); lg_l = lm_head(h.detach()[0, h_th])
                        ce_r = -torch.log_softmax(lg_r.float(), -1).gather(1, tt[0, h_th, None])[:, 0]
                        ce_l = -torch.log_softmax(lg_l.float(), -1).gather(1, tt[0, h_th, None])[:, 0]
                        dlt = (ce_l - ce_r).clamp(-5.0, 5.0); ratio = dlt.exp()
                        own_mask = thm.copy(); own_mask[:, 0, :] = 0.0; own_mask = (own_mask * valid[:, None, None]).reshape(-1).astype(np.float32)
                        w_kl = (args.kl_ref * (ratio - 1.0)).cpu().numpy() * own_mask
                        del hr, lg_r, lg_l
                    w3[hgeo.TH.reshape(-1)] = w3[hgeo.TH.reshape(-1)] + w_kl.astype(np.float32)
                loss, _ = tp.weighted_ce3(h, lm_head, tt, torch.from_numpy(w1[None]).to(device), torch.from_numpy(w2[None]).to(device),
                                          torch.from_numpy(w3[None]).to(device), n1_dev, n1_dev, n1_dev, args.loss_chunks, autocast_ctx)
                bce_pre, bce_post, acc = gate_loss(h, anchors, valid, h_rc, (gain_all > 0).mean(1).astype(np.float32), (gain_all > 0).reshape(-1), MH)
                loss = loss + args.gate_w * (bce_pre + bce_post)
                (loss / n_micro_total).backward()
                torch_xla.sync()
                v = valid; nv = float(v.sum())
                M["n_hint"] += 1.0; M["n_hvalid"] += nv
                if v.any():
                    M["hint_gain"] += float(gain[v].sum()); M["hint_gain1"] += float(gain1[v].sum())
                    M["hint_helps"] += float((gain[v] > 0).sum()); M["hint_helps1"] += float((gain1[v] > 0).sum())
                    thr = hgeo.TH[:, 0, :]
                    M["imit_ce"] += float((ce_det[thr] * thm[:, 0, :])[v].sum() / max(thm[:, 0, :][v].sum(), 1.0)) * nv
                    M["hint_read_ce"] += float(ce_br[v, 0, :].mean()) * nv; M["hint_len"] += float(lens_h[0][v].sum())
                    if args.hint_rl:
                        M["hrl_adv"] += float(A[v].mean(1).sum()); M["hrl_beats"] += float((gain_all[v, 1:] > gain[v, None]).mean(1).sum())
                        M["hrl_own_gain"] += float(gain_all[v, 1:].mean(1).sum()); M["n_hrl"] += nv
                if dump_f and tok is not None and step % args.dump_every == 0 and v.any():
                    a0 = int(np.where(v)[0][0]); n0 = int(anchors[a0])
                    rec_d = dict(kind="hint", step=step, anchor=n0, context=tok.decode(x[max(0, n0 - 50):n0 + 1].tolist()),
                                 span=tok.decode(x[n0 + 1:n0 + 1 + K].tolist()), hint=tok.decode(Zh[0, a0, :lens_h[0, a0]].tolist()), hint_gain=float(gain[a0]),
                                 hint_gain1=float(gain1[a0]))
                    if args.hint_rl:
                        gb = 1 + int(gain_all[a0, 1:].argmax())
                        rec_d.update(own_best=tok.decode(Zh[gb, a0, :max(int(lens_h[gb, a0]), 1)].tolist()), own_best_gain=float(gain_all[a0, gb]),
                                     own_mean_gain=float(gain_all[a0, 1:].mean()), own_beats_hint=float((gain_all[a0, 1:] > gain[a0]).mean()))
                    dump_f.write(json.dumps(rec_d) + "\n"); dump_f.flush()

        if world > 1:
            tp.sync_gradients(params, world, xm, torch_xla.sync)
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
        nm = max(S_["n_micro"], 1.0); nv_ = max(S_["n_valid"], 1.0); nh = max(S_["n_hvalid"], 1.0)
        tokens_seen += args.grad_accum * T * world
        dt = time.time() - t0
        if step % args.log_every == 0:
            rec = dict(step=step, ntp1=S_["ntp1"] / nm, n_cand=S_["n_cand"] / nm, n_valid=S_["n_valid"] / nm, H_anc=S_["H_anc"] / nv_,
                       gain_mean=S_["gain_mean"] / nv_, gain_best=S_["gain_best"] / nv_, gain1_mean=S_["gain1_mean"] / nv_, helps=S_["helps"] / nv_,
                       best_helps=S_["best_helps"] / nv_, r_std=S_["r_std"] / nv_, read_ce_own=S_["read_ce_own"] / max(S_["n_rco"], 1.0),
                       kl_ref=S_["kl_ref"] / max(S_["n_klr"], 1.0), len_gen=S_["len_gen"] / nv_, rep_kept=S_["rep_kept"] / nv_, digit_share=S_["digit_share"] / nv_,
                       open_share=S_["open_share"] / nm, gate_pre_bce=S_["gate_pre_bce"] / max(S_["n_gate"], 1.0), gate_post_bce=S_["gate_post_bce"] / max(S_["n_gate"], 1.0),
                       gate_post_acc=S_["gate_post_acc"] / max(S_["n_gate"], 1.0),
                       hint_gain=S_["hint_gain"] / nh, hint_gain1=S_["hint_gain1"] / nh, hint_helps=S_["hint_helps"] / nh, hint_helps1=S_["hint_helps1"] / nh,
                       imit_ce=S_["imit_ce"] / nh, hint_read_ce=S_["hint_read_ce"] / nh, hint_len=S_["hint_len"] / nh, n_hints=S_["n_hvalid"],
                       hrl_adv=S_["hrl_adv"] / max(S_["n_hrl"], 1.0), hrl_beats=S_["hrl_beats"] / max(S_["n_hrl"], 1.0), hrl_own_gain=S_["hrl_own_gain"] / max(S_["n_hrl"], 1.0),
                       # 2026-10-09 10:07 tick, G209: THESE TWO ARE ALIASES AND NEITHER IS WHAT ITS NAME SAYS.
                       # `drop_full` here is the k=1 gain (gain1_mean), while think_bestof.py:470 and
                       # think_entropy/think_grpo_gate set the SAME name to the full-span k=16 gain.  Verified on
                       # every logged row (agent/scratchpad/coldefs_1007.out): max|drop_full - gain1_mean| = 0.0 on
                       # runs 96/97/98 (5,738 rows), max|drop_full - gain_mean| up to 4.94 nats, mean ratio 0.25-0.36;
                       # and = 0.0 against gain_mean on think95_bestof.  `keep_rate` is `helps` exactly (max diff 0.0,
                       # same rows): this family NEVER gates, it speaks at every eligible anchor, so there is no kept
                       # fraction to report.  Kept under these names only because runs 96-98 are terminal and their
                       # logs are the record; every reader resolves `gain_mean`/`helps` FIRST, so nothing displayed is
                       # wrong today.  A SUCCESSOR IN THIS FAMILY SHOULD EMIT NEITHER ALIAS.
                       drop_full=S_["gain1_mean"] / nv_, keep_rate=S_["helps"] / nv_,
                       dom_mix=[int(d) for d in dom], lr=lr, gnorm=float(gnorm.cpu()), gnorm_h=float(gnorm_h.cpu()), dt=dt, tokens_seen=tokens_seen, world=world)
            rec = {k: (float(v_) if isinstance(v_, (np.floating, np.integer)) else v_) for k, v_ in rec.items()}
            vt.log(f"step {step:6d}/{args.max_steps} ntp1 {rec['ntp1']:.3f} | OWN: gain {rec['gain_mean']:+.2f} (k1 {rec['gain1_mean']:+.2f}) best {rec['gain_best']:+.2f} "
                   f"helps {100*rec['helps']:.0f}% sd {rec['r_std']:.2f} rdCE {rec['read_ce_own']:.2f} klref {rec['kl_ref']:.3f} len {rec['len_gen']:.1f} open {rec['open_share']:.2f} "
                   f"| HINT: gain {rec['hint_gain']:+.2f} (k1 {rec['hint_gain1']:+.2f}) helps {100*rec['hint_helps']:.0f}%/{100*rec['hint_helps1']:.0f}% imitCE {rec['imit_ce']:.2f} rdCE {rec['hint_read_ce']:.2f} "
                   f"| hRL own {rec['hrl_own_gain']:+.2f} beats {100*rec['hrl_beats']:.0f}% | gate {rec['gate_pre_bce']:.3f}/{rec['gate_post_bce']:.3f} acc {rec['gate_post_acc']:.2f} | gn {rec['gnorm']:.2f} {dt:.1f}s")
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
