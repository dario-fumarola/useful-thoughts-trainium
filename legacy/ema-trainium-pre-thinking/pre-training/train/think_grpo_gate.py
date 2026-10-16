"""GROUP-RELATIVE entropy-triggered thinking with a reader that learns what to ignore (Emanuele + owner, 2026-10-04).

Same base, data order and trigger as think_entropy.py (9B rung, the vanilla run's own unseen 3B, entropy > tau, the
--n-think most uncertain spots per sequence). At each spot the model writes a GROUP of --G thoughts (<= --L tokens, it
may close with </T>). Every thought is scored on the TRUE next token by the model itself: l = loss with the thought,
l_shuf = loss with the thought's own tokens shuffled, base = loss without. Reward r = (base - l) + (l_shuf - l):
"did it help" plus "did it help because of what it says". Advantages are group-normalised per spot.
  WRITER: REINFORCE on the thought tokens of the top-k and bottom-k thoughts of each group (negatives scaled by
          --neg-scale), weight --rl-w per token in the T-normalised loss. No imitation.
  READER: on the top-k thoughts with positive advantage, CE of the true token with the thought in context
          (--reader-w); on the bottom-k thoughts, a don't-be-fooled term: KL(clean prediction || prediction with the
          bad thought) (--fool-w) -- the reader learns to use good thoughts and to ignore bad ones.
  BASE:   clean next-token loss on all T tokens.
Invoked as `think_train.py --grpo 1 ...`. Gradients are all-reduced.

RUN 90 (owner, 2026-10-04 14:00), default OFF: `--frozen-reader 1` scores every thought (l, l_shuf, base) with a
FROZEN bf16 copy of the init instead of the live model -- the yardstick cannot move, so the reader can neither
over-trust (runs 86/87) nor go deaf (88RA, 89). With it `--reader-w 0 --fool-w 0` (the live model is not taught to
read; it still does clean NTP). `--kl-ref beta` adds the standard GRPO k3 KL-to-init on the sampled thought tokens
(exp(lp_ref-lp) - (lp_ref-lp) - 1), folded into the per-row REINFORCE weight as beta*(ratio-1): an anti-collapse
pressure (run 89 reached open_share 0.8 by step 140 with negatives alone).

RUN 92 (owner, 2026-10-05 04:07) -- THIS FILE IS A COPY OF think_grpo.py WITH ONE MECHANISM ADDED: ABSTENTION.
Reached by `think_train.py --grpo 2`; `--grpo 1` still runs think_grpo.py byte-identical (run 91 is executing that
file right now and run_think.sh re-execs it on a crash, so it must not be touched).
WHY: at run 91 step 570 the held-out probe says only 33% of thoughts improve the next token (`deploy_share_helps`
0.224 init -> 0.346/0.313/0.331) and the model has ZERO ability to tell which (`oracle_keep` 0.0469 vs `ctrl_keep`
0.0469, three consecutive n=48 reads). `deploy_drop` is the average over ALL thoughts, so it cannot cross zero
while two thirds are liabilities and deployment keeps every one. The lever is a learnable decision NOT to speak.
HOW (`--abstain-fill 1`): SPACE at thought position 0 IS the abstention action. Such a rollout's content is
replaced by a fixed filler (`--pause-frac-len` spaces then </T> padding), so it is SCORED by the same
`r = drop + cont` path as any other thought -- and because the filler is content-free, `cont == 0` exactly and
`drop` is just the small cost of an empty note, so an abstaining rollout sits at r ~ 0 inside its group while a
harmful thought sits at r < 0. The existing group normalisation therefore advantages silence exactly when the
alternative hurts. NOTHING about the reward is hand-designed for abstention -- that is the point.
THE ONE DELICATE PART (and it is NOT the shape, see GGeo.grad_buffer): an abstaining rollout must take policy
gradient at position 0 ONLY. Runs 32/33 trained the writer to emit the filler token by token and it lost its stop
action (G39); runs 13-17 died because shutting the gate entirely was the cheapest policy.
DERAIL registered before launch: `abst` (logged) > 0.9 for 100 consecutive steps = the degenerate optimum, stop.
Note that at full collapse every rollout in a group scores the identical filler reward, so r_sd -> 0, A -> 0 and
the gradient VANISHES: collapse here is a futility fixed point, not a model-destroying divergence.
READ THE LENGTH DERAIL ON `len_live`, NOT `len`: an abstaining row's length is --pause-frac-len (10), well under
the 32 run 91 generates, so `len_gen` falls with the abstention rate for an entirely benign reason. `len_live` is
the same mean over NON-abstaining rows only and is the one a collapse clause may read.
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
from data_onepass import Mixture as MixOP, MixtureBatcher as BatcherOP
from model_init import build_model, untie_lm_head, MODEL_ID
import thinking as th
import think_par as tp
from think_gen import ScatterCache
import think_entropy as te

T_OPEN, T_CLOSE, EOD, NEG = th.T_OPEN, th.T_CLOSE, th.EOD, th.NEG
SPACE = 220        # run 92: the abstention action at thought position 0 (same token as think_train.py:53)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--grpo", type=int, default=1)
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
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--G", type=int, default=16, help="thoughts per spot")
    p.add_argument("--topk", type=int, default=4, help="REINFORCE on the top-k and bottom-k of each group")
    # device memory: the first launch scored all G thoughts in one buffer (5504 rows) and OOM'd; score --score-group
    # thoughts per forward instead (buffer = T + N * score_group * (L+3) rows)
    p.add_argument("--score-group", type=int, default=4)
    p.add_argument("--rl-w", type=float, default=0.1)
    p.add_argument("--neg-scale", type=float, default=0.5)
    p.add_argument("--adv-clip", type=float, default=3.0)
    p.add_argument("--reader-w", type=float, default=1.0)
    p.add_argument("--fool-w", type=float, default=1.0)
    p.add_argument("--ntp-w", type=float, default=1.0)
    p.add_argument("--frozen-reader", type=int, default=0, help="score thoughts with a frozen bf16 copy of the init")
    p.add_argument("--kl-ref", type=float, default=0.0, help="k3 KL-to-init weight on the sampled thought tokens (needs --frozen-reader)")
    p.add_argument("--pad-klr", type=int, default=0, help="DEFAULT OFF. Pad the KL-to-init row gather to a FIXED "
                   "2*topk*n_think*(L+1) rows. len(tr) is data-dependent (it is 33 x the number of spots with a "
                   "non-zero advantage), and on XLA every new value recompiles the WHOLE ~230 MB training graph "
                   "(~25 min on this host, run 90 2026-10-04). The padding rows are dummy copies of tr[0] and both "
                   "their weights and their k3 are discarded, so the parameter update and the logged kl_ref are "
                   "bit-identical to the unpadded path.")
    p.add_argument("--abstain-fill", type=int, default=0, help="run 92: SPACE at thought position 0 is the "
                   "abstention action; the rollout's content becomes a fixed filler and it is scored by the "
                   "unchanged reward, which puts it at r ~ 0. Policy gradient at position 0 only.")
    p.add_argument("--abstain-explore", type=float, default=0.0, help="probability of FORCING position 0 to SPACE "
                   "in a rollout, so the group always contains priced silence even before the writer finds it. "
                   "All G rows are sampled in this trainer (there is no greedy row to protect).")
    p.add_argument("--pause-frac-len", type=int, default=10, help="spaces in the abstention filler, then </T>.")
    p.add_argument("--klr-skip-gate", type=int, default=0, help="run 94 (B275, registered 2026-10-05 10:07). "
                   "DEFAULT OFF, so --klr-skip-gate 0 is run 93's path byte-identical. ON: the KL-to-init "
                   "gradient weight is ZEROED on the position-0 row of ABSTAINING rollouts -- the abstention "
                   "decision itself -- and left untouched on every other weighted row (all thought-content "
                   "tokens, and position 0 of SPEAKING rollouts). WHY: grad_buffer clears an abstaining "
                   "rollout's REINFORCE weight everywhere except that one row, so `tr = where(w3 != 0)` puts the "
                   "k3 weight on the gate decision. The frozen init abstains 1.3% of the time (measured n=384, "
                   "2026-10-05), so there p_live >> p_ref and w_kl = kl_ref*(ratio-1) sits at 97.8% of its "
                   "saturated lower bound: a CONSTANT 41% tax on Aab that does not shrink as the policy "
                   "converges. A KL toward a reference that never takes an action is not a regulariser on that "
                   "action, it is a prior forbidding it, and letting the model take it is the whole programme. "
                   "NOTE: the gather, its shape and the LOGGED klref are deliberately unchanged -- only the "
                   "gradient weight is dropped -- so run 94's klref column stays comparable to run 93's.")
    p.add_argument("--abstain-free", type=int, default=0, help="run 93. DEFAULT OFF, so --abstain-free 0 is run "
                   "92's path byte-identical. ON: an abstaining rollout's reward is set to EXACTLY 0 instead of "
                   "the scored filler. WHY: --abstain-fill scores silence by actually splicing --pause-frac-len "
                   "spaces in front of the target, so r_abstain = (base - ce_filler) + 0. Run 92's config asserts "
                   "that is '~ 0'; the device says it is +0.075 / -1.067 / -1.155 at three dumped anchors, and at "
                   "DEPLOYMENT declining to emit a thought changes the loss by exactly 0. So the training action "
                   "and the deployment action differ, and the training one is the more expensive -- with the mean "
                   "real-thought reward at about -0.39, a filler costing -0.5 makes silence look WORSE than a "
                   "mediocre thought and a correct learner shuts the gate (run 92: p_pol 0.053 -> 0.032). With "
                   "r_abstain == 0 the group z-score asks exactly the gate's question: is this thought better "
                   "than saying nothing? The filler STAYS in the buffer (shape) and the position-0 weight mask is "
                   "untouched; only the reward changes. The logged drop_full/own_shuf still use the scored filler, "
                   "so they remain comparable with runs 91-92.")
    p.add_argument("--margin", type=float, default=1.0, help="for the logged keep-rate only (probe compatibility)")
    p.add_argument("--chunk", type=int, default=32, help="probe compatibility (one check at L)")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--logit-floor", type=float, default=12.0)
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
    args, unknown = p.parse_known_args()
    args.zero1 = False
    args.unknown = unknown
    assert 2 * args.topk <= args.G
    return args


class GGeo:
    """Spliced buffers for N spots x M thoughts. Score buffer: per (spot, thought) a block [<T>, L tokens], one
    closer </T> (after the thought's own length) and one target row (token x[n] at position n predicting x[n+1]).
    Grad buffer: per (spot, thought) a block [<T>, len tokens, </T>, pad] and one target row."""

    def __init__(self, T, N, M, L):
        self.T, self.N, self.M, self.L = T, N, M, L
        self.PA = (L + 1) + 1 + 1
        self.S = T + N * M * self.PA
        self.PG = (L + 2) + 1
        self.S_grad = T + N * M * self.PG

    def _base(self, x, S):
        T = self.T
        buf = np.full(S, T_CLOSE, dtype=np.int64); pos = np.zeros(S, dtype=np.float32)
        m = np.full((S, S), NEG, dtype=np.float32)
        buf[:T] = x; pos[:T] = np.arange(T)
        m[:T, :T] = np.triu(np.full((T, T), NEG, dtype=np.float32), k=1)
        tgt = np.zeros(S, dtype=np.int64); tgt[:T - 1] = x[1:]
        return buf, pos, m, tgt

    def score_buffer(self, x, anchors, Z, lens):
        """Z [M, N, L], lens [M, N]"""
        T, N, M, L = self.T, self.N, self.M, self.L
        buf, pos, m, tgt = self._base(x, self.S)
        slotpos = np.concatenate([[0.0], np.arange(L, dtype=np.float32)])
        for a in range(N):
            n = int(anchors[a])
            for g in range(M):
                B = T + (a * M + g) * self.PA
                k = max(int(lens[g, a]), 1)
                rows = np.arange(B, B + L + 1)
                buf[B] = T_OPEN; buf[B + 1:B + L + 1] = Z[g, a]
                pos[rows] = n + 1.0 + slotpos
                m[rows, :n + 1] = 0.0
                m[rows[:, None], rows[None, :]] = np.triu(np.full((L + 1, L + 1), NEG, dtype=np.float32), k=1)
                rc = B + L + 1; rt = B + L + 2
                buf[rc] = T_CLOSE; pos[rc] = n + 1.0 + k
                m[rc, :n + 1] = 0.0; m[rc, B:B + k + 1] = 0.0; m[rc, rc] = 0.0
                buf[rt] = x[n]; pos[rt] = n; tgt[rt] = x[n + 1]
                m[rt, :n] = 0.0; m[rt, B:B + k + 1] = 0.0; m[rt, rc] = 0.0; m[rt, rt] = 0.0
        return buf, pos, m, tgt

    def target_rows(self):
        a = np.arange(self.N)[:, None]; g = np.arange(self.M)[None, :]
        return (self.T + (a * self.M + g) * self.PA + self.L + 2)          # [N, M]

    def grad_buffer(self, x, y, anchors, Z, lens, w_tok, w_read, fool, ab=None):
        """Z [M, N, L] (selected thoughts), lens [M, N], w_tok [M, N] signed REINFORCE weight per thought,
        w_read [M, N] reader CE weight, fool [M, N] bool -> rows for the don't-be-fooled KL.
        run 92: `ab` [M, N] bool marks ABSTAINING rollouts. The buffer is UNCHANGED for them -- the filler already
        sits in Z and its length in lens -- but their REINFORCE weight is cleared everywhere except the single row
        that predicts thought position 0, the abstention decision itself.
        The shape is safe to do this in: S_grad = T + N*M*PG is fixed in __init__ from constants alone and `lens`
        only picks WHICH rows inside a fixed slot carry content, so nothing here is data-dependent and the ~230 MB
        XLA graph cannot churn (checked against G133/G135 before writing this). What is NOT safe is the tempting
        one-liner `lsel[ab] = 1`: with La == 1 the weighted span w3[G0:G0+La+1] is TWO rows, G0 (predict the SPACE)
        and G0+1 (predict the </T> that _base left there as padding), so it would train the stop action alongside
        the decision -- G39 by a side door, and silently, because the shape and the loss both stay finite."""
        T, N, M, L = self.T, self.N, self.M, self.L
        buf, pos, m, tgt = self._base(x, self.S_grad)
        tgt[:T] = y
        w3 = np.zeros(self.S_grad, dtype=np.float32); w2 = np.zeros(self.S_grad, dtype=np.float32)
        fool_rows, clean_rows, ab_gate_rows = [], [], []
        slotpos = np.concatenate([[0.0], np.arange(L + 1, dtype=np.float32)])
        for a in range(N):
            n = int(anchors[a])
            for g in range(M):
                G0 = T + (a * M + g) * self.PG
                La = max(int(lens[g, a]), 1)
                rows = np.arange(G0, G0 + L + 2)
                buf[G0] = T_OPEN; buf[G0 + 1:G0 + 1 + La] = Z[g, a, :La]
                pos[rows] = n + 1.0 + slotpos
                m[rows, :n + 1] = 0.0
                live = La + 2
                r_live = rows[:live]
                m[r_live[:, None], r_live[None, :]] = np.triu(np.full((live, live), NEG, dtype=np.float32), k=1)
                m[rows[live:], rows[live:]] = 0.0
                tgt[G0:G0 + La + 1] = buf[G0 + 1:G0 + La + 2]
                w3[G0:G0 + La + 1] = w_tok[g, a]
                if ab is not None and ab[g, a]:
                    w3[G0 + 1:G0 + La + 1] = 0.0          # position 0 only (see docstring)
                    ab_gate_rows.append(G0)               # --klr-skip-gate: the abstention DECISION row
                rt = G0 + L + 2
                buf[rt] = x[n]; pos[rt] = n; tgt[rt] = x[n + 1]
                m[rt, :n] = 0.0; m[rt, G0:G0 + La + 2] = 0.0; m[rt, rt] = 0.0
                w2[rt] = w_read[g, a]
                if fool[g, a]:
                    fool_rows.append(rt); clean_rows.append(n)
        return (buf, pos, m, tgt, w2, w3, np.asarray(fool_rows, dtype=np.int64),
                np.asarray(clean_rows, dtype=np.int64), np.asarray(ab_gate_rows, dtype=np.int64))


class GGen(te.ETGen):
    """ETGen with R rows (the group) sharing one prefill."""

    def __init__(self, trunk, lm_head, T, N, L, R, n_layers, device, autocast_ctx, sync_fn):
        super().__init__(trunk, lm_head, T, N, L, n_layers, device, autocast_ctx, sync_fn)
        self.R = R
        self.open_toks = torch.full((R, N), T_OPEN, dtype=torch.long, device=device)

    def _decode(self, args):
        N, L, R = self.N, self.L, self.R
        z = np.zeros((R, N, L), dtype=np.int64)
        stopped = np.zeros((R, N), dtype=bool)
        lens = np.full((R, N), L, dtype=np.int64)
        with torch.no_grad():
            self.dec.broadcast_from(self.pre, R)
            self.sync()
            for t in range(L):
                tok_t = self.open_toks if t == 0 else torch.from_numpy(np.ascontiguousarray(z[:, :, t - 1])).to(self.device)
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
                tok = torch.argmax(lg - torch.log(-torch.log(u)), -1).cpu().numpy()
                tok = np.where(stopped, T_CLOSE, tok)
                newly = (~stopped) & (tok == T_CLOSE)
                lens[newly] = t
                stopped |= newly
                z[:, :, t] = tok
                self.sync()
        return z, lens


def main():
    args = parse()
    torch.manual_seed(args.seed)
    device = torch_xla.device()
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank, world = xr.global_ordinal(), xr.world_size()
    xm.set_rng_state(int(args.seed) * 1000 + int(rank), device=device)
    T, N, L, G, K = args.seq_len, args.n_think, args.L, args.G, args.topk
    SG = args.score_group
    assert G % SG == 0
    geo = GGeo(T, N, SG, L)
    ggeo = GGeo(T, N, 2 * K, L)
    vt.log("=" * 72)
    vt.log(f"GRPO THINKING | world={world} T={T} GA={args.grad_accum} tau={args.tau} spots={N} G={G} topk={K} L={L} "
           f"rl_w={args.rl_w} reader_w={args.reader_w} fool_w={args.fool_w} | S_score={geo.S} S_grad={ggeo.S_grad}")
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
    reader_bb = copy.deepcopy(backbone) if args.frozen_reader else None     # the init, before any update
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True, "preserve_rng_state": False})
    model = backbone.to(device); model.train()
    for p_ in model.parameters():
        if p_.requires_grad:
            p_.grad = torch.zeros_like(p_)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    trunk, lm_head = backbone.model, backbone.lm_head
    if args.frozen_reader:
        reader_bb = reader_bb.to(torch.bfloat16).to(device).eval()
        for p_ in reader_bb.parameters():
            p_.requires_grad_(False)
        r_trunk, r_lm = reader_bb.model, reader_bb.lm_head
        if args.reader_w > 0 or args.fool_w > 0:
            vt.log(f"WARNING: frozen reader with reader_w={args.reader_w} fool_w={args.fool_w}: the live model is being "
                   f"taught to read while a frozen copy scores -- the two readers will diverge")
        vt.log("FROZEN READER: thoughts are scored (l, l_shuf, base) by a frozen bf16 copy of the init")
    else:
        r_trunk, r_lm = trunk, lm_head
        assert args.kl_ref == 0.0, "--kl-ref needs --frozen-reader 1 (the reference is the frozen copy)"
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

    gen = GGen(trunk, lm_head, T, N, L, G, cfg.num_hidden_layers, device, autocast_ctx, torch_xla.sync)
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
        with torch.no_grad():
            bt = torch.from_numpy(buf[None]).to(device); pt = torch.from_numpy(pos[None]).to(device)
            mt = torch.from_numpy(m[None, None]).to(torch.bfloat16).to(device); tt = torch.from_numpy(tgt).to(device)
            with autocast_ctx():
                h = r_trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
            ces = []
            step_ = (h.shape[1] + 15) // 16
            for i in range(0, h.shape[1], step_):
                with autocast_ctx():
                    lg = r_lm(h[0, i:i + step_])
                ces.append(-torch.log_softmax(lg.float(), -1).gather(1, tt[i:i + step_, None])[:, 0])
            ce = torch.cat(ces).cpu().numpy().astype(np.float64)
        torch_xla.sync()
        return ce

    t_start = time.time()
    for step in range(start_step, args.max_steps):
        lr = vt.lr_at(step, args); lr_host.fill_(lr); lr_tensor.copy_(lr_host)
        t0 = time.time()
        M = dict(n_micro=0.0, ntp1=0.0, n_cand=0.0, n_valid=0.0, n_kept=0.0, H_anc=0.0, base_anc=0.0, drop_full=0.0,
                 drop_best=0.0, drop_kept=0.0, own_shuf=0.0, r_std=0.0, len_kept=0.0, len_gen=0.0, rep_kept=0.0,
                 digit_share=0.0, open_share=0.0, read_ce=0.0, fool_kl=0.0, n_ctrl=0.0, ctrl_kept=0.0, ctrl_drop_full=0.0,
                 n_rce=0.0, n_fkl=0.0, n_kv=0.0, kl_ref=0.0, n_klr=0.0,
                 abst=0.0, len_live=0.0, n_live=0.0,
                 drop_ab=0.0, A_ab=0.0, n_ab=0.0, n_dom=np.zeros(3))
        for micro in range(args.grad_accum):
            gidx = args.block_offset + (step * args.grad_accum + micro) * world + rank
            row, di = one_pass_row(gidx)
            x, y = row[:T].copy(), row[1:T + 1].copy()
            for arr in (x, y):
                arr[(arr == T_OPEN) | (arr == T_CLOSE)] = 220
            x_dev = torch.from_numpy(x[None]).to(device)
            rng = np.random.default_rng([args.seed, rank, step, micro, 29])

            h0 = gen.prefill(x_dev); H = gen.entropy(h0); del h0
            cand = np.zeros(T, dtype=bool); cand[args.min_pos:T - 1] = True
            cand &= H > args.tau; cand[:-1] &= x[1:] != EOD; cand &= x != EOD
            idx = np.where(cand)[0]; n_cand = len(idx)
            if n_cand > N:
                idx = idx[np.argsort(-H[idx])[:N]]
            idx = np.sort(idx)
            valid = np.zeros(N, dtype=bool); anchors = np.full(N, args.min_pos, dtype=np.int64)
            anchors[:len(idx)] = idx; valid[:len(idx)] = True
            gen.set_anchors(anchors)
            Z, lens = gen.decode(args)                                   # [G, N, L], [G, N]

            # ---- run 92: ABSTENTION. SPACE at position 0 means "do not speak here". `ab` is read BEFORE the
            # substitution, because the filler itself starts with SPACE and would otherwise match every row on
            # the next pass (the think_train.py:925 bug note, carried over deliberately).
            ab = np.zeros((G, N), dtype=bool)
            if args.abstain_fill:
                if args.abstain_explore > 0:
                    frc = rng.random((G, N)) < args.abstain_explore
                    Z[:, :, 0] = np.where(frc, SPACE, Z[:, :, 0])
                ab = Z[:, :, 0] == SPACE
                FL = min(args.pause_frac_len, L - 1)
                fil_ab = np.full(L, T_CLOSE, dtype=Z.dtype); fil_ab[:FL] = SPACE
                Z = np.where(ab[:, :, None], fil_ab[None, None, :], Z)
                lens = np.where(ab, FL, lens)         # the filler's </T> sits at FL, so that IS its length

            # ---- score the group and its shuffles, --score-group thoughts per forward
            Zs = Z.copy()
            for g in range(G):
                for a in range(N):
                    la = max(int(lens[g, a]), 1); Zs[g, a, :la] = rng.permutation(Z[g, a, :la])
            lt = np.zeros((G, N)); lts = np.zeros((G, N)); base = None
            for g0 in range(0, G, SG):
                sl = slice(g0, g0 + SG)
                buf, pos, m, tgt = geo.score_buffer(x, anchors, Z[sl], lens[sl])
                ce = score_forward(buf, pos, m, tgt)
                lt[sl] = ce[trows_np].reshape(N, SG).T
                if base is None:
                    base = ce[anchors]                                  # [N] clean loss on x[n+1]
                bufs, poss, ms, tgts = geo.score_buffer(x, anchors, Zs[sl], lens[sl])
                lts[sl] = score_forward(bufs, poss, ms, tgts)[trows_np].reshape(N, SG).T
            drop = base[None, :] - lt                                    # [G, N] help
            cont = lts - lt                                              # [G, N] content
            r = drop + cont
            if args.abstain_fill and args.abstain_free:
                r = np.where(ab, 0.0, r)      # run 93: silence is scored as TRUE silence, not as the filler
            mu = r.mean(0, keepdims=True); sd = r.std(0, keepdims=True)
            A = np.where(sd > 1e-6, (r - mu) / (sd + 1e-6), 0.0)
            A = np.clip(A, -args.adv_clip, args.adv_clip) * valid[None, :]
            order = np.argsort(-A, axis=0)                               # best first per spot
            sel = np.concatenate([order[:K], order[-K:]])                # [2K, N]
            ar = np.arange(N)
            Zsel = np.stack([Z[sel[i], ar] for i in range(2 * K)])       # [2K, N, L]
            lsel = np.stack([lens[sel[i], ar] for i in range(2 * K)])
            absel = np.stack([ab[sel[i], ar] for i in range(2 * K)])
            Asel = np.stack([A[sel[i], ar] for i in range(2 * K)])
            w_tok = np.where(Asel < 0, Asel * args.neg_scale, Asel) * args.rl_w * valid[None, :]
            good = (Asel > 0) & valid[None, :]
            bad = (Asel < 0) & valid[None, :]
            w_read = good.astype(np.float32) * args.reader_w
            gbuf, gpos, gm, gtgt, w2, w3, fool_rows, clean_rows, ab_gate_rows = ggeo.grad_buffer(
                x, y, anchors, Zsel, lsel, w_tok, w_read, bad, ab=absel if args.abstain_fill else None)
            w1 = np.zeros(ggeo.S_grad, dtype=np.float32); w1[:T] = args.ntp_w
            bt = torch.from_numpy(gbuf[None]).to(device); pt = torch.from_numpy(gpos[None]).to(device)
            mt = torch.from_numpy(gm[None, None]).to(torch.bfloat16).to(device); tt = torch.from_numpy(gtgt[None]).to(device)
            with autocast_ctx():
                h = trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
            # k3 KL-to-init on the sampled thought tokens, as a per-row weight: d/dtheta beta*k3 = beta*(ratio-1)*dCE,
            # ratio = p_ref/p_live = exp(CE_live - CE_ref). Pushes the writer back toward the init where it drifted.
            kl_ref_v = None
            tr = np.where(w3 != 0)[0]
            if args.kl_ref > 0 and len(tr) > 0:
                n_tr = len(tr)
                if args.pad_klr:                      # fixed shape -> one XLA graph instead of one per n_tr
                    P = 2 * args.topk * args.n_think * (args.L + 1)
                    assert n_tr <= P, f"pad_klr: {n_tr} weighted thought rows exceed the structural max {P}"
                    trp = np.full(P, tr[0], dtype=np.int64); trp[:n_tr] = tr
                else:
                    trp = tr
                trd = torch.from_numpy(trp).to(device)
                with torch.no_grad():
                    with autocast_ctx():
                        hr = r_trunk(input_ids=bt, position_ids=pt, attention_mask=mt, use_cache=False)[0]
                        lg_r = r_lm(hr[0, trd]); lg_l = lm_head(h.detach()[0, trd])
                    ce_r = -torch.log_softmax(lg_r.float(), -1).gather(1, tt[0, trd, None])[:, 0]
                    ce_l = -torch.log_softmax(lg_l.float(), -1).gather(1, tt[0, trd, None])[:, 0]
                    dlt = (ce_l - ce_r).clamp(-5.0, 5.0)
                    ratio = dlt.exp()
                    k3 = ratio - dlt - 1.0
                    w_kl = (args.kl_ref * (ratio - 1.0)).cpu().numpy()[:n_tr]      # drop the padding rows
                    if args.klr_skip_gate and len(ab_gate_rows) > 0:
                        # B275. Zero the weight on the abstention-decision rows ONLY. Done on the HOST after the
                        # device work, so the gather, its padded shape, the XLA graph and `kl_ref_v` below are all
                        # bit-identical to --klr-skip-gate 0; the only thing that changes is w3. No new graph.
                        w_kl = np.where(np.isin(tr, ab_gate_rows), 0.0, w_kl).astype(np.float32)
                    # G135 (2026-10-04 20:07 tick): `k3[:n_tr]` is a DEVICE slice whose LENGTH is n_tr, so it
                    # is one XLA graph per distinct n_tr -- exactly the churn --pad-klr was added to kill,
                    # surviving in the METRIC after the gradient gather was fixed. Run 91 paid 26 s -> 68 s/step
                    # for this one line. k3 has the FIXED padded length, so take the mean on the host.
                    kl_ref_v = float(k3.cpu().numpy()[:n_tr].mean())
                    del hr, lg_r, lg_l
                w3[tr] = w3[tr] + w_kl.astype(np.float32)
            loss, ce_det = tp.weighted_ce3(h, lm_head, tt, torch.from_numpy(w1[None]).to(device),
                                           torch.from_numpy(w2[None]).to(device), torch.from_numpy(w3[None]).to(device),
                                           n1_dev, n1_dev, n1_dev, args.loss_chunks, autocast_ctx)
            # don't-be-fooled: KL(clean || with bad thought) at the bad thoughts' target rows (clean side detached)
            if len(fool_rows) > 0 and args.fool_w > 0:
                fr = torch.from_numpy(fool_rows).to(device); cr = torch.from_numpy(clean_rows).to(device)
                with autocast_ctx():
                    lg_f = lm_head(h[0, fr]); lg_c = lm_head(h[0, cr])
                lp_f = torch.log_softmax(lg_f.float(), -1); lp_c = torch.log_softmax(lg_c.float(), -1).detach()
                kl = (lp_c.exp() * (lp_c - lp_f)).sum(-1)
                loss = loss + args.fool_w * kl.sum() / float(T)
                fool_kl = float(kl.mean().detach().cpu())
            else:
                fool_kl = None      # no bad thoughts this micro: the term does not exist (G129 -- do NOT average a 0 in)
            (loss / args.grad_accum).backward()
            ce_g = ce_det.cpu().numpy()
            torch_xla.sync()

            # ---- stats
            v = valid
            M["n_micro"] += 1.0; M["ntp1"] += float(ce_g[:T].mean()); M["n_cand"] += float(n_cand)
            M["n_valid"] += float(v.sum()); M["n_dom"][di] += 1
            if fool_kl is not None:
                M["fool_kl"] += fool_kl; M["n_fkl"] += 1.0
            if kl_ref_v is not None:
                M["kl_ref"] += kl_ref_v; M["n_klr"] += 1.0
            if v.any():
                M["H_anc"] += float(H[anchors][v].sum()); M["base_anc"] += float(base[v].sum())
                M["drop_full"] += float(drop[:, v].mean(0).sum()); M["drop_best"] += float(drop[:, v].max(0).sum())
                M["own_shuf"] += float(cont[:, v].mean(0).sum()); M["r_std"] += float(r[:, v].std(0).sum())
                M["len_gen"] += float(lens[:, v].mean(0).sum())
                # run 92: `abst` is the REGISTERED collapse detector (> 0.9 for 100 steps = stop). `len_live` is
                # len_gen over NON-abstaining rows: len_gen itself falls with the abstention rate for a benign
                # reason (the filler is --pause-frac-len long), so a length derail must read len_live.
                M["abst"] += float(ab[:, v].mean(0).sum())
                # run 92/93: the mispricing and the gate's learning signal, both free and both the numbers the
                # 06:07 tick had to infer from 3 dumped anchors. drop_ab is what the filler ACTUALLY costs (it
                # stays on the scored filler even under --abstain-free, which is the point: it is the mispricing);
                # A_ab is the mean advantage an abstaining rollout receives, i.e. whether the gate has a gradient.
                abv = ab[:, v]
                if abv.any():
                    M["drop_ab"] += float(drop[:, v][abv].mean()) * float(v.sum())
                    M["A_ab"] += float(A[:, v][abv].mean()) * float(v.sum())
                    M["n_ab"] += float(v.sum())
                liv = (~ab[:, v])
                if liv.any():
                    M["len_live"] += float(lens[:, v][liv].mean()) * float(v.sum()); M["n_live"] += float(v.sum())
                keptm = (drop >= args.margin) & (cont >= args.margin) & v[None, :]
                M["n_kept"] += float(keptm.sum()) / G
                if keptm.any():
                    M["n_kv"] += float(v.sum())      # the weight these three means are actually taken over (G129)
                    M["drop_kept"] += float(drop[keptm].mean()) * float(v.sum())
                    M["len_kept"] += float(lens[keptm].mean()) * float(v.sum())
                    M["rep_kept"] += float(np.mean([te.rep_share(Z[g, a], max(int(lens[g, a]), 1)) for g, a in zip(*np.where(keptm))])) * float(v.sum())
                dg = [((Z[0, a, :max(int(lens[0, a]), 1)] >= 15) & (Z[0, a, :max(int(lens[0, a]), 1)] <= 24)).mean() for a in np.where(v)[0]]
                M["digit_share"] += float(np.mean(dg)) * float(v.sum())
                ops = [tuple(Z[0, a, :3].tolist()) for a in np.where(v)[0]]
                M["open_share"] += float(max(ops.count(o) for o in set(ops)) / len(ops)) if len(ops) > 1 else 0.0
                M["n_ctrl"] += float(v.sum()) * G
                M["ctrl_kept"] += float(((base[None, :] - lts) >= args.margin)[:, v].sum())
                M["ctrl_drop_full"] += float((base[None, :] - lts)[:, v].mean(0).sum())
                rr = w2 > 0
                if rr.any():
                    M["read_ce"] += float(ce_g[rr].mean()); M["n_rce"] += 1.0
            if dump_f and tok is not None and step % args.dump_every == 0 and micro == 0 and v.any():
                a0 = int(np.where(v)[0][len(np.where(v)[0]) // 2]); n0 = int(anchors[a0])
                gb, gw = int(order[0, a0]), int(order[-1, a0])
                dump_f.write(json.dumps(dict(step=step, anchor=n0, H=float(H[n0]), base=float(base[a0]),
                    context=tok.decode(x[max(0, n0 - 50):n0 + 1].tolist()), next_token=tok.decode([int(x[n0 + 1])]),
                    best=tok.decode(Z[gb, a0, :max(int(lens[gb, a0]), 1)].tolist()), best_drop=float(drop[gb, a0]), best_cont=float(cont[gb, a0]),
                    worst=tok.decode(Z[gw, a0, :max(int(lens[gw, a0]), 1)].tolist()), worst_drop=float(drop[gw, a0]), worst_cont=float(cont[gw, a0]),
                    group_drop_mean=float(drop[:, a0].mean()), group_r_std=float(r[:, a0].std()))) + "\n")
                dump_f.flush()

        if world > 1:
            tp.sync_gradients(params, world, xm, torch_xla.sync)
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
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
        nm = max(S_["n_micro"], 1.0); nv = max(S_["n_valid"], 1.0); nc_ = max(S_["n_ctrl"], 1.0)
        # the kept-subset means (drop_kept, len_kept, rep_kept) are accumulated ONLY on micros that kept
        # something, each weighted by that micro's spot count -- so they must be divided by that same weight,
        # not by every spot. Dividing by nv made them true_value x keep-micro-fraction, i.e. keep_rate in
        # disguise, which is exactly how run 88's read_ce lied this morning (G129).
        nkv = max(S_["n_kv"], 1.0)
        tokens_seen += args.grad_accum * T * world
        dt = time.time() - t0
        if step % args.log_every == 0:
            rec = dict(step=step, ntp1=S_["ntp1"] / nm, n_cand=S_["n_cand"] / nm, n_valid=S_["n_valid"] / nm,
                       keep_rate=S_["n_kept"] / nv, ctrl_rate=S_["ctrl_kept"] / nc_, H_anc=S_["H_anc"] / nv,
                       base_anc=S_["base_anc"] / nv, drop_full=S_["drop_full"] / nv, drop_best=S_["drop_best"] / nv,
                       own_shuf=S_["own_shuf"] / nv, r_std=S_["r_std"] / nv, drop_kept=S_["drop_kept"] / nkv,
                       abst=S_["abst"] / nv, len_live=S_["len_live"] / max(S_["n_live"], 1.0), n_live=S_["n_live"],
                       drop_ab=S_["drop_ab"] / max(S_["n_ab"], 1.0), A_ab=S_["A_ab"] / max(S_["n_ab"], 1.0), n_ab=S_["n_ab"],
                       len_kept=S_["len_kept"] / nkv, len_gen=S_["len_gen"] / nv, rep_kept=S_["rep_kept"] / nkv,
                       digit_share=S_["digit_share"] / nv, open_share=S_["open_share"] / nm,
                       read_ce=S_["read_ce"] / max(S_["n_rce"], 1.0), fool_kl=S_["fool_kl"] / max(S_["n_fkl"], 1.0),
                       n_rce=S_["n_rce"], n_fkl=S_["n_fkl"], n_kv=S_["n_kv"], ctrl_drop_full=S_["ctrl_drop_full"] / nv,
                       kl_ref=S_["kl_ref"] / max(S_["n_klr"], 1.0),
                       # 2026-10-07 10:07: RAW denominators (sums over this log window) so a report can tell
                       # "the mean is 0.000" from "the mean was never computed".  Every average here is
                       # `S_["x"] / max(S_["n_x"], 1.0)`, which turns an empty window into a measured 0.0:
                       # run 94 logged read_ce == fool_kl == 0.000 on all 3,554 rows with n_rce == 0, and
                       # every report called it "reader CE 0.000".  Additive only.
                       n_klr=S_["n_klr"], n_ctrl=S_["n_ctrl"], n_micro=S_["n_micro"],
                       dom_mix=[int(d) for d in dom], lr=lr, gnorm=float(gnorm.cpu()), dt=dt, tokens_seen=tokens_seen, world=world)
            rec = {k: (float(v_) if isinstance(v_, (np.floating, np.integer)) else v_) for k, v_ in rec.items()}
            vt.log(f"step {step:6d}/{args.max_steps} ntp1 {rec['ntp1']:.3f} | cand {rec['n_cand']:.0f} spots {rec['n_valid']:.1f} | "
                   f"H {rec['H_anc']:.2f} base {rec['base_anc']:.2f} | group: drop {rec['drop_full']:+.2f} best {rec['drop_best']:+.2f} "
                   f"own-shuf {rec['own_shuf']:+.2f} r_sd {rec['r_std']:.2f} | kept(rule) {100*rec['keep_rate']:.1f}% drop {rec['drop_kept']:+.2f} "
                   f"| len {rec['len_gen']:.1f} live {rec['len_live']:.1f} abst {rec['abst']:.3f} "
                   f"dAb {rec['drop_ab']:+.2f} Aab {rec['A_ab']:+.2f} "
                   f"dig {rec['digit_share']:.2f} open {rec['open_share']:.2f} | ctrl {100*rec['ctrl_rate']:.1f}% "
                   f"| rdCE {rec['read_ce']:.2f} foolKL {rec['fool_kl']:.3f} klref {rec['kl_ref']:.3f} | gn {rec['gnorm']:.2f} {dt:.1f}s")
            if metrics_f:
                metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
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
