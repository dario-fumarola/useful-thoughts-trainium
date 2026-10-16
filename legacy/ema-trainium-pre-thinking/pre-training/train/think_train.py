"""Thinking-in-pretraining trainer (runs 18+): mixed anchors, forced thinking.

Every sequence carries N thought anchors at a fixed stride. At each anchor the
model emits <T> ... </T>; the thought is spliced at overlapping integer
positions and kept apart from the text by attention masks, so one forward
pass scores the true continuation both WITH the thought (window rows) and
WITHOUT it (clean rows) -- the silence baseline comes from the same buffer,
no referee model. A thought's gain G is the discounted log-likelihood
improvement of the next k true tokens.

Anchors are MIXED inside one sequence:
  SSL  (ssl_frac)   the thought is the true continuation, teacher-forced with
                    a jittered length: imitation of the real future is the
                    permanent anchor of the writer (never decays, never
                    handed over).
  RL   (the rest)   thinking is FORCED (no learned gate): R sampled rollouts
                    are scored on raw gain, the best is spliced for the grad
                    pass, and REINFORCE rides the thought rows as signed CE
                    weights with a leave-one-out paired baseline over the
                    R rollouts. A token-shuffled copy of the winner is scored
                    as a MONITOR only (order-borne value = g - gShuf).

Geometry (masks, positions, row indices) is a per-run constant, so there is
one grad graph, one score graph, one prefill and one decode graph. Parameters
are fp32 by default (bf16 parameters freeze every RMSNorm gain, gotcha G11);
compute is bf16 under autocast.
"""

import argparse
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
from data import Mixture, MixtureBatcher
from model_init import build_model, untie_lm_head, MODEL_ID
import thinking as th
import think_par as tp
import synth as sy

SPACE = 220


def sched(step, v0, v1, t0, t1):
    if step <= t0:
        return v0
    if step >= t1:
        return v1
    return v0 + (step - t0) / max(1, t1 - t0) * (v1 - v0)


def main():
    if "--grpo" in sys.argv:
        # run 92 (2026-10-05): `--grpo 2` reaches think_grpo_gate.py, a COPY of think_grpo.py plus abstention.
        # `--grpo 1` still reaches think_grpo.py unchanged -- run 91 is executing that file and run_think.sh
        # re-execs it on a crash, so a half-finished RL mechanism must never land in its recovery path.
        gi = sys.argv.index("--grpo")
        gv = sys.argv[gi + 1] if gi + 1 < len(sys.argv) else "1"
        if gv == "2":
            import think_grpo_gate
            return think_grpo_gate.main()
        import think_grpo
        return think_grpo.main()
    if "--hintmix" in sys.argv:
        # Emanuele 2026-10-06 15:10 / go 2026-10-07: 50/50 paraphrase-hint mixture; separate trainer.
        import think_hintmix
        return think_hintmix.main()
    if "--bestof" in sys.argv:
        # Emanuele 2026-10-06 15:30: massive-group best-of-G, no oracle at eval; separate trainer.
        import think_bestof
        return think_bestof.main()
    if "--oracle" in sys.argv:
        # the ceiling line (Emanuele, 2026-10-06): oracle-gated span thinking + natural gate heads, separate trainer.
        import think_oracle
        return think_oracle.main()
    if "--etrig" in sys.argv:
        # Emanuele's entropy-triggered design (2026-10-03): separate trainer, same dispatch trick.
        import think_entropy
        return think_entropy.main()
    if "--hindsight" in sys.argv:
        # Emanuele's two-pass idea (2026-10-02): a separate trainer, dispatched from here so every launcher,
        # watchdog and worker count (they all match "think_train.py --corpus") keeps working unchanged.
        import think_hindsight
        return think_hindsight.main()
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "corpus"))
    p.add_argument("--init-from", default=None)
    p.add_argument("--resume", default=None)
    # run 73 (the hold): --lr-horizon fixes the cosine length independently of
    # --max-steps (train.lr_at reads it), so a resumed run can be extended without
    # the LR jumping back up; --resume-model-only restores weights but not Adam
    # state (train.load_checkpoint). Both default off = unchanged behaviour.
    p.add_argument("--lr-horizon", type=int, default=None)
    # GRADIENT SYNC (bug found 2026-10-02): since the run-13 rewrite this trainer
    # called opt.step() with NO all-reduce, so each of the 64 ranks trained its own
    # copy on its own 2 sequences/step and rank 0's copy was the one saved. 1 =
    # average gradients across ranks (xm.reduce_gradients) and clip to --grad-clip,
    # as train.py does. 0 = the old (unsynced) behaviour, kept so in-flight runs
    # that restart do not change semantics mid-run.
    p.add_argument("--sync-grads", type=int, default=0)
    p.add_argument("--resume-model-only", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--max-hours", type=float, default=None)
    # geometry
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--first-anchor", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.97)
    # schedules
    p.add_argument("--p-or-start", type=float, default=0.9)
    p.add_argument("--p-or-floor", type=float, default=0.2)
    p.add_argument("--p-or-t0", type=int, default=300)
    p.add_argument("--p-or-t1", type=int, default=1200)
    p.add_argument("--imit-w", type=float, default=1.0)
    p.add_argument("--imit-floor", type=float, default=0.3)
    p.add_argument("--rl-w", type=float, default=1.0)
    p.add_argument("--rl-start", type=int, default=400)
    p.add_argument("--rl-ramp", type=int, default=300)
    p.add_argument("--distill-w", type=float, default=0.3,
                   help="reader window CE weight (both micro kinds)")
    # gates (runs 10-12 lineage)
    p.add_argument("--distill-eps", type=float, default=1.0)
    p.add_argument("--shuffle-margin", type=float, default=1.0)
    p.add_argument("--register-floor", type=float, default=0.5)
    p.add_argument("--reward-clip", type=float, default=3.0)
    # G96 (2026-09-30): `A` below is (adv - mu) / sd(adv), so the advantage is
    # SCALE-INVARIANT -- mean|A| is ~0.8 whether the true between-rollout spread
    # is 2.0 nats or 0.006. REINFORCE therefore pushes just as hard on pure noise
    # as on real signal, and cannot stand down when the reward goes silent (run
    # 65: g_sd 2.18 -> 0.06 while the writer collapsed onto one string). With a
    # floor f the divisor is max(sd, f), so once the real spread falls below f
    # the advantage shrinks with it and the RL term fades out on its own.
    # 0.0 = the old, scale-invariant behaviour.
    p.add_argument("--adv-sig-floor", type=float, default=0.0,
                   help="floor on the advantage z-score divisor, in nats; "
                        "0 = scale-invariant (pre-G96 behaviour)")
    # G100 (2026-09-30, run 66 post-mortem): the reward is own-vs-silence, whose
    # between-rollout differences reward ANY note that helps this anchor --
    # including notes that help generically (format, register, fluency). Run 66
    # raised `own` and `shuf` by the SAME ~+3.7 nats over 1,800 steps and left
    # `own_minus_shuf` -- the anchor-specific part, and the programme's bar --
    # flat at every one of eight offsets. With w>0 the reward becomes
    #     G[r] - w * (gain of the SAME note placed one anchor over)
    # which is exactly probe_span.py's `own_minus_shuf` (its `shuf` column is
    # np.roll(own, 1, axis=0), the MISPLACED-note control). w=1.0 makes the
    # objective the ruler. Costs RO extra scoring passes per step.
    # 0.0 = bit-identical to the pre-G100 path (no extra passes at all).
    p.add_argument("--xanchor-w", type=float, default=0.0,
                   help="weight on the misplaced-note (cross-anchor) control "
                        "inside the REINFORCE reward; 0 = off, 1 = the reward "
                        "IS own_minus_shuf")
    # sampling + compression (model-controlled length)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--freq-pen", type=float, default=1.0)
    p.add_argument("--top-k-thought", type=int, default=512)
    p.add_argument("--topk-transfer", type=int, default=1)
    p.add_argument("--len-cost", type=float, default=0.15,
                   help="nats per thought token: REINFORCE reward = gain - "
                        "len_cost*len; the compression pressure")
    p.add_argument("--min-think", type=int, default=2,
                   help="</T> banned before this many content tokens")
    p.add_argument("--jit-min", type=int, default=3,
                   help="min length of jittered oracle imitation targets")
    # ---- scheduled sampling / DAgger inside the <T> slot (run 30) ----------
    # `probe_tfslot` (run 25 @900, n=616) showed the slot forecasts at 0.347
    # over positions 1-11 when teacher-forced and at 0.019 running free, while
    # break-even against the filler floor is 0.167.  The capability is there;
    # only the decoding regime destroys it.  And --imit-w fills the block with
    # the TRUE continuation, so every run so far trained pure teacher forcing
    # and never the regime it samples in.  These flags substitute, per SSL
    # anchor and per position, the model's OWN sampled token into the block
    # INPUT while the CE TARGET stays the true continuation (DAgger labels).
    # Defaults keep the old behaviour exactly (probability 0 everywhere).
    p.add_argument("--ss-start", type=float, default=0.0,
                   help="per-position prob of feeding the model's own token "
                        "into the SSL block at step <= --ss-t0")
    p.add_argument("--ss-end", type=float, default=0.0,
                   help="same, at step >= --ss-t1; 0 disables the feature")
    p.add_argument("--ss-t0", type=int, default=0)
    p.add_argument("--ss-t1", type=int, default=600)
    p.add_argument("--ss-row", choices=["sample", "spec"], default="sample",
                   help="which rollout supplies the own tokens: an unselected "
                        "sample (row 0) or the greedy speculation row")
    # the learnable WHEN-to-think gate (user directive 2026-09-04)
    p.add_argument("--gate-w", type=float, default=1.0,
                   help="weight of the gate REINFORCE term (<T> logit at "
                        "anchor rows is the learned trigger)")
    p.add_argument("--gate-thresh", type=float, default=1e-3,
                   help="sampling permissiveness: think with prob "
                        "eps + (1-eps)*min(1, p(<T>)/thresh)")
    p.add_argument("--think-cost", type=float, default=0.5,
                   help="fixed nats price of opening a thought at all")
    p.add_argument("--rate-cost", type=float, default=0.25,
                   help="nats per already-active thought in the preceding "
                        "crowd window (the too-many-thoughts tax)")
    p.add_argument("--crowd-win", type=int, default=8,
                   help="anchors looked back for the rate cost")
    p.add_argument("--eps-think-start", type=float, default=0.5)
    p.add_argument("--eps-think-floor", type=float, default=0.05)
    p.add_argument("--neg-scale", type=float, default=1.0,
                   help="scale on NEGATIVE advantages (content and gate): "
                        "<1 damps the unlikelihood gradients that eroded "
                        "ntp1 at the step-430 handover (drift +0.49)")
    # optim (train.py conventions)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # ZeRO-1 shards the optimizer state; its reduce_scatter HANGS at 64
    # ranks (measured; fine at <= 32) and with fp32 parameters plain AdamW
    # fits the core anyway (0.6B x 16 B = 9.6 GB of 24 GB), so it is off.
    p.add_argument("--zero1", action=argparse.BooleanOptionalAction,
                   default=False)
    # R sampled rollouts per anchor (+1 greedy speculation row, always
    # decoded for the len_spec diagnostic). R=1 reproduces runs 13-15
    # exactly. Cost is dominated by the [R+1, N, V] full-logit transfer, so
    # R and the anchor count N trade against each other (k=64 drops N
    # 118 -> 14, which is what pays for R=8).
    p.add_argument("--rollouts", type=int, default=1)
    # fraction of anchors that are self-supervised (true continuation,
    # teacher-forced). The rest are reinforcement anchors where thinking is
    # FORCED -- no learned gate, so an inert thought costs its token price.
    p.add_argument("--ssl-frac", type=float, default=0.25)
    # WHERE the supervised (oracle) thought comes from. "beyond" = the L
    # tokens AFTER the scored window (runs 13-23): a far-future target the
    # writer cannot learn (imitation CE floors at 3.8) and the reader cannot
    # use (oracle gain never positive at k=64). "jitter" = a random L-token
    # span INSIDE the window, offset o ~ U{0..k-L} per anchor: a learnable
    # sketch of what is coming that the reader can exploit. "fixed" = span at
    # --oracle-off. Probe: pre-training/train/probe_oracle_offset.py.
    p.add_argument("--oracle-mode",
                   choices=("beyond", "jitter", "fixed", "bridge", "span"),
                   default="beyond")
    p.add_argument("--oracle-off", type=int, default=0)
    p.add_argument("--oracle-max-off", type=int, default=-1,
                   help="jitter: upper bound of the offset (default k-L)")
    # READER CALIBRATION (run 25). The reader was trained on a mixture it never
    # meets at inference: 25% of hints are the TRUE future (+7 nats) and every
    # self-written hint is the BEST of R. It learned to trust hints; at
    # inference a single unselected own thought then costs -2.2 nats.
    #   --reader-on-ssl 0   : no reader loss on supervised anchors (the writer
    #                         still imitates there)
    #   --reader-on-ssl 2   : (run 55, POSTMORTEM G66/G67) reader window gradient on
    #                         SUPERVISED anchors ONLY. Run 54 widened the sampler and
    #                         the reader collapsed with it, because --rl-content random
    #                         splices a sampled rollout into the rows the reader is
    #                         trained on: a reader shown confident wrong text learns to
    #                         discount the slot. Mode 2 insulates the reader from what
    #                         the writer explores while leaving the reward path alone.
    #   --rl-content random : RL anchors splice a RANDOM rollout, not the winner;
    #                         the reader sees exactly the inference distribution
    #                         and REINFORCE on it is unbiased (mean-zero paired
    #                         advantage) instead of winner-only.
    #   --score-spec 1      : score the greedy row = inference-mode greedy thinking
    p.add_argument("--reader-on-ssl", type=int, default=1)
    p.add_argument("--rl-content", choices=("winner", "random"), default="winner")
    p.add_argument("--score-spec", type=int, default=0)
    # REINFORCE baseline: "loo" = mean of the other R-1 rollouts (default);
    # "greedy" = the greedy thought's reward, i.e. what the model would emit at
    # inference -- a sample is reinforced only if it beats the model's own
    # inference-time thought (requires --score-spec 1).
    p.add_argument("--baseline", choices=("loo", "greedy"), default="loo")
    # PAUSE-TOKEN CONTROL (Goyal et al.): --pause-len > 0 fills every thought
    # slot with content-free filler instead of a written thought -- no decode,
    # no rollouts, no writer loss, reader trained on the same window rows.
    # It measures the value of the EXTRA POSITIONS alone, which is the floor any
    # real thought must beat, and it is ~3x faster per step than a thinking run.
    p.add_argument("--pause-len", type=int, default=0)
    # FRACTIONAL FILLER (run 33). --pause-len is all-or-nothing: it hard-zeroes
    # imit and rl, so there is no writer left at all. --pause-frac puts the SAME
    # content-free filler on a FRACTION of anchors and leaves the writer running
    # on the rest. Those anchors are removed from both the ssl and the rl mask
    # (so no imitation and no REINFORCE weight reaches their block rows) but are
    # ADDED to the reader mask, which is precisely the regime run 26 had at
    # fraction 1.0. WHY: own = filler + (own - filler), and on run 32 @1000 that
    # is -0.899 = -0.769 + -0.130 at m=4 -- 85% of the deficit is the price of
    # inserting the block, not what the thought says. Run 26 @1500 pays
    # -0.083 +- 0.021 for its block on held-out where run 32 @1000 pays
    # -0.926 +- 0.095 (same instrument, same 465 anchors, --pause 10), and run
    # 26's reader is MORE oracle-sensitive, not less (o0 +12.62 +- 0.42 against
    # run 25 @900's +10.72), so a cheap block is not a deaf one.
    # Default 0.0 reproduces runs 13-32 exactly and draws no rng.
    p.add_argument("--pause-frac", type=float, default=0.0)
    p.add_argument("--pause-frac-len", type=int, default=10)
    # ABSTENTION / pmix SUPERVISION (run 32). The supervised thought keeps the
    # TRUE token only at positions whose true token is in the top-N unigram list
    # and emits content-free filler everywhere else. WHY this exact mask: the
    # abstention ladder (runs/probes/abst_r25_900.json) showed the INTERLEAVED
    # pause format is readable and drops break-even from ~4/12 to 2/12 correct
    # tokens, and every run since 28 has shown that a confident WRONG token is
    # more expensive than a blank slot. The top-N set is where a 9B model's
    # next-token accuracy actually lives, and it is model-free, so the mask costs
    # no extra forward and applies identically to the block INPUT and the CE
    # TARGET (both derive from orc_content), which is the train/inference
    # regime-match that DAgger (run 30) tried to buy with an extra pass and lost.
    # Default "none" reproduces runs 13-31 exactly.
    p.add_argument("--abstain-mode", choices=("none", "freq"), default="none")
    p.add_argument("--abstain-top", type=int, default=300)
    # RUN 34.  Imitation weight on the positions the abstention mask BLANKED.
    # At 1.0 (default) the writer is trained to EMIT SPACE there, which is what
    # runs 32/33 did -- and their greedy writer then went mute: every greedy
    # thought in thinkmode_r32_1000.json / _r33_400.json is SPACE*n + </T>, i.e.
    # byte-identical in kind to the probe's own --pause filler (G39).  At 0.0 the
    # blank stays in the block INPUT (the pmix format is unchanged) but carries
    # no CE target, so silence is no longer a trained action and the only
    # supervision the writer gets is "be right where the true token is
    # reachable".  Any value in [0,1] is legal; 1.0 reproduces runs 32-33 exactly.
    p.add_argument("--abstain-loss-w", type=float, default=1.0)
    p.add_argument("--abstain-file",
                   default="/home/ubuntu/ema-trainium/pre-training/train/"
                           "stop_tokens_top300.npy")
    # ---------------------------------------------------------------------
    # RUN 74.  A PRICED ONE-TOKEN ABSTENTION -- the OPPOSITE delivery of run 32.
    # Run 32 rewrote the CE TARGET (`--abstain-mode freq`), so "emit SPACE" became
    # a SUPERVISED action over ~45% of positions and the writer went mute: every
    # greedy thought on that lineage is SPACE*n + </T> (G39/G40).  Here abstention
    # is an RL-PRICED ACTION AT ONE POSITION and no CE target moves.
    # --abstain-fill 1: if a rollout's thought position 0 is SPACE, the WHOLE note
    #   becomes the canonical filler -- byte for byte the string built at l.966-968
    #   for `--pause-frac`, `min(--pause-frac-len, L-1)` spaces then </T> padding,
    #   which is also probe_span.py's `filler` column.  So an abstaining rollout
    #   earns THE CONTROL'S OWN REWARD instead of a wrong note's penalty, and it is
    #   exactly length-matched to the control by construction -- G39's confound (a
    #   control not matched on the one dimension that moves) cannot arise.
    #   The policy gradient for an abstaining anchor is restricted to POSITION 0
    #   (see row_rl below), so the writer trains the DECISION and is never trained
    #   to emit ten spaces token by token, which is how runs 32/33 lost their stop
    #   action.  Why SPACE and not an empty note (</T> at position 0)?  Two reasons,
    #   both measured before this flag existed (agent/abstain_feas.py on run 73
    #   step 2800, o=+1, n=134, n_seq=10, 2026-10-02 16:3x):
    #     (i) REACHABILITY.  </T> at position 0 has median vocab rank 62,270 and is
    #         inside the --top-k-thought 512 candidate set for only 2.9% +- 1.6% of
    #         anchors -- 0.018 expected empty notes per 100 rollouts.  It is legal
    #         under --min-think 0 and UNREACHABLE.  p(SPACE) at position 0 is
    #         3.90e-2 mean, ~2e5x larger in the median.
    #    (ii) VALUE.  An empty note scores EXACTLY 0.0 (the anchor's block row is
    #         then byte-identical to the `sil` row the frozen reader baselines on),
    #         and run 71 @2400's filler is worth +1.05 ABOVE silence, so perfect
    #         abstention-to-silence lands at own_minus_filler -0.01 -- break-even BY
    #         CONSTRUCTION.  Abstention must map to the FILLER to be able to win.
    # --abstain-explore p: at position 0 ONLY, force sampled (non-greedy) rows to
    #   SPACE with probability p.  Needed because the measured on-policy p(SPACE|t=0)
    #   is BIMODAL ACROSS DOCUMENTS (per-sequence means 3.2e-5 .. 1.57e-1, median
    #   2.6e-4): half the anchors would get no exploration at all, and a DISCRIMINATION
    #   cannot be learned from a sample that covers one side of it.  Same device as
    #   run 57's `--stop-explore`, scoped to t=0.  Known bias: the forced token is
    #   credited with log p(SPACE) under the advantage, i.e. epsilon-greedy REINFORCE.
    # BOTH DEFAULT TO OFF AND THE OFF PATH IS A LITERAL NO-OP, rng stream included
    # (every new rng draw is inside `if args.abstain_explore > 0`), so runs 70-73
    # and any keepalive relaunch of them are bit-identical under this source.
    p.add_argument("--abstain-fill", type=int, default=0)
    p.add_argument("--abstain-explore", type=float, default=0.0)
    # SLOT GEOMETRY. "aligned" puts thought token i at the position of the real
    # token x[n+1+i] it stands for, instead of one step further out. Measured
    # zero-update (probe_slotpos.py, run 25 @900, 616 anchors): in-slot accuracy
    # 0.270 -> 0.305 and reader gain on identical content +2.30 -> +3.15 nats.
    p.add_argument("--slot-pos", choices=("built", "aligned"), default="built")
    # THE BRIDGE (run 35). Hide the last --bridge context tokens from the WINDOW
    # rows; the thought block still sees them. The thought becomes the only path
    # from that text to the prediction, so the writer's job changes from "forecast
    # what you have not seen" (measured impossible at this scale: free-running
    # agreement 0.06-0.31, own thought below an empty slot in seven checkpoints)
    # to "carry what you just read", which is a copy and is learnable. With
    # --bridge > 0 the counterfactual baseline changes too: a thought's gain is
    # measured against SILENCE UNDER THE SAME MASK, not against the clean rows,
    # which still see everything. Clean NTP is untouched.
    p.add_argument("--bridge", type=int, default=0)
    # THE LEARNED CODE (run 37 if the bridge fails). The thought alphabet becomes
    # a block of reserved, never-used token ids: the writer signals bits, not words,
    # so a "wrong forecast" cannot cost anything -- only an uninformative message can.
    # Pure RL (there is no ground-truth code, so --ssl-frac 0), forced thinking, and
    # the reward is the same counterfactual gain. A constant code carries nothing and
    # earns nothing, so collapse is punished by the reward rather than by a rail.
    p.add_argument("--code-size", type=int, default=0)
    p.add_argument("--code-base", type=int, default=151670)
    # A TEACHER FOR THE CODE (run 38). Pure RL has to bootstrap a channel whose
    # meaning does not exist yet. This gives the writer a target it can actually
    # hit: a coarse, deterministic property of the upcoming window -- the hashed
    # identity of its most frequent content token, per half-window -- so the
    # thought is a 64-way classification of "what kind of text is coming" instead
    # of a verbatim forecast. The reader learns the symbol's meaning from the same
    # window loss; RL then refines the choice at the unsupervised anchors.
    p.add_argument("--code-teacher", type=int, default=0)
    # SCHEDULED SAMPLING / DAgger for the supervised anchors. The imitation
    # block has always been pure teacher forcing, so the writer only ever
    # trained in a regime it never samples in: in-slot accuracy with a true
    # prefix is 0.30 but free-running collapses to 0.06 by token 12 (probe_
    # tfslot.py). With --dagger-p > 0 each supervised slot token is replaced by
    # the model's OWN greedy token with that probability while the CE target
    # stays the true continuation: the expert labels the model's own trajectory.
    p.add_argument("--dagger-p", type=float, default=0.0)
    p.add_argument("--dagger-t0", type=int, default=0)
    p.add_argument("--dagger-t1", type=int, default=1)
    p.add_argument("--oracle-min-off", type=int, default=0,
                   help="jitter: lower bound of the offset (0 = the immediate "
                        "continuation, i.e. speculative decoding; >0 forces a "
                        "genuine look-ahead)")
    # PARAMETER dtype (compute stays bf16 under autocast either way).
    # bf16 parameters silently FREEZE every normalisation weight: RMSNorm
    # gains sit at magnitude ~1.0 where the bf16 spacing is 2^-8 ~ 0.0039,
    # and an Adam step of ~5e-5 rounds to zero every time. Measured on runs
    # 15/17/18: 113/113 norm tensors bit-identical to the init after
    # thousands of steps, while every projection (magnitude ~0.02, spacing
    # ~8e-5) trained normally. ZeRO-1's fp32 accumulator only half-fixes it
    # -- it writes back to bf16, so norms move in 0.4% ULP jumps. fp32
    # parameters are the actual fix and are what mixed precision means.
    # ---- E1 (NEXT-STEPS.md sec. 4, 2026-09-26): SYNTHETIC ADDITION, the
    # programme's first POSITIVE CONTROL.  A micro is synthetic iff
    # (step*grad_accum + micro) % synth_every == 0 -- the SAME micros on every
    # rank, so the 90% web micros can skip decode and scoring on all ranks at
    # once (a per-rank coin would make every step wait for the slowest rank).
    # Synthetic micros hold one addition problem per 64-token block, anchored
    # on the ' ' after ' =' so the window starts at the answer's first digit
    # (synth.py).  Their anchors are: FILLER (--synth-pause, reader trained),
    # SUPERVISED on the reversed-answer scratchpad (--synth-ssl of the rest,
    # imitation, teacher-forced), or RL anchors carrying the model's own
    # unselected sampled thought (the rest; reader trained on it -- the run-25
    # inference-matched lesson; REINFORCE only if --rl-w > 0).
    # --web-pause 1: every valid anchor of a web micro is content-free filler
    # (run-26 style, reader trained) so the web negative cannot contaminate the
    # test.  Defaults (0) reproduce runs 13-39 exactly.
    p.add_argument("--synth-every", type=int, default=0)
    p.add_argument("--synth-dmin", type=int, default=6)
    p.add_argument("--synth-dmax", type=int, default=12)
    p.add_argument("--synth-pause", type=float, default=0.25)
    p.add_argument("--synth-ssl", type=float, default=0.5)
    # --web-pause 2 (run 44): web micros carry NO slot at all -- no filler, no
    # reader loss, pure NTP. Run 43 @500 showed that filler at every web anchor
    # (plus unreadable content at the synthetic ones) trains the reader to IGNORE
    # the slot: the literal next tokens in the slot went from +11 nats (vanilla
    # init) to +0.0.  1 = filler at every web anchor (runs 40, 43).
    p.add_argument("--web-pause", type=int, default=0)
    p.add_argument("--synth-task", choices=("add", "chase"), default="add",
                   help="chase = 5-hop pointer chasing with a forward scratchpad "
                        "(run 45); add = runs 40-44")
    # ---- E1-DISCOVERY (run 47, 2026-09-27): learn WHAT to write from reward, with
    # little supervision.  --synth-seed-values K supervises only the FIRST K chain
    # values (the seed; 0 = the full scratchpad, runs 45-46).  --score-answer-only A
    # scores G on the first A window tokens only (the answer), and --reward-mode prob
    # makes the REINFORCE reward p(answer | thought) - p(answer | clean): linear in
    # probability (NEXT-STEPS sec. 0, consequence 2), not log-probability.
    # Defaults reproduce runs 13-46 exactly.
    # run 56 (2026-09-28): discovery with a CURRICULUM and a TASK ALPHABET.
    # --thought-vocab digits: thoughts may only use the 10 digit tokens and SPACE (+ </T>);
    #   a structural prior on the alphabet, no content. Default 'all' = old behaviour.
    # --synth-hops-ramp N: the max chain length grows by one hop every N steps from
    #   --synth-hops up to --synth-hops-max (problems ~ U{hops..current max}). 0 = off.
    p.add_argument("--thought-vocab", choices=("all", "digits"), default="all")
    p.add_argument("--synth-hops-ramp", type=int, default=0)
    # THE SLIDE (Emanuele's idea, 2026-09-29): compression first, then prediction.
    # --writer-delay-sched "step:d,step:d,...": from each listed step on, the WRITER
    # sees the context only up to x[n-d] and writes right after it, while the READER
    # (window rows) is denied the last --bridge tokens before the anchor. d = 0:
    # the thought is written AFTER the hidden span (compression, the bridge run);
    # d >= bridge: the writer sees none of it (pure prediction). Masks/positions are
    # updated in place when d changes. Empty = off (old behaviour).
    p.add_argument("--writer-delay-sched", type=str, default="")
    # THE MOVING SPAN (run 61): "step:o,...". The thought stands for the span
    # x[n+o .. n+o+L-2] and sits at ITS positions; the reader is denied the part of
    # it that lies in the past (bridge = max(0,-o)); the writer always sees up to x[n].
    # o = -(L-1): the span is all past (compression); o = +1: all future (the paper's
    # forecast). Use with --oracle-mode span. Empty = off.
    p.add_argument("--span-sched", type=str, default="")
    p.add_argument("--force-close", type=int, default=0)
    # FROZEN READER (run 70, 2026-10-01): every note is SCORED by a frozen copy of
    # the init, so the reward, the silence baseline and every control are read by a
    # reader that cannot change. Only the WRITER (the trained model: generation +
    # REINFORCE + clean NTP) learns. Runs 60-69 measured the loss on the reader side
    # and kept gating against baselines the reader itself was moving. Use with
    # --distill-w 0 (the trained model's window rows then carry no loss). 0 = off.
    p.add_argument("--frozen-reader", type=int, default=0)
    # sample whole blocks from a corpus stored as back-to-back (seq_len+1) blocks
    p.add_argument("--aligned-blocks", type=int, default=0)
    # run 57: per-position probability of forcing </T> in SAMPLED rollouts (after min_think);
    # an exploration prior on thought LENGTH, not content. 0 = off (old behaviour).
    p.add_argument("--stop-explore", type=float, default=0.0)
    p.add_argument("--synth-seed-values", type=int, default=0)
    # run 49: with 1 the seed's CE covers only the VALUE tokens, not the </T> after
    # them. Run 47 showed that supervising the stop right after the first value
    # makes every sampled thought exactly one value long (len 2.0, 350 steps), so
    # the reward never sees a longer thought and cannot teach one.
    p.add_argument("--seed-no-stop", type=int, default=0)
    p.add_argument("--score-answer-only", type=int, default=0)
    p.add_argument("--reward-mode", choices=("gain", "prob"), default="gain")
    p.add_argument("--synth-hops", type=int, default=5,
                   help="chase: hop count (run 45 = 5). Lower bound if --synth-hops-max > it.")
    p.add_argument("--synth-hops-max", type=int, default=0,
                   help="chase: if > --synth-hops, draw hops ~ U{hops..hops_max} per problem "
                        "(run 46's depth ladder). 0 = fixed, run 45's behaviour.")
    p.add_argument("--synth-distract", type=int, default=1,
                   help="chase: distractor pairs per problem (run 45 = 1).")
    p.add_argument("--synth-width", type=int, default=0,
                   help="fixed-width addition: operands exactly W digits, "
                        "answer zero-padded to W+1 (0 = variable 6-12)")
    # consecutive-repeat ban in the SAMPLED rollouts (think_par). 1 = old
    # behaviour. E1 needs 0: reversed digits repeat ~10% of the time.
    p.add_argument("--repeat-ban", type=int, default=1)
    p.add_argument("--param-dtype", choices=("bf16", "fp32"), default="fp32")
    # Soft (advantage-weighted) distillation temperature. The eligibility
    # gate of Eq.(elig) is a hard AND of three thresholds: it fires on ~0.5%
    # of anchors, so 99.5% of the scoring work already paid for is discarded
    # and the reader gets an extremely sparse signal. --distill-beta > 0
    # replaces each hard step with a sigmoid of the SAME margin at
    # temperature beta, so every anchor contributes in proportion to how far
    # it clears the bar. beta -> 0 recovers the binary gate exactly.
    # The margins stay ABSOLUTE (thresholds and a shuffled minimal pair),
    # never batch-relative -- a policy-relative gate is what killed run 10a.
    p.add_argument("--distill-beta", type=float, default=0.0)
    # HORIZON CURRICULUM. gamma sets the EFFECTIVE horizon independently of k:
    # sum_j gamma^j is 28.6 tokens at 0.97 but only ~6.6 at 0.85. Run 16 showed
    # that at a 28.6-token horizon the reader never learns to use thoughts at
    # all (orG plateaus at -1.7 vs +0.25 at k=8), so there is nothing for the
    # writer to earn against. Annealing gamma up starts the reader where run 15
    # proved it CAN learn and lengthens the horizon once it has -- without a
    # geometry change, so no recompile. gamma-end <= 0 keeps args.gamma fixed.
    p.add_argument("--gamma-start", type=float, default=-1.0)
    p.add_argument("--gamma-t0", type=int, default=0)
    p.add_argument("--gamma-t1", type=int, default=2000)
    p.add_argument("--stock-adamw", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--loss-chunks", type=int, default=8)
    # io
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--ckpt-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "think13"))
    p.add_argument("--keep-checkpoints", type=int, default=2)
    p.add_argument("--metrics-file", default=None)
    p.add_argument("--dump-every", type=int, default=25)
    p.add_argument("--dump-file", default=None)
    args = p.parse_args()
    args.thought_allow = (np.array(list(range(15, 25)) + [SPACE], dtype=np.int64)
                          if args.thought_vocab == "digits" else None)

    ABST_KEEP = None
    if args.abstain_mode == "freq":
        ABST_KEEP = np.load(args.abstain_file)[:args.abstain_top]
        assert ABST_KEEP.size > 0, args.abstain_file

    torch.manual_seed(args.seed)
    device = torch_xla.device()
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank, world = xr.global_ordinal(), xr.world_size()
    T = args.seq_len

    layout = tp.ParLayout(T, args.stride, args.k, args.L, args.first_anchor)
    vt.log("=" * 72)
    vt.log(f"THINKING run 13 (parallel dense) | world={world} T={T} "
           f"N={layout.N} anchors stride={args.stride} k={args.k} "
           f"L={args.L} S={layout.S} C={layout.C}")
    vt.log(f"grad_accum={args.grad_accum} -> {args.grad_accum * T * world:,} "
           f"real tokens/step")
    vt.log("=" * 72)

    _pdt = torch.float32 if args.param_dtype == "fp32" else torch.bfloat16
    backbone, cfg, _ = build_model(MODEL_ID, seq_len=T, seed=args.seed,
                                   dtype=_pdt)
    vt.log(f"parameter dtype: {_pdt} (compute is bf16 under autocast)")
    model = backbone                       # plain backbone: vanilla-shaped
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu",
                        weights_only=False)
        untie_lm_head(backbone)
        sd = ck["model"]
        if any(kk.startswith("backbone.") for kk in sd):
            sd = {kk[len("backbone."):]: v for kk, v in sd.items()
                  if kk.startswith("backbone.")}
        backbone.load_state_dict(sd, strict=True)
        vt.log(f"backbone <- {args.init_from} (step {ck.get('step')}, "
               f"{ck.get('tokens_seen', 0)/1e9:.3f}B tokens)")
        del ck, sd

    reader_bb = None
    if args.frozen_reader:
        import copy
        reader_bb = copy.deepcopy(backbone)       # the init, before any update
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": True, "preserve_rng_state": False})
    model = model.to(device)
    model.train()
    for p_ in model.parameters():
        if p_.requires_grad:
            p_.grad = torch.zeros_like(p_)

    lr_tensor = torch.tensor(args.lr, dtype=torch.float32, device=device)
    lr_host = torch.empty((), dtype=torch.float32)
    opt, _, _ = vt.build_optimizer(model, args, lr_tensor)

    autocast_ctx = lambda: torch_xla.amp.autocast(device, dtype=torch.bfloat16)
    lm_head = backbone.lm_head
    trunk = backbone.model
    if reader_bb is not None:
        reader_bb = reader_bb.to(device).eval()
        for p_ in reader_bb.parameters():
            p_.requires_grad_(False)
        r_trunk, r_lm_head = reader_bb.model, reader_bb.lm_head
        vt.log("FROZEN READER: notes are scored by a frozen copy of the init")
    else:
        r_trunk, r_lm_head = trunk, lm_head

    # per-run constants on device
    def _wk(gam):
        return (gam ** np.arange(args.k)).astype(np.float32)
    gam_now = args.gamma_start if args.gamma_start > 0 else args.gamma
    wk_np = _wk(gam_now)
    if args.score_answer_only > 0:
        # scoring weights only (G, the reward); the READER's window weights
        # (gam_win below) stay the discounted full window
        _ws = np.zeros(args.k, dtype=np.float32); _ws[:args.score_answer_only] = 1.0
        wk_t = torch.from_numpy(_ws).to(device)
    else:
        wk_t = torch.from_numpy(wk_np).to(device)
    pos_dev = tp.build_positions(layout, args.slot_pos).to(device)
    mask_dev = tp.build_full_mask(layout, args.bridge).to(torch.bfloat16).to(device)
    rows_win_d = torch.from_numpy(layout.win_rows().reshape(-1)).to(device)
    rows_clean_d = torch.from_numpy(
        tp.clean_rows_for_windows(layout)).to(device)
    win_rows_np = layout.win_rows()                    # [N, k]
    blk_rows_np = layout.block_rows()[:, :args.L]      # [N, L] rows w/ target
    w1_np = np.zeros(layout.S, dtype=np.float32)
    w1_np[:T] = 1.0
    w1_dev = torch.from_numpy(w1_np[None]).to(device)
    n1_dev = torch.tensor(float(T), device=device)
    gam_win = np.tile(wk_np[None], (layout.N, 1))      # [N, k]
    anchor_rows_d = torch.from_numpy(layout.anchors).to(device)
    topen_tgt_d = torch.full((layout.N,), th.T_OPEN, dtype=torch.long,
                             device=device)

    STOPMASK = np.zeros(160000, dtype=bool)
    _sp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "stop_tokens_top300.npy")
    if os.path.exists(_sp):
        STOPMASK[np.load(_sp)] = True

    mixdata = Mixture(args.corpus, "train", aligned=bool(args.aligned_blocks))
    batcher = MixtureBatcher(mixdata, 1, T, seed=args.seed + 7, rank=rank,
                             world_size=world)
    RO = max(1, args.rollouts)          # sampled rollouts per anchor
    SPEC_ROW = RO                       # the greedy speculation row
    gen = tp.ParGenerator(trunk, lm_head, layout, cfg.num_hidden_layers,
                          RO + 1, device, autocast_ctx, torch_xla.sync,
                          topk_transfer=args.topk_transfer,
                          slot_pos=args.slot_pos)
    vt.log(f"rollouts={RO} (+1 spec row) -> {(RO+1)*layout.N} decode rows")

    tok = None
    if vt.is_master():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)

    start_step, tokens_seen = 0, 0
    if args.resume:
        start_step, tokens_seen = vt.load_checkpoint(model, opt, args,
                                                     device, rank, world)
    metrics_f = open(args.metrics_file, "a") if (args.metrics_file and
                                                 vt.is_master()) else None
    dump_f = open(args.dump_file, "a") if (args.dump_file and
                                           vt.is_master()) else None
    elig_f = None
    if dump_f and args.dump_file not in (None, "/dev/null"):
        elig_f = open(args.dump_file.replace(".jsonl", "_eligible.jsonl"),
                      "a")
    t_start = time.time()
    last_dump_step = -10**9
    vt.log("starting (the big grad graph compiles on the first step) ...")

    # ---- THE SLIDE: writer-delay schedule (no-op when the flag is empty)
    _dsched = sorted((int(a), int(b)) for a, b in
                     (kv.split(":") for kv in args.writer_delay_sched.split(",") if kv))
    def _delay_at(st_):
        d_ = 0
        for s_, v_ in _dsched:
            if st_ >= s_:
                d_ = v_
        return d_
    _cur_delay = 0
    _ssched = sorted((int(a), int(b)) for a, b in
                     (kv.split(":") for kv in args.span_sched.split(",") if kv))
    def _span_at(st_):
        o_ = _ssched[0][1] if _ssched else 0
        for s_, v_ in _ssched:
            if st_ >= s_:
                o_ = v_
        return o_
    _cur_span = None
    _cur_bridge = args.bridge
    for step in range(start_step, args.max_steps):
        if _ssched and _span_at(step) != _cur_span:
            _cur_span = _span_at(step)
            _cur_bridge = max(0, -_cur_span)
            pos_dev.copy_(tp.build_positions(layout, args.slot_pos, 0, _cur_span - 1)
                          .to(pos_dev.device))
            mask_dev.copy_(tp.build_full_mask(layout, _cur_bridge, 0)
                           .to(torch.bfloat16).to(mask_dev.device))
            gen.set_writer_delay(0, _cur_span - 1)
            vt.log(f"SPAN: offset -> {_cur_span} (reader denied {_cur_bridge}) at step {step}")
        if _dsched and _delay_at(step) != _cur_delay:
            _cur_delay = _delay_at(step)
            pos_dev.copy_(tp.build_positions(layout, args.slot_pos, _cur_delay).to(pos_dev.device))
            mask_dev.copy_(tp.build_full_mask(layout, args.bridge, _cur_delay)
                           .to(torch.bfloat16).to(mask_dev.device))
            gen.set_writer_delay(_cur_delay)
            vt.log(f"SLIDE: writer delay -> {_cur_delay} at step {step}")
        lr = vt.lr_at(step, args)
        lr_host.fill_(lr)
        lr_tensor.copy_(lr_host)
        if args.gamma_start > 0:
            # anneal the DISCOUNT, not the geometry: wk_t is updated in place
            # so every compiled graph keeps its shapes and nothing recompiles.
            gam_new = sched(step, args.gamma_start, args.gamma,
                            args.gamma_t0, args.gamma_t1)
            if abs(gam_new - gam_now) > 1e-9:
                gam_now = gam_new
                wk_np = _wk(gam_now)
                if args.score_answer_only <= 0:
                    wk_t.copy_(torch.from_numpy(wk_np).to(wk_t.device))
                gam_win = np.tile(wk_np[None], (layout.N, 1))
        p_or = sched(step, args.p_or_start, args.p_or_floor,
                     args.p_or_t0, args.p_or_t1)
        imit_w = sched(step, args.imit_w, args.imit_floor,
                       args.p_or_t0, args.p_or_t1)
        rl_w = sched(step, 0.0, args.rl_w,
                     args.rl_start, args.rl_start + args.rl_ramp)
        ss_p = sched(step, args.ss_start, args.ss_end, args.ss_t0, args.ss_t1)
        eps_th = sched(step, args.eps_think_start, args.eps_think_floor,
                       args.rl_start, args.p_or_t1)
        t0 = time.time()

        st = dict(ntp1=0.0, read_ce=0.0, n_read=0, imit_ce=0.0, n_imit=0,
                  g=[], g_spec=[], g_shuf=[], or_g=[], n_elig=0, n_anch=0,
                  n_sm=0, n_or=0, lens=[], lens_sp=[],
                  think=[], pT=[], gateR=[], crowd=[], boost=[], g_win=[], elig_w=[],
                  adv_sig=[], adv_absA=[],
                  gx_r0=[], gx_sd=[],
                  g_sd=[], g_r0=[], g_shuf0=[], g_buf=[], ss_sub=[],
                  dagger_sub=[], abst=[], pau=[],
                  scr_ex_s=[], scr_ex_g=[], scr_dig_g=[], g_pau_syn=[], scr_pv_g=[],
                  n_syn=0)
        dumped = False

        for micro in range(args.grad_accum):
            x, y, _ = batcher.batch(step, micro)
            for arr in (x, y):
                arr[(arr == th.T_OPEN) | (arr == th.T_CLOSE)] = SPACE
            is_syn = (args.synth_every > 0 and
                      (step * args.grad_accum + micro) % args.synth_every == 0)
            sq = None
            if is_syn:
                _srng = np.random.default_rng([args.seed, rank, step, micro, 555])
                if args.synth_task == "chase":
                    sq = sy.make_chase_seq(_srng, T=T, first=args.first_anchor,
                                           stride=args.stride, n_anchor=layout.N,
                                           L=args.L, hops=args.synth_hops,
                                           hops_max=(min(args.synth_hops_max,
                                                         args.synth_hops + step // args.synth_hops_ramp)
                                                     if args.synth_hops_ramp > 0
                                                     else args.synth_hops_max),
                                           n_distract=args.synth_distract)
                else:
                    sq = sy.make_seq(
                        _srng, T=T, first=args.first_anchor, stride=args.stride,
                        n_anchor=layout.N, L=args.L,
                        dmin=args.synth_dmin, dmax=args.synth_dmax,
                        width=args.synth_width)
                x = sq["x"][None, :T].copy()
                y = sq["x"][None, 1:T + 1].copy()
                st["n_syn"] += 1
            rng = np.random.default_rng([args.seed, rank, step, micro, 13])
            w2 = np.zeros(layout.S, dtype=np.float32)
            w3 = np.zeros(layout.S, dtype=np.float32)
            vw, vo = tp.validity(layout, x[0])
            if args.oracle_mode == "span":
                # the span the thought stands for, at offset _cur_span from the anchor
                orc = np.stack([x[0, int(n) + _cur_span:int(n) + _cur_span + args.L]
                                for n in layout.anchors]).astype(np.int64)
                vo = vw * np.array([float(not (o_ == th.EOD).any()) for o_ in orc],
                                   dtype=np.float32)
            elif args.oracle_mode == "bridge":
                # exactly what the window can no longer see
                orc = np.stack([x[0, int(n) - args.L:int(n)]
                                for n in layout.anchors]).astype(np.int64)
                vo = vw.copy()
            elif args.oracle_mode == "beyond":
                orc = tp.oracle_content(layout, x[0])
            else:
                # span inside the window: x[n+1+o .. n+1+o+L), o+L <= k, so
                # validity is exactly the window's (no EOD in x[n .. n+k]).
                omax = args.oracle_max_off if args.oracle_max_off >= 0 \
                    else args.k - args.L
                offs = (rng.integers(args.oracle_min_off, omax + 1,
                                     size=layout.N)
                        if args.oracle_mode == "jitter"
                        else np.full(layout.N, args.oracle_off))
                orc = np.stack([
                    x[0, int(n) + 1 + int(o):int(n) + 1 + int(o) + args.L]
                    for n, o in zip(layout.anchors, offs)]).astype(np.int64)
                vo = vw.copy()

            # ---------------------------------------------------------------
            # MIXED ANCHORS (run 18). Runs 13-17 split self-supervision and
            # reinforcement across separate MICRO-BATCHES, so the model never
            # saw the two side by side. Here they share one sequence: a
            # fraction `ssl_frac` of anchors carry the true continuation
            # (teacher-forced, variable length) and the rest carry the model's
            # own sampled thought -- same <T>...</T> format, a few dozen tokens
            # apart, so "write RL thoughts that look like the SSL ones" is a
            # within-context pattern rather than a cross-batch one.
            #
            # THINKING IS FORCED at every RL anchor: there is no learned gate.
            # In runs 13-17 the cheapest policy was to shut the gate
            # (p(<T>) -> 1e-10) and stop paying the length price altogether,
            # so an inert thought was FREE. Forced, an inert thought scores
            # gain ~ 0 minus its token cost, i.e. strictly negative: the model
            # can no longer opt out, only think well or bleed.
            # ---------------------------------------------------------------
            ssl = (rng.random(layout.N) < args.ssl_frac) & (vo > 0)
            rl = (~ssl) & (vw > 0)
            # --pause-frac: a fraction of anchors become FILLER anchors. They
            # leave both writer masks here and rejoin the reader mask at `rdf`
            # below; their block content is overwritten after `content` is
            # built. The `if` guard keeps the rng stream bit-identical when the
            # flag is off.
            if args.pause_frac > 0:
                pau = (rng.random(layout.N) < args.pause_frac) & (vw > 0)
                ssl = ssl & (~pau)
                rl = rl & (~pau)
            else:
                pau = np.zeros(layout.N, dtype=bool)
            if is_syn:
                pau = (rng.random(layout.N) < args.synth_pause) & (vw > 0)
                ssl = (~pau) & (rng.random(layout.N) < args.synth_ssl) \
                    & (vw > 0)
                rl = (~pau) & (~ssl) & (vw > 0)
            elif args.web_pause:
                pau = (vw > 0) if args.web_pause == 1 else \
                    np.zeros(layout.N, dtype=bool)
                ssl = np.zeros(layout.N, dtype=bool)
                rl = np.zeros(layout.N, dtype=bool)
            pauf = pau.astype(np.float32)
            think = ssl | rl
            thf = think.astype(np.float32)
            sslf, rlf = ssl.astype(np.float32), rl.astype(np.float32)

            if args.code_teacher and args.code_size > 0:
                orc_content = np.full((layout.N, args.L), th.T_CLOSE,
                                      dtype=np.int64)
                half = args.k // 2
                for a, n_ in enumerate(layout.anchors):
                    n_ = int(n_)
                    if args.code_teacher == 2:
                        # --code-teacher 2 (run 39, the CHANNEL CONTROL): the
                        # target is the hash of a SINGLE token the model can
                        # already see -- the token k/2 back, then the anchor
                        # token itself. Deterministic, in-context, and a pure
                        # lookup (no counting, no argmax over a window), so a
                        # failure here cannot be blamed on the target being
                        # unpredictable OR on the function being hard to
                        # compute. It measures the CHANNEL and nothing else.
                        # Marginal entropy 4.01 of ln(64)=4.16; the two slots
                        # are near-independent (agreement 0.023 vs chance
                        # 0.016). See G57.
                        spans = (slice(max(0, n_ - half), max(0, n_ - half) + 1),
                                 slice(n_, n_ + 1))
                    else:
                        spans = (slice(n_ + 1, n_ + 1 + half),
                                 slice(n_ + 1 + half, n_ + 1 + args.k))
                    for h, sl in enumerate(spans):
                        w = x[0, sl]
                        w = w[~STOPMASK[np.clip(w, 0, len(STOPMASK) - 1)]]
                        if len(w) == 0:
                            w = x[0, sl][:1]
                        vals, cnt = np.unique(w, return_counts=True)
                        top = int(vals[int(np.argmax(cnt))])
                        sym = (top * 2654435761) % args.code_size
                        orc_content[a, h] = args.code_base + int(sym)
                jlens = np.full(layout.N, 2, dtype=np.int64)
            else:
                orc_content, jlens = tp.jitter_oracle(layout, orc, rng,
                                                      args.jit_min)
            if is_syn:
                # the supervised thought is the reversed-answer scratchpad
                orc = sq["scr"].copy()
                orc_content = sq["scr"].copy()
                jlens = sq["slen"].copy()
                if args.synth_seed_values > 0:
                    # the SEED: only the first K values ('ab cd ..' = 3K-1 tokens)
                    kk = 3 * args.synth_seed_values - 1
                    orc_content[:, kk:] = th.T_CLOSE
                    jlens = np.minimum(jlens, kk)
            abst_sup = None
            if ABST_KEEP is not None:
                # pmix: true token at forecastable positions, SPACE elsewhere.
                # </T> padding past the thought's own end is NOT touched, so
                # lengths and the stop action are exactly as before.
                pad = orc_content == th.T_CLOSE
                keep = np.isin(orc_content, ABST_KEEP) | pad
                orc_content = np.where(keep, orc_content, SPACE)
                # positions whose CE target became SPACE; --abstain-loss-w
                # scales the imitation weight there (run 34).
                abst_sup = ~keep
                st["abst"].append(float((~keep).sum())
                                  / max(float((~pad).sum()), 1.0))
            x_dev = torch.from_numpy(x.astype(np.int64)).to(device)
            # E1: nothing is WRITTEN at this micro (every valid anchor is
            # filler or supervised) -> skip decode and every scoring pass.
            # Gated on the E1 flags so older configs keep their rng stream.
            no_write = ((args.web_pause or args.synth_every > 0)
                        and not rl.any() and args.dagger_p == 0
                        and ss_p == 0)
            ab_row = None
            if args.pause_len > 0 or no_write:
                z = np.full((RO + 1, layout.N, args.L), th.T_CLOSE,
                            dtype=np.int64)
                z[:, :, :min(args.pause_len if args.pause_len > 0
                             else args.pause_frac_len, args.L - 1)] = SPACE
                lpT = np.zeros(layout.N, dtype=np.float32)
            else:
                z, lpT = gen.generate(x_dev, rng, args, spec_row=SPEC_ROW,
                                      ban_from=th.EOD)
            # run 74: SPACE at position 0 IS the abstention action (see --abstain-fill).
            # ab_row is computed BEFORE the substitution, because the filler itself
            # starts with SPACE and would otherwise match every row after one pass.
            if args.abstain_fill:
                if args.abstain_explore > 0:
                    frc = rng.random(z.shape[:2]) < args.abstain_explore
                    frc[SPEC_ROW] = False          # the greedy row stays on-policy
                    z[:, :, 0] = np.where(frc, SPACE, z[:, :, 0])
                ab_row = z[:, :, 0] == SPACE       # [RO+1, N]
                fil_ab = np.full(args.L, th.T_CLOSE, dtype=z.dtype)
                fil_ab[:min(args.pause_frac_len, args.L - 1)] = SPACE
                z = np.where(ab_row[:, :, None], fil_ab[None, None, :], z)
            tgt_win_d = torch.from_numpy(np.stack(
                [x[0, int(n) + 1:int(n) + 1 + args.k]
                 for n in layout.anchors]).reshape(-1)
                .astype(np.int64)).to(device)

            imit_target = orc_content.copy()       # CE target stays the truth
            if args.dagger_p > 0 and args.pause_len == 0:
                p_own = sched(step, 0.0, args.dagger_p,
                              args.dagger_t0, args.dagger_t1)
                own = z[SPEC_ROW]                  # the model's own greedy roll
                take = rng.random((layout.N, args.L)) < p_own
                take[:, 0] = take[:, 0] & False    # keep the opener supervised
                orc_content = np.where(take, own, orc_content)
                st["dagger_p"] = p_own
                # realized substitution rate over the SUPERVISED rows (the only
                # ones that keep orc_content; _mix overwrites the RL rows). The
                # 20:07 tick found dagger_p was set here but never logged, so
                # run 30 logged ss_p (the unused mechanism, constant 0.0) and
                # nothing for the mechanism it actually runs on.
                st["dagger_sub"].append(
                    float((take & ssl[:, None]).sum())
                    / max(float(ssl.sum() * args.L), 1.0))

            def _mix(rl_src):
                """SSL anchors hold the true continuation; RL anchors hold
                `rl_src`; everything else is silent."""
                c = orc_content.copy()
                c[rl] = rl_src[rl]
                c[~think] = th.T_CLOSE
                return c

            base_masked = None
            # frozen reader: ALWAYS baseline on silence under the same mask, also at
            # o >= 0 (bridge 0), so the logged gain stays comparable with probe_span's
            # `own` column. At o=+1 the clean rows sit ~11 nats above silence (the block
            # shares positions with the window), which made run 71's g read -12 from step
            # 1800. LOO cancels any per-anchor constant, so training is unchanged.
            if _cur_bridge > 0 or args.frozen_reader:
                sil = np.full((layout.N, args.L), th.T_CLOSE, dtype=np.int64)
                b_s = tp.build_buffer(layout, x[0], sil, SPACE)
                lw_s, _ = tp.score_buffer(
                    r_trunk, r_lm_head, b_s, pos_dev, mask_dev, layout, wk_t,
                    rows_win_d, rows_clean_d, tgt_win_d, autocast_ctx)
                torch_xla.sync()
                base_masked = lw_s

            _last = {}

            def _score(c):
                b_ = tp.build_buffer(layout, x[0], c, SPACE)
                lw, lc = tp.score_buffer(
                    r_trunk, r_lm_head, b_, pos_dev, mask_dev, layout, wk_t,
                    rows_win_d, rows_clean_d, tgt_win_d, autocast_ctx)
                torch_xla.sync()
                # under the bridge the clean rows still see the hidden span, so
                # the only honest baseline is silence under the SAME mask
                _base = base_masked if base_masked is not None else lc
                _last["lw"], _last["lc"] = lw, _base
                return lw - _base

            # R rollouts, scored on RAW gain. The shuffle is no longer in the
            # reward (user directive 2026-09-08): with the SSL channel holding
            # the policy on the manifold of real future text, the reward is
            # simply "did this thought make the true continuation more
            # likely". The shuffle survives as a MONITOR below, because raw
            # gain is winnable by topic priming and without the minimal pair
            # we could not tell priming from anticipation.
            G = np.zeros((RO, layout.N), dtype=np.float64)
            PR = np.zeros((RO, layout.N), dtype=np.float64)
            GX = np.zeros((RO, layout.N), dtype=np.float64)   # G100 misplaced
            if no_write:
                pass                                    # nothing to score
            elif args.pause_len > 0:
                G[:] = _score(_mix(z[0]))[None, :]      # one pass: all rows equal
            else:
                for r_i in range(RO):
                    c_r = _mix(z[r_i])
                    G[r_i] = _score(c_r)
                    if args.reward_mode == "prob":
                        PR[r_i] = np.exp(_last["lw"]) - np.exp(_last["lc"])
                    if args.xanchor_w > 0.0:
                        # G100: the MISPLACED-NOTE control, built exactly as
                        # probe_span.py:56 builds its `shuf` column. roll(+1) on
                        # the anchor axis puts note a-1 at anchor a, so
                        # roll(-1) on the SCORES brings note a's misplaced gain
                        # back to row a. One extra scoring pass per rollout.
                        GX[r_i] = np.roll(
                            _score(np.roll(c_r, 1, axis=0)), -1, axis=0)
            S0 = np.stack([tp.stop_pos(_mix(z[r_i]), args.L)
                           for r_i in range(RO)])
            REW = ((PR if args.reward_mode == "prob" else G)
                   - args.len_cost * S0 - args.xanchor_w * GX)
            if RO > 1:
                PAIR = (REW.sum(0)[None, :] - REW) / (RO - 1)
            else:
                PAIR = np.zeros_like(REW)
            ADV_R = REW - PAIR
            best = G.argmax(0)
            aidx = np.arange(layout.N)
            if args.rl_content == "random":
                best = rng.integers(0, RO, size=layout.N)
            content = _mix(np.stack([z[best[a]][a] for a in aidx]))
            if pau.any():
                # byte for byte the --pause-len filler built at l.491-494:
                # SPACE for pause-frac-len positions, then </T> padding.
                fil = np.full(args.L, th.T_CLOSE, dtype=content.dtype)
                fil[:min(args.pause_frac_len, args.L - 1)] = SPACE
                content[pau] = fil
            s0 = tp.stop_pos(content, args.L)
            gain = G[best, aidx]
            adv_paired = ADV_R[best, aidx]

            # ---- MONITOR ONLY: one extra scoring pass on a shuffled copy of
            # the winner's RL content. Never enters the loss; it is the only
            # way to say how much of any earned gain is order-borne.
            if args.pause_len > 0 or no_write:
                # filler has no order and no writer: the shuffle monitors and
                # the greedy row are all the same content, scored once already.
                gain_shuf = gain_shuf0 = gain_spec = gain
                s1 = s0
            else:
                c_sh = content.copy()
                for a in np.where(rl)[0]:
                    if s0[a] > 1:
                        c_sh[a, :s0[a]] = rng.permutation(content[a, :s0[a]])
                gain_shuf = _score(c_sh)
                # UNBIASED order-borne value (G19): rollout 0 is not selected on
                # anything, so G[0] - gain of its own shuffle carries neither the
                # winner's-curse inflation nor the winner's-shuffle deflation.
                # One more scoring pass; monitor only.
                c_sh0 = _mix(z[0]).copy()
                s00 = tp.stop_pos(c_sh0, args.L)
                for a in np.where(rl)[0]:
                    if s00[a] > 1:
                        c_sh0[a, :s00[a]] = rng.permutation(c_sh0[a, :s00[a]])
                gain_shuf0 = _score(c_sh0)
                s1 = tp.stop_pos(z[SPEC_ROW], args.L)   # spec-row len
                gain_spec = (_score(_mix(z[SPEC_ROW])) if args.score_spec
                             else np.zeros_like(gain))
            if args.baseline == "greedy":
                s_sp = tp.stop_pos(_mix(z[SPEC_ROW]), args.L)
                adv_paired = (gain - args.len_cost * s0) \
                    - (gain_spec - args.len_cost * s_sp)
            elig = (rlf * (gain > args.distill_eps)).astype(np.float32)
            if is_syn and not no_write:
                # E1 live signal: does the writer produce the scratchpad?
                # Measured on EVERY valid synthetic anchor (what it would write
                # there), sampled row 0 and the greedy row, against the truth
                # (digits then </T>).
                okv = vw > 0
                ar_ = np.arange(args.L)[None, :]
                within = ar_ <= sq["slen"][:, None]
                digs = ar_ < sq["slen"][:, None]
                for tag, zr in (("s", z[0]), ("g", z[SPEC_ROW])):
                    hit = zr == sq["scr"]
                    ex = (hit | ~within).all(-1)
                    st[f"scr_ex_{tag}"].extend(ex[okv].astype(float).tolist())
                    if tag == "g":
                        dg = (hit & digs).sum(-1) / np.maximum(sq["slen"], 1)
                        st["scr_dig_g"].extend(dg[okv].tolist())
                        # correct chain VALUES written as a prefix (discovery)
                        for a_ in np.where(okv)[0]:
                            m_ = 0
                            while m_ < sq["slen"][a_] and hit[a_, m_]:
                                m_ += 1
                            st["scr_pv_g"].append((m_ + 1) // 3)

            # ---- weights: SSL rows imitate, RL rows ride the advantage
            ar = np.arange(args.L)[None, :]
            row_ssl = (ar <= jlens[:, None]).astype(np.float32) * sslf[:, None]
            if is_syn and args.seed_no_stop and args.synth_seed_values > 0:
                row_ssl = (ar < jlens[:, None]).astype(np.float32) * sslf[:, None]
            if abst_sup is not None and args.abstain_loss_w != 1.0:
                # run 34: blanked positions keep their place in the block INPUT
                # but lose (all or part of) their CE weight, so "emit SPACE" is
                # no longer a supervised action.  The n3 normaliser is left
                # alone on purpose: the kept positions then keep exactly the
                # per-token weight they had in run 32, and only the total
                # imitation mass falls (by the `abst` fraction, which is logged).
                row_ssl = row_ssl * np.where(
                    abst_sup, np.float32(args.abstain_loss_w),
                    np.float32(1.0)).astype(np.float32)
            row_rl = (ar <= s0[:, None]).astype(np.float32) * rlf[:, None]
            if ab_row is not None:
                # run 74: abstention is ONE decision. An anchor whose SELECTED
                # rollout abstained takes policy gradient at position 0 only --
                # never at the ten substituted spaces. This is the single line
                # that separates this design from runs 32/33, where the writer
                # was trained to emit the filler token by token and lost its stop
                # action (G39). `abst` logs the realized rate over RL anchors and
                # is the registered collapse detector (3b's length clause is DEAD
                # here: an abstaining anchor's content row is the 10-space filler,
                # so s0 == 10 and len_mean does NOT fall when the writer goes mute).
                ab_sel = ab_row[best, aidx]
                row_rl = np.where(ab_sel[:, None],
                                  (ar == 0).astype(np.float32) * rlf[:, None],
                                  row_rl)
                st["abst"].append(float((ab_sel & rl).sum())
                                  / max(float(rl.sum()), 1.0))
            # --reader-on-ssl 2 (run 55, the G66 decoupling): the reader takes
            # window gradient at DEMONSTRATION anchors only, never through a slot
            # holding a sampled rollout. `_score` is a separate forward over the
            # spliced `content`, so the WRITER's reward is unchanged.
            rdf = (sslf if args.reader_on_ssl == 2 else
                   (thf if args.reader_on_ssl else rlf))
            # the whole point: the reader takes window gradient after a filler
            # block. pau is disjoint from ssl and rl, so this stays a 0/1 mask.
            if pau.any():
                rdf = rdf + pauf
            w2[win_rows_np.reshape(-1)] = \
                (gam_win * rdf[:, None]).reshape(-1)
            if rl_w > 0:
                nv = max(rlf.sum(), 1.0)
                mu = float((adv_paired * rlf).sum() / nv)
                sig = float(np.sqrt(((adv_paired - mu) ** 2 * rlf).sum() / nv))
                # G96: max(sig, floor) -- with floor 0 this is exactly the old
                # divisor, so the default path is unchanged bit for bit.
                div = max(sig, args.adv_sig_floor)
                # G96: `sig` is the quantity --adv-sig-floor acts on and it was
                # not observable anywhere. Log it (and how hard the floor bit)
                # so a run can be told "the reward went silent" from metrics.
                A = np.clip((adv_paired - mu) / (div + 1e-6),
                            -args.reward_clip, args.reward_clip) * rlf
                st["adv_sig"].append(sig)
                st["adv_absA"].append(
                    float((np.abs(A) * rlf).sum() / max(rlf.sum(), 1.0)))
                A = np.where(A < 0, A * args.neg_scale, A)
            else:
                A = np.zeros(layout.N, dtype=np.float64)
            if args.pause_len > 0:
                imit_w_eff, rl_w_eff = 0.0, 0.0
            else:
                imit_w_eff, rl_w_eff = imit_w, rl_w
            w3[blk_rows_np.reshape(-1)] = (
                imit_w_eff * row_ssl
                + rl_w_eff * A[:, None] * row_rl).reshape(-1)
            n2 = max(float((gam_win * rdf[:, None]).sum()), 1.0)
            n3 = max(float((sslf * (jlens + 1)).sum() + rlf.sum()), 1.0)

            st["n_elig"] += int(elig.sum())
            st["n_anch"] += int(rlf.sum())
            st["n_sm"] += 1
            st["n_or"] += int(sslf.sum())
            st["think"].append(float(thf.sum()) / max(vw.sum(), 1.0))
            st["pau"].append(float(pauf.sum()) / max(vw.sum(), 1.0))
            st["pT"].extend(np.exp(lpT[vw > 0]).tolist())
            st["g"].extend(G.mean(0)[rl].tolist())
            st["g_shuf"].extend(gain_shuf[rl].tolist())
            st["g_r0"].extend(G[0][rl].tolist())
            st["g_shuf0"].extend(gain_shuf0[rl].tolist())
            if args.xanchor_w > 0.0:
                # G100 live instrument: gx_r0 is rollout 0's MISPLACED gain, so
                # (g_r0 - gx_r0) is own_minus_shuf measured in training, on the
                # same quantity the held-out probe gates on. gx_sd is the
                # between-rollout spread of the DIFFERENCED reward -- the one
                # number that says whether the new objective has any signal
                # left for REINFORCE to use (compare against g_sd).
                st["gx_r0"].extend(GX[0][rl].tolist())
                if rl.any():
                    st["gx_sd"].append(float((G - GX).std(0)[rl].mean()))
            # per-anchor spread of the R rollouts' gains: best-of-R inflates the
            # winner by ~1.42 sd on pure noise, so g_win - g_shuf is only
            # interpretable next to this number (logged as g_sd).
            if rl.any():
                st["g_sd"].append(float(G.std(0)[rl].mean()))
            st["g_win"].extend(G.max(0)[rl].tolist())
            st["g_buf"].extend(gain[rl].tolist())
            st["g_spec"].extend(gain_spec[rl].tolist())
            st["lens"].extend(s0[rl].tolist())
            st["lens_sp"].extend(jlens[ssl].tolist())

            # run 18: thinking is FORCED, so there is no gate to train. The
            # <T> op below is kept only so the compiled graph is unchanged.
            w_gate = np.zeros(layout.N, dtype=np.float32)
            n_gate = 1.0

            # ---- scheduled sampling / DAgger (run 30; no-op when ss_p == 0).
            # The SSL block's INPUT gets the model's own token at a fraction
            # ss_p of positions; the CE TARGET is built from `content`, i.e.
            # still the true continuation.  Per-position Bernoulli (Bengio et
            # al. 2015) rather than whole-prefix replacement on purpose: with
            # most positions still true the target x[n+1+t] stays identifiable,
            # so the model learns to recover from its own errors instead of
            # being asked to predict a continuation it can no longer locate.
            # Substitution is skipped past the source thought's own </T> so we
            # never splice a stop token into the middle of the block.
            content_in = content
            if ss_p > 0.0 and args.pause_len <= 0:
                src = z[SPEC_ROW] if args.ss_row == "spec" else z[0]
                s_src = tp.stop_pos(src, args.L)
                ar_ss = np.arange(args.L)[None, :]
                m_ss = ((rng.random((layout.N, args.L)) < ss_p)
                        & ssl[:, None] & (ar_ss < s_src[:, None]))
                content_in = content.copy()
                content_in[m_ss] = src[m_ss]
                st["ss_sub"].append(float(m_ss.sum())
                                    / max(float(sslf.sum() * args.L), 1.0))
            buf = tp.build_buffer(layout, x[0], content_in, SPACE)
            tgt = tp.build_targets(layout, x[0], y[0], content)
            if args.dagger_p > 0 and args.pause_len == 0:
                # the supervised rows keep the TRUE continuation as target even
                # where the input token is the model's own
                for a in np.where(ssl)[0]:
                    b0 = layout.block_row(a, 0)
                    tgt[b0:b0 + args.L] = imit_target[a]
            bt = torch.from_numpy(buf[None]).to(device)
            tt = torch.from_numpy(tgt[None]).to(device)
            w2_d = torch.from_numpy(w2[None]).to(device)
            w3_d = torch.from_numpy(w3[None]).to(device)
            n2_d = torch.tensor(n2 / args.distill_w, device=device) \
                if args.distill_w > 0 else torch.tensor(1e30, device=device)
            n3_d = torch.tensor(n3, device=device)
            with autocast_ctx():
                h = trunk(input_ids=bt, position_ids=pos_dev,
                          attention_mask=mask_dev, use_cache=False)[0]
            loss, ce_det = tp.weighted_ce3(h, lm_head, tt, w1_dev, w2_d,
                                           w3_d, n1_dev, n2_d, n3_d,
                                           args.loss_chunks, autocast_ctx)
            # gate REINFORCE: signed CE toward <T> at the anchor rows -- the
            # model's own emission probability is the learned think-trigger
            # (zero weights on oracle micros; op kept for one grad graph)
            w_gate_d = torch.from_numpy(w_gate[None]).to(device)
            n_gate_d = torch.tensor(float(n_gate), device=device)
            with autocast_ctx():
                lgA = lm_head(h[0, anchor_rows_d])
            ceA = F.cross_entropy(lgA.float(), topen_tgt_d,
                                  reduction="none")
            loss = loss + (ceA * w_gate_d[0]).sum() / n_gate_d
            (loss / args.grad_accum).backward()
            ce_np = ce_det.cpu().numpy()
            torch_xla.sync()

            st["ntp1"] += float(ce_np[:T].mean())
            ce_win = ce_np[win_rows_np]                    # [N, k]
            ce_cln = ce_np[tp.clean_rows_for_windows(layout)
                           .reshape(layout.N, args.k)]
            # both signals now live in ONE micro, so both stats come from
            # the same pass: orG/imitation CE from the SSL anchors, reader CE
            # over every thinking anchor.
            if sslf.sum() > 0:
                st["or_g"].extend(
                    (((ce_cln - ce_win) * gam_win).sum(-1))[ssl].tolist())
                st["imit_ce"] += float(
                    (ce_np[blk_rows_np] * row_ssl).sum()
                    / max(float((sslf * (jlens + 1)).sum()), 1.0))
                st["n_imit"] += 1
            if is_syn and pau.any():
                # filler gain at synthetic anchors, from the grad pass: the
                # within-weights floor the written scratchpad must beat
                st["g_pau_syn"].extend(
                    (((ce_cln - ce_win) * gam_win).sum(-1))[pau].tolist())
            if thf.sum() > 0:
                st["read_ce"] += float(
                    (ce_win * gam_win * thf[:, None]).sum()
                    / max((gam_win * thf[:, None]).sum(), 1.0))
                st["n_read"] += 1

            # ---- dumps (master only) ----
            if vt.is_master():
                if (elig_f and elig is not None and elig.sum() > 0):
                    for a0 in np.where(elig > 0)[0][:8]:
                        a0 = int(a0)
                        n0 = int(layout.anchors[a0])
                        elig_f.write(json.dumps(dict(
                            src="rl",   # elig = rlf * (...), RL anchors only
                            step=step, anchor=n0, gain=float(gain[a0]),
                            shuf=float(gain_shuf[a0]),
                            spec=float(gain_spec[a0]),
                            context=tok.decode(
                                x[0, max(0, n0 - 40):n0 + 1].tolist()),
                            thought=tok.decode(
                                content[a0, :int(s0[a0])].tolist())
                            + " </T>",
                            window=tok.decode(
                                x[0, n0 + 1:n0 + 1 + args.k].tolist()),
                        )) + "\n")
                    elig_f.flush()
                a0 = -1     # E1 fix: undefined on a resumed non-writing micro
                if (dump_f and not dumped
                        and step - last_dump_step >= args.dump_every
                        and thf.sum() > 0):
                    # first sample micro once due (step % N misses whole
                    # slots: 44% of steps have no sample micro on rank 0)
                    dumped = True
                    last_dump_step = step
                    # CONTAMINATION GUARD: pick from RL anchors ONLY. SSL
                    # anchors hold the TRUE continuation teacher-forced, so
                    # argmax over `think` would almost always return a copy
                    # of the ground truth and label it as the model's own
                    # thought. Every dumped record now carries src=rl|ssl.
                    a0 = int(np.argmax(np.where(rl, gain, -1e9))) \
                        if rl.any() else -1
                    n0 = int(layout.anchors[a0]) if a0 >= 0 else -1
                if vt.is_master() and a0 >= 0:
                    dump_f.write(json.dumps(dict(
                        src="rl",
                        step=step, anchor=n0, gain=float(gain[a0]),
                        len=int(s0[a0]), len_spec=int(s1[a0]),
                        p_think=float(np.exp(lpT[a0])),
                        think_rate=float(thf.sum() / max(vw.sum(), 1.0)),
                        shuf=float(gain_shuf[a0]),
                        context=tok.decode(
                            x[0, max(0, n0 - 40):n0 + 1].tolist()),
                        thought=tok.decode(
                            content[a0, :int(s0[a0])].tolist()) + " </T>",
                        spec_thought=tok.decode(
                            z[SPEC_ROW, a0, :int(s1[a0])].tolist()) + " </T>",
                        window=tok.decode(
                            x[0, n0 + 1:n0 + 1 + args.k].tolist()),
                        oracle_would_be=tok.decode(orc[a0].tolist()),
                        syn=bool(is_syn),
                    )) + "\n")
                    # ...and one SSL anchor for reference, EXPLICITLY TAGGED.
                    # This is what the model is TAUGHT (true continuation,
                    # teacher-forced) as opposed to what it WRITES. Never mix
                    # the two when judging whether the model learned to think.
                    if ssl.any():
                        a1 = int(np.argmax(np.where(ssl, gain, -1e9)))
                        n1 = int(layout.anchors[a1])
                        dump_f.write(json.dumps(dict(
                            src="ssl",
                            step=step, anchor=n1, gain=float(gain[a1]),
                            len=int(jlens[a1]),
                            context=tok.decode(
                                x[0, max(0, n1 - 40):n1 + 1].tolist()),
                            thought=tok.decode(
                                content[a1, :int(jlens[a1])].tolist())
                            + " </T>",
                            window=tok.decode(
                                x[0, n1 + 1:n1 + 1 + args.k].tolist()),
                        )) + "\n")
                    dump_f.flush()

        if args.sync_grads:
            if world > 1:
                tp.sync_gradients([p_ for p_ in model.parameters() if p_.requires_grad],
                                  world, xm, torch_xla.sync)
            torch.nn.utils.clip_grad_norm_(
                [p_ for p_ in model.parameters() if p_.requires_grad],
                args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=False)
        torch_xla.sync()
        if step == start_step + 1:
            # one-off replica check: with --sync-grads 1 every rank must print the
            # SAME checksum; without it they diverge after the first step.
            with torch.no_grad():
                _cs = float(sum(p_.double().sum() for p_ in
                                list(model.parameters())[:4]).cpu())
            print(f"[replica-check] rank {rank} step {step} "
                  f"sync_grads={args.sync_grads} checksum {_cs:.10e}", flush=True)

        tokens_seen += args.grad_accum * T * world
        dt = time.time() - t0
        if step % args.log_every == 0:
            g = args.grad_accum
            gm = float(np.mean(st["g"])) if st["g"] else 0.0
            gp = float(np.mean(np.array(st["g"]) > 0)) if st["g"] else 0.0
            gsp = float(np.mean(st["g_spec"])) if st["g_spec"] else 0.0
            gsh = float(np.mean(st["g_shuf"])) if st["g_shuf"] else 0.0
            gr0 = float(np.mean(st["g_r0"])) if st["g_r0"] else 0.0
            gsh0 = float(np.mean(st["g_shuf0"])) if st["g_shuf0"] else 0.0
            org = float(np.mean(st["or_g"])) if st["or_g"] else 0.0
            rce = st["read_ce"] / max(st["n_read"], 1)
            ice = st["imit_ce"] / max(st["n_imit"], 1)
            lm_ = float(np.mean(st["lens"])) if st["lens"] else 0.0
            lsp = float(np.mean(st["lens_sp"])) if st["lens_sp"] else 0.0
            tr = float(np.mean(st["think"])) if st["think"] else 0.0
            pT = float(np.mean(st["pT"])) if st["pT"] else 0.0
            gR = float(np.mean(st["gateR"])) if st["gateR"] else 0.0
            cw = float(np.mean(st["crowd"])) if st["crowd"] else 0.0
            vt.log(f"step {step:5d}/{args.max_steps} "
                   f"ntp1 {st['ntp1']/g:6.3f} "
                   f"g {gm:+6.2f} g+ {gp:.2f} gSp {gsp:+6.2f} "
                   f"gSh {gsh:+6.2f} ob0 {gr0 - gsh0:+5.2f} "
                   f"el {st['n_elig']:3d}/{st['n_anch']:3d} "
                   f"len {lm_:4.1f}/{lsp:4.1f} "
                   f"thk {tr:.2f} pT {pT:.1e} gR {gR:+5.2f} "
                   f"orG {org:+6.2f} imCE {ice:5.2f} rdCE {rce:5.2f} "
                   f"or/sm {st['n_or']}/{st['n_sm']} "
                   f"pOr {p_or:.2f} rl {rl_w:.2f} "
                   + (f"syn {st['n_syn']} scrEx {np.mean(st['scr_ex_g']):.2f} "
                      f"gPau {np.mean(st['g_pau_syn']) if st['g_pau_syn'] else 0:+.2f} "
                      if st["scr_ex_g"] else "")
                   + f"{dt:5.1f}s "
                   f"{args.grad_accum * T * world / dt:6,.0f} tok/s")
            if metrics_f:
                metrics_f.write(json.dumps(dict(
                    step=step, ntp1=st["ntp1"] / g, g_mean=gm, g_pos=gp,
                    rollouts=RO,
                    gamma=float(gam_now),
                    elig_w=(float(np.mean(st["elig_w"]))
                            if st["elig_w"] else 0.0),
                    boost=(float(np.mean(st["boost"])) if st["boost"] else 0.0),
                    g_win=(float(np.mean(st["g_win"])) if st["g_win"] else 0.0),
                    g_sd=(float(np.mean(st["g_sd"])) if st["g_sd"] else 0.0),
                    adv_sig=(float(np.mean(st["adv_sig"]))
                             if st["adv_sig"] else 0.0),
                    adv_absA=(float(np.mean(st["adv_absA"]))
                              if st["adv_absA"] else 0.0),
                    # 2026-10-07 10:07: `g_best=gm` REMOVED.  It logged the group MEAN under a "best"
                    # name on every row of all 50 runs 20-80 (found and recorded at JOURNAL:10382, then
                    # wired into the "best-of-G gain" row of five instruments by the 08:07 schema fix).
                    # The real best-of-R field here is `g_win`, which those instruments now read.
                    g_spec=gsp, g_shuf=gsh, g_r0=gr0, g_shuf0=gsh0,
                    gx_r0=(float(np.mean(st["gx_r0"])) if st["gx_r0"] else None),
                    gx_sd=(float(np.mean(st["gx_sd"])) if st["gx_sd"] else None),
                    g_buf=(float(np.mean(st["g_buf"])) if st["g_buf"] else 0.0),
                    n_elig=st["n_elig"],
                    n_anch=st["n_anch"], n_th=st["n_anch"],
                    len_mean=lm_, len_spec=lsp,
                    think_rate=tr, p_think=pT, gate_R=gR, crowd=cw,
                    eps_th=eps_th,
                    or_g=org, imit_ce=ice, read_ce=rce,
                    n_or=st["n_or"], n_sm=st["n_sm"], p_or=p_or, rl_w=rl_w,
                    imit_w=imit_w, ss_p=ss_p,
                    ss_sub=(float(np.mean(st["ss_sub"]))
                            if st["ss_sub"] else 0.0),
                    dagger_p=float(st.get("dagger_p", 0.0)),
                    dagger_sub=(float(np.mean(st["dagger_sub"]))
                                if st["dagger_sub"] else 0.0),
                    abst=(float(np.mean(st["abst"]))
                          if st["abst"] else 0.0),
                    pau=(float(np.mean(st["pau"]))
                         if st["pau"] else 0.0),
                    n_syn=st["n_syn"],
                    scr_ex_s=(float(np.mean(st["scr_ex_s"]))
                              if st["scr_ex_s"] else None),
                    scr_ex_g=(float(np.mean(st["scr_ex_g"]))
                              if st["scr_ex_g"] else None),
                    scr_dig_g=(float(np.mean(st["scr_dig_g"]))
                               if st["scr_dig_g"] else None),
                    scr_pv_g=(float(np.mean(st["scr_pv_g"]))
                              if st["scr_pv_g"] else None),
                    g_pau_syn=(float(np.mean(st["g_pau_syn"]))
                               if st["g_pau_syn"] else None),
                    writer_delay=_cur_delay,
                    span_off=_cur_span, bridge_now=_cur_bridge,
                    dt=dt, tokens_seen=tokens_seen)) + "\n")
                metrics_f.flush()

        if (step + 1) % args.ckpt_every == 0:
            vt.save_checkpoint(model, opt, step + 1, args, tokens_seen,
                               rank, world)
        if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
            vt.log(f"time limit {args.max_hours}h reached at step {step}")
            if (step + 1) % args.ckpt_every != 0:
                vt.save_checkpoint(model, opt, step + 1, args, tokens_seen,
                                   rank, world)
            break

    vt.log(f"done: {tokens_seen/1e6:.1f}M real tokens, "
           f"{(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
