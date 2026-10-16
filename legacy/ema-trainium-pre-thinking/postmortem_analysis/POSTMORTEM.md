# POSTMORTEM / BOOTSTRAP — thinking-in-pretraining, handoff to trn2.48xlarge

Written 2026-09-06 by the Claude session that ran runs 12–14 on the old
trn2.3xlarge, for the Claude session that will continue on a **trn2.48xlarge**.
This file is self-sufficient: the old instance's memory files do not travel.
The full scientific narrative is `overleaf/iclr2027/main.tex` (compiles with
`pdflatex main.tex`, 22pp) — read §5 (results), §7 (the scaled plan, THE
roadmap you are executing), and the appendix chronology table before touching
anything. The user is Emanuele (emanuele.lamalfa@cs.ox.ac.uk).

## 1. What this project is

Teach a 0.6B Qwen3-architecture model, pretrained from scratch, to emit short
free-form "thought" tokens *during pretraining*: at an anchor position it may
open `<T>`, write a few tokens, close `</T>`; the next real tokens are then
predicted with the thought visible. Loss lives on real tokens only; thought
content is credited purely through its measured counterfactual effect on the
true continuation's likelihood. Fully unsupervised.

## 2. Where we are scientifically (the compressed verdicts)

1. **The reader skill is real and trainable.** A trained reader converts
   leak-free hints (true text from *beyond* the scored window) into +1.6…1.9
   nats of window likelihood. First demonstrated in run 13. It grows with
   training and never stopped growing.
2. **The writer cannot be bootstrapped by RL at ≤2.1B-token maturity.**
   Three independent falsifications: run 12 (rejection sampling → pass rate
   exactly the calibrated noise floor), run 13 (REINFORCE on raw gain →
   register exploit: shuffled copies gain as much as thoughts), run 13′
   (REINFORCE on the honest shuffle-corrected gain → lawful convergence to
   silence: gate priced to never-think, lengths to the floor, mode collapsed
   to a universal `1)</T>`, and clean NTP eroded +0.25 held-out nats buying
   nothing). Conclusion in the paper: the binding constraint is the
   information content of the policy's own samples, not the mechanism.
3. **The anchored design (run 14, live at handoff) delivers stability.**
   Permanent imitation anchor (oracle micros floored at 50%, imit weight 1.0
   forever) + small always-on RL (0.15, shuffle-corrected, negatives damped
   4×) + recoverable gate (cheap prices, permanent 0.15 exploration floor).
   At 2,500 steps: NO collapse of any kind, lengths flat at ~5.9, drift
   bounded ~+0.3, oracle gain +1.6. Mean order-borne value (g − gShuf) still
   zero — but the **eligible rate tripled 0.007 → 0.025** (12× noise floor):
   the tail is moving, the mean is not. That tail is the thing to fatten.
4. **User's frame (a standing directive):** RL is *not* the fundamental
   teacher — imitation of the real future is — but RL is *necessary*,
   because it is the only mechanism that explores: a modified NTP loss can
   only push toward nameable targets, and the valuable thought is often a
   string no supervised target proposes. Never hand the writer over to RL;
   never remove RL entirely.

Held-out CE ledger (32 held-out seqs, `runs/eval_ckpt_cpu.py <ckpt> 32`,
**run from an activated venv** — see gotcha G7): vanilla 2.07B base 3.0024;
run-13 warm-up ckpt 3.0499; run-13′ post-RL ckpt 3.2553 (the cautionary
number). Recompute the baseline for the new ~6B vanilla ckpt before anything
else — 3.0024 belongs to the 2.07B base only.

## 3. What arrives on the new instance

- This repo (git). Everything dead was deleted 2026-09-06; what remains is
  load-bearing.
- Vanilla checkpoints trained to **~6B tokens** from a *different* sibling
  instance (user copies them; ~3× the maturity of any base we tested).
  Verify what you actually get: `ls checkpoints/`, load one on CPU, check
  tokens-seen metadata if present, and RUN THE HELD-OUT RULER on it first.
- Data/corpus also copied by the user. The corpus manifest is
  `pre-training/data/corpus/manifest.json`; shards live under `data/corpus`
  at repo root on the instance (see `TRAINIUM_SETUP.md` + `BUNDLE_README.md`
  for layout; bundle sits at repo ROOT, venv at `.venv`).

## 4. File map (post-cleanup; everything here is alive)

| file | role |
|---|---|
| `run_think.sh <run>` | THE launcher (generic plumbing: smoke-first, supervisor, memory watchdog, monitor, retries); all run-specific flags live in `runs/<run>/config.sh` |
| `keepalive_think.sh` | cron keepalive (`*/10 * * * *`) for the run named in `.watchdog_run`; stops at DONE or a failed smoke; resurrects the gain plot daemon; quarantines orphans; clears stale compile locks |
| `run_vanilla3.sh` | vanilla-ladder launcher (continue plain pretraining / make new rungs) |
| `pre-training/train/think_train.py` | THE trainer (mixed anchors, forced thinking, R rollouts, fp32 params; runs 18+) |
| `pre-training/train/think_par.py` | core module: parallel layout, masks, generation, scoring, weighted_ce3 — torch-only, CPU-testable |
| `pre-training/train/think_gen.py` | ScatterCache (KV cache used by think_par) |
| `pre-training/train/thinking.py` | token ids (`T_OPEN`, `T_CLOSE`=Qwen `</think>` 151668, `EOD`) + splice utils |
| `pre-training/train/train.py`, `data.py`, `model_init.py`, `evaluate.py`, `eval_ckpt.py` | vanilla training stack (think_train imports train as `vt`) |
| `pre-training/train/test_think_par.py` | CPU equivalence suite — `USE_TORCH_XLA=0 python test_think_par.py` (add `RUN_BIG=1` for full 0.6B geometry). ALL must pass before any launch |
| `pre-training/train/dev_ab_gen13.py`, `dev_topk_matrix.py` | on-device replay / A-B tools (the S3 rule, see G1) |
| `pre-training/train/monitor_think13.py` | status digest; RUNAWAY renamed DRIFT! (auto-kill disabled by user order — keep it that way unless he says otherwise) |
| `runs/eval_ckpt_cpu.py` | held-out CE ruler (the ledger tool) |
| `runs/plot_gain13.py` | 4-panel gain plot daemon; takes `[metrics.jsonl] [out.png]` args |
| `pre-training/think_ntp_monitor.py` | clean-NTP plot daemon |
| `eval/` | benchmark eval (LAMBADA/HellaSwag/SciQ/ARC-E gold-continuation LL, clean vs think mode) |
| `_old-claude-prompts/*.md` | design docs: `PARALLEL_THINKING.md` (run-13 design), `BASELINE_THINKING.md` (oracle-regret + paired-rollout baselines, partially still unimplemented), `SPEEDUP_THINKING.md` (the speed war) |
| `overleaf/iclr2027/main.tex` | the paper. UPDATE IT when results land; user wants the compiled PDF sent to him |

## 5. Hard-won engineering rules (violating these cost us days; do not relearn)

- **G1 — device top-k is broken. Never enable it.** neuronx-cc 2.27.x
  miscompiles (or crashes on) `torch.topk` over the 152k vocab at every
  shape tried — and at one shape it compiled *silently wrong*, feeding 310
  steps of garbage rollouts while CPU tests passed. `--topk-transfer 0`
  (full-logit transfer + host sampling) is load-bearing in the launcher.
  General form of the rule: **CPU equivalence certifies semantics, not the
  compiled artifact** — before trusting ANY newly compiled fast path, replay
  a step on-device against CPU ground truth (`dev_ab_gen13.py` pattern).
- **G2 — geometry is compile-once.** All masks/positions are per-run
  constants (T=1024, stride=8, k=8, L=12, first-anchor 64 → N=118 anchors,
  buffer S=3502, cache C=2440). Changing any geometry flag = full recompile
  (fine, just slow the first time). Batch is 1 sequence/rank by design;
  you scale by ranks, not batch (see §7).
- **G3 — never edit a running bash script's file in place.** bash reads
  incrementally; the safe update is write-new + `mv` over (the running
  process keeps its fd on the old inode). Python helpers that are exec'd
  fresh each cycle (monitor, keepalive) are safe to edit anytime.
- **G4 — pgrep/pkill patterns must be bracketed** (`[t]hink13_train.py`),
  or they match your own shell and kill it (bit us twice).
- **G5 — stale compile-cache locks deadlock the next job** ("Another
  process must be compiling"): a killed compiler leaves
  `/var/tmp/neuron-compile-cache/**/*.lock` with no `.neff` beside it.
  The keepalive clears them; if a job hangs at startup, check this first.
- **G6 — memory watchdog.** The old box (124GB, no swap) livelocked when
  parallel compilers ate RAM; the launcher's watchdog kills compilers below
  4GB avail. The 48xlarge has far more memory but also 12× the ranks — keep
  the watchdog, re-tune the threshold after you see real usage.
- **G7 — eval needs the activated venv**: `source .venv/bin/activate` first;
  bare `.venv/bin/python` fails (`libneuronpjrt-path` lives in `.venv/bin`
  and must be on PATH).
- **G8 — detachment discipline**: long jobs launch with `setsid nohup … &`,
  cron keepalive as the safety net, `--resume latest` semantics in the
  launcher. Background `&` swallows a preceding `cd` — use absolute paths.
  The Claude session must never be load-bearing.
- **G9 — smoke first, cron after.** The launcher runs a 3-step smoke before
  real training when no ckpts exist. Arm the cron keepalive only AFTER the
  smoke passes (a failing smoke + keepalive = relaunch loop).
- **G10 — audits are always-on, not lazy.** Every sampled micro scores its
  shuffled copy; the reward is `gain − gain_shuf` (order-borne value only).
  Whatever you change, keep the minimal-pair audit in the loop — every
  exploit we ever had would have been (or was) caught by it.

## 6. Bootstrapping the trn2.48xlarge (do in this order)

1. Read `TRAINIUM_SETUP.md` / `BUNDLE_README.md`; recreate `.venv` if not
   copied. **Discover the node yourself** — do not assume: `neuron-ls`
   (device/core count), `/proc/meminfo`, `nproc`. A trn2.48xlarge has ~16×
   the accelerator cores of the 3xlarge we tuned on (which exposed 4 logical
   cores; expect ~64, but TRUST neuron-ls, and check
   `NEURON_LOGICAL_NC_CONFIG`-style settings in the setup docs).
2. `USE_TORCH_XLA=0 python pre-training/train/test_think_par.py` and again
   with `RUN_BIG=1` — all green before any device work.
3. Held-out ruler on the incoming ~6B vanilla ckpt → this is your new
   baseline number. Record it in the paper's ledger.
4. Scale the launcher: `NPROC` (the launcher reads `runs/think14_anchor/
   nproc14` if present, else 4 — set it to what neuron-ls supports),
   `--grad-accum` (8 on the old box; with 16× ranks you may want 1–2 for
   the same global batch, or keep it and enjoy a 16× bigger batch — the lr
   5e-5 was tuned at global batch 32 seqs; scale thoughtfully),
   `BASE_M` env var (the tokens-seen offset of the NEW init ckpt, in
   millions — it only affects plot x-axes), and the smoke will recompile
   everything for the new world size (expect a long first compile; G5/G6).
5. Write `runs/<run>/config.sh`, `echo <run> > .watchdog_run`, launch
   `bash run_think.sh <run>`, watch the smoke, then arm cron (G9).
6. Start the plot daemons if the keepalive hasn't (`runs/plot_gain13.py
   <metrics> <out.png>`; `pre-training/think_ntp_monitor.py`).

## 7. The experimental roadmap (= paper §7; user-approved)

Priority order, each gated on the previous looking sane:

1. **Re-run the anchored recipe from the ~6B base.** Same flags first —
   this alone answers "does 3× maturity move the tail or the mean?".
   Watch: g−gShuf (the mean), eligible rate (the tail), oracle gain,
   lengths, drift, and the dumps (user ALWAYS wants verbatim thoughts).
2. **The two probes at every rung** (minutes each, before any training):
   leak-free oracle gain; bootstrap-viability (sample from the rung's own
   policy, score g−gShuf under the audit stack). Plot both vs maturity.
3. **Continue the vanilla ladder in parallel** (`run_vanilla3.sh`) toward
   10–50B tokens, rungs every ~5B — the big node makes this fast; the
   maturity curve is the paper's central future figure.
4. **Widen exploration** (this is where new implementation starts):
   R=4–8 sampled rollouts per anchor with paired-sample baselines (Kool et
   al. — each rollout baselines the others; sketch in
   `_old-claude-prompts/BASELINE_THINKING.md`), an entropy bonus or
   temperature schedule on thought rows, and the oracle-regret baseline for
   the gate (gain(z) − gain(oracle here) → the gate learns "is there
   anything worth knowing HERE"). The generator already decodes 2 rows;
   generalizing rows and scoring variants is straightforward in
   think_par.py — keep CPU tests + on-device replay (G1) for every change.
5. **Lengthen the horizon**: k=8 windows structurally favor paraphrase;
   return to the k=64 discounted objective (§2.3 of the paper) so
   abstraction can out-earn imitation. This is a geometry change (G2):
   bigger buffers, fewer anchors per sequence — the 48xlarge's memory is
   exactly what makes it affordable.
6. **Controls when results appear**: compute-matched vanilla twin, pause
   -token arm, pretrained Qwen3-0.6B endpoint (36T maturity).

## 8. Working with Emanuele (observed, verbatim where possible)

- Direct, fast, allergic to over-engineering: "this is not an engineering
  problem… stick to my intuitions, you put the coding capabilities and
  engineering effort to make it fast and scalable."
- Speed is a first-class requirement (a 10–16× overhead killed run 12's
  design independently of its science).
- Report honestly and lead with the verdict — he has accepted three
  falsifications without complaint; what he does not want is spin, or
  discovering a caveat late.
- He wants to SEE thoughts verbatim (dumps decode content before the stop;
  keep it that way) and asks "how are the thoughts" often.
- Update the paper when concrete results land, recompile, and SEND him the
  PDF; keep the appendix chronology table current — it is his audit trail.
- Big design pivots are his call; bring him the options with a
  recommendation. Small correctness/speed fixes: just do them and report.
- The run on the OLD instance is under a "never stop it, I will tell you"
  order — if you are the new instance's session, that's not your run, but
  the same spirit applies: guards may observe and report, only he decides
  to kill a run (the 13′ auto-kill was the right call then; he revoked
  auto-kill afterwards).

## 9. Open loose ends at handoff

- Run 14 still training on the old instance (step ~2,500+, unbounded); its
  final numbers/dumps are worth harvesting into the paper if the user keeps
  it alive a while — but the new-instance run from the 6B base supersedes
  it scientifically.
- `BASELINE_THINKING.md` items unimplemented: oracle-regret baseline,
  paired sampled rollouts (both green-lit conceptually in paper §7).
- min-think=2 floor: if stop-lengths ever mass exactly at 2, lowering it
  is the lever.
- The gate has never re-opened (p(<T>) at numerical zero in every RL run);
  the oracle-regret baseline is the designed fix. If it re-opens
  *selectively* (specific contexts, not a level shift), that's a headline
  result — check p_think in dumps against context type.
- lr 5e-5 was never swept; neither were len-cost 0.15 / think-cost 0.1 —
  affordable now (roadmap step 4 territory).

---

# CURRENT STATE (2026-09-10) — runs 15-19 on the trn2.48xlarge

Appended by the session that ran 15-19. Read this before touching anything;
several entries below are mistakes made *by that session* and cost real time.

## Where the science stands

| run | config | gap (g - gShuf) | orG | verdict |
|---|---|---|---|---|
| 15 | k=8, R=1, hard gate | -0.066 +/- 0.004 | +0.22 | 4th falsification |
| 16 | k=64, R=8, soft gate | -0.180 +/- 0.013 | -1.47 | reader broken by long horizon |
| 17 | k=64 + gamma curriculum | -0.078 +/- 0.007 | -0.55 | within 1.4 se of run 15's -0.067 (G59) |
| 18 | mixed anchors, FORCED thinking, raw-gain reward | -1.70 | -1.81 | model ACTS on thoughts; ordering costs ~1.7 nats |
| 19 | + L=64, k=128, T=2048, **fp32 params** | -0.45, rising t=5.6 | -1.2 | stopped at step 514 for the three-phase plan |
| 20 | **phase A**: run 18's config + fp32, from the 8.04B vanilla rung, **64 ranks**, 1B tokens | -0.95 (step 2300) | -1.2 | **killed at 2307** (2026-09-11): base degraded +0.56 held-out (G18) |
| 22 | **phase B at ssl-frac 0** (launched by another session 2026-09-13 09:59 from run 21 @7700) | B +0.56 (step 1400) | n/a | **killed at ~1435**: 4th handover failure -- register churn 7%->22% of winners, rollout spread 1.6->0.7, len on floor, held-out +0.042 @1300 |
| 23 | **phase B at ssl-frac 0.10**, otherwise = run 22 (lr 1e-5, 3800 steps, from run 21 @7700) | | | launched 2026-09-13 20:12 |
| 21 | = run 20 at **lr 1e-5** (min 5e-6, warmup 200); held-out probe in cron | A -1.71 / B +0.79 (end) | -1.4 | **DONE** 7700 steps, 1.009B tokens, 50.8h; held-out +0.155; elig 0.50, g_win -0.06; pre-registered MC pool vs vanilla 22990: +0.96 pts p=0.11 (NOT confirmed); LAMBADA -7.0 |
| 25 | reader trained on ONE unselected own thought (`--reader-on-ssl 0 --rl-content random`) | ob0 -0.149 +/- 0.015 | +3.9 | **STOPPED 1565**: calibration is real (held-out greedy -2.45 -> -1.17) and SATURATES; content unmoved |
| 26 | **pause-token control** (`--pause-len`), content-free filler in every slot | n/a (no writer) | n/a | **DONE 1500**: FILLER BEATS THE WRITTEN THOUGHT by 0.43 +/- 0.13 nats on run 25's own weights |
| 27 | = run 25 with `--baseline greedy` | ob0 -0.097 | +4.7 | **STOPPED ~825**: futility rule; filler bar REPLICATES run 25 (-0.355 +/- 0.148) |
| 28 | **near-horizon target** `--oracle-min-off 0 --oracle-max-off 4`, `--len-cost 0.15 --min-think 3` | ob0 -0.127 +/- 0.016 | +4.4 | **STOPPED ~860**: futility rule. Agreement 0.042 -> 0.113 (real!) but own-filler WORSE at every length |
| 29 | = run 28 with `--oracle-mode fixed --oracle-off 0` (speculative copy) | ob0 -0.32 | ~+10 (arithmetic) | **STOPPED ~780**: K2 failed and the K3 escape failed (agreement(m=2) 0.222 < 0.25). Direction real (0.116 -> 0.222), value not |
| 30 | = run 29 + `--slot-pos aligned` + **`--dagger-p 0.8`** (scheduled sampling in the slot) | n/a (G28) | n/a (G28) | **STOPPED ~690**: pre-registered step-600 bar own(1-11) >= 0.15 failed; exposure bias falsified and the teacher-forced ceiling it was meant to unlock FELL |
| 31 | run 30's design run LONG, no DAgger | n/a (G28) | n/a (G28) | **STOPPED ~861**: gate G1. own(1-11) 0.0716 @300 -> 0.0722 @600, FLAT. 4th independent measurement of the free-running floor |
| 32 | **abstention** `--abstain-mode freq --abstain-top 300` | n/a (G28) | n/a (G28) | **STOPPED ~1130**: own-filler -0.110 +/- 0.030 (m=4) -- but the writer was MUTE (space rate 0.88), so this is silence, not value (G39, G40) |
| 33 | = run 32 + `--pause-frac 0.5` (make the block free) | n/a (G28) | n/a (G28) | **STOPPED ~900**: gate M2. Block price +0.376 @400 then REVERSED to +0.262 @800, bar +0.55. Lost its stop action (10.45 -> 13.9 tokens) |
| 34 | = run 32 + `--abstain-loss-w 0.0` (make the writer speak) | n/a (G28) | n/a (G28) | **STOPPED ~886**: N2(b). Speaks (space rate 0.043 vs run 32's 0.881) and own-filler -0.520 +/- 0.067 (m=4), BELOW its own init. Value tracks volume, not precision -- the negative is complete. Think-mode probe @800 agrees: own-filler -0.812 +/- 0.082 PAIRED, -0.687 +/- 0.169 at matched length |
| 35 | **the bridge** `--bridge 32`: hide the last 32 context tokens from the WINDOW rows, leave them visible to the thought block -- the writer CARRIES instead of forecasting | n/a (G52) | n/a (G52) | **STOPPED ~250**: B1 (`g_r0 > +0.5` @300) unreachable, g_r0 -0.32 and falling. The RESULT is `bridge2_r35_200.json` (n=154, vs silence under the same mask): future +9.85+/-0.59, the EXACT hidden text -0.27+/-0.18, own +0.10+/-0.17, filler -0.03+/-0.11. The reader pays for forecast-shaped text and nothing else |
| 36 | bridge anneal (32 -> 0) | -- | -- | **NEVER LAUNCHED**: dead on arrival on run 35's own probe. Config kept unrun in `runs/think36_anneal/` |
| 37 | **learned code, pure RL** `--code-size 64 --code-base 151670`, alphabet init to a neutral token | n/a (G54) | n/a (G54) | **STOPPED ~420**: registered collapse-to-a-constant (g_sd 0.03 vs 0.6), ob0 -0.001+/-0.001 flat. Code rows 0.01532 -> 0.01609 vs 1.016 for normal tokens. A neutral init is a SYMMETRIC one and symmetry is an RL fixed point (G56) |
| 38 | = run 37 + `--code-teacher 1` (64-way hash of the upcoming window, 25% of anchors) | n/a (G54) | n/a (G54) | **STOPPED ~410**: H1 (rows 0.01617 @400 vs 0.05) and H2 (ob0 -0.0018, g_sd 0.031) both FAIL. imCE sat on `2*ln(64)/3 = 2.7726` -- the UNIFORM point -- for 300 steps. An uninformative target is not an outside force (G57) |
| 39 | **channel control**, bounded 300 steps: `--code-teacher 2` (hash of a SINGLE in-context token), `--ssl-frac 0.9 --rl-w 0.0` | n/a (G54) | n/a (G54) | **DONE at 300, K1 and K2 both FAIL**: imCE(150-199) 2.724 vs bar 2.00; code rows 0.01541 @200 / 0.01565 @300 vs bar 0.05 (lm_head 0.03605 -> 0.03760 vs 1.194). Recovered 0.141 of the 0.148 nats/symbol of CONTEXT-FREE marginal and 0.000 of the 4.011 in-context. The channel is the problem (G58) |

### The E1 / synthetic-chain programme, runs 40-59 (added 2026-10-01 12:07, G107 -- the rest of the backlog G105 recorded)

The 10:07 tick cleared runs 60-68 and named runs 40-59 as "pure archaeology I am naming rather than
letting the debt blur". Here they are. **The columns of the run-15..39 table (`gap`, `orG`) do not
apply**: this era's instrument was exact-match on a synthetic task (`EM(clean)` / `EM(filler)` /
`EM(tf)` / `EM(greedy)`, plus `pvls_greedy` for writer content), so that is the column. Four of the
twenty were **never launched** and are marked as such -- nobody may later read a config-only directory
as a run that happened. **Run 53 does not exist**: the number was skipped, there is no `think53*`
directory, and it is recorded here so nobody goes looking for it.

| run | the one change | ended | how it ended |
|---|---|---|---|
| 40 `think40_add` | first E1: synthetic addition, reversed scratchpad, filler at web anchors | ~1020 | **Design flaw, not plumbing.** The reader was never trained on a CORRECT scratchpad (`--reader-on-ssl 0`) and the writer sat at chance digit accuracy after ~200 synthetic updates (`scr_dig_g` 0.15 vs chance 0.10). |
| 41 `think41_addfill` | control: filler at EVERY anchor, no writer | **never launched** | Config-only directory (no `metrics.jsonl`). |
| 42 `think42_addvan` | control: NTP only on the addition mixture | **never launched** | Config-only directory (no `metrics.jsonl`). |
| 43 `think43_add2` | reader on the true scratchpad too, 50% synthetic, R=1 | ~800 | **The reader was trained to IGNORE the slot.** At step 500 the literal next tokens placed in the slot gained **+0.0 nats**, against **+11.0** for the same construction on the vanilla init. Web filler at every anchor plus an unreadable scratchpad taught it the slot was noise. |
| 44 `think44_addw` | fixed-width 8-digit addition, no web slots | ~1030 | Stopped by the read-health rule (`max(fwd,tf) - clean < +2`). Two facts, and the second killed addition as a positive control: **(a)** with aligned operands **one pass learns the addition** (`EM(clean)` 0.000 @500 -> 0.429 @1000 -- alignment, not carries, was the hard part in 40/43); **(b) the reader reads nothing** (`tf` 0.446 vs clean 0.429). A mirror-position test on the init showed why: **native slot reading is INDUCTION, needing forward token order** (forward hint +8.45, reversed-at-mirror-positions -1.80), not position matching. |
| 45 `think45_chase` | 5-hop pointer chasing, forward scratchpad | ~3090 | **The registered UNINFORMATIVE clause fired at its own @2000 gate**: `EM(filler)` **0.893** against a 0.85 ceiling (clean 0.902, tf/greedy/sample 1.000). The task had become solvable without reading the note, so no contrast it produced could mean anything. |
| 46 `think46_hops` | depth ladder, hops ~ U{3..10} | ~4000 | **Question answered, not failed**: the decay curve was read at 1000/2000/2500/3000/3500/4000 and the device moved on. Its `EM(tf) - EM(filler)` of **+0.074** became the yardstick run 52 beat six-fold. |
| 47 `think47_discover` | 1-value seed + reward linear in p(answer) | ~560 | **Design flaw: exploration dead by construction.** The 1-value seed supervised `</T>` immediately after the first value, so from step 200 **every** sampled RL thought was exactly `<v> </T>` (len 2.0, pv 1.00; 14/14 in the dumps). A reward cannot search a space the supervision has closed. |
| 48 `think48_seedonly` | control: seed only, no reward | **never launched** | Config-only directory (no `metrics.jsonl`). |
| 49 `think49_discover2` | 1-value seed with the stop removed + reward | ~1178 | Stopped by **its own AMENDMENT 1**: DEEP `pv_greedy` **0.926** against a 1.1 bar at step 1000 -- the writer never extended past the seed even with the stop supervision gone. |
| 50 `think50_few` | 5% full-chain demos + reward | ~1190 | Early-stopped by the headless supervisor **before** its registered @2000 read, because the run's **precondition was measurably false with the WRONG SIGN** at the registered @1000 trajectory read. |
| 51 `think51_fewctl` | control: 5% demos, no reward | **never launched** | Config-only -- and **this non-launch is the origin of a standing rule**: run 50's writer was at chance, so `run50 - run51` would have been `0 - 0` and 13 h of device would have bought a number with no content. **"A control that cannot distinguish anything is not a control."** That rule is what run 68 had to answer *before* it was allowed to exist (G105(8)), and answering it prospectively is the difference between the two. |
| 52 `think52_readeronly` | reader supervised (40% demos), writer unsupervised (`--imit-w 0`) | ~1160 | **R1a @1000 PASSED BY A LARGE MARGIN and the run still ended** -- by case (i) of its own amendment, registered pre-data. DEEP (hops>=8, n=149) `EM(tf)` **0.779** vs `EM(filler)` **0.329** = **+0.450** at McNemar p=6.8e-17: **six times run 46's +0.074 at the same step.** The reader can be taught to read a chain; that was never again in doubt. |
| 53 | -- | **does not exist** | The number was skipped. No `think53*` directory. Recorded so nobody hunts for it. |
| 54 `think54_explore` | attack exploration (repeat-ban, freq-pen, temp, 8 rollouts, min-think 6) | ~1052 | **X1 @1000 FAILED**: `g_sd` **0.0912+-0.0023** over steps 500-1000 against a 0.30 bar, and X1b `or_g` **0.0909+-0.0300** over 900-1000 against +0.5. The content clause passed at 30.81 tokens but **on off-task prose** (G67) -- length is not exploration. |
| 55 `think55_decouple` | reader decoupled from the sampled rollout (`--reader-on-ssl 2`) | ~1300 | **R @1000 PASSED** (reader DEEP `EM(tf) - EM(filler)` **+0.510**, p=5e-21) **and the WRITER was at exactly zero** (`scr_pv_g` 0.000, `EM(sample)` 0.000, no answer-vocabulary tokens at all) through its registered 1000-1300 futility window. A readable channel with nothing written into it. |
| 56 `think56_curric` | 1-hop curriculum + digit alphabet + answer reward | ~295 | **Check A as registered FAILED** (`scr_pv_g` 0.000 over 150-249) **but the metric was blind to a leading space**: an unselected space-tolerant probe of step 200 read **pv 0.98 greedy and 0.98 sampled**. **The first content discovery from reward in the programme** -- and the corrected read was recorded *beside* the failing one, never substituted for it. |
| 57 `think57_stop` | + stop exploration | ~850 | **Check A' @500 FAILED as registered**: hops 1-2 `EM(greedy)` **0.3646** vs bar 0.4188 (McNemar b=0, c=10). The split is the finding: **1-hop worked** (clean 0.938, `pvls_g` 0.938) and **hops >= 2 was at zero** (EM 0.000-0.021). |
| 58 `think58_mixed` | mixed-depth + complete-value floor | ~655 | **Check A @500 FAILED on BOTH clauses.** (i) 1-hop `pvls_greedy` **0.2708** vs a 0.90 bar -- run 57 read **0.938** at the same step, so **the run-56/57 discovery did not survive the uniform U{1..10} mixture**; (ii) hops 1-2 `EM(greedy)` **0.0000**. |
| 59 `think59_tail` | run 57 + complete-value stop floor, gate read after the length converges | ~1105 | **Check A @1000 FAILED on clause (i)**: 1-hop `pvls_greedy` **0.333** vs 0.90, and **declining** (0.562 @500 -> 0.333 @1000) rather than holding. Clause (ii) PASSED (`EM(greedy)` 0.438 >= `EM(clean)` 0.417 - 0.05). |

**What the twenty rows say together, and it is the reason the slide programme exists.** Runs **40-44**
were a search for a positive control that failed for a reason worth keeping: **the model reads a slot by
induction, needing forward token order, not by position matching**, so a reversed scratchpad is
unreadable no matter how well it is written. Runs **45-46** then showed the other half of the trap --
make the task easy enough to learn and it becomes solvable *without* the note, which is what run 45's
own UNINFORMATIVE clause caught. Runs **47-59** are **thirteen runs spent trying to get CONTENT out of
reward** on a synthetic chain: exactly **one** (run 56) produced a content discovery, it was found by a
probe that was not the selected metric, and it **did not survive a depth mixture** (58) or its own
length convergence (59). Meanwhile runs **52 and 55 both PASSED their reader gates by huge margins** --
`+0.450` and `+0.510` EM -- which is the fact that redirected everything: **the reader was never the
bottleneck; the writer's content was.** The slide programme (60-68) is that redirection, moved off the
synthetic task and onto real text, and it has since produced its own inversion (G107: the writer's
reward term buys specificity and pays for it in raw usefulness).

### The SLIDE programme, runs 60-68 (added 2026-10-01 10:07, G106 -- clearing part of the backlog G105 recorded)

The table above stops at run 39 and the eight-run gap was recorded as a debt at 08:07 rather than
quietly skipped. These rows are the part of that debt that **cannot be invalidated by run 68's
pending result**: what each run was, and how it ended. The columns of the run-15..39 table (`gap`,
`orG`) do not apply here -- the slide programme's verdict quantity is `own_minus_shuf` on held-out
anchors at the trained offset (00S item 4, G99.1), so that is the column.

| run | the one change | ended | how it ended |
|---|---|---|---|
| 60 `think60_slide` | first slide run | ~215 | **Not a test of the idea: two implementation defects.** Sampled thoughts never closed (len 16 = cap, no `</T>`, the G53 unterminated-block cost), and the compressed PAST span was written at FUTURE positions, so the reader read the hidden past as the coming text -- even a TRUE span cost -17..-22 nats. |
| 61 `think61_span` | span at its own positions + forced close | ~280 | **The READER stopped reading the slot.** Matched held-out probe o=-15: true span +15.82 -> +9.40, own +7.10 -> +4.70 over 200 steps. Cause: `--reader-on-ssl 1` with `--rl-content random` trains the reader through slots holding sampled rollouts (the run-54/55 G66 collapse). |
| 62 `think62_read` | `--reader-on-ssl 2` | ~149 | **G0 FAILED at step 100**: true span +11.88 (bar 13.8), own +5.79 (bar 7.1). Reader-on-ssl 2 did not stop the degradation; `or_g` tracked run 61 exactly. Suspicion recorded: writer IMITATION training erodes the reader through shared weights. |
| 63 `think63_noimit` | `--imit-w 0 --imit-floor 0` | ~1030 | **G1 @400 FAILED** (`g_r0`(300-399) -3.17, bar +1.0) and the held-out bar with it: own-filler +0.77+-0.67 against the untrained init's +6.93+-0.85, i.e. training destroyed ~89% of the note's value. Writer collapsed to a 1-2 token note (len 15.0 -> 5.7). **Diagnosis: `--reader-on-ssl 2` trains the reader only behind TRUE spans, so it over-trusts the slot and an imperfect own note costs ~5 nats versus silence -- shutting up is the writer's only defence.** |
| 64 `think64_calib` | `--reader-on-ssl 1` (calibrated reader) | ~485 | Pre-registered **G0 @200** clause fired; runbook 3c. |
| 65 `think65_nofiller` | drop the filler arm | ~1205 | **DERAIL (3b)**: rollout spread 0.51 (bar 0.60) and reg% 50.7 (bar 15); the writer collapsed to a CONSTANT ~3-token note (`g_r0` +0.01, `g_sd` 0.05). Trained-offset gate also failed. **Its headline win is RETRACTED -- it was measured on a broken ruler (G94); the verdict quantity is `own_minus_shuf`.** |
| 66 `think66_nolencost` | remove the length price | 2182 | **R7 @1800 @ o=-1 LOSS**: `own_minus_shuf` +1.16+-0.41 against a +2.30 bar (init +5.30+-1.41). The probe landed 24 min after a tick, so **the run overran its own stop by ~375 steps** -- the ops failure that produced the armed-daemon fix. 3b fired twice as well (reg% 33.5, spread 0.26): run 65's absorbing state, reached 900 steps later. |
| 67 `think67_xanchor` | `--xanchor-w 0.0 -> 1.0` | 1822 | **X5 @1800 FAILED**: +2.01+-0.56 < +2.30, killed by `armed_probes_r67.sh` within a minute, unattended. Passed X0-X4. **But read as a block it is not a gate story: pooled over 7 matched rungs run 67 MINUS run 66 = -0.034+-0.292 (the same run), and run 67 MINUS the untrained init = -1.377+-0.496, 6/7 negative (run 66: -1.343+-0.488).** The flag that made the reward BE the gated quantity bought 0.03+-0.29 nats. |
| -- `think67_advfloor` | `--adv-sig-floor 0.5` (7(f)) | **never launched** | Config-only directory. Registered as run 67's successor, then **RETIRED on 2026-10-01 on the test its own caveat demanded**: run 67 tied while its writer was 2.5x healthier than run 66's at matched steps (len 9.41 vs 3.67, `g_sd` 1.54 vs 0.74), so the failure mode 7(f) repairs did not occur, and a healthier writer bought 0.03+-0.29. Implemented, tested, unrun. |
| 68 `think68_norl` | `--rl-w 0.5 -> 0.0` | 1026 | **A CONTROL, NOT A LEVER, and the first run 60-69 to separate the writer's RL gradient from the reader/SSL machinery.** No win bar was registered. X0 @200 passed 09:23, V @600 passed 11:37 on the mix arm alone, and **P1 @1000 ENDED THE RUN AT 13:43 WITH AN ANSWER: pooled(run68-run67) over 3 rungs = -1.334+-0.417 (z -3.20), i.e. DELETING THE WRITER'S WHOLE RL GRADIENT MAKES THE EROSION WORSE.** H_RL refuted with a sign; the erosion is reader/slot-side and the RL term claws back ~45% of it. Stopped by its own clause 24 min before a tick, not by a supervisor. |
| 69 `think69_rlw1` | `--rl-w 0.5 -> 1.0` | **LIVE** | **DOSE-RESPONSE on the one term measured to have a sign, and the first run 60-69 to carry a win bar with a numerical basis** (`pooled(run69-init) >= +1.00`). X0 @200 **passed 15:13**; D1 @400 **passed 16:16**, matched in-training `D` vs run 67 = **+0.093+-0.070** against a bar of -0.123, which is the zero-dose value -- so on this window the three dose points 0.0/0.5/1.0 read **-0.123 / 0.000 / +0.093**, monotone with a shrinking increment. Remaining: 7 probe rungs from 600, P1 @1000 one-sided, W @1800. |

**What the nine rows say together.** Runs 60-62 were the channel being built and failing to be read;
63-64 established that the reader's calibration, not the writer, decides whether a note can be worth
anything; 65-67 are **three runs that each moved one well-argued flag and each landed on the same
number**, and the pooled reading in run 67's row is the reason that stopped looking like three ties and
started looking like one effect. **The eight "ties" of runs 60-67 were each measured against the
PREVIOUS run and read one rung at a time** -- a previous-run floor plus a multiple-comparison ladder
(G102(c), G104(d)). Pooled against the untrained init instead, both runs that have seven matched rungs
sit **~1.36 nats BELOW the model that was never trained at all**, and the two deficits agree to 0.03
nats. That is the only effect the slide programme has measured cleanly, and it points the wrong way.
**Still outstanding from the G105 backlog: `overleaf/iclr2027/main.tex` has no runs 60-69.** The rows
for runs 40-59 were added at 12:07 (G107) and that half of the debt is paid. The decomposition P1 was
being waited on **landed at 13:43** (G108(2)), so the stated reason for leaving the chapter has now
expired: what remains is that G108(3) changes how every figure in it must be drawn, and that run 69's
verdict is one paragraph and one row. **It is the single outstanding debt and it is a writing job, not a
device job.**

**The invariant**: wherever the reader works, order-borne value of the policy's
own thoughts is a small negative constant, unmoved by base maturity
(0.87->8.99B), horizon (k=8->64), search width (R=1->8) or reward density.
Runs 15 and 17 agree at -0.07 within one s.e. across an 8x horizon change.

**What run 18 changed**: forcing thinking (no gate) removed the "make thoughts
inert" escape. The model now *acts* on thoughts -- and acts wrongly, at a cost
of ~1.7 nats vs their own shuffle. Being wrong-and-costly beats being
ignorable: only the former has a gradient.

**Benchmarks** (benchmark/, CPU-only, runs alongside training): the objective
is capability-NEUTRAL. vanilla 45.8% macro, thinking ckpt 45.6%. The model
knows facts (sciq 72%) and reads (lambada 33%) and CANNOT REASON (winogrande
and arc_challenge at chance). 275M tokens on a 9B base is 3% more data -- no
benchmark could move.

## NEW GOTCHAS (G11-G16). G11-G14 were self-inflicted.

- **G11 - bf16 parameters FREEZE every normalisation weight.** RMSNorm gains
  sit at magnitude ~1.0 where bf16 spacing is 2^-8 = 0.0039; an Adam step of
  5e-5 rounds to zero EVERY step. Measured: 113/113 norm tensors bit-identical
  to init after thousands of steps in runs 15-18, while every projection
  (magnitude ~0.02) trained fine. `--zero1` only half-fixes it (fp32
  accumulator writes back to bf16 -> 0.4% ULP jumps, 112/113 still frozen).
  **Use `--param-dtype fp32`**; compute stays bf16 under autocast. Verified:
  0/113 frozen, values off the bf16 grid.
- **G12 - ZeRO-1 checkpoints are SHARDED: `step_N.optim_rank0.pt`**, not
  `step_N.optim.pt`. An orphan-checkpoint guard testing the exact non-sharded
  name declares every healthy ZeRO-1 checkpoint an orphan. This deleted three
  good checkpoints and restarted run 19 from zero, losing 500 steps. Always
  glob `"${m%.pt}".optim*.pt`.
- **G13 - a watchdog must not treat a MISSING log as an infinitely stale one.**
  During the smoke phase `think<N>.log` does not exist (the run writes
  `smoke_think<N>.log`), so an `AGE=999` default read as "stalled" and killed a
  healthy run mid-compile. A cold compile of new geometry legitimately runs
  40+ min silent; stall threshold is now 75 min over the NEWEST of both logs.
- **G14 - eval must replicate the TRAINING splice exactly, or it measures
  nothing.** Benchmark think-mode scored lambada at 0.0% until it reproduced
  both (a) the overlapping positions (thought t -> n+1+t, window j -> n+j) and
  (b) the mask that hides ctx[n] from the window row that re-emits it. Plain
  concatenation is a geometry the model has never seen -- the Section-3
  fractional-position failure in a new costume.
- **G15 - `NRT_RESOURCE` on Load = device resources exhausted**, usually a new
  graph compiled mid-run with no room. Seen at step 514 of run 19 (fp32 params
  + T=2048). Mitigation if it recurs: `--grad-accum 2 -> 1`.
- **G16 - the device-owner guard greps `[t]rain.py`, which matches
  `think1N_train.py` inside YOUR OWN shell command line.** Any command
  containing that string makes the next launcher abort with "another Neuron job
  is running". Launch with a command line that contains no `*train.py`
  substring. (Documented in BUNDLE_README; hit three more times anyway.)
  FIXED 2026-09-10: the guard now requires `_train\.py --corpus`, which no
  interactive shell command contains. The same trap exists for `pkill -f
  <name>` -- it kills the shell running it (exit 144); use `pkill -f "[p]lot"`.
  COROLLARY (2026-09-13, cost 4 aborted launches): the bracket trick only helps
  if the SAME command line does not also contain the plain string. A shell that
  runs `pgrep -fc '[t]hink_train.py --corpus'` contains "_train.py --corpus"
  and trips the launcher's guard while it lives; a shell that both greps
  `[r]un_think.sh` and launches `run_think.sh` kills itself. Rules: kill by
  numeric PID from a command that names nothing; launch from a command that
  contains no pgrep/pkill at all; count workers in a separate command.

- **G17 - an orphan guard must give a checkpoint time to finish.** The model
  `.pt` appears (atomic rename) up to ~2 min BEFORE its 6 GB fp32 optimizer
  file. A guard that runs in that gap quarantines a healthy checkpoint --
  watchdog.sh did exactly this to run 20's step 100 (2026-09-10 16:00).
  Both guards now ignore model files younger than 10 min (`find -mmin +10`).

- **G18 - bf16 parameters + plain AdamW froze HALF of every projection, not
  just the norms.** Round-to-nearest drops any update smaller than half a ULP;
  at lr 5e-5 that is every weight with |w| >= 2^-6 = 0.0156: 52% of q_proj,
  42% of down_proj, 64% of lm_head (init magnitudes ~0.02-0.04), and weights in
  [2^-7, 2^-6) moved by a full ULP (+22%). Runs 15-18 therefore trained the
  SMALL half of the network at an lr that was never really 5e-5. Run 20 (fp32)
  is the first run in which every weight moves at the nominal rate, and the
  base degraded: held-out mix CE 2.786 (init) -> 3.157 (step 1000) -> 3.351
  (step 2300), +0.56 nats, monotone; run 15's total tax at 2500 steps was
  +0.13. Weights moved 2x more than run 18's at the same step (q_proj 8.3% vs
  4.3%, lm_head 35% vs 20%); norms moved 0.3% and are NOT the cause. The
  training loss ntp1 hid this: it is measured on a deterministic batch stream
  (same seed -> runs 15 and 20 correlate 0.97 step by step) whose difficulty
  jumps at step ~150 in every run. Paired on identical batches, run 20 - run 15
  grows +0.11 -> +0.42 over 2300 steps. The vanilla ladder ran at a constant
  3e-5 with ZeRO-1 fp32 masters (all weights moving). Consequences: (1) every
  lr "tuned" under bf16 is void; (2) heldout_probe.sh now scores every
  checkpoint on the 32-sequence ruler into runs/<run>/heldout.tsv -- the ONE
  number for base health; (3) run 21 = run 20 at lr 1e-5, warmup 200.

- **G19 - since run 18 "g - gShuf" compares the rollout MEAN with the WINNER's
  shuffle.** Runs 16-17 scored a shuffle per rollout, so g_mean - g_shuf was
  like-for-like. Run 18 demoted the shuffle to one pass on the WINNER only, but
  digest/monitor/paper kept subtracting it from g_mean: that difference is
  biased negative (the winner's shuffle keeps the winner's topical advantage)
  and the "-1.7 nats confident wrong commitments" story was built on it.
  g_win - g_shuf (same thought both sides) is +0.8 in runs 18/20/21, +1.4 in
  run 19 -- biased UPWARD (a winner is selected partly for a lucky order), and
  it SHRINKS over training (+3.5 -> +0.8) while shuffle damage shrinks
  -13 -> -1: the reader learns to tolerate disorder faster than ordered
  thoughts gain. The unbiased instrument is the shuffle of a NON-SELECTED
  rollout (one more pass, monitor only) -- not yet in think_train.py.
  Corrected in the paper 2026-09-12. Run 21 @3600 = run 18 @2200 on every
  thinking metric (elig 0.46, len 10.6, imCE 3.82, orG -1.8, B +0.84).

## PRE-REGISTERED test for the end of phase A (written 2026-09-12 12:10, step ~4400)

Run 21 @4200 vs token-matched vanilla step_0021280 (clean mode, paired): macro
-0.3, LAMBADA -6.6 (p<0.001), but the five multiple-choice tasks
(arc_challenge, arc_easy, boolq, sciq, winogrande) pooled: 108 vs 73
discordant, +1.4 pts, McNemar p=0.011 -- post hoc. Run 19 (other rung) had the
same sign on arc_easy/sciq. The test that counts: `benchmark/pooled_test.py`
on run 21 @7700 vs vanilla step_0022990 (9.052B vs 9.038B tokens), the same
five-task pool, correct_norm. `benchmark/bench_final_r21.sh` is armed and runs
it when the checkpoint lands (-> benchmark/results/report_r21_7700.txt).
p<0.05 there = a result; otherwise today's number is noise and is reported as such.

- **G19 - the 'order-borne value' has been quoted with two different
  definitions.** `g_shuf` is the shuffle of the WINNER (best of R), so the
  same-thought minimal pair is `g_win - g_shuf` (+0.80 at the end of run 21).
  The number in the run table for runs 18-20 (and in the paper's tables up to
  2026-09-13) is `g_mean - g_shuf` (mean rollout vs the winner's shuffle),
  which mixes in the winner's curse (g_win - g_mean = +2.1 at R=8). Neither is
  an unbiased estimate of a typical thought's order-borne value: the winner
  was SELECTED on g (order + bag), so win-shuf is inflated by selection; at
  step 0, before any learning, it is already +3.5. Fix (one extra scoring
  pass): also shuffle ONE RANDOM rollout -> `g_rand - g_shuf(rand)`, unbiased.
  Not yet implemented.
- **G20 - at k=64 the supervised hint is text 64-80 tokens ahead
  (`oracle_content` = the L tokens BEYOND the window, by design, to be
  leak-free), and the reader cannot convert it into likelihood: orG plateaus
  at -1.3 (run 21, 1B tokens; runs 16/17: -1.45/-0.49). At k=8 (run 15,
  hint 9-20 ahead) orG crossed zero by step 1350. A negative ceiling means NO
  writer can be rewarded, so the RL channel's only reachable optimum is
  'harmless' -- which is exactly where runs 18, 20 and 21 all converged
  (winner gain -> 0, eligible -> 0.5 = best-of-8 of a distribution centred
  below zero). Phase B (lower rho) and phase C (k=128) cannot change this.
  A hint INSIDE the window is copyable (run 7) or already visible to the
  scored rows; the only lever is hint distance, i.e. k. Proposal: k=32 probe.

## Phase A/B outcomes (2026-09-14)

- Run 21 (phase A, lr 1e-5) complete: 7,700 steps, held-out +0.155 (flat from 5,000);
  eligible 0.14->0.50; winner = silence (-0.06); winner-vs-own-shuffle +1.6->+0.8;
  orG never positive; imCE floor 3.8 from step 1,000. PRE-REGISTERED MC pool vs
  token-matched vanilla: +0.96 pts, McNemar p=0.11 (NOT significant); LAMBADA -7.0
  (p<1e-3); SciQ +2.6 (p=0.035); think-mode neutral. Same sign on 3 checkpoints /
  3 rungs, ~+1 pt, unconfirmed. Compute-matched vanilla (@4200) beats it (p=0.002).
- Run 22 (phase B, ssl-frac 0) killed at 1,435 by the other session: register churn
  (`0 at least...` -> `in this` -> `and time`) 0 -> 20% of winners, rollout spread
  1.7 -> 0.9, lengths at the 8 floor, while eligible/g_win ROSE. Fourth handover failure.
- Run 23 (phase B, ssl-frac 0.10) stable and static at 1,900: register 1%, spread 1.7,
  len 10, eligible 0.55, win-shuf +0.79, held-out +0.064. Nothing moving for 1,000 steps.
- NOTE: two Claude sessions drive this box (the other launched runs 22/23 and the final
  benchmark). Coordinate via .watchdog_run and STATUS; never pkill -f a pattern that
  appears in your own command line (G16).

- **G20 - `pgrep -f`/`pkill -f` match YOUR OWN shell whenever the same command
  line also LAUNCHES the thing.** The `[p]attern` bracket trick only protects
  the pattern token itself; a later `python probe_x.py` in the same heredoc
  still matches `[p]robe_x` and the shell kills itself (exit 144). Three
  self-kills on 2026-09-13/14. Rule: search in one command, kill by numeric
  PID in a second, launch in a third. Never combine. **See also G50 (2026-09-22): a second `pgrep` on the same line defeats the bracket exactly as a launch does, and a bare `pgrep -c` is uninterpretable.**

- **G21 - `reg%` in `agent/status.py` is a top-3-prefix SHARE and is not
  comparable across blocks of different size.** The denominator is the number
  of RL thoughts logged in the block (~2 per step), so the top-3 share falls
  mechanically as the block grows. A 15-step partial block showed reg% 7.1 on
  2026-09-17 (run 25 step 600-614) next to 1.5 for the preceding full 300-step
  block, and the 16:07 tick flagged it as possible register collapse. It was
  an artifact: that window had **35 distinct thoughts out of 35 samples**, and
  the apples-to-apples 100-step series was *falling* (5.0% -> 2.0% from step 0
  to 600, 199/200 distinct). Before acting on the reg% derail criterion, always
  recount over equal-size windows and check the distinct/total ratio; a real
  collapse shows distinct << total (run 22: one prefix took 22% alone).

- **G22 - killing a 64-rank run is NOT instantaneous, and a process count is
  not proof that it stopped** (2026-09-18, stopping run 25). SIGTERM went to
  the torchrun elastic agent and all 64 rank PIDs at ~00:29. The agent takes
  the signal inside `time.sleep(monitor_interval)` and then runs a *graceful*
  teardown, so rank 0 kept genuinely computing: `metrics.jsonl` advanced from
  step 1524 to **1565** and was last written at **00:37:55**, roughly eight
  minutes after the kill. Meanwhile `ps -eo cmd | grep -c 'think_tr[a]in.py
  --corp[u]s'` returned **0** at ~00:30 -- a reading flatly contradicted by the
  metrics timeline. (It was not `ps` truncation: the same idiom returns the
  right count when re-tested. Whatever the cause, a single count taken during
  teardown cannot be trusted.) Nothing was harmed only because run 26's smoke
  started at 00:38:33, 21 s after run 25's last write. **Rule: verify a stop by
  watching `runs/<run>/metrics.jsonl` stop growing for ~a minute, then launch.**
  Two 64-rank jobs on 64 NeuronCores is a night lost.
  Found in the same incident: **`keepalive_think.sh` did not honour `STOPPED`.**
  Its guard was `^(DONE|FAILED \(.*smoke)`, while `agent/SUPERVISOR.md` 3b has
  always instructed the supervisor to write `STOPPED (...)` *first* "so keepalive
  won't relaunch". A stopped run still named in `.watchdog_run` would therefore
  have been resurrected within 10 minutes; run 25 escaped only because
  `.watchdog_run` was repointed to think26_pause first. Guard now reads
  `^(DONE|STOPPED|FAILED \(.*smoke)` (backup `/tmp/keepalive_think.sh.bak`).

- **G23 - `ps | grep -c <pattern>` counts YOUR OWN shell, and the `[p]attern`
  trick does not save you.** Two consecutive ticks were misled by it in opposite
  directions. 2026-09-18 00:30: a count of 0 while run 25's ranks were still
  computing (G22). 2026-09-18 02:21: a count of 4 `run_think.sh` supervisors that
  looked like duplicate-supervisor corruption. Both were artefacts.
  The bracket idiom `grep "[t]hink_train"` only stops the *grep* from matching
  itself. It does nothing about (a) the enclosing `sh -c`/`bash -c` process, whose
  cmdline contains every literal string in your compound command -- so writing
  `grep think_train` anywhere earlier in the SAME command line makes the shell a
  match; and (b) the shell from an EARLIER tool call that launched the run, which
  stays alive as the supervisor's parent with the full `setsid nohup bash
  run_think.sh <run>` text in its cmdline.
  What to do instead:
  * never mix a literal and a bracketed form of the same pattern in one command;
  * count with `ps -eo pid,ppid,etime,cmd | grep "[p]attern"` and LOOK at the rows
    -- a real 64-rank run is 1 supervisor + 2 child subshells + 64 ranks + 1 agent;
  * for "is the supervisor healthy", trust `grep -c "supervisor start"
    runs/<run>/supervisor.log` (must be 1 per launch), not a process count;
  * for "is the run alive or dead", trust `runs/<run>/metrics.jsonl` growing or
    static (G22). Process counts are the least trustworthy instrument on this box.

- **G24 - the probe scripts keep bare single-letter globals at module scope; a
  loop variable you add WILL collide with one.** `probe_oracle_offset.py` holds
  `T, K, L, GAMMA` (K = 64 = the scored-window length), plus `LEN, pos, mask, rng,
  SPACE`. Adding the precision ladder on 2026-09-18 with `for K in MIXK` rebound
  K to 12; `layout` had already been built with K = 64, so the FIRST sequence was
  fine and the SECOND crashed in `score_buffer` with
  `shape '[14, 64]' is invalid for input of size 168` (168 = 14 anchors x 12).
  A second edit nearly shadowed the `pos` positions tensor with `pos = []`.
  Two lessons: prefix any new loop variable (`MK`, `posacc`), and note that BOTH
  bugs were invisible on sequence 1 -- **always smoke a probe at NPER=1, which is
  3 sequences across the 3 domains, never at 0 and never by eyeballing the diff.**

- **G25 - `run_think.sh`'s device-owner guard matches YOUR OWN monitoring shell,
  and it aborts the launch.** The guard is `pgrep -f "_train\.py --corpus"`. On
  2026-09-18 08:13 the launch of run 28 died instantly with
  `ABORTED (another Neuron job is running -- stop it first)` while zero trainer
  processes existed: the match was a *backgrounded verification loop I had started
  myself*, whose command line contained the literal string
  `pgrep -cf 'think_train.py --corpus'`. This is G23's twin -- and note the comment
  in `run_think.sh` claims a shell that "merely MENTIONS a trainer file name never
  matches", which is only true for the file name alone, not for the name plus
  `--corpus`. **Rule: when you write a loop that polls for trainer processes, kill
  that loop before launching the next run, and check with
  `pgrep -f '_train\.py --corpus'` -- the guard's own pattern -- not with a pattern
  of your own.** Count real trainers with
  `ps -eo pid,args --no-headers | awk '/think_train\.py/ && !/awk/ && !/bash -c/'`,
  which excludes shell wrappers.
  **2026-09-19 00:14, SECOND OCCURRENCE, and it widens the rule.** Run 31's launch
  aborted the same way with zero trainers alive. There was no polling loop this
  time: the launch and a verification `pgrep -c -f 'think_train.py --corpus --stride'`
  were in ONE `bash -c`, so `run_think.sh`'s guard -- which runs ~1 s after the `&`
  -- matched the *launching shell's own command line*, which contains the pattern as
  literal text. **The rule is not "kill your polling loops first", it is: the string
  `_train.py --corpus` must not appear ANYWHERE in the command line of the shell that
  launches `run_think.sh`, including in a verification chained after the `&`.** This
  is runbook G20 generalised from `pkill` to any process-matching text. Launch in one
  call; verify in the NEXT call, which is a different shell. Cost: one wasted minute
  and a spurious `ABORTED` in `runs/think31_long/supervisor.log` at 00:14:21 --
  harmless because the guard fails CLOSED, but it will look like a real abort to a
  later tick reading the log, so it is written down here.

- **G26 - the SUCCESS BAR is a DIFFERENCE of two probe runs, and until 2026-09-18
  10:07 the probe threw away everything needed to compute its error bar properly.**
  `probe_thinkmode.py` wrote only `summary` (mean/se/n) plus six example texts, so
  own-vs-`--pause` could only be combined as if the two runs were INDEPENDENT:
  `se = sqrt(se_own^2 + se_pause^2)`. They are not independent -- the traversal is
  deterministic (fixed domains, fixed `eval_batches`, fixed NPER; with `--pause` the
  generator is never called, so the rng cannot desynchronise it), so the two runs
  score the SAME anchors in the SAME order and the comparison is PAIRED. Anchor
  difficulty is the dominant variance component, so the unpaired se is materially
  too wide: run 27's bar had to be reported as -0.355 +- 0.148 (2.4 se) when the
  paired se is likely far smaller. The script now also dumps `per_anchor` (the six
  raw lists, traversal order); for any probe pair run from now on, compute the bar
  as the mean of the elementwise difference. **The old JSONs (runs 25, 26, 27)
  cannot be repaired without re-running them -- do not silently quote a paired se
  for those three.**


- **G27 - `CCOM WARN Timeout waiting for RX (waited 120 sec) - retrying` is the
  CHECKPOINT WRITE, not a dying run. Do not kill a run for it.** Run 28 logged it
  twice and both times the step counter stalled ~4 minutes and then resumed
  normally. The timestamps line up exactly with the 9 GB checkpoint write (3 GB
  weights + 6 GB optimiser):
      step 200: .pt 09:35:39, .optim.pt 09:39:23, CCOM WARN 09:37:39
      step 300: .pt 10:15:30, .optim.pt 10:19:18, CCOM WARN 10:17:30
  Rank 0 blocks on the write, the collective's 120 s soft timeout fires, it
  RETRIES and recovers. The give-away is the word "retrying" and a `metrics.jsonl`
  that resumes on its own within ~5 minutes. **Diagnosis procedure: watch
  `metrics.jsonl` grow (the same instrument as G22), do not read a process count
  and do not read the warning.** Note the interaction worth remembering: step
  100's checkpoint took 65 s and produced NO warning, while steps 200 and 300 took
  3 min 44 s each and both warned -- the difference is heavy CPU probes competing
  for IO and page cache. Probes do not merely share cores with training, they
  lengthen its checkpoint writes into the collective's timeout window. That is the
  real reason for the runbook's 3-heavy-jobs cap.

- **G28 - A CONFIG COPIED FROM THE PREVIOUS RUN INHERITS THE PREVIOUS RUN'S KILL
  RULES, AND A MONITOR READ OFF A MUTATED BUFFER STOPS MEASURING WHAT ITS NAME
  SAYS.** Both halves of this were found in run 30's config by the 2026-09-18
  20:07 tick, hours before either could do damage, and either one alone would
  have killed a healthy run.
  (a) Runbook rule 5 says "copy the latest config, change what you mean to
  change". Run 30's header was written on top of run 29's and kept run 29's
  pre-registered `(K1) step 300: imCE > 3.10 -> STOP`. But run 30 sets
  `--dagger-p 0.8`, which SUBSTITUTES the model's own tokens into the SSL block
  input while the CE target stays the true continuation -- so imCE is the cost of
  a FIXED target under a progressively HARDER input and must RISE as p_own ramps
  0 -> 0.8 over steps 100-900. K1 would have stopped run 30 at step 300 for
  doing exactly what it was designed to do. `MAX_STEPS`'s own comment still said
  "kill rules fire at 300 and 800".
  **Procedure: when a config is copied, the inherited PREDICTIONS and KILL RULES
  are stale until re-read line by line. Void them explicitly and in place rather
  than leaving them to be discovered by whichever tick trips over them.**
  (b) The same substitution silently voids `orG`. `or_g` (think_train.py l.677)
  and `imit_ce` (l.679) are both read off the buffer built from `orc_content`,
  which l.479 has already overwritten with own tokens. So orG is no longer "the
  gain of the true continuation" -- it must FALL, and run 29's header note that
  "with offset 0 orG will read ~+10" is void for run 30, which opened at +8.4.
  **Generalisation worth keeping: every monitor named after the quantity it used
  to measure keeps its name after you change what is in the buffer. When a run
  mutates a tensor that a monitor reads, list the monitors it invalidates in the
  config header in the same edit.**
  (c) Third instance of the same class, found in the same read: `st["dagger_p"]`
  was assigned at l.480 but was NOT in the explicit keyword list that builds the
  `metrics.jsonl` record, while `ss_p` (the earlier, UNUSED scheduled-sampling
  mechanism) was. So run 30 logged a constant 0.0 for a mechanism it does not
  run and logged nothing at all for the ramp its entire hypothesis rests on -- a
  later tick reading `ss_p: 0.0` would have concluded scheduled sampling was off.
  Fixed the same tick: the record now carries `dagger_p` and `dagger_sub` (the
  realized substitution rate over the supervised rows). **Setting a key on the
  stats dict is not logging it; the record is an explicit keyword list.**

- **G29 - A PROBE HARDCODES THE DEFAULTS OF THE TRAINER IT WAS WRITTEN AGAINST,
  SO THE FIRST RUN THAT CHANGES A DEFAULT IS MEASURED IN THE OLD REGIME -- i.e.
  the probe silently stops testing THE ONE THING THE RUN CHANGED.** Found by the
  2026-09-18 22:07 tick, ~90 minutes before the number would have decided run 30.
  `probe_tfslot.py` was written at the 16:07 tick against run 25 and called
  `tp.build_positions(layout)` and `tp.ParGenerator(...)` with no `slot_pos`,
  both of which default to `"built"` (thought token i at n+2+i, runs 13-29).
  Run 30 is the first run to train with `--slot-pos aligned` (token i at n+1+i).
  So `runs/probes/tfslot_r30_300.json` scored run 30's weights, AND generated its
  free-running thought, one position out of the register the run was trained and
  is deployed in. The pre-registered SECONDARY bar for run 30 is `own@1 >= 0.3617`
  and its stated purpose is "this bar tests `--slot-pos aligned`, not DAgger" --
  measured with built positions it cannot test it at all, by construction.
  Magnitude is not hypothetical: `probe_slotpos.py` measured the same mismatch in
  the other direction on run 25 @900 and it was worth 0.035 of in-slot accuracy
  and 0.85 nats of reader gain.
  Fixed: `probe_tfslot.py` takes `slot_pos` as a 4th argv, default `"built"` so
  every runs-13-29 probe stays reproducible (verified byte-identical positions),
  and both the scoring pass and the generator are switched together, as
  `build_positions`' own docstring demands. `agent/tfslot_r30.sh` now takes each
  checkpoint in ITS OWN register -- run 29 built, run 30 aligned, which is the
  apples-to-apples comparison because it is the register each model is deployed
  in -- and takes the built reading second, only to decompose register vs DAgger.
  **Procedure, and it is the general form: a run that changes a TRAINER DEFAULT
  must, in the same edit, grep every probe that will judge it for that default.
  The instruments were written against the old default and will not error --
  they will return a plausible, wrong, LOWER number, which is the most expensive
  kind of wrong because it reads as a negative result.**
  **SAME-TICK CORRECTION, and it cuts the other way -- record it so the next
  reader does not inherit my overstatement.** Running both conventions on run 30
  @300 side by side showed the `own` column barely moves (own@12 0.047 built vs
  0.045 aligned; own@1 IDENTICAL to three decimals at every sequence). Only
  `tfslot` moved (@12 0.305 -> 0.322). So the mismatch was real and the fix is
  right, but at step 300 it was worth ~0.002 on the deciding column, not the
  0.035 the run-25 slotpos probe suggested. **Two lessons, and the second is the
  bigger one.** (i) A default-mismatch bug is worth measuring before it is worth
  believing; I asserted the magnitude from an analogous probe and the direct
  measurement was ~20x smaller. (ii) The side-by-side exposed something the
  magnitude question would never have reached: `own@1` is the argmax of the <T>
  ROW, and `build_positions` puts <T> at n+1 in BOTH conventions (`off` shifts
  only block positions 1..L-1, think_par.py l.93-94). So run 30's pre-registered
  SECONDARY bar (own@1 >= 0.3617), whose stated purpose in the config header was
  "this bar tests --slot-pos aligned, not DAgger", is invariant to `--slot-pos`
  BY CONSTRUCTION -- and it was already known to be invariant to `--dagger-p`
  (`take[:, 0] &= False`). It therefore tests NEITHER of run 30's interventions
  and is a drift monitor on the opener, nothing more. Retracted in the config
  header the same tick; run 30 rests entirely on its primary bar.
  **The general rule: before pre-registering a bar, check that the quantity it
  measures can be moved by the thing you changed. An invariant bar will still
  produce a number, and the number will still look like a result.**

## THE PROBE THAT CHANGED THE DESIGN (2026-09-14, runs/probes/oracle_offset_*.json)

Zero-update, 36 held-out sequences, 458 anchors, training-faithful splice, 12-token
hint + </T>. Gain of a PERFECT hint over silence (nats, discounted k=64 window) by
how far ahead the hint's text comes from:

    offset ->        0      8     16     24     32     48     64(=training oracle)  shuf0  shuf64  empty
    reader run23  +8.85  +6.74  +6.29  +5.13  +4.00  +1.46  -1.64 (win 27%)        -0.14  -1.81   -0.82
    vanilla       +8.10  -1.02  -2.67  -4.73  -6.28  -9.29 -12.22                  -7.76 -12.20   -6.12

The reader is strong and ORDER-sensitive for near-future content (o0 +8.9 vs its
shuffle -0.1), generalises to unlocated spans it was never given as oracles, and
values the one target every run 13-23 imitated -- the L tokens BEYOND the window
-- at less than nothing. imCE floored at 3.8 because that target is unlearnable;
orG never crossed zero because it is unusable. The writer was taught the wrong
thing for 3B tokens. Fix = run 24 (`--oracle-mode jitter --oracle-min-off 8
--oracle-max-off 48`): supervised thought = a span from inside the window, 8-48
tokens ahead. First steps: orG +2 to +6 at supervised anchors (was -1.4).
Unbiased order-borne value `ob0` (rollout-0 shuffle) now logged: -0.8 at start.

Phase B closed: rho=0 (run 22) -> register churn collapse, killed at 1435;
rho=0.10 (run 23) -> 3,800 steps, static at the phase-A plateau, held-out 3.036
(+0.095). Phase C (L=64 sparse budget) DEFERRED: ~80 s/step and pointless
until thoughts have value.

## AUTONOMY (2026-09-17)

Emanuele delegated ownership: launch, stop, redesign without asking; the goal is to BUY a capability
(own unselected thought improves prediction at inference), NTP loss acceptable. Headless supervisor:
`agent/tick.sh` (cron `7 */2 * * *`) runs `claude -p` with `agent/SUPERVISOR.md`; memory across wake-ups
is `agent/JOURNAL.md`; state dump `agent/status.py`. It stands down while `agent/INTERACTIVE_HEARTBEAT`
is < 100 min old (interactive monitors touch it).

Run 24 (sketch target) DONE 2026-09-15: orG +7.4, own sampled thought -2.2, ob0 -0.2, held-out +0.086.

Run 25 (`think25_calib`, 2026-09-17) STOPPED 2026-09-18 at step ~1530 (futility, runbook 3c).
Reader trained only on one random unselected own thought (`--reader-on-ssl 0 --rl-content random`),
RL 0.5 / neg 0.5, greedy row scored (`--score-spec 1`), from run 24 final.
**It is the biggest held-out win the project has had, and it bought the wrong thing.**
Held-out inference probe (`probe_thinkmode.py`, 465 anchors, unselected own thought):

    run 24 final (= run 25's starting line)   greedy -2.446 +- 0.154   sample -2.894 +- 0.207
    run 25 @  900                             greedy -1.169 +- 0.102   sample -1.287 +- 0.117
    run 25 @ 1500                             greedy -1.257 +- 0.137   sample -1.621 +- 0.164

The cost of thinking at inference HALVED (-2.45 -> -1.17), and it is real: unselected, no oracle,
held-out. But it all happened by step 900 and then stopped dead (900 -> 1500 is -0.09 +- 0.17).
Training-side, the whole move was a level shift between steps 300 and 500, in lockstep with
`orG` falling +6.3 -> +3.9: the reader learned to DISCOUNT hints it should not trust. After step 900
`gSp` slope was -0.013 +- 0.012 /100 steps (flat) and `ob0` sat at -0.149 +- 0.015, never once
positive across 1,500 steps -- a SHUFFLED own thought reads as well as the ordered one.
Verdict: what run 25 bought is reader CALIBRATION (and possibly just extra compute), not thought
CONTENT. Thinking still costs ~1.2 nats/window; it is merely much cheaper than it was.

Run 26 (`think26_pause`, 2026-09-18 00:38 -> 02:01) DONE at 1,500. The pause-token control
(`--pause-len 10`), same init as run 25: content-free filler in every thought slot, no decode,
no rollouts, no writer loss, reader trained on the same window rows. **VERDICT: FILLER WINS,
and it is not close.** Training-time `g_r0`, block by block, against run 25 at the same steps:

    block        run 25 (own written thought)     run 26 (content-free filler)
    0-299              -2.12                            -0.49
    300-599            -1.71                            -0.14
    600-899            -1.71                            -0.07
    900-1199           -1.46                            -0.05
    1200-1499          -1.53                            -0.03

Filler is ~1.5 nats better at EVERY block and converges to ~zero cost, i.e. the extra positions
are free while the written thoughts are not. Held-out (`probe_thinkmode.py --pause 10`, the
condition added 2026-09-18) confirms it on run 25's OWN step-900 weights -- same reader, same
splice geometry, the only thing removed is what the thought says:

    run 25 @900, its own written thought       greedy -1.169 +- 0.102  sample -1.287 +- 0.117
    run 25 @900, content-free filler           greedy -0.736 +- 0.087  (same weights, det.)
    run 26 @1500, content-free filler          greedy -0.083 +- 0.021  (its native condition)
    run 26 @1500, a real written thought       greedy -10.355 +- 0.375 sample -11.640 +- 0.353
  ALL FINAL (n=465, 2026-09-18 04:0x tick). The 02:07 tick quoted partials (~-0.40 / ~-0.03 /
  ~-10.4) from a fineweb-edu-only prefix; only the run-25 filler moved materially (-0.40 ->
  -0.736), because dclm drags every mean down. THE WITHIN-WEIGHTS NUMBER, which is the bar:
    disc_greedy(own) - disc_greedy(filler) = -0.433 +- 0.134  ->  the thought LOSES to filler
  by >3 se even unpaired (the two conditions share the same 465 windows, so this understates
  the precision). Lengths are matched: own greedy 9.4 tokens vs a 10-token filler.

Honest caveat: run 26's reader trains on 100% of anchors (`--ssl-frac 0`) vs run 25's 75%, an
asymmetry that favours the control -- but nowhere near 1.5 nats' worth.
So the pre-registered consequence stands: **run 25's celebrated -2.45 -> -1.17 was extra compute
plus recalibration, not thinking.** A written thought is worse than leaving the slot blank.

One more result from the same batch, and it corrects the run-25 diagnosis: feeding run 26's
filler-trained reader a REAL written thought costs **-10.4 nats greedy / -11.6 sampled** (final; the 02:07 tick's -12.8 was a fineweb-edu partial). So the reader is *order*-blind,
not *content*-blind -- `ob0` ~ 0 says shuffling a thought's tokens is free, yet changing the
register that occupies the slot costs 12 nats. It reads a thought as a bag of words, never as a
sequence. Do not write that the reader "ignores" thoughts; it does not.

**THE CENTRAL FACT, everything on one "nats over silence" scale.** Perfect 16-token future hint
+11.0; content-free filler ~0; the model's own written thought -1.2; a best-overlap in-context
*recall* span -3.09 and a *random* context span -2.69 (`runs/probes/recall_r24_3800.json`).
The reader has enormous unused capacity, and every content source we can actually produce at
inference is worse than silence. That is the bottleneck -- not calibration, not reward shaping.
This also **falsifies the queued "recall thoughts" hypothesis before it ever ran**: its own
precondition was "reader values it >= +2 nats"; it fails by 5.

Run 27 (`think27_greedy`, launched 2026-09-18 02:15): greedy-baseline RL from run 25's final
(calibrated) checkpoint, exactly one flag changed vs run 25 (`--baseline greedy`), so a sample is
reinforced only when it beats the thought the model would actually emit. 2,000 steps, with a
pre-registered futility rule at step 800 (`ob0`(500-799) >= +0.05 OR `g_r0` +0.30 over block 0-299)
and a pre-registered prediction of its failure mode, both written into
`runs/think27_greedy/config.sh` before launch.
**THE SUCCESS BAR IS NOW THE FILLER FLOOR** (also written into SUPERVISOR.md §1): a checkpoint must
beat `probe_thinkmode.py <ckpt> ... --pause 10` on its OWN weights. "Better than the previous run"
and "g_r0 > 0" are both retired -- run 26 showed they can be cleared by buying compute.

**Run 27 STOPPED 2026-09-18 08:10 at step 826 by its own pre-registered futility rule.**
Block 500-799 vs block 0-299:
    ob0(500-799)  -0.097 +- 0.017   (rule needed >= +0.05)            FAIL
    g_r0 gain     +0.278            (rule needed >= +0.30)            FAIL
    ob0 gain      +0.054            (amended AND-clause needed +0.10) FAIL
The **unamended** rule fails too, narrowly -- recorded because the 06:07 tick predicted it would
pass. Run 27 reproduced run 25's signature exactly: g_r0 -1.730 -> -1.452 while the thought's own
SHUFFLE moved -1.579 -> -1.355, i.e. +0.224 of the +0.278 is register drift, and ob0 was never
positive in 826 steps. Held-out mix +0.051 and decaying; reg% 1.3, spread 1.20, no NaN, no NRT_.
Verdict: the REINFORCE baseline is not the bottleneck. Do not carry `--baseline greedy` forward.
**Run 27's within-weights filler bar (2026-09-18 10:07 tick, n=465 each, step_0000800.pt):**
    own    disc_greedy -0.857 +- 0.112   (len 9.55)
    filler disc_greedy -0.503 +- 0.096   (len 10.00, `--pause 10`)
    own - filler  **-0.355 +- 0.148** (se treated as unpaired, so this is the conservative
    reading; the probe stored no per-anchor rows when these two JSONs were written -- it
    does now, so future pairs are compared PAIRED. See G26).
Run 25 @900 gave -0.433 +- 0.134. Two runs, two reward shapes, two independently trained
checkpoints, same verdict: the written thought is WORSE than leaving the slot blank. The
filler result of run 26 is not a one-checkpoint accident.

## The three probes that closed the "write informative content" branch (2026-09-18)

**(a) `runs/probes/oracle_r25_900_self16.json`** (n=458 anchors, run 25 @900, one checkpoint and
one scale, so the whole ladder is internally comparable):

    offset ladder   o0 +10.72   o8 +10.19   o16 +9.49   o24 +7.62   o32 +5.99   o48 +2.89   o64 -0.90
    controls        silence_ctl -0.689   pause_ctl (filler floor) -0.735   shuf64 -1.224
    SHUFFLED true hint (shuf0)  +0.251     <- order carries +10.5 of the hint's +10.7 nats
    teacher-forced self-forecast (self16)  -1.347,  positional accuracy 5.30/12, set overlap 6.40/12
    precision ladder (K true tokens in TRUE positions, rest = this sequence's own context):
      mix2 -1.131   mix4 -0.758   mix6 +0.234   mix8 +2.086   mix10 +5.479   mix12 +10.716

Two conclusions. **Queue #5 (self-forecast writers) is CLOSED**: `self16 - pause_ctl = -0.61`, and
self16 is a teacher-forced UPPER BOUND on any free-running writer, so no writer trained toward its
own forecast can clear the floor. And **break-even against the filler floor is ~4/12 positionally
correct tokens** (mix4 -0.758 sits on the floor, mix6 is +0.97 above it) -- but at its OWN matched
accuracy of 5.3/12 the model's forecast scores **1.2 nats BELOW the ladder**. Same amount of truth,
opposite sign. The penalty is not inaccuracy, it is *what the model writes when it is wrong*:
greedy-argmax errors are confident, high-frequency, plausible tokens the reader treats as evidence,
while context-drawn errors are visibly noise it discounts.
(Correction to the 04:07 journal entry: its n=37 smoke reported "break-even is 2/12". That rested
on a smoke filler floor of -0.03 that was wrong by 0.7 nats. Break-even is ~4/12. Never quoted in
the paper.)

**(b) `runs/probes/draft_r25_900.log`** (n=310 anchors, run 25 @900) -- the number that reframes
the project. Own = the model's own greedy thought truncated to m tokens:

    m      own      filler    oracle    own-filler    token agreement(own, true)
    2    -0.49     -0.44     +0.43        -0.05            0.042
    4    -0.56     -0.43     +2.35        -0.13            0.035
    6    -0.64     -0.42     +4.50        -0.22            0.033
    8    -0.75     -0.43     +6.81        -0.32            0.029
   12    -0.85     -0.53    +11.14        -0.32            0.021

The writer's free-running thought agrees with the true continuation on **4% of tokens**. The same
model's teacher-forced next-token argmax is right ~44% of the time (probe (a), 5.30/12). **The
writer is not a bad forecaster; it is not forecasting at all.** And the damage is monotone in
length: every extra written token costs, because every extra token is confident junk.

**(c) The cause, found in our own configs.** Runs 24-27 all supervised the writer with
`--oracle-min-off 8 --oracle-max-off 48`: a 16-token span 8-48 tokens AHEAD of the anchor. Run 24
chose that to exclude "the trivial copy of the immediate continuation (speculative decoding)".
Probe (a) says tokens that far ahead are essentially unforecastable at 9B. We spent four runs
training the writer on an impossible target, and it learned register phrases instead -- which is
exactly what probe (b) measures. **The "trivial copy" we excluded is the only content the model
can actually produce.** Run 28 aims the writer at offset 0-4 and lets it be short.

Inference probe: `pre-training/train/probe_thinkmode.py`. Recall-thought probe: `probe_recall.py`.
Offset/precision ladder: `probe_oracle_offset.py` (4th arg 1 = self16, 5th arg 1 = mix ladder).
Draft-length probe: `probe_draft.py`.

Run 28 (`think28_near`, launched 2026-09-18 08:14) from the same init as run 27
(`think25_calib_archive/step_0001500.pt`). Two changes, one hypothesis:
`--oracle-min-off 0 --oracle-max-off 4` (was 8/48) and `--len-cost 0.15 --min-think 3 --jit-min 3`
(was 0.05/8/8); `--baseline loo` reverted. Pre-registered futility rule at step 800 is a SINGLE
clause -- `ob0`(500-799) >= +0.05 -- with no OR, because runs 25 and 27 both passed a `g_r0` clause
while being dead. Pre-registered derail exemption: short thoughts are the intended outcome here, so
the runbook's "length within 0.5 of --min-think" axis is replaced by "length at the floor AND
ob0 < 0 for two blocks". All of it is in `runs/think28_near/config.sh`, written before launch.

**VERDICT (2026-09-18 14:12): STOPPED at step ~860 on the pre-registered rule.**
`ob0`(500-799) = **-0.1265 +- 0.0164** against a bar of >= +0.05: eleven s.e. below it, and
flat since step 0 (-0.185 @0-299, -0.160 @300-599, -0.127 @500-799). `g_r0` did gain +0.369
(-1.921 -> -1.551) but `g_shuf0` gained +0.311 of that (-1.735 -> -1.425): the same shared rise
that runs 25 and 27 produced. The single-clause rule existed precisely so this would not be
read as progress, and it was obeyed with no OR clause added.

**What run 28 nevertheless proved, and it is the most important writer result we have.** The
near-horizon target IS the right diagnosis of the register collapse: `draft_r28_200.json`
(n=310) puts free-running token agreement at m=2 at **0.113**, against run 25 @900's 0.042 --
2.7x, and the first time any run's free-running thought is distinguishable from the ~0.04
unigram base rate. **And the model got WORSE by exactly the amount the ladder predicts.**
own-filler is worse at every length (m=2: -0.24 vs run 25's -0.05; m=12: -0.79 vs -0.32).
This is ladder consequence #4 running in the open: **partial correctness is the worst place to
be.** At 11% the writer is far below break-even, so its tokens are wrong most of the time, but
they are now *plausible near-horizon* wrong tokens instead of visible noise, and the reader
believes them. Run 25's babble was harmless; run 28's near-miss is harmful.
Caveat kept on the record: run 28 @200 and run 25 @900 are different checkpoints, so only the
within-checkpoint own-filler column is comparable; and `probe_draft.py` hard-codes
`min_think=8`, above run 28's trained floor of 3, so its "own" thought is longer than the run
was trained to emit. Neither rescues own-filler, which is paired on one checkpoint.

**RUN 28'S CEILING WAS REACHED IN 200 STEPS AND THEN HELD (2026-09-18 16:07 tick,
`runs/probes/draft_r28_800.json`, n=465).** Re-probing run 28's *step-800* checkpoint:

    m      own      filler    own-filler    token agreement(own, true)
    2    -1.07     -0.96        -0.11            0.116
    4    -1.27     -0.98        -0.30            0.083
    6    -1.52     -1.01        -0.51            0.067
    8    -1.75     -1.05        -0.70            0.059
   12    -1.89     -1.20        -0.70            0.043

Agreement at m=2 is **0.116 at step 800 against 0.113 at step 200**: six hundred further steps of
training moved it by three thousandths. That is the third independent trace of the offset-lottery
cap, and the most direct one -- a capability limit is approached over a run, an ambiguity floor is
hit immediately and held. It also confirms the cap arithmetic (0.44 x 0.2 = 0.088 predicted, 0.113
and 0.116 measured) was not a step-200 coincidence. own-filler is still negative at every length,
so run 28 joins 25 and 27: three runs, three reward shapes, the written thought below the empty
slot. (Same `min_think=8` caveat as the r28@200 probe; the 200-vs-800 comparison is internally
consistent because both probes carry it.)

**THE FREE-RUNNING GAP IS NOT A DECODING ARTEFACT -- IT IS THE POLICY (2026-09-18 16:07 tick,
read out of `think_par.py`, not guessed).** Two mechanical suspects had to be cleared first, and
both are cleared:
  1. `think_par.py` `sample_step_topk` (l.209) and `sample_step_full` (l.249) both special-case the
     speculation row: `if r == spec_row: argmax`. **The greedy row gets NO top-k, NO frequency
     penalty, NO previous-token hard ban and NO temperature.** `probe_draft.py` and
     `probe_thinkmode.py` both call `generate(..., spec_row=1, ...)` and read `z[1]`, so every
     "own greedy" number this project has ever reported is a pure argmax.
  2. The penalty the SAMPLED rollouts do carry (`thinking.penalize_repeats`, freq_pen 1.0 plus a
     hard ban on the immediately previous token), measured against the true continuation on
     held-out text (L=16, n=232 spans, 3712 positions): the hard ban would block **0.46%** of true
     positions (0.86% of the first two); 8.0% of positions carry a mean 1.24 nat penalty (0.099
     nats averaged over all positions). Under a percentage point. Neither suspect can move 44%->4%.

## THE RESULT THAT NAMES THE NEXT RUN (2026-09-18 16:07 tick, `runs/probes/tfslot_r25_900.json`)

New instrument `pre-training/train/probe_tfslot.py`. Three positional accuracies against the TRUE
continuation, **same checkpoint (run 25 @900), same anchors, same forward pass, n=616**:
  `ntp`    clean row n+i predicting x[n+1+i]                -- the base model's own top-1
  `tfslot` the <T> block filled with the TRUE continuation  -- teacher-forced INSIDE the slot
  `own`    the model's free-running greedy thought          -- what we actually get

    prefix m     ntp   tfslot     own          per position i = 0 .. 15:
          1    0.433   0.047   0.047    ntp     .43 .41 .43 .45 .44 .45 .43 .45 .46 .47 .45 .46 .43 .44 .45 .47
          2    0.421   0.155   0.041    tfslot  .05 .26 .34 .37 .39 .42 .41 .40 .32 .31 .29 .31 .25 .21 .09 .01
          4    0.429   0.255   0.035    own     .05 .03 .03 .03 .03 .03 .02 .01 .01 .00 .01 .01 .00 .00 .00 .00
          8    0.437   0.330   0.029
         12    0.444   0.322   0.022    Built-in alignment check that PASSED: tfslot[0] == own[0]
         16    0.445   0.277   0.017    identically (0.047), as it must -- at position 0 the
                                        prefix is empty, so the two conditions are the same row.

**THE SLOT IS NOT BROKEN. Given a true prefix it forecasts at 0.42 by position 5 -- statistically
indistinguishable from the base model's 0.44 -- and averages 0.347 over positions 1-11.** The
`ntp` row is flat at 0.43-0.47 everywhere, which is the 0.442 that `self16` reported in the central
ladder, re-derived by an independent instrument: the probe is calibrated.

**So the entire 44% -> 2% collapse is EXPOSURE BIAS, and it has TWO compounding parts:**
1. **Position 0 is broken on its own.** The `<T>` row sees exactly the context the clean row n sees
   and is right **0.047** against the clean row's **0.433** -- a 9x collapse on the FIRST token,
   before any self-conditioning can have happened. The opener has been trained to emit a register
   token, not a forecast.
2. **And nothing recovers from it.** Positions 1-11 are worth 0.347 with a TRUE prefix and 0.019
   with the model's own -- an 18x collapse that is purely the cost of conditioning on its own
   wrong tokens.
The model cannot start, and it cannot recover from not starting.

**Why this closes the arithmetic.** Break-even against the filler floor is ~4/12 with context-noise
padding and **2/12 with pause padding** (abstention ladder). tfslot's 0.347 over positions 1-11 is
4.2/12 -- **above break-even**. A writer that ran free at its own teacher-forced accuracy would
clear the filler floor. The capability is present and measured; only the decoding regime destroys it.

**THE CAUSE IS, AGAIN, IN WHAT WE TRAIN.** `--imit-w 1.0` supervises the writer with the block
buffer holding the TRUE continuation as `content` -- i.e. pure teacher forcing. We have trained
exactly the regime that works and never trained the regime we sample in. The RL term samples
on-policy but is scored by a moving reader and weighted 0.5. **Run 30 should be scheduled sampling
/ DAgger: build the block from the model's OWN sampled thought prefix while keeping the TRUE
continuation as the CE target, ramping the own-prefix probability from 0 to ~0.8.** Judge it by
`probe_tfslot`'s `own` column, which is measured against true held-out text by a fixed instrument
and cannot rise with its own shuffle. (Implemented 2026-09-18 18:07 behind `--ss-start/--ss-end/
--ss-t0/--ss-t1/--ss-row`, all default-off; config drafted at `runs/think30_dagger/config.sh`.)

## RUN 29 MOVED THE FREE-RUNNING WRITER, AND IT RETRACTS THE "BROKEN OPENER" (18:07 tick)
`runs/probes/tfslot_r29_500.json`, run 29 @500, **n=616**, the same instrument and the same scale
as `tfslot_r25_900.json`. Read the two side by side:

              pos       0     1     2     3     4     5     6     7     8     9    10    11
  r25@900  tfslot   0.047 0.263 0.339 0.370 0.394 0.422 0.406 0.401 0.321 0.307 0.289 0.308
  r25@900     own   0.047 0.034 0.031 0.028 0.029 0.028 0.018 0.015 0.006 0.005 0.011 0.008
  r29@500  tfslot   0.297 0.396 0.412 0.401 0.391 0.391 0.351 0.338 0.333 0.364 0.325 0.330
  r29@500     own   0.297 0.185 0.123 0.102 0.071 0.054 0.057 0.042 0.031 0.023 0.019 0.015

  prefix m      1      2      4      8     12     16        own(1-11)   tfslot(1-11)   ratio
  r25@900   0.047  0.041  0.035  0.029  0.022  0.017          0.0193        0.3474     18.0x
  r29@500   0.297  0.241  0.177  0.116  0.085  0.065          0.0657        0.3664      5.6x

1. **THE OPENER WAS MISTARGETED, NOT BROKEN. Retract the 16:07 tick's reading.** That tick called
   position 0 "broken by itself" from run 25's 0.047-against-0.433 on identical context. It was
   CONFOUNDED: run 25 trained with `--oracle-min-off 8 --oracle-max-off 48`, so its position 0 was
   never supervised on x[n+1] at all. Run 29 supervises exactly that and position 0 goes to
   **0.297** against the clean row's 0.427 -- a 1.4x gap, not 9x. Run 30 needs no special
   position-0 machinery; the fix was the target, and run 29 already ships it.
2. **The largest free-running move this project has measured**: own(1-11) 0.0193 -> 0.0657 (3.4x),
   own@2 0.041 -> 0.241. **And the base was not spent for it**: `ntp` is 0.427-0.466 against run
   25's 0.433-0.466, i.e. unchanged.
3. **The teacher-forced ceiling did NOT move** (tfslot(1-11) 0.347 -> 0.366). Run 29 bought
   free-running FIDELITY, not capability -- which is exactly what the exposure-bias diagnosis says
   a better-specified target should buy, and it leaves run 30's premise intact.
4. **It is still short of break-even, by about half.** Expected correctly-positioned true tokens in
   a 12-token thought: 0.26 (r25) -> **1.02** (r29), against the abstention ladder's ~2 with pause
   padding. And run 29's remaining 11 tokens are wrong CONTENT, not pause, which the ladder says is
   worse than pause. The honest headline is "halfway to the bar", not "over it".
5. Exposure bias is still the dominant remaining loss but no longer the only story: the tf/own
   ratio fell 18.0x -> 5.6x, and `own` now decays smoothly (0.297, 0.185, 0.123, 0.102, 0.071 ...)
   instead of collapsing at position 1. That decay curve IS the exposure-bias signature, and it is
   what run 30 is built to flatten.
Caveat kept on the record: tfslot decays after position ~7 and collapses at 14-15 (0.09, 0.01),
past run 25's trained thought length of ~10 -- that tail is out of distribution, not a forecast limit.

**Do not read `imCE` across runs with different oracle offsets** -- run 28's 3.45 vs run 27's
1.6 is a statement about how predictable each target is, not about writer quality (see the
10:07 entry; the rule is now in the runbook).

## RUN 30: SCHEDULED SAMPLING IS FALSIFIED, AND IT COST THE CEILING IT WAS MEANT TO UNLOCK
(2026-09-19 00:07 tick. STOPPED at step ~690 on its pre-registered step-600 bar.)

Run 30 = run 29 + `--slot-pos aligned` + `--dagger-p 0.8` (ramped over steps 100-900), from the
same `think25_calib @1500` init as runs 27/28/29. Its single pre-registered bar, set at 20:07 on
2026-09-18 against a control computed at 18:42 and not read until after: `probe_tfslot`'s
**own(positions 1-11) >= 0.15 at step 600**, read in run 30's own ALIGNED register.

    tfslot probe, n=616, all columns same anchors and same forward
                        own(1-11)   own@1   tfslot(1-11)
      run 29 @300 built    0.0605  0.2760      0.3918
      run 29 @500 built    0.0657  0.2971      0.3664
      run 29 @600 built    0.0751  0.3117      0.3346    <- THE CONTROL
      run 30 @300 built    0.0608  0.2744      0.3682
      run 30 @300 aligned  0.0704  0.2744      0.3850
      run 30 @600 built    0.0617  0.3052      0.2395
      run 30 @600 aligned  0.0689  0.3052      0.2741    <- THE BAR: 0.0689 vs 0.15

1. **FAILED, and not narrowly: run 30 finished BELOW its own control** (0.0689 vs 0.0751) while
   p_own ramped 0.25 -> 0.63. Its trajectory was 0.0704 @300 -> 0.0689 @600, i.e. FLAT-to-down
   across the 300 steps in which scheduled sampling was actually biting.
2. **DAgger's isolated contribution is negative.** Comparing the same register on both sides so
   `--slot-pos aligned` cancels: @300 built run 30 0.0608 vs run 29 0.0605 (+0.0003, p_own still
   ~0.25); @600 built run 30 0.0617 vs run 29 0.0751 (**-0.0134**). The sign flips exactly as the
   ramp climbs.
3. **THE NEW FACT, AND THE REASON THIS IS A FALSIFICATION RATHER THAN A NULL: THE TEACHER-FORCED
   CEILING COLLAPSED.** tfslot(1-11) went 0.3850 -> 0.2741 aligned and 0.3682 -> 0.2395 built
   between steps 300 and 600, against run 29's much gentler 0.3918 -> 0.3346 over the same steps.
   Scheduled sampling destroyed ~0.11 of the very capability run 30 existed to make usable, and
   bought nothing free-running. **The mechanism is that DAgger's target is often not inferable
   from DAgger's input**: it asks the writer to emit the TRUE token x[n+1+i] given its OWN WRONG
   prefix, and when the prefix is wrong that token frequently is not a function of what the model
   just wrote. The gradient is noise on a shared pathway, and the pathway it degrades is the
   teacher-forced one.
4. **GENERAL RULE (G30 below).** Run 30's premise -- "the capability is there teacher-forced, only
   the decoding regime destroys it" -- is *still true as a measurement* and is now *false as a
   route*: you cannot reach the teacher-forced number by training on own-prefix/true-target pairs,
   because that pair is not a learnable mapping. Any future exposure-bias fix must keep input and
   target consistent (e.g. a free-running objective scored by the reader, or a slot that does not
   self-condition at all), not splice them.
5. The run was healthy on every derail axis when stopped: held-out mix +0.010 at step 600 against a
   bar of +0.25, ntp1 2.81, reg% 2.4, spread 1.61, 0 NaN / NRT_ / Traceback. It died on its bar,
   which is what a pre-registered bar is for.
6. `--slot-pos aligned` SURVIVES run 30 and is kept in run 31: measured side by side on one
   checkpoint it is worth +0.0096 on own(1-11) at step 300 -- about the size of run 29's entire
   step-500-to-600 gain -- and nothing in run 30's failure is attributable to it.

**WHAT IS STILL OPEN AFTER RUN 30.** Exposure bias is falsified *as something scheduled sampling
can fix*. Two routes remain and they are being separated by measurement, not by a run:
  (a) **train longer** -- run 29's own(1-11) was rising +0.0094/100 steps and monotonically when it
      was killed at ~780 by `ob0 >= +0.05`, a gate the abstention ladder has since shown cannot
      fire below ~2-3 correct tokens of 12. No run has ever been allowed to continue on that
      trajectory. **That is run 31** (`runs/think31_long/config.sh`, launched 2026-09-19 00:15),
      which is run 30 with `--dagger-p 0.0` and nothing else changed.
  (b) **stop self-conditioning structurally** -- a non-autoregressive / parallel slot, where block
      row (a,i) predicts x[n+1+i] from context alone and error propagation is impossible. This is
      cheap to MEASURE before it is run: `probe_tfslot.py --nar` (added this tick, default off)
      fills the block with content-free filler instead of the true continuation, leaving positions,
      mask and context identical. Running on run 29 @600 and run 30 @600 (`agent/nar_probe.sh`).
      **THE TWO DIRECTIONS ARE NOT SYMMETRIC EVIDENCE, and the 00:07 tick wrote them as if they
      were before catching it.** `nar >> own` would be strong: it would mean the capability is
      sitting there unused, exactly as `tfslot >> own` did. `nar ~ own` is WEAK: the nar regime
      has never been trained, whereas tfslot is literally what `--imit-w 1.0` trains (the true
      continuation in the slot), so a low `nar` is a LOWER bound on a trained parallel slot, not
      an upper bound on one. Read a low `nar` as "there is no free lunch here", never as "parallel
      decoding is dead".
      **FIRST PARTIAL READ (n=105 of 616, run 29 @600 built, 00:21 -- NOT a verdict):** the
      built-in alignment check passes exactly (`nar@1 == own@1 == tfslot@1 == 0.219`, as it must,
      since at position 0 all three conditions are the same row with an empty prefix), and
      `nar@12 = 0.024` against `own@12 = 0.063` and `tfslot@12 = 0.274`. On this partial read the
      parallel slot is WORSE than free-running autoregression -- i.e. the model's own wrong prefix
      is worth more to it than no prefix at all. If that holds at full n it says the free-running
      collapse is not error propagation in the usual sense, and it removes the free-lunch version
      of route (b) without killing the trained version.
      **FULL READ, 2026-09-19 02:07 tick, n=465 per checkpoint (not 616; the probe's anchor
      budget at 12 seqs -- the partial read's n=105 above is superseded and must not be quoted).**
      `runs/probes/nar_r29_600_built.json`, `runs/probes/nar_r30_600_aligned.json`:
                          nar(1-11)   own(1-11)   tfslot(1-11)   alignment nar@1==own@1==tfslot@1
        r29 @600 built      0.0202      0.0714        0.330               0.3054  PASS
        r30 @600 aligned    0.0176      0.0686        0.274               0.3054  PASS
      **THE PARTIAL READ HELD AND THE VERDICT IS THE PRE-REGISTERED `nar ~ own` BRANCH, only more
      so: nar is 3.5-4x BELOW own**, and below the ~0.04 unigram base rate. Both checkpoints agree
      to within 0.003. The pre-registered asymmetry applies verbatim and is the whole reading:
      this is a LOWER bound on a trained parallel slot (the nar regime has never been trained;
      tfslot is what `--imit-w 1.0` trains), so it says **there is no free lunch on route (b)** --
      a parallel slot is not one probe away from working -- and it does NOT say parallel decoding
      is dead. Consequence for the queue: route (b) is not the next run, and run 31 (route (a),
      simply more training on run 29's rising trajectory) is the only live route. In the paper as
      a new paragraph in sec:run30 and chronology row P6.
Honest arithmetic that bounds both: the abstention ladder puts break-even against the filler floor
at ~2 correctly-positioned tokens of 12 with pause padding, and the *teacher-forced* ceiling is
only ~4.1/12 (0.31 + 11x0.345). So even a perfect exposure-bias fix lands at roughly +0.6 nats over
filler, not at the +10.7 of a true hint. Nothing on this path is going to be a landslide.

- **RUN 30's FILLER FLOOR, on the project's actual success bar (02:07 tick, PAIRED per G26,
  n=465 shared windows).** `thinkmode_r30_600.json` vs `..._600_pause.json`, both on run 30's own
  step-600 weights, both `--slot-pos aligned` (G29): own greedy **-2.251** against filler's
  **-1.612**, paired difference **-0.639 +- 0.121** (t=-5.3; own 8.6 tokens vs 10-token filler;
  own beats filler on only 41.5% of anchors). Sampled: -2.325 vs -1.612, -0.714 +- 0.130.
  **FOURTH consecutive run below its own filler floor** (r25 @900 -0.433, r27 @800 -0.355,
  r28 own-filler -0.11..-0.70 by length, r30 @600 -0.639) and the largest margin yet. Note also
  that DAgger displaced the FLOOR itself: run 30's filler scores -1.612 where run 25's was -0.735
  and run 27's -0.503, i.e. the whole checkpoint is worse in think mode, consistent with the
  tfslot ceiling collapse. Nothing here changes a decision -- run 30 is already stopped and
  falsified -- it is the honest number for the paper's N2 paragraph, now in it.

- **G30 - A FIX THAT SPLICES AN INPUT AND A TARGET WHICH DO NOT DETERMINE EACH OTHER
  WILL DEGRADE THE CAPABILITY IT WAS MEANT TO UNLOCK.** Run 30's scheduled sampling fed
  the model its own sampled thought prefix while keeping the TRUE continuation as the CE
  target. That pair is not a function: given a wrong prefix, the true next token is often
  unrecoverable. The result was not a null -- the teacher-forced in-slot accuracy the run
  was built to exploit fell 0.385 -> 0.274 in 300 steps while free-running accuracy did
  not move. **Before building a training signal, ask whether a perfect model could achieve
  zero loss on it.** If it could not, the irreducible part is noise and it lands on shared
  weights. (Same family as the run-28 offset lottery: there the target was a 5-way mixture
  the writer could not observe, and imCE sat +0.70 above ntp1 forever. Both times the
  config, not the model, was the thing that was wrong.)

## RUN 31 IS THE FOURTH INDEPENDENT MEASUREMENT OF THE FREE-RUNNING FLOOR, AND ITS OWN
## TRAJECTORY IS FLAT (2026-09-19 06:07 tick, STOPPED at step ~861 on gate G1)
VERDICT FIRST. `runs/probes/tfslot_r31_600_aligned.json`, n=616 (checked), aligned register
(G29), the pre-registered instrument and the bar untouched since 00:07:
    own(1-11) = 0.0722   against G1's 0.0751   -> STOP
That shortfall is inside one se and it is NOT the interesting number. **The decisive reading
needs no other run: at MATCHED REGISTER and MATCHED n, run 31's own trajectory is FLAT.**
    r31 @300 aligned n616   own(1-11) 0.0716   own@0 0.2711   tfslot(1-11) 0.4029
    r31 @600 aligned n616   own(1-11) 0.0722   own@0 0.3117   tfslot(1-11) 0.3589
+0.0006 over 300 steps x 131,072 tokens. The aligned register delivers its whole contribution
by step 300 and then nothing happens. **Run 29's "+0.0094 per 100 steps" was a different
checkpoint climbing to the SAME LEVEL from a lower start, not a rate of learning.** Run 31 was
launched specifically because that slope had never been allowed to continue; it continued and
there was no slope. Built-register cross-check, same direction: r31 @600 built 0.0695 vs
r29 @600 built 0.0751; aligned-minus-built on the identical r31 @600 checkpoint is +0.0027
(it was +0.0090 at step 300 and +0.0096 on run 30), so even the register gain shrinks.
FOUR MEASUREMENTS, FOUR DIFFERENT LEVERS, ONE ANSWER:
    run 28  training length      agreement(m=2) 0.113 @200 -> 0.116 @800
    run 30  exposure bias        own(1-11) 0.0704 @300 -> 0.0689 @600 (below its control)
    nar probe  decoding order    nar(1-11) 0.018-0.020 vs own 0.069-0.071
    run 31  optimization time    own(1-11) 0.0716 @300 -> 0.0722 @600
**Stop treating the free-running writer's accuracy as a quantity that will improve. It is a
floor.** No derail axis fired: held-out mix 3.1115 -> 3.1298 over 800 steps (7% of the +0.25
allowance), reg% 1.0, spread 1.60, len 7.0, ntp1 2.78, 0 NaN / 0 NRT_ / 0 Traceback. The stop
was verified per G22: metrics.jsonl static at step 861 for 100 s, 0 trainer processes.

### THE ACCURACY-HARM RELATION REVERSES BETWEEN CHECKPOINTS AND WITHIN ONE (2026-09-19 08:07
### tick, `runs/probes/draft_r31_800.json`, run 31 @800, aligned, n=465, PAIRED per G26)
Run 31 @800 is the most accurate free-running writer the project has ever had and its thought is
the most harmful. PAIRED own-filler, with t on 465 anchors:
    m         2        4        6        8       12
    own-fil  -0.325   -0.485   -0.648   -0.834   -0.806
    t        -4.96    -6.10    -7.38    -8.71    -7.74
    agree     0.260    0.190    0.148    0.127    0.094
**Fourth checkpoint, fourth confirmation that the written thought is worse than an empty slot**
(25, 27, 28, 31). And ACROSS checkpoints the harm grows with the accuracy, monotonically, at m=2:
    r25 @900  agree 0.042  own-filler -0.05
    r28 @800  agree 0.116  own-filler -0.111
    r31 @800  agree 0.260  own-filler -0.325
**But WITHIN run 31 @800 the correlation has the OPPOSITE sign, and this is new.** Per-anchor
r(agree, own-filler) = **+0.180 at m=2 (t=+3.94)**, +0.202 / +0.132 / +0.113 / +0.139 at m=4/6/8/12.
Split by agreement quartile at m=2:
    top quartile (agree >= 0.50)   own-filler  -0.082 +- 0.105   <- break-even, indistinguishable from 0
    bottom quartile (agree = 0)    own-filler  -0.470 +- 0.083
Read it as the first POSITIVE evidence for selective speaking: on the anchors where the writer is
actually right, its thought already costs nothing. The two signs reconcile if accuracy gains arrive
bundled with a larger plausibility cost on the ~74-90% of anchors still wrong -- a better writer is
confidently wrong in more believable ways, which is exactly the self16 mechanism.
**State the ceiling honestly, and do not let a later tick inflate it:** perfect anchor-level gating
on this checkpoint reaches BREAK-EVEN (-0.08 +- 0.11), not a positive number, and r=+0.18 is weak, so
a realistic selector captures a fraction of that. Anchor-level gating is a direction with headroom,
not a win in hand.

## RUN 32: STOP PAYING FOR WRONG GUESSES (launched 2026-09-19 06:19, smoke passed, 64 ranks)
`runs/think32_abstain/config.sh`. Every design since run 25 tried to make the writer RIGHT more
often; the four rows above say that is the term that does not move. Run 32 attacks the COST term,
which is larger and which we have measured repeatedly without acting on:
  - self16 is right 5.30/12 and scores 1.2 nats BELOW the mix rung at its own accuracy;
  - run 28 RAISED agreement 0.042 -> 0.113 and its own-filler gap got WORSE at every length.
  The writer emits ~12 confident tokens of which ~10.9 are wrong and pays for all of them.
MECHANISM (new flag `--abstain-mode freq`, default "none" = runs 13-31 unchanged; trainer backup
`think_train.py.bak_pre_abstain_20260919_0607`): in the imitation block the TRUE token is kept only
where it is one of the top-300 unigrams (`stop_tokens_top300.npy`), and every other position becomes
SPACE. `</T>` padding past the thought's own end is untouched, so lengths and the stop action are
exactly as before. Measured coverage on the val mixture: top-300 = 55% of the next 12 tokens, so the
supervised thought holds ~6.6 content positions and ~5.4 silences (top-100 46%, top-200 53%).
WHY A UNIGRAM MASK AND NOT A CONFIDENCE HEAD: it is model-free, so (a) zero extra forward passes and
(b) -- the point -- it applies IDENTICALLY to the block INPUT and the CE TARGET, because both derive
from `orc_content`. That train/inference regime-match is what run 30 spent an extra generate pass
buying and lost 0.013 of own(1-11) on.
WHY THE FORMAT IS WORTH TRAINING: the abstention ladder (`abst_r25_900.json`) already measured
interleaved pause as READABLE -- pmix2 -0.71, pmix4 -0.16, pmix6 +0.77 against the filler floor
-0.74 -- which drops break-even from ~4/12 correct tokens to 2/12.
GATES (in the config verbatim; instrument = probe_tfslot `own`, aligned, n=616):
  D1 @300 DIAGNOSTIC, NO KILL (an abstaining writer must learn WHERE to speak; a transient dip is
     expected -- this clause exists so no future tick invents a gate at 300)
  G1 @600  own(1-11) >= 0.0722  (= run 31 @600 same register same step: do no harm)
  G2 @1000 own(1-11) >= 0.105   (run 31's G2 verbatim, so 31 and 32 share ONE ladder)
  G3 @1500 own(1-11) >= 0.154   (= own@0 0.31 + 11*0.154 = 2.0/12 = pmix2 = break-even; if it
     clears, the PAIRED thinkmode own-vs-pause probe is then the headline, not own(1-11))
  G4 runbook 3b derail axes, and reg%/spread/len ARE live here (unlike a --pause-len run: run 32
     still decodes real thoughts, only its SUPERVISION is masked).
REGISTERED FAILURE MODE: the cheapest policy under this mask is all-filler, which is run 26 and
scores -0.74 while buying nothing. Filler drives own(1-11) -> 0 so the gates exclude it, but if the
run lands near zero the write-up must say "collapsed to silence", not "failed to learn".
INIT = run 31 @800, NOT run 25 @1500. Run 31 bought own@0 0.047 -> 0.312 and is flat, so re-buying
those 800 steps would cost ~300 steps for nothing. **COMPARABILITY CAVEAT: run 32 does not share an
init with runs 27-31, so matched-step comparisons to them are LEVEL comparisons only.** Acceptable
because the instrument is absolute (agreement against true held-out text), not reader-relative.
NEW DIAGNOSTIC TO WATCH AT 300, and it is a READER question, not a writer one: `orG` is now the gain
of the masked pmix hint, so it is a legitimate measure of whether the READER can read this format
(the content is fixed and not model-generated). The ladder says a ~5-7-true-token pmix hint should be
worth roughly +0.5 to +2. The 3-step smoke read -0.06 over THREE SSL anchors, which is noise, but if
orG is still ~0 at step 300 the reader has not learned the format and that is a real problem.
G28 still applies to the writer: never judge run 32 by orG, g_r0, gSp or imCE.

- **G33 - A LADDER RUNG IS ONLY A REFERENCE FOR A TARGET BUILT THE SAME WAY. CHECK HOW
  THE CONTROL SELECTS ITS SURVIVING TOKENS, NOT JUST HOW MANY IT KEEPS.** (2026-09-19
  08:07 tick.) Run 32 was justified by the abstention ladder: `pmix6 +0.77` against the
  -0.74 filler floor, so a ~6.6-true-token masked hint should be comfortably readable.
  But `probe_oracle_offset.py`'s pmix<K> chooses which positions to blank with
  `rng.choice(LEN, LEN-PK)` -- **uniformly at random**, so its survivors include content
  words -- while `--abstain-mode freq` keeps a true token only where it is a **top-300
  unigram**, i.e. precisely the tokens the reader can already predict. Same count, a
  systematically less informative subset, and the rung had never been measured for the
  target the run actually trains. Measured this tick at matched count and matched length
  (new default-off 7th arg to the ladder probe, `pmixfreq<m>` / `pmixfreqrand<m>` /
  `pausectl<m>`; run 31 @800 = run 32's own init; smoke n=37, full n=465 in flight):
  frequency selection costs **+0.05 / -0.41 / -1.70 nats at m=4 / 7 / 12** against a
  count-matched RANDOM selection -- nothing at short lengths, where almost every token is
  high-frequency anyway, and 1.7 nats at 12, where the mask strips the content words.
  **The run survives this**: at m=7, its operating length, the target is still +1.46 +- 0.59
  (t=+2.46) ABOVE a blank slot of the same length, so run 32's ceiling is above its floor
  and the live `orG` does not justify a kill. Two lessons: the cheap defence is to measure
  the rung with the run's OWN mask before citing a rung as a bar, and the length matters as
  much as the count -- the trainer's SSL hint is the thought's own trained length (~7), not
  the ladder's fixed 12, which is most of why `orG` reads ~-0.5 where the m=12 rung implied +2.

- **G32 - AN APPARENT SLOPE ACROSS A RUN CAN BE A CLIMB TO A FIXED LEVEL. GET A
  TWO-POINT READ INSIDE ONE RUN, AT MATCHED REGISTER AND MATCHED n, BEFORE YOU CALL IT
  A LEARNING RATE AND SPEND A RUN EXTRAPOLATING IT.** (2026-09-19 06:07 tick.) Run 31
  existed because run 29's own(1-11) read 0.0605 @300 -> 0.0657 @500 -> 0.0751 @600 and
  the 18:07 tick quantified that as "+0.0094 / 100 steps, still rising when it was
  killed". Run 31 reproduced the design and got 0.0716 @300 -> 0.0722 @600: the level is
  the same, the slope is gone. What run 29's three points actually traced was a transient
  approach to a fixed asymptote, and two of the three differences were also confounded
  with the register change (run 31 is aligned, run 29 was built) and, briefly, with n
  (G31). The cheap defence is the one that decided this run: always hold ONE run's own
  early and late checkpoints against each other in the SAME register at the SAME n, and
  treat a bar borrowed from another run as a level check only. Corollary for kill rules:
  a gate written as "match the control" can be passed or failed by a fraction of an se,
  so pair every such gate with an internal trajectory read that does not depend on
  another run at all.

- **G31 - A PROBE'S ANCHOR COUNT IS PART OF ITS BAR. NEVER COMPARE TWO `probe_tfslot`
  NUMBERS WITHOUT CHECKING `n_anchors` FIRST.** (2026-09-19 04:07 tick, caught BEFORE the
  gate fired, which is the only reason it is a gotcha and not a wrong decision.)
  `probe_tfslot.py` takes NPER = sequences per domain as argv[3]; NPER=16 gives n=616
  anchors, NPER=12 gives n=465, and because `eval_batches` takes the FIRST NPER sequences
  the smaller set is a strict nested subset -- same anchors, fewer of them. The means are
  NOT interchangeable. On run 29's own step-600 checkpoint, same register, same file:
      n=616  own(1-11) = 0.0751   (`tfslot_r29_600.json` -- the source of run 31's G1 bar)
      n=465  own(1-11) = 0.0714   (`nar_r29_600_built.json` -- identical checkpoint)
  Run 31's gates (G1 0.0751 @600, G2 0.105 @1000, G3 0.15 @1500) are all calibrated
  against the n=616 series, but `agent/tfslot_r31.sh` was written with NPER=12. Holding
  the run to 0.0751 while measuring it over 465 anchors demands strictly MORE than "match
  run 29", which is what G1 says it is -- and run 31 @300 aligned read 0.0684, close enough
  to the bar that the 0.0037 would have decided the run. Fixed by raising run 31's probes
  to NPER=16 (`agent/tfslot_r31_n16.sh`) rather than by editing a pre-registration after
  seeing data (G28). **The general rule: a bar quoted from an old file inherits that file's
  n, its register (G29) and its flags; re-read all three before holding a run to it.**
  Checked while fixing it: `--nar` does NOT perturb the `own` column, because
  `rng = default_rng(3)` is consumed by `gen.generate()` before the NAR forward.

## RUN 32 D1 @300: THE WRITER WENT SILENT, AND THE GATE CANNOT TELL SILENCE FROM FAILURE
## (2026-09-19 10:07 tick, `runs/probes/tfslot_r32_300_{aligned,built}.json`, n=616 each)

D1 was pre-registered as a DIAGNOSTIC WITH NO KILL and it is read as one here: run 32
continues. What it says, verdict first:

      file                        own[0]   own(1-11)        own_space_rate(1-11)
      tfslot_r31_300_aligned_n616  0.2711   0.0716 +-0.0031   (not instrumented)
      tfslot_r31_600_aligned       0.3117   0.0722 +-0.0031   (not instrumented)
      tfslot_r32_300_aligned       0.1510   0.0323 +-0.0021    0.827   (stop rate 0.068)
      tfslot_r32_300_built         0.1510   0.0298 +-0.0021    0.839   (stop rate 0.075)

Run 32 at step 300 sits at **less than half its own G1 bar of 0.0722**, the opener has
**halved** (0.151 against run 31's 0.31), and the writer emits SPACE at **83%** of
positions 1-11. On the gate's own axis this is the registered failure mode, "collapsed
to silence", arriving at step 300.

**BUT THE GATE'S AXIS IS NOT NEUTRAL BETWEEN THE TREATMENT AND THE FAILURE, AND THAT IS
THE FINDING (see G34).** Run 32 is *trained* to emit SPACE wherever the true token is not
a top-300 unigram. Abstaining lowers token agreement mechanically, so `own(1-11)` scores
the treatment and the failure identically. Decompose it and the two separate:

      speak rate = 1 - space rate;  precision when speaking = own(1-11) / speak rate
      run 31 @600   speak ~1.00   own 0.0722   precision 0.072
      run 32 @300   speak  0.173  own 0.0323   precision **0.187**

Run 32 speaks one position in six and is **2.6x more accurate when it does**. That is the
design working as specified. Whether it is worth anything is a question about VALUE (nats
against the filler floor), not about agreement -- and value is the axis runs 25, 27, 28 and
31 all failed on, so it is the harder test, not the softer one.

**PRE-REGISTERED AT 10:07, BEFORE STEP 600 EXISTED, IN `agent/draft_r32_600.sh`:** G1 is
NOT weakened. If run 32 clears G1 it continues regardless. If it FAILS G1, it continues
only if the paired own-minus-filler difference against **its own init** (draft_r31_800,
n=465, same anchors, aligned) is positive at BOTH m=4 and m=8 by more than 2 se; otherwise
G1 stands and the run stops and is reported as collapsed to silence, not as a near miss.
Reference (08:07 tick): own-filler at m=2/4/6/8/12 = -0.325 / -0.485 / -0.648 / -0.834 / -0.806.

**G1 @600 FAILED, AND THE BAR WAS TOO GENEROUS (12:07 tick, n=616 each, aligned).**

      file                          own[0]   own(1-11)          space(1-11)  stop(1-11)  precision
      tfslot_r31_600_aligned        0.3117   0.0722 +-0.0031      --          --          0.072   <- the bar
      tfslot_r31_800_aligned        0.3117   0.0775 +-0.0032     0.025       0.228        0.080   <- the INIT
      tfslot_r32_300_aligned        0.1510   0.0323 +-0.0021     0.827       0.068        0.187
      tfslot_r32_600_aligned        0.1640   0.0397 +-0.0024     0.881       0.001        0.335
      tfslot_r32_600_built          0.1640   0.0353 +-0.0022     0.904       0.001        0.368

1. **G1 FAILS, decisively.** 0.0397 against 0.0722 is a factor of 1.8 and >10 se, not a near miss.
2. **The bar was set BELOW the run's own starting point.** G1/G2/G3 were calibrated on run 31 @600,
   but run 32 initialises from run 31 @**800**, which had never been probed with tfslot and scores
   **0.0775**. Had the bar been set where it belonged, run 32 would have failed it by more. Record it
   as a calibration correction, never as a rescue. (Honest caveat the other way: run 31 @800 emits
   `</T>` from position 8 on, 0.228 of positions 1-11, so neither writer attempts eleven forecasts.)
3. **Precision when speaking is the one quantity in this run that moves**: 0.187 @300 -> 0.335 @600,
   against the teacher-forced ceiling of 0.347 (run 25, `tfslot_r25_900`). Space rate 0.827 -> 0.881
   and stop rate 0.068 -> 0.001: it no longer ENDS its thought, it FILLS it. Run 32 is becoming a
   better speaker and a rarer one at about the same rate, and agreement falls because the second is
   larger. Everything now rides on VALUE (`draft_r32_600.json`, ~13:10), not agreement.

- **RUN 32 STOPPED AT 1130: THE AMENDED RULE SPLIT, AND G35's WORRY WAS HALF-RIGHT.**
  (2026-09-19 14:07 tick, `runs/probes/draft_r32_600.json` vs `draft_r31_800.json`, n=465, PAIRED,
  aligned, same anchors.)
      m     (own-fil) r32@600      (own-fil) r31@800     D = paired difference
      2     -0.0673 +-0.0257       -0.3251               +0.2578 +-0.0615  (t +4.19)
      4     -0.1102 +-0.0302       -0.4853               +0.3751 +-0.0777  (t +4.83)
      8     -0.1319 +-0.0329       -0.8343               +0.7023 +-0.0935  (t +7.51)
     12     -0.1487 +-0.0360       -0.8060               +0.6573 +-0.1014  (t +6.48)
  **(b-i) PASSES by 4-7 se; (b-ii) FAILS by 3-4 se in the wrong direction.** The registered branch
  is therefore "(b-i) only": **it stopped harming; it did not start helping.** G2's agreement bar
  0.105 is WITHDRAWN in the paper as unreachable, not quietly missed. Run stopped; the hard stop was
  registered at 1000 and the run reached 1000 at ~13:15, i.e. BETWEEN ticks -- see the new G36.
  **THE DECOMPOSITION, which is the part worth keeping.** Split the anchors by whether run 32 spoke
  at all (own == filler exactly <=> it emitted filler and nothing else):
      m=4: 230/465 IDENTICAL (r31@800: 0/465). speaking n=235: r32 -0.2180 +-0.0590 vs
           r31 on the SAME anchors -0.3215, paired D +0.1035 +-0.0980 (t +1.06, NOT significant)
      m=8: 230/465 identical. speaking: r32 -0.2611 +-0.0641 vs r31 -0.7108, D +0.4497 +-0.1215 (t +3.70)
  So **half the gain is mechanical (silence cannot harm) and half is real (the thoughts it does
  write are less damaging), and at m=4 essentially ALL of it is the mechanical half.** G35 predicted
  the first half and was right to; it did not predict the second, which is a genuine improvement.
  **SELECTIVITY: suggestive, NOT established.** Measured on the init, the anchors run 32 chose to be
  silent on are the ones where speaking cost more: -0.4621 vs -0.1777 (m=2, t -2.18), -0.6527 vs
  -0.3215 (m=4, t -2.09), -0.9604 vs -0.7108 (m=8, t -1.30), -0.9132 vs -0.7010 (m=12, t -1.02).
  Same sign at all four lengths, 2 se only at the short ones. Where it speaks its agreement is
  0.2532 (m=2) -- what its init got EVERYWHERE -- and where it is silent, 0.0174, the base rate.
  That is anchor-level gating, whose ceiling was registered in advance as BREAK-EVEN, not positive
  (run 31 entry, sec.2); run 32 landed inside that ceiling. Do not re-run anchor gating as a thesis.

- **RUN 32 @1000 IS FLAT AGAINST @600: IT ASYMPTOTED BELOW THE FILLER FLOOR, AND THE STOP WAS
  RIGHT** (2026-09-19 16:07 tick, `runs/probes/draft_r32_1000.json`, n=465, PAIRED against both
  `draft_r32_600` and `draft_r31_800`, same anchors).
      m        own-fil @600        own-fil @1000       paired 600->1000
      2      -0.067 +-0.026      -0.084 +-0.031      -0.017 +-0.030  (t -0.56)
      4      -0.110 +-0.030      -0.130 +-0.033      -0.020 +-0.033  (t -0.60)
      8      -0.132 +-0.033      -0.148 +-0.034      -0.016 +-0.036  (t -0.45)
     12      -0.149 +-0.036      -0.041 +-0.040      +0.107 +-0.041  (t +2.60)
  **G2v (own-filler itself > 0 at m=4 or m=8) fails at 1000 exactly as it failed at 600, and the
  400 extra steps moved it by less than one se.** "It stopped harming, it did not start helping"
  is the run's final verdict, not an interim one. On the 204 anchors that speak at BOTH steps the
  m=4/m=8 movement is +0.021 +-0.061 / -0.002 +-0.066: nothing.
  **The one column that moved is m=12, and it moved for a reason the silence artefact cannot
  explain**: identical-anchor count at m=12 COLLAPSED 230/465 -> 71/465 while m=4 held (230 ->
  212). The stop token returned between 600 and 1000 (`tfslot_r32_1000`: `</T>` 0.25 at position 8,
  1.00 at 15), so the 12-token draft now ends instead of running as pure filler, and those endings
  are less damaging than the init's: on the 224 anchors speaking at m=12 in both, +0.177 +-0.069
  (t +2.55). Selectivity at 1000 survives at short lengths only (init cost where run 32 is silent
  vs where it speaks: -0.689 vs -0.315 at m=4; -0.947 vs -0.741 at m=8; REVERSED at m=12).

- **QUEUE #14 IS FALSIFIED AT FULL n, AND CONFIDENCE IS A MEASURABLY WORSE SELECTOR THAN RANDOM**
  (2026-09-19 16:07 tick, `runs/probes/pmixconf_r31_800.json`, run 31 @800 = run 32's init, n=465,
  paired, gate registered at 14:25 before the file existed).
      m=7   pmixconf - pmixfreq  (THE RUN 33 GATE)   **+0.021 +- 0.066  (t +0.32)  FAILS**
            pmixconf - pmixfreqrand                   -0.609 +- 0.087  (t -7.00)
            pmixanti - pmixfreq                       +1.263 +- 0.105  (t +11.99)
            pmixanti - pause(m)                       +2.014 +- 0.119  (t +16.94)
  The 14:07 tick's mechanistic objection was right and is now measured: confidence is a better
  proxy for "the reader already predicts this" than frequency is, so selecting on it buys nothing
  over frequency (+0.02) and loses 0.61 nats to a count-matched RANDOM mask. **The value lives in
  the positions the model is LEAST sure of** -- pmixanti clears the matched blank slot by 2.01
  nats at m=7 and beats random by 0.65. The alignment check passed at seq 1 (every published
  column reproduced exactly), so the patch changed nothing it was not meant to.
  **THE ACHIEVABLE HALF IS FINAL TOO (`selfanti_r31_800.json`, landed 16:16, n=458), AND BOTH
  PRE-REGISTERED CLOSING CLAUSES FIRE.** Priced at the token the writer would really emit:
      C        selfconf   prec@C     selfanti  precanti@C   anti-conf   anti-floor
      2         -1.46      0.812      -1.55      0.152        -0.09       -0.27
      4         -1.53      0.718      -1.79      0.207        -0.26       -0.51
      8         -1.74      0.556      -1.99      0.301        -0.25       -0.71
  (floor `pause_ctl` -1.28.) selfanti LOSES to selfconf at every width and BOTH sit below the
  filler floor by 0.18-0.71, so: **the information does live in the low-confidence positions and
  the writer cannot deliver it, and POSITION-LEVEL SELECTION IS CLOSED, queue #14 included.**
  The 14:07 tick predicted precanti@4 ~ 0.21 from the published ladder before the code existed;
  it measured 0.207. The per-domain split (provisional, de-cumulated) closes it on finemath as
  well as on web text, so it is not a property of unstructured prose.

- **THE DECOMPOSITION THAT REFRAMES THE PROGRAMME: 60-85% OF THE DEFICIT IS THE COST OF THE BLOCK,
  NOT OF WHAT THE THOUGHT SAYS** (2026-09-19 16:07 tick; no new compute, all from files on disk).
      run 32 @1000, draft instrument, m=4:  own -0.899  =  filler -0.769  +  (own-filler) -0.130
      thinkmode instrument, own vs its own filler: r25@900 -1.169/-0.736 (63%), r27@800
      -0.857/-0.503 (59%), r30@600 -2.251/-1.612 (72%)
  Every run of this project has attacked the 15-40%. **And the 60-85% is trainable, in two
  independent measurements:**
   (a) `thinkmode_r26_1500_pause.json`, written 2026-09-18 02:38 and never read back: run 26
       @1500 -- 1500 steps in which every anchor was a content-free filler block -- pays
       **-0.083 +- 0.021** for that block, against its own sibling run 25 @900's **-0.736 +-
       0.087**, same instrument, same 465 anchors, same init (`think24_sketch` @3800), the two
       configs differing only in `--pause-len 10`. A 0.65-nat difference: 5x the entire content
       gap run 32 closed.
   (b) run 32 raised its own filler level over its init by **+0.501 +-0.065 (m=4)**, +0.550 (m=8),
       +0.580 (m=12), paired, draft instrument -- so a run WITH a writer also cheapens its block.
       But it plateaued at -0.77 between 600 and 1000 (+0.023 +-0.03), nowhere near run 26's -0.08.
  **THE FORK, pre-registered in `agent/ladder_r26_1500.sh` and `agent/pausecost_r32_1000.sh`
  before either file exists.** Two accounts predict different things about run 26's reader:
   (H1) TRAINABLE OVERHEAD - the block is merely out of distribution; then run 26's reader is
        cheap AND still hears a true hint (o0 >= +5 against run 25 @900's +10.72), and run 33
        makes the block free while keeping the writer.
   (H2) THE LISTENING TAX - the 0.7 is what a reader PAYS to attend to the block at all; then run
        26 is cheap because it is DEAF (o0 < +3), cost and sensitivity are one parameter, and the
        programme's answer is the negative result, not another run.
  Already suggestive for H2 and recorded so it is not forgotten: run 26's filler-trained reader
  fed a real written thought scores **-10.36** (`thinkmode_r26_1500_own.json`). That is garbage in
  an unfamiliar register, not proof of deafness -- the true-hint ladder is what decides it.
  Deciding files: `ladder_r26_1500.json` (~18:30) and `thinkmode_r32_1000_pause.json` (~17:45).

- **THE FORK RESOLVED TO H1 ON ITS OWN PRE-REGISTERED DISCRIMINATOR: A CHEAP BLOCK IS NOT A DEAF
  READER, AND "THE LISTENING TAX" IS FALSIFIED** (2026-09-19 18:07 tick; both deciding files
  landed FINAL, rc=0, with their bars written into the launcher headers before they existed).
      `ladder_r26_1500.json`   (18:00, n=458)  run 26 @1500, the reader that saw NOTHING but
                                filler for 1500 steps:  **o0 +12.624 +- 0.417**
      reference (fair, built-trained and built-probed, same init, n=458): run 25 @900  o0 +10.72
      branch bars, registered 16:20:  A = o0 >= +5.0   |   B = o0 < +3.0
  It is **+12.62**, i.e. ABOVE the reference, not below it: the cheapest block this project has
  ever measured (-0.083 +- 0.021) sits under the MOST oracle-sensitive reader it has ever
  measured. Its whole ladder is intact and ordered -- o8 +8.02, o16 +6.39, o24 +4.38, o32 +2.73,
  o48 -0.67, o64 -4.41, shuf0 -1.45 -- so the sensitivity is to CONTENT AT THE RIGHT OFFSET, not
  a level shift, and the ladder's break-even is **2/12** (pmix2 +0.04, pmix4 +0.74) against
  pause_ctl -0.083, the lowest break-even ever recorded here. **Cost and sensitivity are not one
  parameter. H2 is dead; queue #15 branch A fires.** (This also retracts the "suggestive for H2"
  note above: run 26's -10.36 on a real written thought was register shock, exactly as the
  registered caveat said, and the true-hint ladder decided it the other way.)
  **THE SECOND FILE SAYS A WRITER DOES NOT GET THE CHEAPNESS FOR FREE.**
  `thinkmode_r32_1000_pause.json` (16:36, n=465, --pause 10, aligned): run 32 @1000 pays
  **-0.926 +- 0.095** for its block -- bar `<= -0.6` of the three registered at 16:35, the worst
  one. Its clause required a LOW o0 to read as the listening tax and o0 came in at +12.62, so the
  conjunction does not fire; what remains is its literal content: **cheapness needs the regime run
  26 had, and 1130 steps of run 32 did not produce it.** Ordered, one instrument, `--pause 10`,
  n=465: r26@1500 -0.083 | r27@800 -0.503 | r25@900 -0.736 | r32@1000 -0.926 | r30@600 -1.612.
  Measured headroom on this axis: **~0.84 nats, 6x the content gap run 32 closed.**

- **RUN 33 = MAKE THE BLOCK FREE (launched 2026-09-19 18:12 from run 32 @1000, 64 ranks).**
  ONE flag differs from run 32 plus the init: **`--pause-frac 0.5`** (new, default-off; trainer
  backup `think_train.py.bak_pre_pausefrac_20260919_1807`; `test_rollouts.py` passes; the NEFF
  cache hit on relaunch, which confirms the compiled graph is unchanged -- the flag is host-side
  numpy masking only). Mechanism: a fraction of anchors carry run 26's content-free filler, leave
  BOTH writer masks (no imitation, no REINFORCE on their block rows) and JOIN the reader mask.
  That is run 26 at fraction 1.0 and run 32 at fraction 0.0; the ssl:rl ratio among the remaining
  anchors is unchanged, and the writer therefore gets half the gradient per step -- read run 33's
  writer curve against run 32's at HALF the step count.
  **THE LIMIT OF THIS RUN, STATED BEFORE IT, because it is the thing a later tick would otherwise
  discover as a success:** the §1 bar is own MINUS filler, and the block cost is the term that
  CANCELS in that difference. If run 33 does exactly what it is built to do -- filler -0.93 ->
  -0.08, content term held at -0.13 -- then `own` goes -0.90 -> -0.21 and **own-filler is still
  -0.13 and the primary bar still fails.** The run buys the secondary bar (absolute held-out gain)
  outright; it reaches the primary bar only through an unmeasured hypothesis, **(H1') the content
  term is -0.13 BECAUSE the reader is in a high-surprise regime, so a reader that pays nothing to
  read the block can also use what is in it.** H1' is the real question. Gates, in
  `runs/think33_cheap/config.sh`: M1 @400 paired improvement >= +0.30 over -0.926 (mechanism
  fires; derived from run 26's own curve, g_r0 -0.756 @0-99 -> -0.198 @300-399 at fraction 1.0);
  M2 @800 >= +0.55; **P1 @800, which OVERRIDES M1/M2: own-filler must not be worse than the
  init's**, against the baseline pair completed by `agent/own_r32_1000.sh` this tick; G3 @1500
  own-filler > 0. Registered failure mode: the reader learns "block = filler = ignore it", whose
  signature is M1/M2 passing while own-filler gets WORSE -- which is what P1 is for.

- **RUN 33'S P1 BASELINE EXISTS NOW, AND IT IS THE SIXTH CHECKPOINT BELOW ITS OWN FILLER FLOOR --
  THIS TIME ON THE INFERENCE INSTRUMENT THAT DECIDES SEC.1's BAR.** (2026-09-19 20:07 tick.)
  `agent/own_r32_1000.sh` landed `thinkmode_r32_1000.json` at 18:48 (n=465, rc=0) and this tick
  paired it per anchor (G26) against `thinkmode_r32_1000_pause.json` (n=465, 16:36). Same weights,
  same 465 anchors, aligned register (G29), lengths 10.45 (own, greedy) vs 10 (filler):
      own          -1.1265 +- 0.1111   filler   -0.9262 +- 0.0950
      own - filler PAIRED  **-0.2003 +- 0.0421  (t -4.76)**, frac_pos 0.432
      (sample column, for completeness: own -1.4935, own-filler -0.5673 +- 0.0710)
  Two things follow.
  (1) **The P1 bar is now a number and is registered before run 33's step 800 exists**, in
      `agent/m2p1_r33_800.sh` (written and launched this tick, self-firing like M1): run 33 @800's
      paired own-filler must not be worse than -0.2003 by more than 1 se of its own paired diff.
  (2) **It is a NEW measurement, not a restatement.** Run 32's own-filler has only ever been
      quoted from `probe_draft` at fixed prefix lengths -- -0.130 +-0.033 (m=4), -0.148 +-0.034
      (m=8). On the instrument that sec.1's bar is actually written in, at the full ~10-token
      thought the model really emits, the deficit is **-0.200 +- 0.042**, i.e. ~40% larger than
      the draft probe's m=8 figure and 4.8 se below zero. The direction of the difference is the
      expected one (harm is monotone in length on every checkpoint measured), so read it as the
      draft ladder extrapolated to m~10, not as a contradiction. **"Stopped harming, did not start
      helping" survives the change of instrument -- with the caveat that the honest headline number
      for run 32 @1000 is -0.200, not -0.130.**
  Sixth checkpoint below its own filler floor: 25 @900, 27 @800, 28 @800, 31 @800, 32 @1000 (draft),
  32 @1000 (thinkmode). No exceptions in the project's record.

- **RUN 33'S M1 GATE PASSES: THE BLOCK IS GETTING CHEAPER, AND THAT IS THE TERM THAT CANCELS.**
  (2026-09-19 22:07 tick. `thinkmode_r33_400_pause.json`, rc=0 21:17:58, n=465, PAIRED per anchor
  against `thinkmode_r32_1000_pause.json`, aligned register, filler length 10 on both sides.
  Alignment verified, not assumed: all 6 dumped examples have byte-identical `ctx` and `window`.)
      r33 @400 pause  -0.5507 +- 0.0822      r32 @1000 pause (init)  -0.9262 +- 0.0950
      **M1 paired improvement +0.3755 +- 0.0527 (t +7.13), cheaper on 62.8% of anchors**
      BAR +0.30 -> **PASSES**, by 1.4 se.  (sample arm identical to greedy: filler needs no decode.)
  Closed 45% of the 0.84-nat headroom between run 32's block and run 26's. That is ABOVE the gate's
  36% and BELOW the ~65% the fraction-scaling derivation expected, so **M2 @800 (+0.55, = 65%) is
  NOT already in hand** -- the mechanism is live but under the assumed pace. No kill; M1 was the
  only @400 gate and it passed.
  **G38 APPLIES TO THIS ENTRY AND IS THE REASON IT IS WORDED THIS WAY.** What moved is `filler`.
  `own - filler` at step 400 is not measured by this gate and the init's value is -0.2003 +- 0.0421.
  If the block reaches run 26's -0.08 with the content term held, `own` improves by ~0.85 nats and
  **sec.1's bar still fails by -0.20**. The honest one-line statement of run 33's position tonight:
  *the slot is 0.38 nats cheaper and nothing has yet been said about whether a thought is worth
  putting in it.*
  DIAGNOSTIC LAUNCHED THIS TICK, WITH ITS STATUS REGISTERED BEFORE THE DATA: `agent/own_r33_400.sh`
  -> `thinkmode_r33_400.json` (~22:50), the own arm on the SAME step-400 weights, which completes
  the own-filler pair 400 steps early for ~40 min of idle CPU (0 heavy jobs were running). **It
  cannot kill or save the run and it does not touch P1's bar** (-0.2003, registered in
  `agent/m2p1_r33_800.sh` before the step-800 checkpoint existed). Precedent: run 32's D1 @300, G34.

- **RUN 33 HEALTH AT STEP 590, AND THE TWO AXES THAT MOVED (2026-09-19 22:07).** All derail rules
  clear, but two are worth naming so the next tick does not read them fresh.
  (1) **Held-out mix has FLATTENED.** 3.1577 @0 -> 3.1558 @100 -> 3.2160 @200 -> 3.2423 @300 ->
      3.2482 @400 -> **3.2455 @500**, i.e. per-100 increments +0.026 / +0.006 / **-0.003** after the
      one-off +0.060 step at 100->200. The 20:07 tick registered "if steps 300 AND 400 continue at
      >= +0.02/100 the rule fires": 300 did (+0.026), 400 did not (+0.006), 500 is negative. The
      runbook's own rate rule over the most recent 300 steps (200->500) reads **+0.0098/100 against
      a bar of +0.02** -- does not fire. Total drift +0.088 against the +0.25 level bar.
  (2) **Writer length and rollout spread are drifting down, and run 32 calibrates them as normal.**
      r33 len_mean 5.90 (300-399) -> 4.91 -> **4.52** (500-590); g_sd 1.18 -> 0.79 -> **0.78**.
      Run 32 ended at len 4.88 / g_sd 0.69 over 1000-1100 and never derailed, so this is the same
      regime, not a collapse. Bars: length derails at <= 3.5 (within 0.5 of `--min-think 3`) for
      2 blocks; spread derails at < 0.6. **Neither has fired; both are within ~1 block of firing if
      the trend continues, so the 02:07 tick should read them before anything else.**
  Also clear: block ntp1 2.517 (bar 4.0), reg% 4.2 (bar 15, and the abstention qualifier applies),
  no NaN and no `NRT_` in the last 2000 log lines, 65 workers, ~21 s/step, `pau` 0.535 over the last
  30 steps (the flag is live at its intended 0.5).

- **G37 - A NUMBER THAT WAS MEASURED AND FILED IS NOT A NUMBER THAT WAS READ.**
  (2026-09-19 16:07.) The run-26 filler cost (-0.083, the largest single effect in the project's
  probe record) sat unread in `runs/probes/` for 38 hours. The tick that launched it quoted only
  the fineweb-only partial ("~-0.03 -> free"), drew the opposite lesson from it -- "the filler
  floor is the BAR TO BEAT" -- and never reopened the finished file; six subsequent ticks inherited
  the framing. Two habits follow, and this tick paid for both: **(1) before designing a probe,
  `ls runs/probes/` for one that already answers it** -- the launcher written this tick to measure
  run 26's filler cost correctly no-op'd on its own idempotency guard, which is the only reason it
  was found; **(2) every partial you quote must be re-read from the FINAL file by the tick that
  uses it.** A floor and a cost are the same number read with opposite intent, and which one you
  wrote down decides what you spend the next week on.

- **G36 - A HARD STOP AT A STEP NUMBER IS NOT A HARD STOP; THE CRON TICK IS THE ONLY ACTUATOR.**
  (2026-09-19 14:07.) The 12:07 tick registered "hard stop at step 1000" for run 32. At 22 s/step
  the run passed 1000 at ~13:15, an hour before any tick could act, and this tick found it at 1129 --
  129 steps, ~45 min of 64 NeuronCores, spent past a stop that was already binding. Nothing was
  learned from them and the checkpoint that mattered (step 1000) had been written and probed on time.
  **Register stops the supervisor can actually enforce: either a step number that `--max-steps`
  enforces in-process, or a step the NEXT tick will still be short of.** When you write a rule whose
  trigger falls between two ticks, either set `--max-steps` to it at launch or state in the journal
  that the run will overshoot by one tick and that the overshoot is accepted.

- **G35 - TAKE EVERY REGISTERED RULE TO ITS LIMIT BEFORE THE DATA ARRIVES: A CLAUSE THAT
  RESCUES A TREATMENT FROM A FAILURE MODE CAN BE MAXIMALLY SATISFIED BY THAT FAILURE MODE.**
  (2026-09-19 12:07 tick, `agent/AMENDMENT_r32_value.md`.) The 10:07 rule continued run 32 if
  `D(m) = (own-filler)_r32 - (own-filler)_init > 0` at m=4 and m=8. Evaluate at the limit: a writer
  that fills EVERY position has own == filler exactly, so own-filler == 0, so D(4) = +0.485 and
  D(8) = +0.834 -- **the largest pass the rule can score is total silence, which is the registered
  failure mode the companion clause was written to catch.** The clauses did not partition the
  outcomes. This is not hypothetical: run 32's space rate is RISING (0.827 -> 0.881) and its stop
  token has vanished, i.e. it is moving along the axis that makes the rule easier to pass and less
  meaningful. Fix, registered before the probe finished: keep D(m) > 0 as NECESSARY and add that
  `own - filler` must itself be > 0 at m=4 or m=8 by 2 se -- the project's actual bar (sec.1) --
  before the format may be called buying. D(m) > 0 alone means only **"it stopped harming; it did
  not start helping"**, which is a real result (four runs were below an empty slot) but is not a
  positive one, and it earns at most 168 more steps to a hard stop at 1000 (gate G2v), never an
  open-ended continuation. **The general defence: for every gate you write, ask what score the
  degenerate limit of the failure mode would get. If it passes, the gate is measuring the wrong
  thing.** Same family as G34, one level up: G34 is a metric that cannot separate treatment from
  failure, G35 is a *decision rule* that cannot.

- **G34 - A GATE BORROWED FROM AN UNMASKED RUN CANNOT SCORE AN ABSTAINING RUN, BECAUSE
  ITS METRIC CHARGES THE TREATMENT AND THE FAILURE THE SAME PRICE. DECOMPOSE THE METRIC
  INTO RATE x PRECISION, AND REGISTER THE TREATMENT'S OWN AXIS BEFORE THE GATE'S DATA
  EXISTS.** (2026-09-19 10:07 tick.) Run 32's G1/G2/G3 are `own(1-11)` bars calibrated on
  run 31, which speaks at every position. `own(1-11)` is a *count* of true tokens in true
  positions, so a writer that correctly stays silent and a writer that has stopped working
  both drive it to zero. At D1 the decomposition above separates them and they point
  opposite ways. This is the same error class as G33 -- a reference measured a different
  way than the thing it is the reference for -- and the same defence applies: check how
  the control BEHAVES, not just what it scores. The rule that keeps this honest rather
  than convenient (G28): the alternative axis must be registered BEFORE the gate's data
  exists, must not lower the original bar, and must demand a POSITIVE result on an
  instrument that is at least as hard. All three hold here; if they had not, the right
  move would have been to let G1 kill the run.

- **RUN 33 IS STOPPED AT ~900. GATE M2 FAILED BY 6 se AND THE MECHANISM REVERSED**
  (2026-09-20 00:07 tick.) `thinkmode_r33_800_pause.json` (n=465, rc=0 23:51, PAIRED per
  anchor against the init `thinkmode_r32_1000_pause.json`, aligned register, filler length
  10 on both sides, all 6 dumped `ctx` byte-identical):
      r33 @800 pause  -0.6641 +- 0.0808   |   init (r32 @1000)  -0.9262 +- 0.0950
      **paired  +0.2621 +- 0.0478 (t +5.49)**, against a pre-registered bar of **+0.55**
  That is 6 se below the bar. It is also a **REGRESSION**: the same gate at step 400 read
  +0.3755 +- 0.0527. The filler-anchor mechanism peaked at step 400 and went backwards over
  the next 400 steps, so M2 did not fail by being slow, it failed by turning around. The
  block's price is 0.26 of the 0.84 nats available and no longer moving toward run 26's
  -0.083. `runs/think33_cheap/config.sh` says "STOP if below"; the run was stopped at step
  899, verified per G22 (0 `think_train.py` processes, `metrics.jsonl` static at 899 for
  100 s before anything else was launched).
  P1 also fails, hard -- own-filler paired **-1.1466 +- 0.2840** (greedy) and **-0.8538
  +- 0.1482** (sample) against the init's -0.2003 +- 0.0421 / -0.5673 +- 0.0710 -- but
  **do not quote P1's greedy arm as a content result**, because of G39.

- **G39 -- ON AN ABSTAINING LINEAGE, `probe_thinkmode`'s OWN ARM *IS* ITS PAUSE ARM, SO
  "own - filler" MEASURES FILLER LENGTH. READ `examples`, NOT ONLY `summary`.**
  (2026-09-20 00:07 tick; found by printing the token ids instead of the means.)
  `probe_thinkmode.py --pause N` builds the control as SPACE (id 220) x N then `</T>`
  (l.77-79). Every greedy thought on the run-32 lineage is *the same string*:
      r32 @1000 greedy = [220]x11 + [</T>]x5        (mean greedy length 10.45)
      r33 @400  greedy = [220]x14 + [</T>]x2        (13.42)
      r33 @800  greedy = [220]x14 + [</T>]x2        (13.93)
  So `disc_greedy(own) - disc_greedy(pause)` on these checkpoints is **a comparison of two
  identical fillers at different lengths**, and it decomposes exactly that way. Run 32 @1000,
  binned by its own greedy length against the fixed 10-token control:
      len  8 (n=121) +0.023 | len 10 (n=71, MATCHED) **-0.131 +- 0.089, n.s.** | len 13
      (n=56) -0.498 | len 14 (n=46) -0.656 | r(len, own-filler) = -0.270
  **At matched length run 32 has no measurable content effect at all**, and the -0.2003
  headline that P1's bar was built on is ~two-thirds a length artifact. Run 33 @800 is the
  same story louder: 390/465 anchors emit 14-15 tokens (-0.494 and -1.201 respectively), and
  **3 anchors hit the L=16 cap where the thought never closes and cost ~66 nats each**,
  which is what inflates the greedy se to 0.284. 5%-trimmed, run 33 @800 is -0.719 +- 0.067.
  TWO CONSEQUENCES, both acted on this tick:
  1. **The block's price is steeply superlinear in length.** Pure whitespace at 14 tokens
     costs -0.49 and at 15 costs -1.20 against the *same* 10-token control on the *same*
     weights. Run 33's real defect is that it **lost its stop token** (greedy length 10.45
     -> 13.9), which is what the halved writer gradient bought.
  2. **Use `probe_draft.py` for value on this lineage, never `probe_thinkmode`'s greedy
     arm.** probe_draft's filler is length-matched at each m by construction, so it does not
     have this defect. Same family as G33/G34/G35: a control that is not matched on the one
     dimension that moves.

- **THE DEEPER FINDING, AND IT RETIRES RUN 32'S HEADLINE: RUN 32's WRITER IS MUTE, AND THAT
  IS *WHY* ITS own-filler LOOKED GOOD.** `--abstain-mode freq` writes SPACE into the CE
  TARGET wherever the true token is not a top-300 unigram (~45% of positions), so
  `--imit-w 1.0` trains "emit SPACE" as an action. The writer over-generalised it:
  `own_space_rate` over positions 1-11 is **0.827 @300, 0.881 @600, 0.736 @1000**, against
  **0.026** for run 31 @800 -- which is run 32's own init. Run 32's registered failure mode
  was "collapse to all-filler". It happened, and the instrument that was supposed to catch
  the value consequence (own-filler) is maximised by exactly that collapse (G35) *and* blind
  to it on the greedy arm (G39). **"Run 32 stopped harming" should be read as "run 32 stopped
  speaking".** What run 32 genuinely bought is narrower and still real: where it *does* speak
  it is 4x more accurate than any other writer here -- precision when speaking 0.187 @300 ->
  **0.335** @600 against run 31's 0.080 and the teacher-forced ceiling 0.347.

- **RUN 34 = MAKE THE WRITER SPEAK AGAIN (launched 2026-09-20 00:20, smoke passed, 64 ranks,
  22 s/step, NEFF cache hit so the compiled graph is unchanged).** Exactly one flag differs
  from run 32 and the init is **run 32's own init** (run 31 @800), so run 32 is a matched
  control at every step and every probe run 32 already has is a reference computed before
  run 34 existed. New flag `--abstain-loss-w` (default 1.0 = literal no-op; backup
  `think_train.py.bak_pre_abstlossw_20260920_0007`; `py_compile` OK; `test_rollouts.py`
  PASSED): at 0.0 the blanked positions keep their place in the block INPUT, so the pmix
  format is unchanged, but carry no CE weight, so **silence is no longer a trained action**
  and the imitation gradient survives only where the true token is reachable -- the "never
  train an impossible target" lesson of runs 24-27, applied at position level instead of
  offset level. The n3 normaliser is deliberately left alone so the kept positions carry
  exactly the per-token weight they had in run 32. Live-flag check: `imCE` reads 3.05-3.39
  in the smoke where run 32 sat at ~1.5-1.8, which is the mask's easy SPACE targets leaving
  the average.
  **THE QUESTION, in one line: is a spoken token worth more or less than a blank one?**
  Run 32 scored -0.110 +- 0.030 (m=4) by saying nothing; run 31, speaking freely at precision
  0.080, scored -0.485 +- 0.080. Run 34 is the untested cell -- speaking AND trained only on
  reachable targets. Honest prior from six checkpoints: less.
  GATES, in `runs/think34_speak/config.sh`, registered before any data and self-firing
  (`agent/n1_r34_300.sh`, `agent/n2_r34_600.sh`, `agent/link_r34_ckpts.sh`, all three verified
  alive at 00:23):
    (N1) step 300, MECHANISM: `own_space_rate(1-11)` <= 0.50 (controls 0.827 / 0.026). STOP above.
    (N2) step 600, PRIMARY, OVERRIDES N1 IN BOTH DIRECTIONS, two clauses BOTH required:
         (a) NOT SILENT, `own_space_rate(1-11)` <= 0.50 -- **the G35 defence written in
             advance**, because total silence maximises clause (b) at exactly 0.000;
         (b) WORTH ITS SPACE, `probe_draft` own-filler at m=4 AND m=8 no worse than run 32
             @600's (-0.110 / -0.132) by more than 2 se of run 34's own paired difference.
    (G3) step 1200: own-filler > 0 at m=4 by 2 se -- the sec.1 bar no run has ever cleared.
  Registered reading of N2, before the data: land at run 32's level while speaking -> content
  is finally free; land between -0.110 and -0.485 -> precision did not survive volume, run 32's
  0.335 was a selection effect, and the programme's negative is complete.

- **RUN 34 STOPPED 2026-09-20 06:12 AT STEP ~886 ON N2 CLAUSE (b). THE FLAG WORKED, THE WRITER
  SPOKE, AND SPEAKING COST IT. THE PROGRAMME'S NEGATIVE IS COMPLETE.**
  VERDICT FIRST, both clauses, all files rc=0, n checked (G31):
    (a) MECHANISM, PASSED TWICE BY A FACTOR OF ~20: `own_space_rate(1-11)` = **0.0459 @300**
        and **0.0428 @600** (`tfslot_r34_{300,600}_aligned.json`, n=616, aligned, identical
        invocation to every control), bar <= 0.50. Controls at the SAME steps, ONE flag apart:
        run 32 **0.827 / 0.881**; run 31 @800 (= run 34's own init) 0.026. `--abstain-loss-w 0.0`
        does exactly what it was built to do. The slot is not damaged either: own[0] 0.2662
        (r32@600 0.1640, r31@800 0.3117) and the teacher-forced ceiling tfslot(1-11) 0.2435
        @600 / 0.3377 @300 (r32@600 0.2056, r31@800 0.3601).
    (b) VALUE, FAILED BY 6 AND 9 SE: `draft_r34_600.json`, n=465, PAIRED per anchor (G26):
          **m=4 -0.520 +- 0.067   m=8 -0.892 +- 0.083**
        against bars -0.110 / -0.132 (+2 se allowed), and **below run 31 @800's -0.485 +- 0.080
        / -0.834 +- 0.096** -- i.e. 600 steps of the treatment left the writer speaking as much
        as its init and worth slightly LESS.
  **THE DECOMPOSITION, AND IT IS THE REGISTERED READING, NOT A NEW HYPOTHESIS.** own-minus-filler
  against speaking rate (= 1 - `own_space_rate(1-11)`), three checkpoints, one instrument:
        run 32 @600   rate 0.119   precision 0.335   own(1-11) 0.0397   own-filler(m=4) -0.110
        run 34 @600   rate 0.957   precision 0.062   own(1-11) 0.0596   own-filler(m=4) -0.520
        run 31 @800   rate 0.975   precision 0.080   own(1-11) 0.0775   own-filler(m=4) -0.485
  Three points on one line at ~-0.49 nats per unit of speaking rate; the 4x precision advantage
  run 32 held WHERE IT SPOKE contributes nothing detectable. And run 34 is the MORE accurate
  writer on the draft instrument's own token agreement -- **0.149 vs run 32's 0.089 at m=4** --
  while being worth 0.41 nats LESS. More true tokens in true positions, less value.
  HEALTH AT THE STOP, so the record is clean: **no derail axis fired.** len_mean settled at
  4.18/4.18/4.11 over the last three 100-blocks (bar <= 3.5 for 2 blocks) after falling from
  6.65, g_sd 0.766/0.750/0.724 (bar 0.60), len_spec FLAT at 8.97/9.06/9.12, held-out mix +0.048
  over the whole run and FLAT from step 300 (3.1775 at 600/800), ntp1 block 2.69-2.83 (bar 4.0),
  reg% ~1, no NaN, no `NRT_`. **This run was stopped by its gate, not by its health.**

- **G40 -- PRECISION MEASURED ON AN ABSTAINING CHECKPOINT DOES NOT TRANSFER WHEN THE SAME
  DESIGN IS MADE TO SPEAK. NEVER EXTRAPOLATE A SELECTED-SUBSET ACCURACY TO FULL VOLUME.**
  Run 32 @600 was right 0.335 of the time on the 12% of positions where it chose to speak, 4x
  any other writer this project has had, and that number is what motivated run 34. Made to
  speak everywhere by one flag, the same format and the same loss produce precision **0.062** --
  below its own init's 0.080. The 0.335 was not a capability the abstention format created; it
  was a property of the positions the mask selected (top-300 unigrams: the easy ones). Same
  family as G33/G34/G35/G39: a number measured under a selection and then quoted as if it
  applied to the population. **When a gate's rationale is "X is high on the subset we act on",
  the run that tests it must measure X on the FULL set, and the instrument that decides is the
  value probe, never the precision that motivated the run.** Run 34's N2 was written this way
  on purpose -- clause (a) forced the writer to speak before clause (b) was allowed to read --
  and that is the only reason this cost one run and not three.

- **RUN 34's HEADLINE-INSTRUMENT NUMBER (2026-09-20 08:07 tick, `thinkmode_r34_800{,_pause}.json`,
  n=465 each, PAIRED per anchor per G26).** No gate was attached to it; it exists so run 34 has a
  number on the instrument sec.1's bar is written in, and it AGREES with the draft probe that
  decided the gate:
      own greedy  -1.500 +- 0.123     own sampled -1.452 +- 0.128
      filler (--pause 9, length-matched to len_spec 9.0)  -0.688 +- 0.093
      **own - filler, PAIRED: -0.812 +- 0.082 (greedy), -0.765 +- 0.089 (sampled).**
  **G39's length objection does NOT apply on this lineage and the check was actually run:** the
  `examples` block shows run 34's greedy thought is ordinary text (e.g. ids 28670 527 3010 315 264
  501 2272 304 ...), not the space-token control string that runs 32/33 emitted, and the tfslot
  space rate is 0.043. Binned by own greedy length anyway, the gap survives at matched length:
  **-0.687 +- 0.169 on the 93 anchors with a 9-token thought** (len bins 8/9/10: -0.729/-0.687/-1.007).
  So run 34 is below its own filler floor by four standard errors at EXACTLY matched length --
  the widest verified content gap in the project, against run 25's -0.433 and run 27's -0.355.
  Paper: sec:run34 has a "headline instrument agrees" paragraph, N2 and the chronology row carry
  the number (`main.tex.bak_pre_r34final_20260920_0807`, pdflatex x2 rc=0, 58 pp, 0 undefined).

- **G41 -- `benchmark/score.py` SCORED THE THOUGHT BLOCK IN THE OLD "built" REGISTER, AND EVERY
  CHECKPOINT FROM RUN 30 ONWARD TRAINED "aligned". FIXED 2026-09-20 08:07, FLAG-GUARDED.**
  `think_par.build_positions` puts thought token i at n+2+i under `--slot-pos built` (runs 13-29)
  and at n+1+i under `aligned` (runs 30+). `score.py` hard-coded `n+1+i for i in range(B)` in
  `think_logprob`, in the lambada greedy block, AND implicitly in `write_thought` (which passed no
  `position_ids` at all, so the decode ran in the built register too). Scoring run 34 in think mode
  without fixing this would have repeated mistake #5 -- the 0.85-nat out-of-register failure -- in a
  new costume, and the benchmark would have reported "thinking hurts" for a geometry reason.
  New `--slot-pos {built,aligned}` (default `built`) + `block_positions()` helper; `write_thought`
  now passes explicit positions. VERIFIED behaviour-preserving: on 3 arc_easy examples the new
  script in `built` mode reproduces the old script's `gold_logprob` to **max |delta| = 0 exactly**,
  while `aligned` moves it by up to 1.41 nats, so the flag is live. Backup
  `benchmark/score.py.bak_pre_slotpos_20260920_0807`. Every think-mode row now records `slot_pos`.
  **Any re-score of runs 30-34 must pass `--slot-pos aligned`; runs 19/21 must not.**

- **G42 -- THE BENCHMARK VERDICT ON THE FINAL CHECKPOINT (2026-09-20 10:07). THE RUN-21
  DISCRIMINATION GAIN IS GONE, THINK MODE COSTS, AND THE CAPABILITY LOSS IS INHERITED --
  NOT CAUSED BY MAKING THE WRITER SPEAK.** Four clean arms + one think arm, all CPU, all
  paired per example (500/task), vanilla control `step_0026600` (9.985B) token-matched to
  run 34 @800 (9.957B) and 28M tokens AHEAD of it, i.e. favoured.
    task            vanilla   r31@800   r34@800   r34@800 think(L=16, aligned)
    LAMBADA           34.6      20.8      21.2      15.2
    OpenBookQA        35.2      31.0      29.6      28.4
    HellaSwag         36.0      34.6      34.0      32.4
    macro             45.9      43.8      43.4      41.9
  1. **THE PRE-REGISTERED 5-TASK POOL IS DEAD: +0.12 points, 163 vs 160 discordant,
     McNemar p=0.91.** It was +1.4 on run 21 @4200 and +0.96 (p=0.11) on run 21 @7700, and
     the paper reported it as "suggestive and unproven". On the checkpoint with the MOST
     thinking tokens behind it, it is zero. **Retired in the paper, not re-hedged.**
  2. **THE LAMBADA COLLAPSE IS INHERITED. THIS IS WHY THE r31@800 ARM WAS RUN AND ITS
     READING WAS REGISTERED IN `agent/bench_r31_800.sh` BEFORE IT WAS SCORED.** The two
     branches written down were "r31 ~= 21 => inherited" and "r31 ~= 26 => run 34 doubled
     it". r31@800 came back at **20.8**, so the first branch fires: the paired contrast
     r31@800 -> r34@800 (the 105M tokens that un-muted the writer) is macro **-0.4**,
     LAMBADA **+0.4**, pool -0.32 (p=0.47), all 4000 -0.40 (p=0.19). **NOTHING.** The fall
     from run 21's 26.4 to ~21 happened in runs 24/25/31 -- reader recalibration and moving
     the supervised target inside the scored window. **Making the writer speak is
     downstream-free; only the held-out value of what it says is negative.**
     *Do not attribute the -13.4 to run 34 or to `--abstain-loss-w 0.0`.*
  3. **THINK MODE LOSES, AS REGISTERED IN ADVANCE.** The prediction `think <= clean` was
     written at 08:07 from the held-out probe (-0.81 +- 0.08 nats vs length-matched filler)
     before the arm finished. Result: macro -1.4, all 4000 examples -1.43 points
     (McNemar **p=4e-4**), LAMBADA -6.0 (p<1e-3), no task up beyond noise, 5-task pool
     -0.52 (p=0.35). No first positive. The G41 fix was load-bearing: all 4000 think rows
     carry `slot_pos: aligned` (CHECKED, not assumed).
  4. **NEW AND INDEPENDENT: THE THOUGHT IS A REGISTER, MEASURED ON A SECOND CORPUS.** The
     greedy think arm emitted **258 distinct strings over 4000 examples**; one string
     (`The first step in the process is to determine the type of`) at **1083** of them; BoolQ's
     500 questions got **22** distinct thoughts; LAMBADA (prose, not question stems) was the
     most varied at 137. That is run 25's order-borne-value result (ob0 never positive,
     shuffling a thought costs nothing) arriving from a different direction on a different
     corpus, and it is the clearest single artefact the programme has for "the writer learned
     a register, not a forecast".
  Written into the paper this tick: `sec:bench` retitled ("a calibration loss that lasts, a
  discrimination gain that does not"), a new 4-column table + four paragraphs, `sec:evalproto`
  gained the built/aligned register requirement, chronology row 34 carries the numbers, and the
  closing discussion paragraph now states the verdict instead of promising the test.
  Backups `main.tex.bak_pre_bench34_20260920_1007`, `POSTMORTEM.md.bak_pre_bench34_20260920_1007`;
  pdflatex x2 rc=0, **59 pp, 0 undefined refs**.
  Artefacts: `benchmark/results/report_r34_800.txt`, `report_r31_800.txt`, and the five `.jsonl`.

- **G43 -- THE EDITORIAL AUDIT OF 2026-09-20 12:07, AND THE FOUR CLAIMS IN THE PAPER THAT HAD
  GONE STALE BEHIND THEIR OWN EVIDENCE.** The 10:07 tick left the device idle, the hypothesis
  queue empty and the evidence queue empty, and named the only remaining honest work: read the
  paper end to end against this file and fix what no longer holds. Done this tick; five things
  were wrong and none of them was visible from the section they sat in, because each was a
  forward-looking sentence written before the run that answered it.
  1. **THE TOKEN COUNT WAS ~20% TOO HIGH, AND IT IS NOW AUDITABLE.** The abstract said
     "roughly 6B thinking-mode tokens". Summing each run's own `tokens_seen` counter:
       runs 20-34  **3,823,632,384**  (the fp32 lineage -- every result in the paper)
       runs 15-18  **1,099,825,152**  (15: 509.3M, 16: 183.2M, 17: 118.8M, 18: 288.5M)
       runs 15-34  **4,990,959,616 = 4.99B**
     Run 19's counter is not trustworthy -- `think19_metrics.jsonl` is 0 bytes and only
     `think19_metrics.restart_from_zero.jsonl` survives (max step 514, 67.5M) -- while the paper's
     contemporaneous text says 275M at step 500; runs 1-14 are on the retired 4-core trn2.3xlarge
     and not in this tree at all (the paper's own figure for run 14 is 82M over 2,500 steps, i.e.
     32,768 tokens/step). So the defensible total is **~5.3B and the abstract now says "roughly
     5B"**, with the derivation written into `sec:setup` so no future tick has to guess it again.
     The per-run numbers the paper already quotes check out against the counters (run 18 288M,
     run 15 511M), so it was the SUM that was invented, not the parts.
  2. **THE ABSTRACT'S own-minus-filler RANGE, "$0.4$ to $1.5$ nats", IS NOT A RANGE OF THAT
     QUANTITY.** The 1.5 is run 25 vs run 26's TRAINING-TIME `g_r0` gap -- two different runs'
     weights -- while the abstract's sentence says "on the same weights". Every same-weights
     paired measurement in this project falls in **-0.11 (run 28 @800, m=2) to -0.89 (run 34
     @600, m=8)**, with -0.812 +- 0.082 on the headline instrument for the final checkpoint.
     Abstract now reads "$0.1$ to $0.9$ nats ... and $0.81 +- 0.08$ on the last checkpoint",
     and "every length from 2 to 16" is now "2 to 12" because 12 is where the draft ladder stops.
     N2's closing count went "Five runs, five designs" -> **"Seven checkpoints -- runs 25, 27,
     28, 30, 31, 32 and 34"**, which is what the body already claims two sections later
     ("the sixth checkpoint" for run 32, run 34 after it). Run 33 is deliberately NOT in that
     list: G39 retires its value reading.
  3. **"Run 31, live"** (sec on exposure bias) and **"Where the project stands, and what is
     running now (run 32)"** (the closing paragraph's title) were both stale by two and three
     runs. Worse than the tense: the same sentence still carried run 29's "+0.0094 per hundred
     steps ... no run has ever been allowed to continue on that trajectory" as an OPEN route,
     when run 31 is exactly that run and G32 retired the slope (0.0716 @300 -> 0.0722 @600).
     Fixed in place, with the retraction stated where the claim is made rather than only in
     sec:run31 six pages later.
  4. **`sec:whatwouldwork` WAS A LIST OF FUTURES THAT HAVE ALL SINCE HAPPENED.** "Our next run
     applies it together with the position fix" = run 30, falsified. "A model that could sort its
     own thoughts into keep and discard would convert a negative mean into a positive one" is
     contradicted by the paper's OWN run-31 measurement four pages earlier: r(agreement,
     own-filler) = +0.180 at m=2 and the top agreement quartile scores -0.082 +- 0.105, i.e.
     perfect anchor-level gating reaches BREAK-EVEN. Both now state the measured ceiling.
     Related work's "a full pause-token training arm at a ladder rung remains the cleanest way
     to separate compute from content" was written before run 26, which IS that arm at fraction
     1.0 for 1,500 steps.
  **THE GENERAL LESSON, which is why this is a G and not a journal line: a pre-registration and
  a stale claim are the same sentence at different times.** This paper's method is to register a
  bar before the data and then report against it, so the body is full of future-tense sentences
  that are correct when written and wrong a day later. Nothing in the run table catches them --
  they are prose, not numbers. **Every tick that writes up a run must also grep the body for the
  previous registration it just answered.** Cheap detectors that worked here: `grep -n "live\|is
  running\|next run\|we will\|remains the"` over `main.tex`, and reading every paragraph TITLE in
  the discussion section.
  Backups `main.tex.bak_pre_edit_20260920_1207`, `POSTMORTEM.md.bak_pre_edit_20260920_1207`,
  `SUPERVISOR.md.bak_20260920_1207`; pdflatex x2 rc=0, **59 pp, 0 undefined refs**, verified
  after the first batch of edits and again after the second.

- **G44 -- G43's DETECTORS WERE RUN AGAIN AT 14:07 AND FOUND SIX MORE, ALL IN THE APPENDIX.
  The editorial pass of 12:07 audited the BODY; the lab-record appendices were never swept, and
  they are where contemporaneous tense survives longest.** The 12:07 tick registered its own
  greps as "the one cheap check worth repeating" -- repeating them cost a minute and paid:
  1. `sec:run13obs` announced **"(run 15, live)"** and "The scaled experiment is now running"
     -- and then reported run 15's completion and fourth falsification TWO PARAGRAPHS LATER.
     A self-contradiction inside one subsection, surviving nineteen runs.
  2. The same subsection closed a paragraph with "the subject of ... **the run now in
     progress**", meaning run 16, which is the next paragraph and has been done since Sep 8.
  3. `sec:densepar` said run 13 "**trains from the 2.07B base at the time of writing**".
  4. Chronology table **row 13** verdict was "**In progress.**"; it now records what the run
     did (warm-up delivered, corrected handover converged to silence, killed at step 1,174).
  5. Chronology table **row 14** was "**In progress; stability claim delivered**" and ended
     "the run continues on the new node"; it now names run 15 as where the recipe continued
     and that the eligible rate came back flat there -- which is the paper's own text at
     sec:run13obs, so the row was contradicting the section it points at.
  6. "has now trained ~2,500 steps" (run 14) -> "trained".
  **No number, bar or verdict was touched -- only tense and status markers**, and each
  replacement states what a later section of the same paper already establishes.
  **THE ADDITION TO G43: run the detectors over the WHOLE file, and add `In progress`,
  `at the time of writing` and `, live)` to the pattern.** A longtable row is the worst case
  of all: it is prose (so the run table's numbers do not catch it), it is bold (so it reads as
  a verdict), and nobody re-reads an appendix table after the run it describes ends.
  Verified: `pdflatex` x2 rc=0, **59 pp, 0 undefined refs**; backups
  `main.tex.bak_pre_appendixedit_20260920_1407`, `POSTMORTEM.md.bak_pre_appendixedit_20260920_1407`,
  `SUPERVISOR.md.bak_20260920_1407`. After the edits the detectors return only benign hits:
  "argument in progress" (prose), the position-selection registration at l.983 that is answered
  22 lines later, the run-11/12 criteria paragraph that answers itself, and the closing
  "why nothing is running", which is true.

- **G45 -- THE THIRD DETECTOR RUN (2026-09-20 16:07) FOUND FOUR MORE, AND THEY ARE A CLASS THE
  FIRST TWO PATTERNS COULD NOT SEE: A STALE *REFERENT*, NOT A STALE TENSE.** G43 and G44 hunted
  future-tense promises and `In progress` markers. What survived both is the word **"current"**:
  prose that is grammatically fine, states nothing false about the past, and yet points a
  2026-09-20 reader at run 34 when it means run 10c. **Add `current run`, `our current` and
  `currently` to the detector pattern.**
  1. `sec:think10` (run 10c's own appendix) opened **"The current run instantiates the full
     method..."** -- twenty-four runs stale, and the very next sentence is already past tense
     ("This configuration was reached in four same-day iterations"). Now "Run 10c instantiates".
  2. `sec:distill-old`, titled *"Eligibility by conjunction (runs 7--17)"*, said the writer's
     signal **"in our current runs"** is best-of-$m$ self-distillation. That method is superseded
     -- the appendix exists to say so. Now "in runs 7--17".
  3. `sec:gate-old` (same appendix) priced the oracle ceiling at "+32 nats per discounted window
     **in the current run**". Now "in those runs". No number touched.
  4. **THE ONE THAT MATTERED: run 10c's held-out table closed with "A pause-token arm REMAINS
     THE DISCRIMINATOR"** -- an open route to a test that has since been run and has since become
     this paper's central negative (run 26, `sec:run26`: content-free filler in the same slots
     BEATS the written thoughts). G43 fixed the identical sentence in related work ("a full
     pause-token arm remains the cleanest way") and this copy survived because the 12:07 pass was
     body-only and the 14:07 pass changed only tense markers. It now states that the arm was run
     later on a DIFFERENT lineage as run 26 and what it found, and -- honestly -- that it was
     never run on *that* checkpoint, so on that rung the confound stands unresolved. The health
     signal (thinking training did not damage the base) is unchanged.
  **No number, bar or verdict was touched.** Verified: `pdflatex` x2 rc=0, **59 pp, 0 undefined
  refs**; backups `main.tex.bak_pre_referents_20260920_1607`,
  `POSTMORTEM.md.bak_pre_referents_20260920_1607`, `SUPERVISOR.md.bak_20260920_1607`. After the
  edits the extended detectors return only benign hits: run 12's "remains the base LM's tail
  distribution" (prose), `sec:scaled` Step 3's "the current runs" (that section is explicitly
  fenced -- "reproduced as written after run 14, before runs 15--19"), and the closing
  "why nothing is running", which is true.
  **THE LESSON, which is now three for three: a tick that believes the editorial backlog is empty
  should still spend one minute on the greps, and should EXTEND the pattern rather than re-run it
  unchanged.** Each sweep found what the previous pattern was blind to: body promises (G43),
  appendix status markers (G44), stale deictics (G45). Do not treat "the detectors return only
  benign hits" as proof the paper is clean -- it is proof only that the current pattern is spent.

- **G46 -- THE FOURTH SWEEP (2026-09-20 18:07) CHANGED THE DETECTOR FROM WORDS TO UNITS, AND
  FOUND A WRONG LENGTH IN THE PAPER'S OWN ONE-SCALE SUMMARY.** G43/G44/G45 all matched on
  ENGLISH (promises, status markers, deictics). Those are spent. The new pattern matches on a
  NUMBER WITH A UNIT -- `grep -o '\$[0-9]\+\$-token[a-z ]*'` over the whole file, then check
  each hit against the constant in the probe that produced it. Twenty-one such phrases; one
  paragraph was wrong, and it is the paragraph a reader is most likely to quote.
  1. `sec:probes2609`'s **"Everything on one scale"** opened: *"a perfect $16$-token hint from
     inside the window is worth $+11.0$ ... the $16$-token span of context with the highest
     content-word overlap ... is worth $-3.1$"*. **Both hints are TWELVE tokens.**
     `probe_recall.py` l.17 is `T, K, L, LEN, GAP = 1024, 64, 16, 12, 32`: `LEN = 12` is the
     span, `L = 16` is the THOUGHT SLOT it is spliced into. The `16` in "16-token hint" is
     neither -- it is the OFFSET (`o16` = the true span 16..28 tokens ahead, l.61). Three
     different 16s in one probe and the prose took the wrong one twice.
  2. **It contradicted the paper's own table two paragraphs later**: `tab:ladder`'s caption says
     "a perfect $12$-token hint taken $o$ tokens after the anchor", the precision ladder is
     $K$ of $12$, `probe_oracle_offset.py` l.92 is `LEN = 12`, and the abstract says
     "a true $12$-token hint". The summary paragraph was the only place in the paper claiming 16.
  3. Fixed to "$12$-token hint from inside the window (the span $16$ to $28$ tokens after the
     anchor)" and "$12$-token span of context". **No value was touched** -- $+11.0$ (recall
     `o16` $10.991 \pm 0.392$, n=382), $-3.1$ (`recall` $-3.094$), $-2.7$ (`randctx` $-2.691$)
     and $-1.2$ (run 25 @900 own) were each re-read out of `runs/probes/recall_r24_3800.json`
     and `oracle_r25_900_self16.json` this tick and all four are right.
  4. Also swept and CLEAN, so do not redo: the abstract's other numbers against their artefacts
     -- $+4$ to $+11$ and $-1.6$ (`oracle_offset_r23_3600.json`: o32 $3.998$, o0 $8.849$,
     o64 $-1.64$; $+11$ from recall's o16), $-2.45 \to -1.17$, $0.30$/$0.41$/$0.06$ (N5),
     $0.81 \pm 0.08$, $1.4$ macro / $p=4\times10^{-4}$ / $258$ strings / $1{,}083$ (G42),
     0.6B against `sec:setup`'s architecture line -- and the hedge detector
     (`we believe|suggests that|we suspect|presumably|should be able to`), which returns
     **zero hits** in the whole file.
  5. The other twenty `$N$-token` phrases check out: $64$-token window/horizon (k=64), $16$-token
     thought and $16$-token supervised span (L=16, runs 24--27), $8$-token floor (`--min-think 8`),
     $10$-token filler (`probe_oracle_offset.py` `PAUSE_LEN = 10`), $8$-token window in
     `sec:run13obs` (that run's k).
  **THE RULE, now four for four: extend the detector into a class it has never matched on.**
  Words were mined out after three passes; units were untouched and paid immediately. The next
  unswept classes, for whoever wants a cheap minute: an $n$ or a $\pm$ quoted for the wrong
  probe, a step number quoted for the wrong run, and every "$X \to Y$" arrow re-read against
  the metrics file it came from. Verified: `pdflatex` x2 rc=0, **59 pp, 0 undefined refs**;
  backups `main.tex.bak_pre_units_20260920_1807`, `POSTMORTEM.md.bak_pre_units_20260920_1807`,
  `SUPERVISOR.md.bak_20260920_1807`.

- **G47 -- THE FIFTH SWEEP (2026-09-20 20:07) CHECKED THE SAMPLE SIZE AGAINST THE PROBE THAT
  PRODUCED IT, AND FOUND ONE INFLATED BY 2.7x.** G46 named this class and it paid on the first
  pass. The detector is not a grep over English or over units: it is a JOIN. Dump `n` /
  `n_anchors` from all 62 files in `runs/probes/*.json`, then check every `$n=...$` and every
  bare "<N> anchors" in `main.tex` against the artefact the surrounding numbers came from.
  1. **THE HIT: `sec:slotpos` credited the slot-position result to "$616$ anchors". Its probe
     ran on 232.** `runs/probes/slotpos_r25_900.json` reports `n: 232` on all four of the
     quantities that paragraph quotes -- `built_acc` $0.2701 \pm 0.0099$, `shifted_acc`
     $0.3049 \pm 0.0105$, `built_gain` $2.2957$, `shifted_gain` $3.1491$ (and `ntp_acc`
     $0.4132$, the clean ceiling). $616$ is `probe_tfslot.py`'s anchor count at NPER=16 -- the
     instrument in the NEXT section -- so the wrong n came from the paper's most-quoted n,
     not from nowhere. **No value moved**: $0.270 \to 0.305$, $0.413$, $+2.30 \to +3.15$ and
     the $0.85$-nat headline ($3.1491 - 2.2957 = 0.853$) are all correct as printed; only the
     sample size was, and it overstated the evidence behind a number the paper lists as a named
     cost in `sec:whatwentwrong`. Fixed to "$232$ anchors".
  2. **WHY A WORD OR UNIT DETECTOR COULD NEVER SEE IT.** $616$ is a true, current, correctly
     unitised anchor count that appears 11 more times in this paper, every one of them right.
     It is only wrong relative to the FILE the sentence's other numbers were read from. That
     join is the detector; nothing cheaper reaches it.
  3. **ALSO SWEPT AND CLEAN -- do not redo.** (a) Every other `$n=...$` in the file against its
     artefact: $458$ (`oracle_r25_900_self16`, `abst_r25_900`, `pmixfreq`/`selfanti_r31_800`),
     $310$ (`draft_r25_900`), $465$ (all 18 `thinkmode_*` and the `draft_*`/`nar_*` probes),
     $616$ (every `tfslot_*` except the two `tfslot_r31_300_*` files, which are $465$ and are
     already handled in the text at `sec:run31`), $232$ (the freq-pen span check), $37$ (named
     in the text as the superseded smoke), $382$ (`recall_r24_3800`, quoted only in G46's
     numbers, not as an $n$ in the paper). (b) **Every "run NN @ step" attribution in the
     paper resolves to a checkpoint that exists**: 21@4200, 21@7700, 24@3800, 25@900, 25@1500,
     26@1500, 29@300/500/600, 30@300/600, 31@800, 32@1000, 33@800, 34@800, and 19@500
     (`checkpoints/think19_archive/step_0000500.pt`) -- checked against `ls` of every
     `think*_archive`. No step number is attributed to the wrong run.
  **THE RULE, now five for five, and the class is again the one nobody had matched on.** Still
  unswept, for whoever wants a cheap minute: every "$X \to Y$" arrow re-read against the
  metrics file it came from (~40 of them, the largest remaining class), and each $\pm$ re-read
  as a PAIRED vs UNPAIRED standard error (G26 changed that convention mid-project, so a figure
  quoted from before 2026-09-18 10:15 may be the unpaired combination). Verified: `pdflatex`
  x2 rc=0, **59 pp, 0 undefined refs**; backups `main.tex.bak_pre_nquote_20260920_2007`,
  `POSTMORTEM.md.bak_pre_nquote_20260920_2007`, `SUPERVISOR.md.bak_20260920_2007`.

- **G48 -- THE SIXTH SWEEP (2026-09-20 22:07) TOOK THE TWO CLASSES G47 LEFT AND THE SECOND ONE
  CAUGHT THE PAPER BREAKING G26 IN ITS OWN HEADLINE NEGATIVE.** Both classes were swept; the
  ~40 arrows came back clean, the $\pm$ join did not.
  1. **THE HIT: N2 (`sec:negatives`) said all three of its filler gaps were "paired anchor by
     anchor", and two of them cannot be.** The sentence read: "$-0.43 \pm 0.13$ against run 25's
     own checkpoint, $-0.36 \pm 0.15$ against run 27's, and $-0.64 \pm 0.12$ against run 30
     @600's---paired anchor by anchor on the $n=465$ windows both conditions share, $t=-5.3$".
     `thinkmode_r25_900{,_pause}.json` and `thinkmode_r27_800{,_pause}.json` have **no
     `per_anchor` key** -- they predate the 2026-09-18 10:15 dump -- so $0.13$ and $0.15$ are
     necessarily the INDEPENDENT combination: $\sqrt{0.1017^2+0.0866^2}=0.1336$ and
     $\sqrt{0.1124^2+0.0960^2}=0.1478$, reproduced exactly. Only run 30's $0.12$ is paired
     (recomputed from `per_anchor`: $-0.6393 \pm 0.1209$; its unpaired combination is $0.193$,
     so the paper is quoting the paired one there). **G26's last sentence is literally "do not
     silently quote a paired se for those three" and N2 did exactly that.** `sec:run26` (l.443)
     already discloses it for run 25 -- "even treating the two conditions as independent, which
     understates the precision" -- so N2 was also contradicting the paper's own body.
  2. **THE $t$ AND THE LENGTH IN THAT CLAUSE BELONG TO RUN 30 ALONE.** $t=-5.3$ is
     $0.6393/0.1209 = 5.29$, i.e. run 30's paired $t$; runs 25 and 27 are $3.24$ and $2.40$ on
     their unpaired bars (G26 itself says "2.4 se" for run 27). "the thought $8.6$ tokens long"
     is run 30 @600's greedy length ($8.634$); run 25 @900 is $9.38$ and run 27 @800 is $9.55$,
     both already correct elsewhere in the body. Three gaps listed, the strongest $t$ and the
     shortest thought quoted as if they described the set.
     **FIXED, no value moved**: the three gaps still read $-0.43$/$-0.36$/$-0.64$ with the same
     bars; the pairing, the $t$ and the $8.6$ are attributed to run 30, runs 25/27 are stated as
     the conservative independent combination at $3.2$ and $2.4$ se and lengths $9.4$/$9.55$,
     and the paragraph now says we do not quote a paired bar we cannot compute.
  3. **TWO SMALLER HITS, same join.** (a) `sec:run29`: "\textsc{ntp} is $0.427$--$0.466$ against
     run 25's $0.433$--$0.466$" -- both pairs are [`per_pos[0]`, max(`per_pos`)], not ranges.
     True per-position minima are $0.4172$ (`tfslot_r29_500`) and $0.4091$ (`tfslot_r25_900`).
     Fixed to $0.417$--$0.466$ and $0.409$--$0.466$; the claim (the base is not spent) is
     unchanged. (b) Run 31 @800's filler rate over positions 1--11 is $0.0249$
     (`tfslot_r31_800_aligned.json`, `own_space_rate`), i.e. $0.025$, not the $0.026$ quoted at
     `sec:run33` and in chronology row 34. Fixed in both places.
  4. **THE ARROWS ARE CLEAN -- DO NOT REDO THEM.** Every numeric "$X \to Y$" that this tree can
     check was recomputed from its own artefact and is right: run 18 imCE $5.04 \to 3.79$ / orG
     $-11.7 \to -1.8$ (these are the first and last **hundred rows** of
     `think18_metrics.jsonl`, which has 15 duplicate rows from a relaunch -- by deduped step the
     same window gives $4.94$/$-11.48$, so the convention, not the number, is what wobbles);
     run 20 elig $1\% \to 47\%$ (0.0087 / 0.4712 at 2201--2300), imCE $4.6 \to 4.2$, orG
     $-11 \to -1.2$, winner-over-shuffle $+0.8$ (=$+0.78$), and the whole held-out chain
     $2.786/3.040/3.157/3.246/3.317/3.351$ with $+0.56$; run 22 elig $0.50 \to 0.63$, winner
     over silence $-0.02 \to +0.25$, spread $1.7 \to 0.9$, held-out $+0.044$; run 24 orG
     $+4.6 \to +7.4$ (300- and 500-blocks), $g_{r0}$ $-2.5 \to -2.2$ (300-blocks), imCE $3.7$
     (whole-run mean $3.713$), held-out $2.941 \to 3.027$; run 25 $\to$ run 29 tfslot own(1--11)
     $0.0193 \to 0.0657$ ($3.40\times$) and ceiling $0.347 \to 0.366$; ALL of run 30's
     ($0.0704 \to 0.0689$, $0.3850 \to 0.2741$, $0.3682 \to 0.2395$, $0.3918 \to 0.3346$, and
     the $+0.0096$ / $+0.0003$ / $-0.0134$ decomposition); run 28 agreement
     $0.113 \to 0.116$; the three-run ladders $0.042 \to 0.116 \to 0.260$ and
     $-0.05 \to -0.111 \to -0.325$; run 32 stop rate $0.068 \to 0.001$ and precision
     $0.187 \to 0.335$ (pooled own/(1-space) over positions 1--11 = $0.1865$/$0.3346$); run 33
     greedy length $10.45 \to 13.9$ with 390 of 465 at 14--15 tokens and 3 at the cap; and row
     34's speaking-rate line $0.119 \to -0.110$, $0.957 \to -0.520$, $0.975 \to -0.485$.
     Also re-derived and correct: every post-G26 $\pm$ in the paper is genuinely paired
     ($-0.200 \pm 0.042$, $-1.147 \pm 0.284$, $-0.81 \pm 0.08$, $+0.376 \pm 0.053$,
     $+0.262 \pm 0.048$, $-0.520 \pm 0.067$, $-0.892 \pm 0.083$, $-0.110/-0.132$,
     $+0.501 \pm 0.065$), and l.181 was the ONLY place in 59 pages asserting pairing for runs
     25/27.
  **THE RULE, six for six.** The class that pays is always the one no previous pattern could
  match: English (G43--G45), a unit (G46), a sample size (G47), and now **the STATISTICAL
  TREATMENT behind an error bar** -- a $\pm$ that is correct as a number and wrong about how it
  was computed. The named seam is now closed: the arrows are swept, the $\pm$ are swept, the
  $n$ are swept, the units are swept, the words are swept. A tick that cannot name a genuinely
  NEW class should hold quiet rather than re-run any of them. Verified: `pdflatex` x2 rc=0,
  **59 pp, 0 undefined refs**; backups `main.tex.bak_pre_paired_20260920_2207`,
  `POSTMORTEM.md.bak_pre_paired_20260920_2207`, `SUPERVISOR.md.bak_20260920_2207`.

- **G49 -- THE SEVENTH CLASS, FOUND ONLY BECAUSE NEW TEXT ARRIVED (2026-09-21 10:07): A SCALAR
  QUOTED AT THE WRONG GRANULARITY. Three errors in the four figure captions added at 08:45,
  and a fourth of the same class sitting in the paper since 09-19.** Five quiet ticks
  (22:07--08:07) correctly refused to re-run G43--G48 on an unchanged file. The gate those ticks
  pre-registered -- "only if `main.tex` mtime moves off 2026-09-20 22:16:24" -- opened at 08:45
  when the interactive session added four figures. **The detectors did not need extending; the
  INPUT changed.** ~1.3 kB of brand-new caption prose, dense with numbers, had never been swept.
  1. **THE CLASS.** G43--G48 all join a claim to an artefact and ask *is this number right?*
     The new class asks *is this number the right KIND of thing?* -- an aggregate over positions
     quoted as a single position, a set of runs quoted as "the same" set, a range quoted for
     runs it does not cover. Every one is a true number from the right file, wrong about what
     it is a number OF. No word, unit, $n$, arrow or $\pm$ detector can see it. **The detector
     is: for every scalar in a caption, name the axis it was reduced over, then re-derive it.**
  2. **HIT (new text, `fig:freerunning`): an aggregate over positions 1--11 printed as the
     position-12 value.** The caption read "the model writing its own prefix---$0.31$ at the
     first token ... and $0.07$ by the twelfth". `tfslot_r31_600_aligned.json` `own.per_pos`:
     position 1 is $0.3117$ (correct), **position 12 is $0.0179$**. $0.0722$ is `own` averaged
     over positions 1--11 -- the project's standard aggregate, the one run 30's bar was written
     against. So one sentence quoted a per-position value and a mean over positions as if both
     were per-position, and **the figure directly above it plots the true $0.018$**: the caption
     contradicted its own picture, 4x. Fixed to "$0.072$ averaged over positions $1$--$11$, and
     $0.018$ by the twelfth". The correction makes the collapse LOOK WORSE, i.e. it strengthens
     the paper's own negative result -- the error was against us, which is why nothing flagged it.
  3. **HIT (new text, `fig:ledger`): "the same checkpoints" names a set that does not overlap.**
     Panel (b) opened "The same checkpoints measured at inference". Panel (a) plots runs
     **20, 21, 25, 29** (`heldout.tsv`); panel (b) plots **25, 27, 30, 32, 33, 34**
     (`thinkmode_*`). **Run 25 is the only member of both.** Fixed to "Six later checkpoints"
     and the runs named explicitly.
  4. **HIT (new text, `fig:ledger`): a cost range that excludes two of the three runs it
     describes.** "at $10^{-5}$ the cost is $0.10$--$0.16$ nats per run" -- of the $10^{-5}$
     runs plotted, only run 21 ($+0.155$) is in that band; run 25 is $+0.085$ and run 29 is
     $+0.015$. Across all fourteen runs at $10^{-5}$ the true span is $+0.010$ to $+0.155$, so
     the lower bound was over-stated by 10x at the bottom of the range. This number appears
     **nowhere else in the paper** -- it was born in the caption. Fixed to name run 21's
     $0.16$ over $7{,}700$ steps and the shorter runs' $0.09$/$0.02$. "Decelerating" was
     CHECKED and kept: run 21 pays $+0.037, +0.027, +0.024, +0.023, +0.014, +0.014, +0.011$
     per thousand steps.
  5. **SWEPT AND CLEAN, do not redo.** `fig:hintvalue` entirely: $458$ anchors (= `n` in
     `oracle_offset_r23_3600`), $12$-token hint (G46), $+8.9$ (o0 $8.849$), $+4.0$ (o32
     $3.998$), shuffled $-0.1$ (shuf0 $-0.143$), negative at $o{=}64$ ($-1.64$), and **"value
     falls to zero near $o \approx 56$"** -- interpolating o48 $1.460$ to o64 $-1.640$ gives
     $55.5$. `fig:belowfiller`: $465$ (= `draft_r29_600` `n`), own below filler at all five $m$,
     all four designs below zero at every $m$, run 32 closest. `fig:ledger` (a) run 20
     $+0.565$ in $2{,}300$ steps; (b) all six pairs own < filler, all $n=465$. `fig:freerunning`
     $616$ anchors (correct here -- G47's $616$ hit was `sec:slotpos`, a different probe),
     violet "never left $0.05$" (r25 own max $0.0471$), and "the slot itself costs about $0.12$
     at position one" ($0.4351 - 0.3117 = 0.1234$).
  6. **NOT FIXED, FLAGGED FOR EMANUELE -- the same class, in old text, and it reaches the
     abstract.** `N5` (l.208) reads "it reaches $0.06$ by the twelfth token, and $0.226$ at the
     very first token". **$0.226$ is a phantom: it is not an `own` value or aggregate at any
     position in ANY of the 24 `tfslot_*.json` files** -- and the journal already established
     this on 09-18 ("0.226 does not appear in the probe file"; run 29 @500 own@1 is $0.2971$)
     when it RETIRED that number as a run-30 bar. It has been in `main.tex` since at least
     `main.tex.bak_pre_r30_20260919` and survived all six sweeps. $0.06$ is likewise
     `own(1--11)` for run 29 @500 ($0.0657$), not a twelfth-position value. **The repair is not
     mechanical**, which is why this tick did not make it: N5's other two numbers ($0.30$
     teacher-forced, $0.41$ ceiling) are `slotpos_r25_900`'s `shifted_acc`/`ntp_acc` ($n=232$),
     its table `tab:tfslot` is `tfslot_r25_900` ($n=616$, own@1 $= 0.047$, printed twice
     twenty lines below), and its new figure is `tfslot_r31_600_aligned` (own@1 $= 0.31$).
     **Three probes, three checkpoints, one paragraph**, and the abstract repeats
     $30\%/41\%/6\%$. Choosing which checkpoint N5 speaks for is a call that propagates into
     the abstract, so it is HIS, with every number laid out here.
  7. **ALSO MINOR, left alone:** `fig:belowfiller` says the own-filler gap "widens with $m$";
     run 29's is $-0.262, -0.420, -0.551, -0.806, -0.742$ -- it narrows at $m{=}12$. The body
     makes the same overstatement at l.530 ("monotone in length"), so this is one editorial
     decision, not two, and it belongs with item 6.
  **THE RULE, seven for seven, with a correction to how six of them were found.** The previous
  six were found by EXTENDING the detector against a fixed file. This one was found by the file
  CHANGING: the honest trigger is not "invent a seventh pattern" but **"new prose has landed,
  sweep it"** -- and the five quiet ticks that refused to invent one were right. **Standing
  instruction: any tick that finds `main.tex` mtime moved must diff it and audit only the diff.**
  Verified: `pdflatex` x2 rc=0, **61 pp, 0 undefined refs**, 3 overfull boxes **which are
  pre-existing** (byte-identical count and line numbers from the pre-edit backup -- checked, not
  assumed); backups `main.tex.bak_pre_figcaptions_20260921_1007`,
  `POSTMORTEM.md.bak_pre_figcaptions_20260921_1007`.

- **WHERE THE PROGRAMME STANDS AFTER RUN 34 (queue state, for the next tick).** Every axis this
  project has opened is now closed by a measurement, not by an opinion:
    writer accuracy      runs 28/30/31 + the nar probe -- FOUR levers, one flat floor (~0.07).
    decoding regime      run 30 (DAgger) falsified; nar measured below free-running.
    selection / format   abstention freq (run 32), conf and anti (closed without a run, #14).
    block cost           run 33: the mechanism peaked at step 400 and reversed.
    speak vs abstain     run 34, THIS ENTRY: value tracks volume, precision was a selection effect.
  **DO NOT LAUNCH A SUCCESSOR ON ANY OF THEM.** The honest deliverable is the write-up, which is
  what the 06:07 tick did, and the CPU-only evidence the 08:07 tick then spent the idle device on
  (the sec:bench control on run 34 @800: vanilla step_0026600 clean / r34 clean / r34 think). Phase C (roadmap item 8) is gated on an L=16 own thought being net
  positive held-out and that has never happened. Re-occupying the device is a PROGRAMME-LEVEL
  call and is flagged for Emanuele in the journal rather than improvised at 06:07.

- **G50 -- G20's BRACKET TRICK FAILS WHEN THE UNBRACKETED LITERAL SITS ELSEWHERE ON THE SAME
  COMMAND LINE, AND A BARE `pgrep -c` COUNT OF A WORKER PATTERN IS UNINTERPRETABLE
  (2026-09-22 00:07).** The 00:07 tick's close check ran `pgrep -c -f 'think_train.py --corpus'`
  and got **1** on a host with no trainer running. Re-run as `-af`, the single hit was the
  tick's own `/bin/bash -c` (pid 1398350). The tick then ran the G20-approved bracketed form
  `think_trai[n]\.py --corpu[s]` **in the same shell command as the unbracketed one** -- and it
  matched too, because that shell's own cmdline contains the plain string `think_train.py
  --corpus` as the first pgrep's quoted argument. **The bracket protects the PATTERN token, not
  the command line.** G20 already says this for a heredoc that LAUNCHES the program; this is the
  same failure with a second pgrep in place of the launch, and it is more insidious because the
  bracketed form is the one the runbook tells you to trust.
  Rules, in order of reliability:
  1. Never put a bracketed and an unbracketed form of the same pattern in one shell command.
  2. Never read a bare `pgrep -c <worker pattern>`; the count cannot be audited. Always `-af`,
     and check whether each hit is your own shell.
  3. For a definitive idle reading, do not use pgrep at all:
     `ps -eo pid,ppid,etime,cmd --no-headers | grep -E 'think_train|run_think|torchrun|score\.py|probe_' | grep -v "^ *$$ " | grep -v shell-snapshots`
     -- on an idle host this leaves only the grep itself. Cross-check against the expected
     background set: 14 `runs/plot_gain.py` daemons (one per run think20..think34, oldest 11d),
     plus `networkd-dispatcher` and `unattended-upgrade-shutdown`. Anything else is real work.
  This is the SEVENTH self-measurement trap the quiet ticks have found. The full list, all of
  which read at a glance as "something changed" and none of which was: `.git/objects` blobs
  (judge commits by HEAD/refs/index + reflog), `pgrep -c neuron`, `heldout.tsv` mtime (read the
  last ROW), `~/.bash_history` mtime (the harness's pty spare touches it; judge an interactive
  session by a NEW `/dev/pts` entry + `INTERACTIVE_HEARTBEAT` + the spawner pid),
  `~/.claude.json` mtime (the tick's own start), and the two in this entry.

- **G51 -- `ps > file` IN THE SAME SHELL RE-INJECTS THE PATTERN INTO THE FILE, AND THE WRAPPER
  PID IS NOT `$$` (2026-09-22 08:07).** The 08:07 tick's CLOSE check ran
  `ps -eo pid,cmd > /tmp/ps_close.txt; grep -Ec 'think_train|think_par|torchrun|probe_thinkmode'
  /tmp/ps_close.txt` and got **1** on an idle host. Looked at the line instead of trusting the
  count: it was **the tick's own `/bin/bash -c ... eval '...'` wrapper (pid 1432681)**, whose
  cmdline contains the grep pattern verbatim because the pattern is an argument inside the
  eval'd string. Dumping `ps` to a file first does NOT escape G50 -- the dump is taken while the
  wrapper is running, so the pattern lands in the file and the later grep finds it there.
  Two additions to G50:
  1. **`grep -v shell-snapshots` is the load-bearing filter, not `grep -v "^ *$$ "`.** The
     matching wrapper was pid **1432681** while `$$` was **1432641** -- a *child* of the
     reporting shell, so the `$$` filter alone misses it. Every harness wrapper cmdline contains
     `shell-snapshots`; that is the reliable discriminator.
  2. **Apply the full item-4 form at CLOSE too, not just for the definitive reading.** The
     temptation is a quick `grep -c` to confirm what you already measured five calls earlier;
     that quick form is precisely the one that lies. And never report a bare count: when a count
     is non-zero, print the matching LINES before drawing any conclusion.
  The real worker count that tick was **0**, unchanged, confirmed by the item-4 filtered reading
  (14 `runs/plot_gain.py` daemons + `networkd-dispatcher` + `unattended-upgrade-shutdown`).
  This is the EIGHTH self-measurement trap the quiet ticks have found.

## Live infrastructure (cron, all independent of any shell)

All five read the run name from `.watchdog_run` (currently `think34_speak`).

    */10 keepalive_think.sh   relaunches run_think.sh <run> until STATUS is DONE (or a failed smoke)
    */5  watchdog.sh          -> runs/<run>/WATCHDOG.md ; quarantines orphans, clears stale
                              compile locks, kills genuine stalls (75 min), never loops on a failed smoke
    */3  archive_ckpts.sh     hard-links every 500th ckpt to checkpoints/<run>_archive/
    */15 rsync benchmark/results -> /home/ubuntu/bench_backup
    */30 heldout_probe.sh     -> runs/<run>/heldout.tsv ; held-out CE of every checkpoint (CPU)

## Files (after the 2026-09-10 clean-up)

    run_think.sh <run>                      the ONE launcher; runs/<run>/config.sh holds INIT/NPROC/GA/FLAGS
    keepalive_think.sh watchdog.sh archive_ckpts.sh   cron trio (above)
    runs/<run>/{train.log,smoke.log,STATUS,metrics.jsonl,dumps.jsonl,dumps_eligible.jsonl,MONITOR.md,WATCHDOG.md,gain.png,ntp_loss.png}
    pre-training/train/think_train.py       THE trainer (was think18_train.py); fp32 params and no ZeRO-1 by default
    pre-training/train/think_par.py         geometry, generator, scoring, weighted_ce3 (unchanged)
    pre-training/train/test_rollouts.py     CPU gate (paired baselines, soft gate, curriculum, mixed anchors)
    pre-training/train/monitor_think.py, runs/plot_gain.py, runs/digest.py <metrics.jsonl>
    benchmark/                              zero-shot eval; METRICS.md is the one summary file
    checkpoints/think18_archive/step_0002000.pt, think19_archive/step_0000500.pt   the two benchmarked ckpts
    checkpoints/_superseded/                runs 15-19 live checkpoints (78 GB); safe to `rm -rf`
    runs/think15_9b .. think19_long         metrics/dumps/logs of runs 15-19 (the paper's evidence; small)

Deleted: think13/think16 trainers, run_think14-19.sh, keepalive_think14-19.sh,
dev_*.py, eval/ (superseded by benchmark/), qwen0.6b-inference.py.

## 64 ranks: verdict (2026-09-10)

Works. `--param-dtype fp32 --no-zero1`, R=8, T=1024: host RSS peaks at
**6.0 GB/rank (384 GB of 2 TB)** -- the 32.9 GB/rank OOM of run 16 was ZeRO-1 +
16 scoring passes, both gone. Steady state **26-28 s/step at 64 x GA 2** vs
36 s at 32 x GA 4 (same 131k tokens/step): 1.33x, not 2x. Halving the
per-rank micro count removed ~9 s, so ~17 s/step is a fixed cost that does not
scale with GA -- the fp32 gradient all-reduce over 16 devices and host-side
contention of 64 samplers (CPU ~55% busy, load ~100) are the candidates.
Phase A ETA: 7,700 steps x 27 s = 58 h.

## Reporting order-borne value honestly (2026-09-13)

Two numbers are logged and they disagree: `g_win - g_shuf` (the winner vs its OWN
shuffle, +0.8 at the end of run 21) is inflated by best-of-8 selection (~+1.42 sd
on noise); `g_mean - g_shuf` (mean rollout vs the winner's shuffle, -1.4) is
biased the other way. The trainer now logs `g_sd` (mean per-anchor sd of the R
gains) so the correction 1.42*g_sd can be applied. The paper's run-18/19 tables
report g_mean - g_shuf; fix the column label.

## The three-phase plan (user, 2026-09-10)

    A  run 20   1B tokens from vanilla step_0019190 (8.04B total), L=16, k=64, stride 64,
               1/4 SSL anchors, forced thinking elsewhere, fp32, 64 ranks x GA 2 = 131k tok/step,
               7,700 steps. Run 18's exact config minus the bf16 defect.
    B           0.5B tokens, SSL fraction lowered (user asked 0%; 5-10% argued -- 0% is the
               handover that failed three times).
    C           2B mid-training, L=64, thoughts SPARSE under a fixed per-sequence budget:
               rank anchors by the <T> logit, open exactly B of N (top-B or sample without
               replacement), supervise placement densely with per-anchor oracle gain,
               validate with a random-placement control of the same B thoughts.

Data: the sampler draws random windows (data.py), so the 8.04B rung has
touched ~43% of the 9.0B corpus once; 3.5B more tokens is mild repetition,
not epochs. The vanilla ladder rung at ~11.5B total is the equal-token control.

- **G52 -- UNDER `--bridge`, `orG` IS SCORED AGAINST THE *CLEAN* ROWS AND IS THEREFORE
  MEANINGLESS AS A "READER" SIGNAL (2026-09-22 18:07, run 35 @100).** Run 35's config comment is
  explicit that with the bridge "a thought's gain is measured against SILENCE UNDER THE SAME
  MASK, never against the clean rows, which still see everything". That conversion was done for
  the *reward* path and only there: `think_train.py` builds `base_masked` (silence under
  `mask_dev`) and `_score()` returns `lw - base_masked`, so **`g_r0`, `g_spec`, `ob0`, `g_win` --
  every number B1/B2 are judged on -- are correct.** The `orG` monitor was NOT converted. It is
  computed separately at `think_train.py:816` as `((ce_cln - ce_win) * gam_win).sum(-1)` over the
  SSL anchors, and `build_full_mask` (`think_par.py:121`) applies the bridge to the WINDOW rows
  only -- clean rows keep the plain causal mask. So under `--bridge 32`:
      orG = (value of the oracle thought)  -  (cost of hiding 32 context tokens)
  and the second term dominates by an order of magnitude. Run 35 logs orG **-10.9 -> -12.4**,
  where the oracle content is *by construction the exact hidden span* and should score hugely
  POSITIVE against the honest baseline. Nothing is broken: orG enters **no loss**, only
  `metrics.jsonl`, `STATUS` and `MONITOR.md`.
  **The trap is in the reading, and it is pre-loaded to fire.** `MONITOR.md`'s auto-generated
  gloss is still the run-13 boilerplate -- *"reader: oracle gain -10.91 -> -12.36 nats (rising =
  learns to use a good thought)"* -- which (a) calls a number that is getting MORE negative
  "rising", and (b) invites a tick to read the bridge handicap as the reader decaying and stop a
  healthy run on it. **Under a `--bridge` run, orG is not a live axis; judge the reader by
  `rdCE` and the run by the `_score`-derived gains only.** Recorded rather than fixed: the
  correct monitor needs an extra scoring pass on the oracle content, a code change cannot reach
  the running ranks anyway, and a restart to buy a logging line would cost the run's first
  ~100 steps and its comparability. Fix it in the next run that uses `--bridge`.

## RUN 35 (THE BRIDGE) IS A NEGATIVE, AND `probe_bridge2` SAYS WHY: THE READER PAYS FOR
## FORECAST, NOT FOR CARRIED CONTEXT (stopped 2026-09-22 ~19:08 at step ~250)
Run 35 hid the last 32 context tokens from the WINDOW rows (`--bridge 32`) while leaving them
visible to the thought block, so the writer's job became CARRYING (a copy, in-slot teacher-forced
accuracy 0.30-0.41) instead of FORECASTING (which seven checkpoints could not do). Its
pre-registered B1 was `g_r0 > +0.5` at step 300. It was stopped at ~250 with g_r0 **-0.32** and
falling; the smoke's +1.20/+1.47/+1.67 `g_spec` was 2-micro noise at init and did not survive
100 steps.
**The instrument that makes it a result rather than another flat line is
`runs/probes/bridge2_r35_200.json` (run 35 @200, bridge=32, n=154 held-out anchors, nats vs
SILENCE UNDER THE SAME MASK — the honest baseline):**
```
  hidden_raw    -19.63 +- 0.75    the 32 hidden tokens dropped in the slot, NO </T>
  hidden_term   -0.27  +- 0.18    the same tokens, terminated with </T>
  hidden_tail   +0.57  +- 0.19    the last 16 of them, terminated
  future_term   +9.85  +- 0.59    the true FUTURE tokens, terminated
  own           +0.10  +- 0.17    the model's own thought
  filler        -0.03  +- 0.11    content-free filler
```
Hiding the 32 tokens costs ~8 nats of window CE. Carrying **the exact hidden text** back into the
slot recovers **-0.27 +- 0.18, i.e. nothing** (its 16-token tail, +0.57 +- 0.19, is the only
positive and it is 3 se), while the true future tokens are worth **+9.85** under the very same
mask. **The reader does not pay for information it is missing; it pays for text that looks like
the continuation.** That is a stronger and more general negative than "the writer cannot forecast":
it says the slot's only exploitable currency at this scale is forecast-shaped text, so re-aiming
the writer at an easier *carrying* target cannot work, no matter how well it carries. Run 36
(the bridge anneal) is therefore dead on arrival and was never launched.

### G53 -- A THOUGHT WITHOUT `</T>` COSTS ~19 NATS; TERMINATION IS NOT COSMETIC
Same probe, same tokens, same weights: `hidden_raw` **-19.63** vs `hidden_term` **-0.27**. The only
difference is the closing `</T>`. An unterminated block leaves the reader inside the thought, and
the window rows pay for every token of it. Consequences: (a) any probe that splices content into
the slot MUST terminate it or its number is meaningless (~19 nats of pure framing error);
(b) the superlinear length price recorded in G39 is the same effect seen from the other side --
a writer that loses its stop action does not merely get worse, it falls off a cliff;
(c) when reading a new writer, check `len_mean` and whether `</T>` is reachable at all BEFORE
reading any gain.
**CONFIRMED ON THE TRAINING PATH, not just in a probe (2026-09-22 22:07 tick, run 37's smokes).**
Two smokes, the SAME init (`code_init_from_r25_1500.pt`), the same three batches, the same 4 code
symbols, differing ONLY in whether `</T>` was forced into the last slot:
    unterminated (`--L 4 --min-think 4`)          g_r0  -13.62 / -17.03 / -19.47
    terminated   (`--L 5 --min-think 4` + force)  g_r0   -2.58 /  -4.21 /  -5.12
**+11.0 nats at step 0 from one token.** Third independent measurement of G53 and the first on the
training loop's own scorer, so the effect is not an artifact of `probe_bridge`. It also means the
run-37 pre-registered sanity gate C0 (`g_r0 > -3.0`) is met at the smoke's step 0 (-2.58).

### G54 -- IN A `--code-size` RUN, THREE DERAIL AXES GO DEAD BY CONSTRUCTION, AND FRESH IDS
### NEED AN EMBEDDING INIT (2026-09-22, run 37)
`--code-size 64 --code-base 151670` restricts the thought alphabet to 64 reserved ids. Two
consequences a tick must know before it reads a code run:
1. **The implementation lives ONLY in the full-vocab sampling path.** `sample_step_full`
   (`think_par.py:265-277`) masks everything outside `[code_base, code_base+code_size)`;
   `sample_step_topk` (l.212) does NOT — it only bans `id >= ban_from` (EOD 151643), which would
   ban every code symbol. **A code run therefore REQUIRES `--topk-transfer 0`** (runs 37/38 set
   it). With topk transfer on, the writer would be unable to emit any code symbol at all.
2. **`</T>` is unreachable when `--min-think >= L`.** `sample_step_full` only un-masks `T_CLOSE`
   at `t >= min_think`, and runs 37/38 use `--L 4 --min-think 4`, so the thought is a FIXED
   4-symbol word and `stop_pos` always returns L. Hence `len_mean == 4.0` at every step and the
   3b derail clause *"thought length within 0.5 of `--min-think` for 2 blocks"* fires on a
   perfectly healthy run from step 1. Dead too: the `top RL thought prefixes` line and `reg%`
   read prefixes of ids that decode to nothing, and `imCE`/`orG` are 0 whenever `--ssl-frac 0`.
3. **Reserved ids start worth about -14 nats.** Run 37's first smoke (19:09, init = run 25 @1500
   untouched) logged `g_mean` **-14.1 / -17.9 / -21.3** on its three steps: an untrained embedding
   in the slot is not neutral, it is catastrophic, and no RL signal is visible underneath it.
   The fix (interactive owner, 20:09) is `checkpoints/code_init_from_r25_1500.pt` — the same
   weights with the 64 code embeddings set to the SPACE embedding, so the alphabet starts at the
   filler floor (~0) instead of -14. **Any future run that introduces fresh token ids must
   initialise their embeddings to a known-neutral token first.**
   (The same smoke's `ntp1` 3.15 -> 3.39 -> 3.95 is NOT degradation: at step 2 of a 200-step
   warmup the lr is ~1.5e-7. It is 2-micro noise. Do not read a smoke's ntp1 as a derail.)

### G55 -- OUR OWN WATCHDOG DELETED A LIVE COMPILE LOCK AND KILLED RUN 37's SMOKE
### (2026-09-22 22:07 tick; fixed in `watchdog.sh`)
`watchdog.sh` step 2 implemented G5 ("a killed compiler leaves a `.lock` with no `.neff`") as
**age alone**: `find /var/tmp/neuron-compile-cache -name '*.lock' -mmin +20` and delete if no
`.neff` sits beside it. **A cold compile of a new graph shape takes far longer than 20 minutes,
and the `.neff` is written only at the very end**, so a healthy compile looks exactly like a
stale lock for the whole middle of its life.

What happened, on the clock:
- 20:15:48 run 37 relaunched with the new `--L 5` shapes => an all-new graph => cold compile.
- 20:45:01 the watchdog "cleared stale compile lock MODULE_11671161841538108445+fad94d7c".
- 21:28:20 that same module reported **`Compiler status PASS`** -- it had been alive the whole
  time -- and then rank 14 died in `think_train.py:838` with
  `FileNotFoundError: .../MODULE_11671161841538108445+fad94d7c/model.hlo_module.pb.lock`.
  Wrapped by Neuron as `RunNeuronCCImpl: error condition !(error != 400)`, which reads like a
  compiler bug and is not one.
- 21:28:22 torchrun took the job down; STATUS `FAILED (... smoke)`; the watchdog then correctly
  refused to relaunch (8 ticks of "NEEDS A HUMAN") and the device sat idle ~40 minutes.

**THE FIX (this tick, backup `watchdog.sh.bak_pre_locksafe_20260922_2207`):** the lock sweep now
runs only when **no compiler process exists at all** (`pgrep -f 'neuronx-cc|walrus_driver'`) and
the age floor is 30 min. That is what "stale" actually means -- a killed compiler leaves no
process behind, a slow one does. Age can never distinguish the two.

**Reading rule for any future tick:** a `FileNotFoundError` on a `*.lock` inside
`/var/tmp/neuron-compile-cache` is OUR DELETION, not a Neuron fault. Check
`runs/<run>/watchdog.log` for a "cleared stale compile lock <MODULE>" line naming the same module
hash as the traceback. The compile's output usually survives (run 37's `model.neff` and
`model.done` were both on disk afterwards), so the relaunch is warm and cheap -- just relaunch.

## RUN 37 (THE PURE-RL CODE) IS A NEGATIVE WITH A MECHANISM: A SYMMETRIC INIT IS A
## FIXED POINT THAT RL CANNOT LEAVE (stopped 2026-09-23 00:12 at step ~420)
**Verdict first: the 64-symbol alphabet never became an alphabet.** Run 37 gave the writer 64
reserved ids with no pretrained meaning and asked pure RL to invent a code. After 420 steps the
symbols are still the same vector as each other, so nothing the writer emits can carry a message.

The instruments, all from `runs/think37_code/metrics.jsonl`, blocks of 100 steps:
```
            0-99    100-199  200-299  300-399   400+
  g_r0     -1.2125  -0.4918  -0.4956  -0.4496  -0.3823
  g_shuf0  -1.2125  -0.4922  -0.4954  -0.4482  -0.3802
  ob0      +0.0000  +0.0004  -0.0002  -0.0014  -0.0021   (se 0.001)
  g_sd      0.037    0.032    0.033    0.032    0.034
```
`g_r0`, `g_spec`, `g_shuf0`, `g_best` agree to three decimals at every block, `g_win` (best of 8)
beats them by 0.05, and the rollout spread is 0.03 where C2 asked for 0.6. **Shuffling the code
changes nothing because there is nothing to shuffle.**

**THE MECHANISM, measured directly on the checkpoints rather than inferred** (embedding rows
151670-151733 of `code_init_from_r25_1500.pt` vs `checkpoints/think37_code/step_0000400.pt`):
```
                       pairwise code-code distance     dist to SPACE row
  init                 mean 0.01532  max 0.01656       mean 0.01083
  step 400             mean 0.01609  max 0.01786       mean 0.01228
  NORMAL tokens, same metric, for scale:  mean 1.016
```
G54's fix -- initialise the new ids to the neutral SPACE embedding so they start worth ~0 instead
of -14 nats -- **also made all 64 of them the SAME VECTOR** (pairwise separation 1.5% of a normal
token pair). That closes a loop RL cannot open:

> identical embeddings -> all 8 rollouts score identically (spread 0.03)
> -> the paired advantage is ~0 -> no gradient that distinguishes symbol i from symbol j
> -> the embeddings stay identical.

400 steps moved the pairwise distance by **+0.0008**. At that rate a normal token's separation is
~10^5 steps away, and the rate cannot rise, because the only force that would raise it is the
reward variance it is itself suppressing. **C2 (`g_r0 > 0` AND `g_sd > 0.6`) at step 800 was
unreachable, and this is why** -- not a forecast, an arithmetic property of the fixed point.

**C1 PASSED AND THE RUN STILL FAILED; THIS IS G38 CATCHING A GATE WE WROTE OURSELVES.** C1 asked
for mean `g_r0`(300-399) >= mean `g_r0`(0-99) + 0.3, and it cleared it by **+0.763**. But
`g_r0 = (block price) + (content)`, and the decomposition is unambiguous here: the content term
is `ob0` = -0.001 +- 0.001, flat since step 0, while the whole +0.763 is the reader learning to
pay less for a 5-token block of an indistinguishable symbol. That is precisely the term G38 says
cancels out of the success bar, and run 26 already bought it more cheaply (filler floor -0.083).
**A gate written on `g_r0` alone cannot see the difference between the two terms and will pass a
dead run.** Gate a code run on `ob0` and `g_sd`, never on `g_r0`.

Run 37 was stopped on SUPERVISOR.md 3b's live code-run axis (`rollout spread < 0.6`), which is
also the registered failure mode of a pure-RL code ("collapse to a CONSTANT code"), with the
embedding measurement above as the mechanism. Health was otherwise fine at the stop: held-out mix
3.1683 against a 3.36 derail bar, ntp1 block mean 2.75, no NaN, no `NRT_`, 64 ranks.

### G56 -- A NEUTRAL INIT FOR NEW TOKEN IDS IS ALSO A SYMMETRIC ONE, AND PURE RL CANNOT BREAK IT
### (2026-09-23 00:07 tick; this is the sequel to G54 and it partly retracts its advice)
G54 said: "any future run that introduces fresh token ids must initialise their embeddings to a
known-neutral token first." **That is still right about the LEVEL and wrong about the SPREAD.**
Setting every new id to the same neutral vector makes them mutually indistinguishable, and under
a pure-RL objective that is a fixed point (see run 37 above): equal embeddings give equal rewards
give zero differentiating advantage give equal embeddings.

**The corrected rule: initialise new ids to a neutral token PLUS symmetry-breaking noise**, at the
scale of real token separation (~1.0 pairwise here, i.e. per-row noise of order 0.5-0.7 of the
embedding norm), not the ~0.015 of numerical jitter run 37 had. **Or supply a supervised target
that breaks the symmetry from outside**, which is what run 38's `--code-teacher` does: a CE
gradient toward a specific symbol id differentiates rows on step 1 regardless of where they start,
because the gradient on output row j is `(p_j - 1[j=target]) * h` and that already depends on j.
**Diagnostic for any future alphabet run, cheap and decisive, run it at the first checkpoint:**
load the new embedding rows and print their mean pairwise distance against a normal-token
baseline. If it is not moving toward the baseline, the alphabet is one symbol wearing 64 hats and
no amount of training time will change that.

### RUN 38 -- THE HASHED-WINDOW TEACHER: STOPPED 2026-09-23 02:12 AT STEP ~410
### H1 FAIL AND H2 FAIL, BOTH PRE-REGISTERED AT 00:07 BEFORE THE FIRST TRAINING STEP
Run 38 was run 37's alphabet plus an outside symmetry-breaker: `--code-teacher 1` supervised a
quarter of the anchors on a 64-way hash of the most frequent content token in each half of the
UPCOMING 64-token window. Per G56 that gradient differentiates output rows on step 1 by
construction, so it should have separated the alphabet where run 37's pure RL could not.

**It did not, and the reason is arithmetic rather than a matter of training time.**

    pre-registered gate            bar               measured              verdict
    H1 code-row pairwise dist      >= 0.05, rising   0.01543 @200          FAIL
      (init 0.01532, normal 1.016)                   0.01617 @400          FAIL
    H2 ob0 (300-399)               >= +0.05          -0.0018 +- 0.001      FAIL
    H2 g_sd (300-399)              >= 0.30           0.0313                FAIL

`g_r0`, `g_shuf0`, `g_spec` and `g_best` again agreed to three decimals at every block, exactly as
in run 37: the content term was zero from step 0 and never moved.

**THE MECHANISM, and it is a new one -- read the imitation CE against its own baselines.** With
three supervised positions (sym, sym, `</T>`) and a trivially learnable close, the scalar has
three reference points:

    guessing uniformly over the alphabet     2*ln(64)/3          = 2.7726
    knowing only the target's marginal       2*3.922/3           = 2.6147
    knowing the window                       -> 0

`imit_ce` fell 11.11 -> 2.79 by step 100 and then **sat at 2.776 / 2.758 / 2.758 / 2.770 for the
next 300 steps**. Decomposed per symbol position, the information recovered about the hash was
**-0.005, -0.015, +0.009, +0.022, +0.022, +0.004 nats out of the 4.159 available** -- non-monotone,
smaller than a tenth of the marginal skew, and indistinguishable from zero. The entire 11.11 ->
2.78 drop was the FORMAT (emit two symbols, then close), which is worth nothing.

The target was not degenerate; that was checked on the real corpus rather than assumed: 64/64
symbols used, entropy 3.922 of ln(64)=4.159, top-symbol share 0.102.

**So the teacher never applied a separating force.** A CE whose Bayes-optimal output is the uniform
distribution pushes every output row by the same expected amount: the gradient on row j is
`(p_j - 1[j=target]) * h`, and if the target is independent of the context then `E[h | target=j]`
is the same for all j, so the expected gradient is identical across rows and only noise separates
them. **The supervised target that was supposed to break the symmetry from outside was itself
sitting at the symmetric point**, because a 64-way hash of text the writer has not read is not
predictable from the writer's context. Run 37 died of symmetry under RL; run 38 died of symmetry
under a teacher that carried no information. Both are downstream of the same fact.

**WHY THE CODED-THOUGHT PROGRAMME IS CLOSED, and it is not closed on these two runs.** It is closed
on run 35's bridge probe (`runs/probes/bridge2_r35_200.json`, n=154, nats vs silence under the same
mask): the reader pays **+9.85 +- 0.59** for the true FUTURE tokens and **-0.27 +- 0.18** for the
exact context text it is missing. The reader values forecast-shaped text and nothing else. A
64-symbol code is not forecast-shaped text, so even a perfectly separated, perfectly informative
alphabet would have been paid the filler floor. **Runs 37 and 38 were dead before they started, for
a reason neither of them measured**; their own failures are symmetry, which is a shallower problem
than the one that actually closes the route. Do not launch another coded-alphabet hypothesis run.

### G57 -- A SUPERVISED TARGET ONLY BREAKS A SYMMETRY IF IT IS PREDICTABLE
### (2026-09-23 02:07 tick; the sequel to G56, and it sharpens the second half of its rule)
G56 offered two ways to separate fresh token ids: symmetry-breaking noise at init, **or** "supply a
supervised target that breaks the symmetry from outside". Run 38 took the second and failed, so the
second clause needs a condition attached:

**A cross-entropy target separates output rows only to the extent that it is PREDICTABLE from the
model's context.** If the target is independent of the context, the Bayes-optimal output is the
marginal -- which for a hash is the uniform distribution, i.e. precisely the symmetric point the
target was introduced to escape. The per-row gradients then differ only by noise. An "outside
force" that is uninformative is not an outside force.

**How to check it in one line, before spending a run:** compute the target on the real corpus and
compare the achieved CE to its three baselines (uniform, marginal-only, zero). If the run sits at
uniform or marginal-only, the target is carrying nothing and the run is measuring its own format.
`2*ln(code_size)/3` for a 2-symbol-plus-close block is a number worth memorising; run 38 sat on it
to three decimals for 300 steps and the scalar still looked like it was "converging".

**Corollary, and it is why run 39 exists:** an uninformative-target failure and a dead-channel
failure look IDENTICAL in `imit_ce`. They are separated by giving the same channel a target that is
in-context and trivially computable. If the CE still will not move, the channel is the problem.

### RUN 39 -- THE CHANNEL CONTROL: DONE AT 300, K1 FAIL AND K2 FAIL
### (both pre-registered in `runs/think39_past/config.sh` before the first training step)
Run 39 existed to remove the one confound runs 37 and 38 leave behind: per G57, an
uninformative-target failure and a DEAD-CHANNEL failure look identical in `imit_ce`. It gave the
same 64-symbol alphabet a target that is in-context and trivially computable -- `--code-teacher 2`,
slot 0 = hash(x[n-32]), slot 1 = hash(x[n]) -- and made every other condition maximally favourable
to the channel: `--ssl-frac 0.9` (near-total supervision) and `--rl-w 0.0` (no reward to fight the
teacher). Bounded at 300 steps with no extension clause.

    pre-registered gate                       bar            measured             verdict
    K0 sanity, g_r0 block 0-19                > -3.0         -3.19 (see caveat)   see below
    K1 THE GATE, mean imit_ce over 150-199    < 2.00         2.724                FAIL
    K2 cross-check, code-row pairwise dist    >= 0.05        0.01541 @200         FAIL
      (init 0.01532, normal tokens 0.994)                    0.01565 @300         FAIL
      untied lm_head rows (normal 1.194)                     0.03605 -> 0.03760   FAIL

K0's -3.19 is not a terminator failure (G53 reads ~-14, and `len 4.0/2.0` at step 0 is the greedy
row closing at 2 as taught); with `--ssl-frac 0.9` the sampled rows are `or/sm 24/2`, so every
g-axis on this run is a mean over ~2 anchors and is near-useless by construction. `imit_ce`, over
the ~24 oracle rows per step, was the instrument and it is the one that decided the run.

**THE SHAPE OF THE FAILURE IS THE RESULT, AND IT IS SHARPER THAN THE GATE.** Target entropies were
computed on the real corpus BEFORE launch (4.012 and 4.009 of ln 64 = 4.1589), so:

    guessing uniformly          2*ln(64)/3   = 2.7726
    knowing only the marginal   2*4.0105/3   = 2.6737
    knowing the context         -> 0

    steps       50-74  100-124  150-174  200-224  250-274  275-299
    imit_ce     2.844   2.771    2.728    2.693    2.694    2.679
    of the 0.148 nats/symbol of CONTEXT-FREE marginal skew, recovered:
               -0.107  +0.002   +0.067   +0.119   +0.119   +0.141
    of the 4.011 nats/symbol of IN-CONTEXT information, recovered:  <= 0 at every block

**The channel learned 0.141 of the 0.148 nats per symbol that a fixed unigram table over the
alphabet is worth, and none of the 4.011 nats that were sitting in plain sight in its own context.
It asymptotes at the marginal floor FROM ABOVE and stops 0.007 short of it.** Rollout spread
`g_sd` 0.032, as in runs 37 and 38. Health was clean at the end and no derail axis ever fired:
held-out mix 3.256 -> 3.173 (it FELL), ntp1 block mean 2.893, no NaN, no `NRT_`, 39.3M tokens.

Run 38's failure was its target sitting at the symmetric point. Run 39's target does not: it is a
pure lookup of a visible token, and the separating force is measurably larger (lm_head code rows
moved +0.00155 in 300 steps against run 38's +0.00098 in 400). It is still ~1/700th of what would
be needed, and the CE says why -- the row separation that DID happen bought exactly the
context-free part of the target and nothing else, which is what a per-row bias is worth.

### G58 -- THE CHANNEL, NOT THE TARGET. A FRESH ALPHABET DOES NOT BECOME ONE IN A SHORT RUN,
### EVEN UNDER A PREDICTABLE IN-CONTEXT TEACHER (2026-09-23 04:07 tick; closes G56/G57)
G56 offered two routes to separating fresh token ids: symmetry-breaking noise at init, **or** an
outside supervised target. G57 attached a condition to the second (the target must be predictable).
**Run 39 satisfied that condition in the cleanest form the design admits and the second route still
failed. Keep G56's first clause; the second is retracted.**

The diagnostic that reads this correctly, and the rule:
1. **Read `imit_ce` against uniform AND marginal-only, per block, as nats per symbol** (G57's rule,
   now with the worked case). A run that descends and then flattens at the marginal floor has
   learned a per-row bias -- that is what a small, uniform row separation buys -- and has learned
   nothing about the context. Run 38 flattened at UNIFORM (target unpredictable); run 39 flattened
   at MARGINAL (target predictable, channel unable). Those are different diseases with the same
   headline symptom, and only the three-baseline decomposition tells them apart.
2. **A supervised target cannot substitute for a separated init.** Both runs that tried it moved
   the code rows by <0.002 against a normal-token scale of ~1.0. If a future design needs a fresh
   alphabet, initialise it with noise at the scale of real token separation (~0.5-0.7 of the
   embedding norm per row) and verify the pairwise distance at the FIRST checkpoint. Do not launch
   a run whose success requires embeddings to travel ~1.0 in a few hundred steps at lr <= 1e-5;
   under Adam that is within a small factor of the per-step speed limit and the gate will fail on
   arithmetic rather than on the hypothesis.
3. **The coded route is closed anyway, and NOT on runs 37/38/39.** It is closed on run 35's bridge
   probe: the reader pays +9.85 +- 0.59 for forecast-shaped text and -0.27 +- 0.18 for the exact
   text it is missing. A 64-symbol private code is not forecast-shaped text, so a perfectly
   separated, perfectly informative alphabet would have earned the filler floor. **That measurement
   was on file before run 37 launched and none of the three runs consulted it.** The rule this
   leaves: before opening a new MEDIUM for the thought, check what the reader pays for that medium,
   not only whether the writer could produce it. Do not launch another coded-alphabet run.

### G59 -- A METRIC COMPUTED WHILE A RUN IS LIVE IS A *PREFIX*, NOT THE RUN'S VALUE; AND STEPS
### WITH NO SAMPLED THOUGHT DILUTE IT TOWARD ZERO
### (2026-09-26 12:07 tick; this RETRACTS the 10:07 tick's recommendation #2)

The 10:07 tick reported "POSTMORTEM's chronology row 17 is stale: `-0.078 +/- 0.007`, where
main.tex:2021 and summary.tex both carry `-0.070 +/- 0.008` for the same run-17 ob" and advised
"the summary followed the paper, which is the right source; reconcile the row". **That is
backwards. Row 17 is correct and the paper's number is the partial one.** Recomputed this tick
from `runs/think17_curric/think17_metrics.jsonl` (906 rows, steps 0-905), quantity
`g_mean - g_shuf`:

      window                                          n      mean       se
      step>=400, thought sampled (n_th>0)  FULL RUN   396   -0.0778   0.0071   <- POSTMORTEM row 17
      step>=400, thought sampled, FIRST 249 steps     249   -0.0697   0.0079   <- main.tex / summary
      step>=400, NO sampled-step filter               506   -0.0609   0.0057
      run 15, step>=400, thought sampled, full run   2570   -0.0666   0.0035   <- the paper's -0.067

All three published numbers reproduce to four decimal places. Nothing is wrong; they are
different windows, and **the paper does not say its run-17 value is a partial.** The 249 steps
are the first 249 of the run's 396 sampled steps, i.e. the number was computed live when run 17
had reached step 744 of 905, and it was never recomputed when the run ended.

Two rules follow, and the second is the one that will bite again:

1. **Any `+/-` quoted for a LIVE run is a prefix statistic. Recompute it at run end.** The
   quantity here is not stationary -- it drifts monotonically from -0.045 (steps 400-648) to
   -0.091 (steps >=850) -- so the prefix is not an unbiased estimate of the run's value, it is
   an estimate of a different window. Every chronology row written while its run was live is
   suspect in the same way; row 17 is simply the one that got caught because a later tick
   recomputed it.
2. **Filter to steps where a thought was actually sampled.** On a step with `n_th == 0` the
   trainer logs `g_mean == g_shuf == 0`, so the difference contributes an exact zero and pulls
   the mean toward zero. Unfiltered, run 17 reads -0.0609 against the true -0.0778 -- a 22%
   attenuation, larger than the effect several runs were judged on. `n_th>0`, `g_shuf!=0` and
   `think_rate>0` all select the same 396 steps here; `n_elig>0` does NOT (it selects 3) and
   `n_or>0` does not either (494). Use `n_th>0`.

**THE CLAIM ITSELF SURVIVES, and say so rather than quietly restating it.** The paper's
invariance argument is that run 17 sits within one s.e. of run 15 across an 8x horizon change
and an 8x search-width change. On the FINAL values that gap is
`-0.0778 - (-0.0666) = -0.0112 +/- 0.0079`, i.e. ~1.4 se -- wider than the "within one s.e."
the paper claims, still the same conclusion. The honest fix in `main.tex` (l.1812, l.1821,
l.2021, l.2023) and `summary.tex` (l.146) is to quote `-0.078 +/- 0.007 (n=396)` and soften
"within one s.e." to "within 1.5 s.e."; the alternative -- keeping -0.070 and labelling it
"steps 400-744, run in progress" -- is also honest but pointlessly preserves a partial.
**NOT DONE THIS TICK ON PURPOSE:** `summary.tex` is Emanuele's own deliverable of 2026-09-26
09:28 and was under his review at the time, and the change alters a headline sentence in a
document he is about to send. Flagged in the journal for him, not silently applied.

### G60 -- `probe_slotpos` AND `probe_tfslot` MEASURE THE SAME-SOUNDING QUANTITY WITH AND
### WITHOUT THE BROKEN OPENER. 0.30 AND 0.35 ARE BOTH RIGHT; DO NOT "RECONCILE" THEM
*(2026-09-26 14:07 tick, applying rule (6). Second consecutive tick in which a cross-document
"discrepancy" dissolved once the metrics were consulted. The 10:07 tick asked for one
agreement triple to be standardised across `main.tex` and `summary.tex`; that request is
RETRACTED -- carrying it out would have deleted a real measurement.)*

The two documents carry two teacher-forced-in-slot triples and they look contradictory:

      main.tex l.42 (abstract) and l.208 (N5)   0.30 TF / 0.41 clean / 0.06 free
      summary.tex l.207, main.tex l.610-618     0.35 TF / 0.43 clean / 0.019 free

**Both reproduce exactly, from two different probes on the same checkpoint** (run 25 @900):

      probe_slotpos.py -> runs/probes/slotpos_r25_900.json      n=232
        shifted_acc 0.30489 +/- 0.0105   ("0.30")
        ntp_acc     0.41322 +/- 0.0118   ("0.41")
      probe_tfslot.py  -> runs/probes/tfslot_r25_900.json       n=616
        tfslot per_pos mean 1-11  0.3474 ("0.35")
        ntp    acc1               0.4334 ("0.43")
        own    per_pos mean 1-11  0.0193 ("0.019")

**The 0.305-vs-0.347 gap is not noise and not staleness -- it is position 0.**
`probe_slotpos.py:56` averages accuracy over ALL in-slot positions
(`np.mean(lg[ok, :L-1] == tg[ok, :L-1], axis=1)`), so it INCLUDES the broken opener that
scores 0.047; `probe_tfslot`'s headline deliberately reports positions 1--11 and excludes it.
Averaging a 0.047 first position into eleven ~0.35 positions costs about the 0.04 observed.
The 0.413-vs-0.433 clean gap is anchor sampling: 232 anchors against 616, a 0.020 difference
on an se of 0.0118, i.e. ~1.7 se.

**Rules.**
1. **Never quote an in-slot accuracy without saying whether position 0 is in it.** The opener
   is a 9x outlier against every other position, so its inclusion moves any average over a
   short span materially. "Teacher-forced in-slot accuracy" is two numbers, not one.
2. **`summary.tex` l.270 is the model to copy, not a bug**: "$0.30$--$0.35$ agreement" spans
   exactly these two instruments and is the honest one-line statement. The defect in the two
   abstracts is a MISSING LABEL, not a wrong number -- so the fix is to name the probe, and
   standardising to a single value would throw away the position-0 decomposition that N5's
   whole argument rests on.
3. **Two probes with the same noun are two instruments.** Before declaring either stale, load
   both JSONs and find which fields the numbers are. Rule (6) exists because this is the
   second time in two ticks that the document-vs-document comparison picked the wrong winner.

**ONE REAL DEFECT DID SURVIVE, and it is one clause.** `summary.tex` l.207 opens "On one
checkpoint and one slot, next-token agreement is $0.43$ ... $0.35$ ... and $0.02$--$0.07$
free-running." The first two numbers ARE that one checkpoint. The third is not: on run 25 @900
free-running is 0.0193 over positions 1--11 and its maximum under any aggregation is 0.0471
(position 0, where `own.acc1 == tfslot.acc1` by construction). **The 0.07 is a different
checkpoint** -- run 29 @500 `own` mean 1--11 = 0.0657, @600 = 0.0751. So the sentence's
"one checkpoint" framing is contradicted by its own third number. Honest fixes: quote 0.02
alone under that framing, or keep the 0.02--0.07 range and drop "On one checkpoint".
l.270 carries the same range WITHOUT the one-checkpoint claim and is therefore already fine.
**NOT APPLIED:** `summary.tex` is Emanuele's deliverable of 2026-09-26 09:28, under his review.
Flagged for him.

**Re-confirmed, not new:** the `0.226` in N5 l.208 is still a phantom (POSTMORTEM l.2112,
first flagged 09-18). An independent numeric sweep of every `runs/probes/*.json` for any float
in [0.2250, 0.2270] this tick returned 28 hits, all per-anchor entries or unrelated fields, and
NO summary-level first-token `own` accuracy. Nearest summary value is
`tfslot_r31_300_aligned.summary.own.acc2 = 0.2258` -- run 31, m=2, n=465, i.e. neither this
checkpoint nor "the very first token". Run 29 @500's own@1 is 0.2971. Still Emanuele's to fix.

## RUN 45 (POINTER CHASING) IS THE PROJECT'S FIRST POSITIVE AND IT HAS A MEASURED HALF-LIFE
## (2026-09-27 02:07 headless tick, STOPPED at step ~3090 on its own UNINFORMATIVE clause)

The E1 arm of the 2026-09-26 programme asks a deliberately easier question than runs 15-39 did:
not "can a writer invent useful content on web text" (answered no, 38 runs) but **"where the
computation certainly exists, can this pipeline harvest it?"** Run 45's task is 5-hop pointer
chasing: shuffled `ab>cd` pairs, a query `Q s0:`, answer = the end of the chain; the supervised
scratchpad is the chain in FORWARD order, so writing and reading are both copy-shaped.

**The answer is YES, and it decays to nothing in 2,500 steps.** Held-out, `probe_e1.py`, n=112
problems per row, own greedy thought against a length-matched filler block on the same weights:

      step   EM clean  EM filler  EM tf  EM greedy | own-filler EM |  own-filler gain (nats)
       500     0.616     0.545    0.964   0.946    |    +0.402      |   +0.889 +- 0.090
      2000     0.902     0.893    1.000   1.000    |    +0.107      |   +0.239 +- 0.054
      3000     0.991     0.991    1.000   1.000    |    +0.009      |   +0.064 +- 0.017  (McNemar p=1.0)

The training rows say it from the other side: `or_g`, the gain of the TRUE scratchpad, fell
+0.63 (500-999) -> +0.006 (3000), and `g_spec - g_pau_syn` +0.65 -> +0.08. The writer was
perfect throughout (scratchpad exact 1.000, `imit_ce` 0.000) and health never moved (held-out
mix +0.066 at step 2300 against a +0.25 derail bar, reg% 0.3-0.5%).

**The mechanism is not "thinking stopped working"; it is that the QUESTION stopped existing.**
Five sequential 2-digit lookups fit inside 28 layers, so one pass learns the task, and once the
clean path has it there is no residual uncertainty for any slot content -- not even the correct
scratchpad -- to remove. The registered UNINFORMATIVE clause (EM filler >= 0.85 at step 2000)
fired exactly as written, at 0.893.

**RULE.** A positive on a task the base can absorb in one pass is a SAMPLE-EFFICIENCY transient,
and its half-life is the measurement, not its peak. Never report such a peak without the
trajectory beside it, and pre-register an uninformativeness clause on the FILLER arm (not on the
own arm) before launching: filler EM rising toward ceiling is the signature, and it is visible
~1,500 steps before the own-minus-filler gap closes.

**Run 46 (`think46_hops`, launched 02:23) is the minimal follow-up**: the same task with
hops ~ U{3..10} per problem and 2 distractors, from the clean vanilla rung, so the harvest is
read AGAINST DEPTH on one set of weights (`probe_e1.py`'s new `verdict.by_hops`). Registered
prediction: the advantage decays at shallow hops and persists at deep ones. Registered honest
verdict if it decays at every depth: the harvest is always transient, the chase family is
exhausted, stop. Gates in full in `runs/think46_hops/config.sh`.

### G61 -- A PROBE THAT HAS NEVER BEEN RUN AT THE LIVE RUN'S OWN LAYOUT IS NOT AN INSTRUMENT
*(2026-09-27 04:07 tick. Cost: the step-500 read of run 46 crashed on its first sequence and the
first depth data arrived ~40 min later than it needed to.)*

`probe_e1.py` grew `--L/--stride/--fill-len` at 02:07 so it could follow run 46 off run 45's
geometry, and the patch was regression-tested at DEFAULT flags (byte-identical output on run 45's
anchors). It still could not probe run 46 at all: **`first=64` was hardcoded in two places** --
`tp.ParLayout(T, K, K, L, 64)` and the `eval_set` kwargs -- while run 46 trains with
`--first-anchor 96`. At hops 10 with 2 distractors the prompt is 89 tokens, which does not fit
before position 64, so `synth.make_chase_seq` asserted (`AssertionError: (-24, 0)`) on sequence 1.
Fixed with a `--first-anchor` flag (default 64, so every runs-40..45 probe still reproduces);
backup `pre-training/train/probe_e1.py.bak_pre_firstanchor_20260927`.

Two rules, both about the class of bug rather than this instance:
1. **A default-flag regression test proves only that you did not break the OLD run.** When a
   patch exists to follow a NEW geometry, the test that matters is the new one: assert the
   probe's layout equals the run's (here: N=9 anchors at 96..864 and S=2185, the two numbers
   `config.sh` itself states) before trusting a single number it prints.
2. **Anything a config passes to the trainer, the probe needs too.** `--first-anchor` was in run
   46's FLAGS from the start; the probe simply had no way to receive it. Diff the run's flag list
   against the probe's `argparse` when a run changes geometry, and prefer one shared layout helper
   over two copies of the same five integers.

### G62 -- A GATE THAT FAILS ITS MAGNITUDE CLAUSE AND PASSES ITS SIGNIFICANCE CLAUSE IS NOT THE FAILURE IT WAS WRITTEN FOR
*(2026-09-27 06:07 tick. Run 46's G1 @1000. The tick did NOT stop the run; it amended the gate
and flagged the amendment as post-data. Amendment text: `runs/think46_hops/config.sh`, block
"AMENDMENT 1".)*

G1 was registered as a **plumbing** check with a single compound bar -- DEEP (hops >= 8) anchors,
`EM(tf) - EM(filler) >= 0.20` with McNemar `p < 0.01` -- and a single registered remedy: "FAIL ->
the reader cannot read a correct deep scratchpad; STOP and debug the splice, nothing downstream."
The data (n=432 anchors, 149 DEEP):

| band | n | EM clean | filler | tf | greedy | tf - filler (McNemar p) | gain(gr)-gain(fil) |
|---|---|---|---|---|---|---|---|
| DEEP >=8 | 149 | 0.503 | 0.483 | 0.557 | 0.557 | **+0.074 (p=0.0034)** | +0.103 +- 0.031 |
| MID 5-7 | 166 | 0.542 | 0.536 | 0.566 | 0.566 | +0.030 (p=0.23) | +0.215 +- 0.033 |
| SHALLOW <=4 | 117 | 0.667 | 0.692 | 0.735 | 0.735 | +0.043 (p=0.12) | +0.124 +- 0.026 |
| ALL | 432 | 0.562 | 0.560 | 0.609 | 0.609 | +0.049 (p=0.00019) | +0.152 +- 0.018 |

The magnitude clause fails; the significance clause passes on the same anchors. **The two clauses
were doing different jobs and the compound bar hid that.** Significance answers the plumbing
question the gate was actually written to ask (*does slot content reach the answer at all?*);
magnitude answers a scientific question about how much the harvest is worth. Writing them as one
`and` meant a small-but-real effect was scored as "the splice is broken", whose registered remedy
is to stop the run and go debug code that is demonstrably fine.

Three things made the correct reading available rather than a matter of taste, and all three were
in place BEFORE the gate data existed:

1. **A pre-registered discriminator for the FAIL branch.** The 04:07 tick had already noticed that
   "G1 fails" has two causes with opposite remedies -- the splice is broken, or the splice is fine
   and the reader has not learned -- and ran the zero-update vanilla init (run 46's own init, never
   trained on our slot) at run 46's exact geometry. `EM(fwd) = 0.324` against `EM(clean) = 0.000`:
   the slot reaches the answer position. Splice sound, remedy discharged, *before* anyone could be
   tempted to discharge it by argument.
2. **The training-row trajectory, which named the mechanism as a lag.** Block means of `or_g` (gain
   of the TRUE scratchpad) and `g_spec - g_pau_syn`: `-0.03/-0.12` at 500, `+0.05/+0.08` at 900,
   `+0.32/+0.31` at 1000, `+0.83/+0.84` at 1100, `+0.94/+1.04` at 1200. **The reader took off
   between steps 1000 and 1200 -- i.e. the gate was read on the last checkpoint before the rise.**
   A gate step is a guess about a learning curve, and this one guessed wrong by ~200 steps.
3. **Depth-banded reads.** The pooled number (+0.049) and the json's own `verdict` would have been
   read as a weak global effect; the DEEP band is where the gate was written and where the effect
   is largest in EM.

The rules:
- **Register the two clauses of a compound gate separately, with separate remedies.** "Is the
  channel open?" (significance, plumbing, remedy = debug code) and "is it worth anything?"
  (magnitude, science, remedy = stop the line of work) are not the same question and must not share
  an `and`.
- **Every FAIL branch that says "go debug X" should carry a pre-registered instrument that decides
  whether X is actually broken**, run before the gate data lands. Without it, the tick that reads a
  failure is choosing between "obey the letter" and "argue from the numbers in front of it", which
  is exactly the position pre-registration exists to prevent.
- **A gate step is itself a guess.** Where the quantity is a learning curve rather than a fixed
  property, register the gate as a *trajectory* clause (e.g. "rising and above X by step N, else
  stop") rather than a single-step threshold, or accept in advance that one extension may be needed
  and register its bar and its deadline at the same time as the original.

**What this tick did, and its cost to the record:** run 46 continued, with a replacement clause
G1' @2000 -- the same `>= 0.20` bar on DEEP with `p < 0.01`, one extension only, no G1'' -- and its
probe armed to run itself (`agent/e1_gate_r46_2000.sh`). That amendment was written after seeing
the data. It is contaminated in the pre-registration sense and is reported as an amendment
everywhere it appears. The clean statement for the paper is: *G1 as written failed at step 1000;
its significance clause passed; the pre-registered splice discriminator cleared the splice; the run
was extended once to step 2000 under the same bar by post-data amendment.*

**A candidate killed by the same data.** The 04:07 tick had registered a run 47 ("right-align the
supervised scratchpad so the answer stops moving with problem depth") licensed by {G1 fails} AND
{layout sound} -- a combination that has now occurred. It is dead anyway, because `fwd` is already
that experiment: `probe_e1.py` l.133-135 writes the bare answer at a FIXED slot offset, and on run
46's own weights `fwd` is not better than the variable-length scratchpad (step 1000: EM 0.616 vs
0.609 pooled; step 500 DEEP: `fwd - clean` = +0.014 +- 0.038). The moving-target hypothesis
predicted `fwd >> tf` and it is falsified. **Before designing a run around a hypothesis, check
whether one of the probe's existing control modes already tests it.**

### G63 -- ONE KNOB WAS TRAINING THE READER AND THE WRITER AT ONCE, SO "LESS SUPERVISION FOR THE WRITER" SILENTLY MEANT "A READER THAT CANNOT READ"
*(2026-09-27 22:07 headless tick. Runs 47, 49 and 50 -- the whole discovery family to date -- are
one failure, not three. Run 50 stopped at ~1190, 810 steps before its registered read; run 52
`think52_readeronly` launched in its place. Registration: `runs/think52_readeronly/config.sh`,
written before run 50's stop data existed.)*

`--synth-ssl` sets the fraction of synthetic anchors that carry a supervised (true) scratchpad. It
does **two** unrelated things, and nothing in the flag, the config comments or three run designs
said so:

- the true scratchpad is spliced into the block INPUT (`think_train.py` l.688-692), and with
  `--reader-on-ssl 1` those anchors take window gradient (l.894, `rdf = thf`). **That is what
  teaches the READER to cash a chain.**
- the same anchors carry block cross-entropy weighted by `imit_w` (l.913-916). **That is what
  teaches the WRITER to write one**, i.e. it is SFT.

Emanuele's goal is "learn WHAT to write with little supervision", so runs 47, 49 and 50 each turned
this one knob down (a 1-value seed; a 1-value seed with the stop unsupervised; full chains on 5% of
anchors). Every one of them cut the reader's training set by the same factor, and a blind reader
makes the reward `p(answer|thought) - p(answer|clean)` content-free. The writer then does the
rational thing and collapses onto whatever the reader still pays for: `'<v1> </T>'` forever (47),
`'<v1>'` + newline padding to the cap (49), a single random 2-digit number (50).

**The instrument that shows this is the teacher-forced column of `probe_e1.py`, and it does not
depend on the writer at all**: splice the CORRECT chain in yourself and ask whether the answer gets
easier than after length-matched filler. On the same 149 DEEP (hops >= 8) held-out puzzles at the
same step:

| run | `--synth-ssl` | EM(clean) | EM(filler) | EM(tf) | `EM(tf) - EM(filler)` (McNemar p) |
|---|---|---|---|---|---|
| 46 | 0.375 | 0.503 | 0.483 | 0.557 | **+0.074 (p=0.0034)**, and +0.302 by step 2000 |
| 50 | 0.07  | 0.054 | 0.128 | 0.087 | **-0.040 (p=0.11)** -- the wrong sign |

The rules:

- **Name, for every flag you turn down to make a run "less supervised", every gradient it feeds.**
  A knob that feeds two learners cannot be used to weaken one of them. Run 52 separates them the
  only way the code allows: `--synth-ssl 0.40` (run 46's own density, for the reader) with
  `--imit-w 0 --imit-floor 0` (the writer gets no cross-entropy, ever). Verified before launch that
  `imit_ce` and `or_g` are computed from `row_ssl` and not from `imit_w` (l.1011-1015), so both
  instruments survive `--imit-w 0`, and confirmed on the live command line after launch.
- **A run's PRECONDITION is not its hypothesis, and it needs its own early-stop clause.** Run 50's
  hypothesis was "how much does the reward add on top of a few demonstrations?"; its precondition
  was "the reader can cash a chain". The precondition failed, with the wrong sign, at the
  registered step-1000 *trajectory* read -- which carried no action, because the registration put
  all the actions at 2000. A run whose precondition is measurably false cannot ask its question, and
  the 2.5 h to its registered read would have bought a confirmed flat line. **Register the
  precondition as a gate with a stop, at the earliest step where it is readable.** Run 52 does
  (R1a@1000 trajectory, R1@2000 stop, at run 46's own bar).
- **Calibrate a gate's STEP on a run that worked, not on optimism.** Run 52's reader bar is
  `>= 0.20` at step **2000**, not 1000, precisely because run 46 -- the run that worked -- read
  +0.074 at 1000 and only +0.302 at 2000. A `>= 0.20 @1000` bar would have killed run 46. If your
  registration would have failed your own best run, it is the registration that is wrong.
- **Runbook 3c's futility clause (`g_r0`/`gSp` flat) does not fire on these runs and must not be
  relied on.** Run 50's `g_r0` went `-0.11 -> +1.03` while `ob0` sat at `+0.00` and the thought was
  2.4 tokens against an 18-token filler: block price and reader recalibration, exactly the failure
  G56 recorded for code runs. On an E1 run the futility instruments are `scr_ex_g`, `scr_pv_g` (is
  it falling?), the rollout spread `g_sd` (was 0.115 -- the LOO advantage is ~0, so REINFORCE has
  almost nothing to act on) and the teacher-forced column above.
- **A control that cannot distinguish anything is not a control.** Run 51 (`--rl-w 0`, run 50's
  registered control) was dropped: with run 50's writer at chance, `run50 - run51` is `0 - 0`, and
  the registered fallback (steps-to-criterion at DEEP `EM(own) >= 0.90`) is undefined because
  neither run reaches it. 13 h of device for a number with no content. Its question is subsumed by
  run 53 = run 52 with `--rl-w 0`, licensed only if run 52's D2 passes.

### G64 -- A LENGTH METRIC THAT COUNTS WHITESPACE CANNOT BE AN "IS THE WRITER STILL OPEN" GATE
*(2026-09-28 00:07 tick, run 52 at step ~560. Written while the run is ALIVE and 440 steps before
the gate it concerns, so this gotcha is not a post-hoc excuse for a stop.)*

Run 52's registered EXPLORATION@1000 stop reads *"mean `g_sd` >= 0.30 OR mean own thought length
>= 6; BOTH FAIL -> the writer collapsed, stop"*. `len_mean` is `len(thought_tokens)` and **counts
whitespace**. What run 52's writer actually emits, in every single `src=rl` row of
`runs/think52_readeronly/dumps.jsonl` from step 300 onward, is

      '12' + 30 x '\n' + ' </T>'          (step 525; 500 is '10', 550 is '19', 450/475 are '33')

so `len_mean == 32.0` -- the `--L 32` cap, the maximum the metric can report -- while the writer is
saying ONE number. Non-whitespace token count by block: **5.00 (0-100), 2.75, 1.50, 1.00 (300-400),
1.00 (400-550)**, with `g_sd` 0.207 -> 0.110 monotonically and `scr_ex_g` 0.000 throughout. The
clause written to *detect* the collapse is *satisfied by* the collapse. And this is the same
signature run 49 was stopped on ("'<v1>' + newline padding to the cap"), so the trap had already
fired once and was re-registered anyway.

**Rules.**
1. **Any gate on "the writer is still producing something" must be in CONTENT units** -- tokens that
   are not whitespace, or better a task-specific count (`scr_pv_g`: how many correct chain values
   the thought carries). A cap-valued length metric is indistinguishable between "maximally
   verbose" and "degenerate padding", which are opposite verdicts.
2. **Read the dumped strings before you trust any writer-side scalar.** `dumps.jsonl` had the answer
   in plain text at step 300; the scalars said `len_mean` 32.0 and looked healthy for 250 steps.
   This is G39 ("check what the writer actually emits before you use this bar") arriving a second
   time from the opposite direction: there the CONTROL was secretly filler, here the TREATMENT is.
3. **Padding is an attractor with a mechanical cause, and there is a zero-code lever against it.**
   With `--imit-w 0` nothing supervises `</T>` placement, and the base model's continuation inside
   an empty slot is newlines; `--repeat-ban 1` makes a repeated newline illegal. But
   `think_par.py` l.209/l.249 special-case the speculation row (`if r == spec_row: argmax`), so
   repeat-ban / freq-pen / top-k / temperature reach the SAMPLED rows only -- the greedy thought
   every EM headline is computed on can still pad. A run that relies on the ban must register
   SAMPLED thoughts as its primary readout.
4. **When you find a defective clause mid-run, amend it before the data exists and flag exactly how
   much of the window you have already seen.** Run 52's AMENDMENT 1 was written with 55 of its 500
   gate steps observed (11%) and says so in the config, in `agent/E_STATE.md` and in the journal; it
   can only make the gate harder. That is the honest version of what G63 caught being done badly.

*(Filing note: `POSTMORTEM.md` and its 36 backups were MOVED from the repo root into
`postmortem_analysis/` at 22:33 on 2026-09-27, by hand and not by this supervisor. Nothing was lost
-- G63 is present at line 2861 -- but the runbook and several tick scripts still say
`POSTMORTEM.md` at the root. Future ticks: the live ledger is `postmortem_analysis/POSTMORTEM.md`.
Do not move it back without being asked.)*

### G65 -- THE BOTTLENECK MOVED: A COMPETENT READER PLUS A CORRECT REWARD IS STILL NOT A QUESTION THE WRITER IS BEING ASKED
*(2026-09-28 02:23 headless tick, run 52 stopped at ~1160 on its registered AMENDMENT 1 case (i);
run 54 `think54_explore` launched. This is the entry that supersedes G63's diagnosis, so read the two
together: G63 identified reader supervision as the broken half, and run 52 FIXED that half and
measured the fix. Believe this one about where the problem now is.)*

**What run 52 was and what it settled.** It separated the two jobs `--synth-ssl` had been doing at
once (G63): demonstrations at run 46's density for the READER (`--synth-ssl 0.40 --reader-on-ssl 1`)
and ZERO cross-entropy on the WRITER (`--imit-w 0 --imit-floor 0`). The reader column of the
held-out probe -- teacher-forced, i.e. we splice the CORRECT chain in ourselves, so it does not
depend on the writer at all -- on the same 149 deep held-out problems at the same step 1000:

                                                              EM(tf) - EM(filler), DEEP, n=149
    run 46  --synth-ssl 0.375, writer SFT on the full chain   +0.074 (p=0.0034)
    run 50  --synth-ssl 0.07 , writer SFT on 5% of chains     -0.040 (p=0.11, wrong sign)
    run 52  --synth-ssl 0.40 , writer SFT NONE (imit-w 0)     +0.450 (p=6.8e-17)

Run 52's +0.450 is six times run 46's margin at the same step and already clears run 52's own
step-2000 bar at half the steps. **So "you cannot buy the reader without also supervising the
writer" is FALSE. Reader supervision is off the suspect list, and it should not be re-litigated.**

**And the writer still did nothing, with every excuse removed.** It had a reader that demonstrably
cashes a correct chain and a reward that would have paid for one, and from step 300 to 1160 every
sampled thought was one random 2-digit number followed by newline padding to the `--L 32` cap:

    mean g_sd, steps 500-1000                 0.0487   (gate bar 0.30), by block 0.207/0.108/0.073/0.037/0.044
    mean CONTENT length of the src=rl thought  1.00    (gate bar 6), against len_mean 31.94 -- G64 again
    probe @1000, greedy vs sampled thought    BYTE-IDENTICAL ('16 23' + 30 newlines)
    DEEP scr_ex_g 0.000, DEEP pv 0.060, EM(greedy)-EM(filler) +0.007 (p=1), window gain -0.160 +- 0.051

**THE RULE.** With `--baseline loo` the REINFORCE advantage is O(`g_sd`). At `g_sd` 0.05 the rollouts
are the *same string*, so the loss is identical whatever they contain and the gradient carries no
information about content. **A writer in that state is not failing to learn; it is not being asked
anything.** Before concluding "the reward cannot teach X", check that the rollouts DIFFER -- `g_sd`
and, better, whether the dumped sampled strings are literally identical. Four runs (47, 49, 50, 52)
were read as writer-capability failures and at least run 52 is an exploration failure.

**THE OTHER HALF OF THE RULE, registered in advance so run 54 is not over-read.** Widening the
sampler makes rollouts DIFFER; it does not make them CORRECT. A 5-value chain drawn from a ~100-value
vocabulary is not reachable by chance, so if run 54's exploration gate passes and its discovery gate
fails, the honest conclusion is that this loop needs a reward DENSE IN PARTIAL CHAIN CORRECTNESS --
a design change, not a knob -- and that is a question for Emanuele, not a fifth variant of the loop.

**TWO INSTRUMENT LESSONS FROM ARMING RUN 54, both of which would have silently corrupted it.**
1. **A probe whose SAMPLER does not match the run's sampler is not an instrument** -- the layout
   version of this is G61, and this is the policy version. `probe_e1.py` HARDCODED
   `top_k_thought=512, temp=1.0, freq_pen=0.0, min_think=2, repeat_ban=0` (old l.66). Run 54 trains
   with `--temp 1.15 --freq-pen 0.3 --repeat-ban 1 --min-think 6` and its PRIMARY readout is the
   `sample` arm, so unpatched the probe would have drawn that arm from a different policy than the
   run trains. Five default-off flags added, defaults EXACTLY the old hardcoded values (backup
   `probe_e1.py.bak_pre_sampler_20260928`; the diff is those five `add_argument` lines plus the
   `gargs` line), so every probe run before 2026-09-28 reproduces bit-for-bit. **Whenever a run
   changes a sampler flag, the probe command must carry the same flag, and the config must say so.**
2. **NEVER EDIT A SHELL SCRIPT THAT LIVE PROCESSES ARE EXECUTING -- COPY IT.** bash re-reads a
   script from the file at a byte offset between commands, so inserting lines into a file whose
   watchers are parked in a `for i in $(seq 1 1200); do ... sleep 60; done` loop for hours can
   corrupt their control flow. An `EXTRA` pass-through was added to `agent/e1_autoprobe_run.sh`
   while four run-52 watchers were inside it, then REVERTED and shipped as a new file
   `agent/e1_autoprobe_run2.sh` instead. This is the same family as G20 (never combine a
   `pgrep`/`pkill` pattern with a launch of the matching program): the long-running shell plumbing
   of this project is mutable state, and editing it in place is an action on running processes.
   A note to that effect is at the top of `e1_autoprobe_run2.sh`.

**One G20 sighting, for the record, because it nearly bit again.** Listing PIDs for the stop with
`pgrep -f "torchrun --nproc_per_node"` returned TWO pids: the real torchrun and *my own shell*,
because the pattern appeared in the command line of the shell running the pgrep. The PIDs were
verified with `ps -o pid,ppid,etime,cmd` before any kill. Always identify before you kill.

### G66 -- AN EXPLORATION KNOB IS NOT READER-NEUTRAL: THE ROLLOUT YOU WIDEN IS ALSO THE TEXT THE READER IS TRAINED ON
*(2026-09-28 04:07 headless tick, run 54 `think54_explore` at step ~306. This is an EARLY,
PRE-GATE observation on a live run, not a verdict: run 54's gates are at step 1000 and 3000 and
nothing here licenses a stop. It is written down now because the mechanism was confirmed by reading
the trainer, not inferred from a curve, and the next tick must not mistake it for a reader failure.)*

**The good half first, because it is real and it is what the run was for.** Run 54 = run 52 with six
sampler flags changed (`--repeat-ban 1 --freq-pen 0.3 --temp 1.15 --rollouts 8 --min-think 6
--len-cost 0.0`). Run 52's writer emitted ONE 2-digit number plus newline padding for 900 steps.
Run 54's writer, measured the same way on the dumped `src=rl` thoughts (new instrument
`agent/content_len.py`, calibrated against run 52 before use):

                                        mean CONTENT tokens        distinct strings
    run 52, steps 500-1000 (n=21)       2.00 tok / 1.00 chunk      7 of 21
    run 54, steps 100-300  (n=9)       27.44 tok / 9.78 chunk      9 of 9

and some of it is in the task's own shape (`'71>29, 29>14, 70>89, 61>71, '`). **The newline-padding
attractor is broken.** So the content clause of the exploration gate is not in doubt.

**The cost, which nobody costed before launching.** `think_train.py` l.810: on RL anchors
`--rl-content random` splices a RANDOM rollout's tokens into `content`, and `content` is what the
GRAD pass sees -- the same rows the reader is trained on. So the sampler does not only decide what the
writer proposes, **it decides what the reader is shown at every RL anchor.** Run 52 was showing the
reader a near-empty slot (2 content tokens); run 54 shows it 32 tokens of fluent, plausible,
task-shaped, and WRONG text. `or_g` -- the discounted window gain at SSL anchors, i.e. the one column
that asks "does the reader cash a CORRECT chain" (l.1011) -- has separated from run 52 at matched
steps, in the wrong direction and well outside noise:

                        or_g, steps 100-310          or_g, steps 200-310
    run 52             -0.035 +- 0.029               +0.033 +- 0.035   (and +0.28/+0.49 by step 350-400)
    run 54             -0.330 +- 0.041               -0.215 +- 0.050

**THE RULE.** In a loop where one policy's samples are spliced into another head's training data,
there is no such thing as a knob that only touches exploration. Widening the sampler is also a
DISTRIBUTION SHIFT ON THE READER, and a reader shown confident wrong text at 30 tokens a slot has
every reason to learn to discount the slot -- which is the same phenomenon as the central ladder's
"a confident wrong token is worse than noise", applied to training rather than to inference. Before
changing a sampler flag, ask which OTHER consumers of the sampled tokens exist (`grep` for the
variable the rollout is spliced into), and pre-register the axis you expect to pay on.

**How the next tick must read it, so the two failures are not confused.** Run 54's X1b READER CHECK
@1000 (`or_g >= +0.5` over steps 900-1000 AND DEEP `EM(tf)-EM(filler)` > 0 at p<0.01) already exists
for exactly this, and this entry is the mechanism behind a failure of it. If X1 passes and X1b fails,
the run is uninterpretable AS A DISCOVERY RUN and stops -- but the finding is NOT "run 52's reader
result was wrong"; run 52's +0.450 stands, it was measured on run 52's weights. The finding would be
that **exploration and reader quality are coupled through the spliced content, and the coupling is a
design problem, not a knob.** Decoupling them needs code that this repo does not have (score the wide
rollouts for the reward but splice something else for the reader gradient); `--rl-content` offers only
`winner` and `random`, and both splice a sampled rollout. That question goes to Emanuele together with
the reward-density question G65 registered -- one design decision, two symptoms, not two more variants.

**Do not act on this before the gate.** `or_g` at step 300 is 6 dumps and 200 noisy steps; run 52's own
`or_g` was -0.03 at step 300 and +0.49 by step 400, so a late climb is exactly what this axis did last
time. The number to decide on is the steps-900-1000 mean, as registered.

### G67 -- BREAKING A COLLAPSE ATTRACTOR MOVED IT, IT DID NOT REMOVE IT; AND A CONTENT-TOKEN BAR IS GAMEABLE BY FLUENT OFF-TASK PROSE
_2026-09-28 06:07, run 54 at step ~690, from `dumps.jsonl` + the @500 probe json. Measured, not inferred._

Run 52 collapsed to ONE 2-digit number plus 30 newlines. Run 54 banned the repeated token, penalised
frequency, raised temperature and doubled the rollouts; **the newline padding is gone and the slot now
holds ~30 content tokens of fluent web-math prose with ZERO task-shaped content.** The matched
measurement is a regex over the sampled `src=rl` thoughts (count of `\d+\s*>\s*\d+`, the chase task's
own pair shape):
      run 52 : steps 0-300  1.08 pairs/thought, 50% of dumps >= 1   | steps 300+     **0.00**, 0%
      run 54 : steps 0-300  1.42 pairs/thought, 50% of dumps >= 1   | steps 300-675  **0.00**, 0%
`scr_pv_g` agrees: run 54 reads 0.000 over steps 400-680 where run 52 read 0.033-0.045.
**BOTH runs write the task's shape early and BOTH abandon it after step ~300.** The six flags changed
WHERE the policy goes when the reward stops discriminating, not WHETHER it leaves the task. Widening a
sampler cannot create content; it can only change which zero-content attractor is cheapest.

**THE GATE CONSEQUENCE, and it is G64 recurring in a new disguise.** G64's fix was "count CONTENT
tokens, not whitespace, bar 6". On run 54 that measure reads **30.19** and would PASS while the thought
contains nothing about the problem. **No length measure of any kind can be an "is the writer exploring
the task" gate.** The honest measure is task SHAPE, and it costs nothing: `dumps.jsonl` rows carry
`src`, `step`, `context`, `thought` and `gain`, so pairs-per-thought is a regex, and the held-out version
(`scr_pv_g` / `EM(sample)` on the SAMPLED arm) is already in the probe json. Report shape BESIDE any
registered length clause; do not silently replace the registered clause with it mid-window.

**AND CHECK THE ANCHOR TYPE BEFORE CALLING PROSE A COLLAPSE.** On `--synth-every 2` half the anchors
are web text, where prose in the slot is correct behaviour. Every `src=rl` dump in runs 52 and 54 turns
out to be at a SYNTH anchor -- but that had to be verified from the `context` field, not assumed.

**REFINEMENT TO G66: the decoupling is ONE LINE, not missing code.** G66 said "decoupling needs code
this repo does not have". It does not: `think_train.py` l.894 is
`rdf = thf if args.reader_on_ssl else rlf`, where `thf` = all think anchors and `rlf` = RL anchors only,
and the mode G66 asks for -- reader window gradient on DEMONSTRATION anchors only, `rdf = sslf` -- is a
third value of that existing flag. The reward path does not touch `rdf` (`_score` is a separate forward
over the spliced `content`), so a wide sampler can be scored for the WRITER while the READER never takes
gradient through a garbage slot. Cost: one line plus an arg-type change.

**NEW MATCHED EVIDENCE FOR G66's mechanism, at step 500, same 432 problems, same probe:**
      run 52 @500  EM(tf)-EM(filler):  ALL **+0.146 (p=3.3e-08)**  MID +0.265  SHALLOW +0.188  DEEP -0.020
      run 54 @500  EM(tf)-EM(filler):  ALL **-0.019 (p=0.18)**     MID -0.012  SHALLOW -0.026  DEEP -0.020
Run 52's reader was already visibly cashing the chain at step 500 on the shallow and mid bands (DEEP came
later, +0.450 at 1000); run 54's shows nothing in any band. And run 54's CLEAN arm is down as well (ALL
EM 0.255 vs run 52's 0.414 at the same step from the same init), so **what the spliced garbage damages is
not confined to the slot -- it is the model's competence on the task.** `or_g` says the same:
run 52 +0.385 at step 500 and +1.2 by 700-1000; run 54 +0.006 +- 0.023 over steps 500-680.

### G68 -- YOUR OWN MONITORING SHELL IS A PROCESS, AND `run_think.sh` COUNTS IT AS THE DEVICE OWNER
*(2026-09-28 08:11, headless tick, stopping run 54 and launching run 55. G20's rule in a new disguise,
and it cost a launch.)*

To verify the run-54 stop per G22 I started a detached wait-loop:

    for i in $(seq 1 40); do n=$(pgrep -fc 'think_train.py --corpus') ; ... ; sleep 3; done &

Every process on this box carries its whole command line in `/proc/<pid>/cmdline`, so **that loop's own
shell matches the string it is searching for.** Two consequences, both observed within three minutes:

1. **The loop can never see zero.** It counted itself and ran to its 120 s timeout while the workers had
   in fact been dead since the first 45 s. Two separate readings in that window printed `workers=3` and
   `torchrun=3` with nothing actually alive -- the count was my own pipeline stages.
2. **It blocked the next launch.** `run_think.sh` l.68 enforces one device owner at a time with
   `pgrep -f "_train\.py --corpus"`, and the comment above it says the pattern requires `--corpus`
   precisely so a shell that merely mentions a trainer filename never matches (G16). My wait-loop
   contained `think_train.py --corpus` *with* the flag, so it matched, and run 55 aborted at 08:11:18
   with `ABORTED (another Neuron job is running -- stop it first)` while the device was idle.

**THE RULE, and it generalises past `pgrep`: never put the literal process pattern in the command line of
anything that outlives the call.** Verify a stop by the artefact, not by a self-matching search:

    # right: the file the trainer writes is the only honest witness (G22)
    wc -l runs/<run>/metrics.jsonl ; stat -c %y runs/<run>/metrics.jsonl     # two calls, minutes apart
    # right: one-shot listing, read it yourself
    ps -eo pid,ppid,stat,etime,cmd | grep -E "think_train\.py|torchrun" | grep -v grep
    # wrong: a loop, a `watch`, or a background job whose argv contains the pattern

`ps | grep` in a *foreground* call is safe in practice only because the call is gone by the time the next
one runs -- and even then the grep child itself shows up, which is why `grep -v grep` is in every example
above. A search whose own footprint can change the answer is not a measurement.

**AND IT IS WORTH SAYING WHAT SAVED IT:** the failure was loud (`STATUS` said `ABORTED` in one line) and
the device sat idle for 44 seconds, not for two hours. Write launches so they fail with a STATUS line;
then read `STATUS` after every launch, always, even when the command returned 0.

### G69 -- `reg%` IS AN *OPENER* MEASURE, NOT A COLLAPSE MEASURE. READ THE DISTINCT-THOUGHT FRACTION BESIDE IT.
(2026-09-28 10:07, run 55 at step ~360.)

`status.py` l.49 computes `reg%` as the share of the top-3 **8-character prefixes** among the `src=rl`
thoughts in `dumps_eligible.jsonl`. Runbook 3b then says *"register share (reg%) > 15% in the last block"*
is a derail. On run 55 that fired at step 300-360 -- `reg% 18.8`, top prefixes `ogibaii\n` x9,
`ogothick` x4, `ozhmandi` x2 -- and it would have been a wrong stop:

    steps 300-400, run 55, src=rl eligible dumps:  n = 83,  DISTINCT FULL THOUGHTS = 83 of 83 (1.00)

Every thought is different; what is shared is the first token or two (an `o...` opener). **An opener
collapse with distinct bodies is not the run-22 register collapse**, which was context-free *whole
phrases* standing in for the thought. The two states are indistinguishable on `reg%` and trivially
distinguishable on the distinct-thought fraction, which is one line of python over the same file.

Calibration from the family, all at steps 0-400, so no tick reads 15% as a bright line again:
`reg%` = 27.1 / 38.7 / 12.7 / 16.1 by 100-step block in **run 52** -- the run whose reader half succeeded
and whose health was called green throughout -- against 3.8 / 2.4 / 1.8 / 6.7 in run 54.

RULE: quote `reg%` **with** the distinct fraction and the prefix line, always. Stop on `reg%` only when
the distinct fraction is also low; a high `reg%` at distinct ~1.0 is a shared opener and belongs in the
journal, not in a kill decision. Same family of error as G64/G67: a summary statistic passing (or firing)
on a state it cannot see.

### G69.1 -- AND RUNBOOK 3b's `rollout spread < 0.6` IS A DEAD AXIS IN THE E1-DISCOVERY FAMILY (runs 45-55).
Declared explicitly here because it fires literally on every run of this family from step ~100 and has
never once been acted on as an independent kill. `g_sd` is 0.05-0.6 throughout: run 52 sat at 0.13-0.20
over steps 0-400 and bought the best reader the project has had; the family's own pre-registered
exploration gates are written at a **0.30** bar (X1), i.e. below 3b's 0.6 by design. Taking 3b literally
here would stop every run in the family before its first gate. It is legitimate to quote as
**over-determination when a registered gate has already failed** (that is how run 52's stop used it) and
it is never on its own a reason to stop. The web-text family (runs 25-39) it was written for is history.

### G70 -- THE OPENER STATISTIC I REGISTERED AT 10:07 WAS BLIND BY CONSTRUCTION, AND THE NEXT WINDOW PROVED IT: MEASURE THE OPENER AT THE TOKEN, NOT AT 8 CHARACTERS
_added 2026-09-28 12:07 (headless tick, run 55 step ~730). Sharpens G69, which was written 2 h earlier and
was itself already too generous to the statistic it was defending._

At 10:07 I declined to stop run 55 on runbook 3b's `reg% > 15%` and registered a replacement clause over
steps 400-700: **stop if the top-3 8-char prefix share >= 15% AND the distinct-full-thought fraction < 0.80.**
That clause returned, on its own window, with its own one-line command:

      steps 400-700, src=rl eligible dumps, n = 340
        top-3 8-CHAR prefix share   2.9%   [('tichesei',4) ('tichawut',3) ('tichunar',3)]
        distinct full thoughts      1.00   (340 of 340)
      -> NO STOP on either arm, by a wide margin on both.

And here is what the same 340 thoughts actually are:

      top-3 4-CHAR prefix share   73.5%   [('tich', 246) ('sege', 2) ('troy', 2)]
      share starting with 'tich'  72.4%
      top-3 6-char prefix share   14.1%   [('tichun',21) ('tichin',15) ('tichan',12)]

`tich` is TWO tokens (83 `'t'` + 713 `'ich'`). Nearly three quarters of the policy's sampled thoughts open
with the same two tokens and then draw a **random** suffix, so:
  - the 8-char statistic reads the suffix and sees noise -> 2.9%, a 25x understatement;
  - the distinct-full-thought fraction, G69's own proposed correction, reads 1.00 -- also blind, because
    random suffixes make every string distinct no matter how frozen the opener is;
  - the 4-char statistic reads 73.5%, and the 6-char one reads 14.1%, i.e. the measured "collapse" falls
    off a cliff between 4 and 6 characters. **The number you get is a pure function of where you truncate.**
The same window's greedy arm makes it unmistakable (`e1_0000500_n48.json`, `examples[].greedy`, decoded):
`ticheseich"tich" (tich"tich) (tich"tich) (tich"tich) (tich` -- a 32-token `tich` loop that fills the block.

**THE RULE.** A prefix-share statistic at a FIXED CHARACTER WIDTH cannot measure opener concentration,
because the thing being measured is a number of TOKENS and the character width that corresponds to it
changes with the tokens the policy picks. Do not register another one. Report, at every tick, the
**first-2-token share** and the **first-4-token share** of `src=rl` thoughts, computed on ids, beside a
distinct-full-thought fraction; the pair (high opener share, high distinct fraction) is exactly the state
run 55 is in and it has no name in runbook 3b at all.

**AND THE HONEST HANDLING OF A MIS-SPECIFIED CLAUSE (this is the part that matters more than the metric).**
I did not convert this into a stop. G63 forbids amending a gate on the data it is being judged against, and
that prohibition does not weaken when the amendment would make the gate STRICTER or when I am the one who
mis-wrote it: a clause that can be re-derived after the fact into whichever verdict the reader prefers is
not a pre-registration. The clause returned NO STOP, so run 55 continued to its real gate. What I did
instead: reported both the registered numbers and the correct ones in the same breath, wrote this gotcha,
and registered the replacement over a window whose data does not yet exist (steps 1000-1300, reachable only
if R @1000 passes). **The cost of mis-specifying a derail clause is paid in the next window, not in this
one.** Related: G69 (the original, too-generous correction), G69.1, G63, G67, G64.

**WHY IT WAS RIGHT TO CONTINUE ANYWAY, SEPARATELY FROM THE PRE-REGISTRATION ARGUMENT.** Run 55 exists to
test G66, i.e. to measure the READER. Its R gate's `tf` arm splices the TRUE continuation into the slot and
never reads the writer's output at all, so a frozen opener cannot contaminate it -- and run 52, the healthiest
reader this project has produced, ran with a writer whose greedy thought was 2 content tokens plus 29
newlines. A writer collapse is this run's *predicted* outcome (its registered prediction is "R passes,
D fails"), not a reason to destroy the reader measurement before it lands.

## G71 (2026-09-28 14:07) -- `dumps_eligible.jsonl` IS SELECTED ON GAIN, SO NO SHAPE STATISTIC COMPUTED FROM IT DESCRIBES THE POLICY
`think_train.py:861` defines `elig = (rlf * (gain > args.distill_eps))` and `l.1042-1059` writes a row to
`<run>/dumps_eligible.jsonl` **only for anchors where `elig > 0`**, capped at 8 per step. The file is therefore
the set of rollouts that BEAT a gain threshold -- the same winner's-curse selection the project already knows
not to headline (`g_win`, G19). Two consequences, both of which bit today:
  1. **Any opener / register / distinctness statistic read off that file is a statistic of the best rollouts**,
     not of the writer. `status.py`'s `reg%` and the `top RL thought prefixes` line are both read from it.
  2. **Its n collapses exactly when the writer collapses.** Run 55's `n_elig` fell 1.24 -> 0.16 per step between
     steps 800 and 1000 as every rollout's gain went below `distill_eps`, so the window 1000-1300 held **24 rows**
     where 400-700 held 340. A shape clause over that window is computed on the surviving tail of a dying policy.
**This is the SECOND independent way the 12:07 futility clause (i) is blind** (G70 was the first: a fixed
character-width prefix cut). It stands as registered and is read at its own window; it is not amended (G63).
**RULE: measure writer shape from the PROBE's `examples` block** -- `probe_e1.py` dumps greedy and sampled ids for
its example anchors unconditionally, with no gain filter -- **and state n.** On run 55 @1000 that reads: first
token `83` in **8/8** on both arms, greedy 2-token opener `[83, 12590]` **7/8**, and **0.0% of the 256 emitted
tokens on each arm lie in the task's 12-id answer vocabulary** `{15..24, 220, 151668}`. The writer emits no digits.

## G71.1 -- AND `n_elig` COLLAPSING DOES *NOT* TURN THE RL LOSS OFF; DO NOT READ IT AS ONE
Recorded because this tick formed that hypothesis and the code refutes it. `elig` appears at `l.861` (definition),
`l.932` (`st["n_elig"] +=`) and `l.1042` (the dump filter) and **nowhere else**: it is a monitor and a dump filter,
never a loss term. The RL gradient rides `A` (`l.912-917`), the per-batch z-score of `adv_paired` clipped to
`reward_clip` with the negative side scaled by `--neg-scale`. Because `A` is standardised **per batch**, it is
invariant to the level of the gains: a batch of 8 uniformly catastrophic rollouts produces the same gradient scale
as a batch of 8 good ones, and its direction is whatever noise separates them. So "el 0/4" in the step line means
"nothing cleared `distill_eps`", not "training stopped" -- and a run whose rollouts are all terrible is still being
pushed, up the noise gradient among them. Check `g_sd` (run 55: 0.069 -> 0.740 over steps 700-1100) for whether
there is a gradient at all; `n_elig` answers a different question.

## G72 -- `--min-think` IS A FLOOR ON TOKENS, AND THE TASK'S ANSWER IS SEVERAL TOKENS: SET THE FLOOR TO A *COMPLETE* UNIT OR STOP EXPLORATION WILL TRUNCATE
Run 57 (2026-09-28) added `--stop-explore 0.1` so the reward could teach the writer WHERE to stop, and it worked:
greedy thought length fell **22.1 @500 -> 2.0 @850** and then sat still. It sat on the FLOOR. `--min-think N`
makes `</T>` legal at slot position `t >= N` (`think_par.py:285`, and `:225` on the top-k path), i.e. N is a count
of **tokens**, not of task units. In `chase`, a value is a 2-digit number emitted as TWO digit tokens
(`synth.py:_two`, l.102-129), and this writer has independently learned to OPEN WITH A SPACE (dumps at step 800-850:
`' 4 </T>'`, `' 9 </T>'`; the space-tolerant `pvls_*` counter added at 17:07 exists only because of that habit).
So under `--min-think 2` the shortest legal thought is **SPACE + ONE DIGIT** -- half an answer -- and a reward that
prefers short thoughts drives the policy straight onto it. Measured consequence: 1-hop `pvls_greedy` **0.938**
(the writer knows the right value) while hops 1-2 `EM(greedy)` **0.365** against `EM(clean)` 0.469 -- it knows the
answer and truncates it. **RULE: when you turn on a length-shortening pressure, compute the token length of ONE
complete task unit -- including any separator the writer has learned to prepend -- and set `--min-think` to that
(here 3), so the shortest legal thought is a whole answer rather than a fragment.** The companion error is the
opposite one and it is already in the ledger: the price of a thought is steeply superlinear in its length (G39).

## G73 -- AN EASY-FIRST CURRICULUM CAN INSTALL A SHORTCUT THE HARD RUNGS THEN HAVE TO UNLEARN: KEEP THE HARD RUNG IN THE MIXTURE FROM STEP 0
Runs 56 and 57 used `--synth-hops 1 --synth-hops-max 10 --synth-hops-ramp 250`, so hops are drawn
`U{1 .. min(10, 1 + step//250)}` (`think_train.py:589-593`): **only 1-hop problems for the first 250 steps.** At
one hop the first value of the chain IS the answer, and that is exactly what the model learned. Run 57 @500
(`runs/think57_stop/e1_0000500_n48.json`, n=432, per-hop bands), after a further 250 steps of 2-hop training:
      hops      1      2     3-4    8-10
      clean  0.938  0.000  0.039   0.035
      tf     0.958  0.021  0.039   0.026
**The teacher-forced column is the damning one.** `tf` hands the model the TRUE chain `v1 v2` and asks for `v2`,
and it scores **0.021**. A reader that cannot cash a chain it is GIVEN makes the reward
`p(answer|thought) - p(answer|clean)` content-free, which is the run-52 lesson arriving by a new road. For scale,
at the SAME step from the SAME init, run 55 (no curriculum, hops `U{3..10}` from step 0) read `tf` **0.846** at
hops 3-4 and 0.403 DEEP. **RULE: an easy rung is for EXPLORATION -- keep it, but keep it as a FRACTION of a mixture
that always contains the hard rung (`--synth-hops-ramp 0` -> `U{1..10}` every step), never as a PHASE.** The
gradient argument for the phase does not hold either: with hundreds of anchors per step, 10% of them at 1 hop is
still dozens of easy problems per step, which is all the discovery in run 56 ever needed. Run 58 tests this.

### G73 AMENDMENT, 2026-09-29 02:0x -- **THE RULE ABOVE IS FALSIFIED AS WRITTEN. DO NOT APPLY IT.**
Run 58 did exactly what the rule says (`--synth-hops-ramp 0`, hops `U{1..10}` at every step, everything else
run 57's) and **the discovery did not survive it.** At the same step, on the same probe, from the same init:
      run 57 @500, 1 hop (n=48):  pvls_greedy **0.938**    run 58 @500, 1 hop (n=48):  pvls_greedy **0.271**
The "dozens of easy problems per step is all it needed" argument is wrong because it counts the easy anchors and
not the REWARD LANDSCAPE. With `U{1..10}` the easy rung is ~10% of anchors and the policy gradient on the other
90% pays nothing for "emit the looked-up value" -- a 5-value chain is not reachable by chance from a ~100-value
vocabulary (the run-52 exploration lesson) -- so the behaviour that the easy rung would reward is swamped before
it is ever found. **The OBSERVATION behind G73 stands** (run 57's reader really did read `EM(tf)` 0.021 at 2 hops
after a 250-step 1-hop phase, and that is a real defect), **but its CURE is worse than its disease.** See G75.

## G74 -- READ THE GATE'S OWN PROBE AT THE GATE'S OWN STEP BEFORE YOU NAME THE CAUSE OF ITS FAILURE
The 22:07 tick of 2026-09-28 stopped run 57 on check A' @500 and diagnosed the failure as G72 truncation, citing
the greedy thought length collapse `22.1 @500 -> 2.0 @850` and the step-800-850 dumps (`' 4 </T>'`, `' 9 </T>'`).
**Those dumps are from 300-350 steps AFTER the number the gate was read on.** Re-reading
`runs/think57_stop/e1_0000500_n48.json` -- the file the gate was computed from -- at 1 hop, n=48:
      EM clean 0.938   filler 0.938   tf 0.958   **greedy 0.729**   pvls_greedy 0.938   **len_greedy 22.2**
and the length distribution over those 48 anchors has its minimum at **19**: not one anchor was truncated. The
greedy string is `[220, 19, 17, 15, 15, ... x19 ..., </T>]` against a true chain beginning `23 15 | 19 17`. So the
mechanism that cost run 57 its EM at the gate is **not truncation: the writer emits the CORRECT value and then
does not stop, and the ~19 junk digits that follow are what the reader answers from.** Run 58, which moved the
other flag, shows the same mechanism in its pure form: `len_greedy` **32.0 on ALL 432 anchors** (the `--L 32` cap,
no `</T>` ever emitted) and `EM(greedy)` **0.000 in every hop band**.
**RULE: when a pre-registered gate fails, the diagnosis must be computed from the artefact the gate was computed
from, at the step the gate names.** Training dumps and length curves from later steps describe a DIFFERENT policy;
using them to explain an earlier gate is how run 58 came to be designed around a defect that had not yet happened.
Corollary for the surviving half of G72: the rule "set `--min-think` to one complete task unit" is still right and
run 59 applies it -- but its stated MEASURED CONSEQUENCE in the G72 entry above is wrong and is retracted here;
truncation was run 57's state at step ~850, not at step 500, and it was never measured against a gate.

## G75 -- AN EXPLORATION RUNG HAS TO BE DOMINANT, NOT MERELY PRESENT
Run 56 (1-hop only for 250 steps) and run 57 (same) each produced this project's only positive writer result --
`pvls_greedy` 0.938-0.98 at 1 hop, the correct looked-up value written with `--imit-w 0`, i.e. no demonstration of
what to write, ever. Run 58 kept the rung but diluted it to a tenth of a uniform `U{1..10}` mixture and the
discovery fell to 0.271, which is near what 32 junk digits hit by chance. The two symptoms of run 58 are one
symptom: with nothing discovered, no short thought earns anything, so `--stop-explore`'s pressure toward stopping
has nothing to pull toward either, and the greedy row never leaves the length cap (32.0/32.0 against run 57's 22.2
and falling at the same step).
**RULE: size an exploration rung by the fraction of the REWARD it carries, not by the count of anchors it occupies.
If a behaviour is only reachable on rung A, rung A must dominate the mixture while the behaviour is being found,
and may be thinned only afterwards.** The design this implies -- and the successor run 59 licenses if its check A
passes -- is a NON-UNIFORM depth mixture (e.g. 50% at 1 hop, 50% spread over 2..10) so the easy rung keeps its
density while the reader still sees the deep chains it demonstrably learns from (G73's real observation). That
needs a new default-off flag in `think_train.py`; the existing `--synth-hops-ramp` can only express "phase" (>0)
or "uniform" (0), and run 57 and run 58 have now each failed at one of those two extremes.

### G74 SUB-NOTE, 2026-09-29 -- `len_spec` IN THE TRAINING LOG IS **NOT** THE GREEDY THOUGHT'S LENGTH
The step line prints `len {lm_}/{lsp}` (`think_train.py:1159`) and it is easy to read the second number as the
greedy/speculation row, because the metric is called `len_spec`. It is not. `lsp` comes from
`st["lens_sp"].extend(jlens[ssl].tolist())` (l.967): the lengths of the **DEMONSTRATION scratchpads at the SSL
anchors**. On a `chase` run that number is a function of the HOPS MIXTURE and nothing else -- one value per hop
plus separators -- so it moves when you change `--synth-hops-ramp` even if the writer is frozen. The 00:07 tick of
2026-09-29 compared "run 57's `len_spec` was 6.1 and falling at step 300; run 58 sits at 12.0" as evidence about
the two writers; run 57 at step 300 was drawing `U{1..2}` and run 58 `U{1..10}`, so that difference is the
CURRICULUM, not the policy. Run 59's smoke makes it unmistakable: at steps 1-2 it printed `len 15.7/2.0` and
`len 14.2/17.0` was run 58's at the identical steps with the identical data seed -- the first number (sampled RL
thought) is byte-identical between the runs, and only the demonstration length differs.
**RULE: the greedy thought's length is `len_g` in a `probe_e1.py` json (`tp.stop_pos`, l.180) and `len_spec` in
`dumps.jsonl`. Never `len_spec` in the step line.**

## G76 (2026-09-29 04:07) -- `scr_dig_g` IS **NOT** "THE DIGIT RATE IN THE GREEDY THOUGHT". IT IS POSITIONAL AGREEMENT WITH THE TRUE SCRATCHPAD DIVIDED BY THE SCRATCHPAD'S LENGTH, SO THE LENGTH FIX AND THE HOPS RAMP BOTH PUSH IT DOWN ON A HEALTHY RUN
The 22:40 addendum of 2026-09-28 introduced `scr_dig_g` as "(digit rate in the greedy thought) ... the live
training-side proxy and it is rising: 0.054 / 0.058 / 0.070 / 0.084" -- offered as the honest substitute for
`scr_pv_g`, which is blind to this writer's leading space. The substitute is real but the NAME is wrong and the
misreading is load-bearing. `think_train.py:884-892`:
```
within = ar_ <= sq["slen"][:, None];  digs = ar_ < sq["slen"][:, None]
hit    = zr == sq["scr"]                      # zr = z[SPEC_ROW], the greedy row
dg     = (hit & digs).sum(-1) / np.maximum(sq["slen"], 1)
```
So `scr_dig_g` = **(number of positions i < slen where the greedy token equals the TRUE scratchpad token at i) /
slen**. It is a positional exact-match rate against the full true chain, not a property of the thought's alphabet.
Two consequences, and both of them apply to run 59 right now:
1. **It is bounded above by `min(len_g, slen)/slen`.** A greedy thought of 4 tokens can match at most 4 positions
   however right it is, so the moment `--min-think 3` + `--stop-explore` does its job and `len_g` falls 32 -> 4,
   `scr_dig_g` MUST fall. Run 59's block means are 0.056 / 0.047 / 0.053 / 0.049 / 0.045 / 0.045 / 0.025 while its
   greedy row goes from the L=32 cap to ` 18  </T>`. **That fall is the fix working, not the discovery dying.**
2. **The denominator grows with the curriculum.** `slen = len(S)` in `synth.make_chase_seq` is the token length of
   the whole chain, so `--synth-hops-ramp 250` (hops_max = 1 + step//250) inflates `slen` every 250 steps. Like
   `len_spec` in the step line (G74 sub-note), `scr_dig_g` therefore moves when you change the hops mixture with
   the policy frozen. It is not comparable across runs with different ramps, nor across steps within a ramped run.
**RULE: `scr_dig_g` is a chain-COMPLETION measure, so it can only be read on a writer that is trying to emit the
whole chain. On a run whose design target is ONE complete value (run 59), it is structurally capped near
`3/slen` and carries no information about whether the value is correct. The correct-value question is answered by
`pvls_greedy` in a `probe_e1.py` json -- prefix-of-correct-VALUES, space-tolerant, length-independent -- and by
nothing in `metrics.jsonl`.** Companion to G74: name the instrument from its code before you quote its trend.

---

## G77 (2026-09-29, 06:07 headless tick, run 59). `reg%` IS A DEAD AXIS ON A `--thought-vocab digits` RUN WITH A SHORT `--min-think` FLOOR, AND ITS 15% DERAIL BAR SITS *BELOW* THE UNIFORM-POLICY NULL.

Runbook 3b stops a run on *"register share (reg%) > 15% in the last block, or > 8% and rising for 3 blocks"*.
On run 59 that clause fires on a run doing exactly what it was designed to do, and it must not be used.

`status.py:50` computes reg% as the share of the **top-3 eight-character prefixes** among the RL (sampled)
thoughts in the block. The bar was calibrated on free-text thoughts, where the prefix alphabet is effectively
unbounded and a 15% top-3 share really does mean a handful of memorised phrases (run 22). Run 59's thought is
`--thought-vocab digits` with `--min-think 3`, i.e. SPACE + two digits + `</T>`: **the 8-char prefix IS the whole
thought, and the alphabet of possible thoughts is ~90 two-digit values.** Measured, `dumps_eligible.jsonl`, RL rows:

      steps      n   distinct  top-3 share
      300-349   97     94        6.2%
      450-499   76     51       14.5%
      550-599   93     50       17.2%     <-- literal clause "> 15% in the last block" FIRES here
      600-649   95     54       13.7%

and the null, 2000 draws of n=95 from a **perfectly uniform** policy (`np.random.default_rng(0)`):

      V = 90 distinct values : top-3 share mean 11.8%, p95 14.7%
      V = 60                 : mean 14.2%, p95 16.8%
      V = 50                 : mean 15.5%, p95 18.9%

**A uniform writer over this alphabet scores 11.8% by chance, and its 95th percentile is 14.7% -- so the 15% bar
is roughly the null's own p95, and the "> 8% and rising" bar is below the null's mean.** Run 59's 13.7% with 54
distinct values among 95 samples is not distinguishable from uniform.
The rise 6.2% -> 13.7% is also mechanical and not a content change: at step 300 the thought was still ~8 tokens so
an 8-char prefix carried extra entropy; from step 450 the length is pinned at 3 (G72/G76), so the same statistic is
now measuring a 90-symbol alphabet instead of a long one. **The length fix moved reg% by construction.**

**RULE: before stopping a run on reg%, compute the null for that run's actual thought alphabet and length.**
If the thought is one token-pair from a small closed vocabulary, reg% has no headroom above chance and is not
evidence of anything. The collapse it is meant to catch -- the writer emitting a CONSTANT, context-free string --
is caught directly by `pvls_greedy` in a `probe_e1.py` json: a constant thought can match the true first value on
at most ~1/90 of anchors, so run 59's 1-hop `pvls_greedy` **0.562 (27/48)** at step 500 is positive proof that the
writer is conditioning on the context. Companion to G54 (length/reg%/prefix dead on a `--code-size` run) and to
G76 (`scr_dig_g`): **three of runbook 3b's live axes are now known to be artefacts of the thought's alphabet, not
properties of the policy. Read 3b's clauses as written FOR FREE TEXT, and re-derive each one for the run in hand.**

## G78 (2026-09-29 08:5x) -- DO NOT PRE-WRITE THE NARRATIVE OF A GATE FAILURE YOU EXPECT. READ THE ARTEFACT COLD, AND CHECK WHETHER THE QUANTITY IS *HOLDING* OR *DECLINING*
The 06:07 tick correctly predicted that run 59's check A would fail, and then wrote down, 2.5 hours in advance,
what the failure would MEAN: *"not 'the fix destroyed the discovery' but 'the converged writer is right about half
the time, and the 0.90 bar was set from a transient'... IF A FAILS, DO NOT REVERT `--min-think 3`."* When A did
fail, half of that was right and half was **measured wrong**:
      1-hop `pvls_greedy`   run 59 @500  0.562      run 59 @1000  **0.333**
      1-hop `pvls_sample`                0.667                    **0.312**
      training arm, 1-hop, sampled (`agent/pvls_live.py`): steps 200-249 **1.000** -> 500-899 0.55-0.67 -> 900-999 0.294/0.400
The bar claim was right (0.90 came from run 57's @500 transient, and run 57's own converged value is 0.000).
**"The converged writer is right about half the time" was not: run 59 had no converged plateau. It was on the same
slide as run 57, more slowly, and the pre-written sentence named it a plateau because a plateau was the reading
that justified the flag.** The mechanism of the error is ordinary and worth naming: a narrative written before the
number is a PRIOR, and it was written in the same file that the next tick reads as memory, so it arrives looking
like a conclusion. **RULES. (1) It is right and good to pre-register the GATE (bars, arms, step, and the
stop/continue action) -- that is what keeps a failing run from being talked into continuing. It is not right to
pre-register the DIAGNOSIS. (2) Two probes of the same quantity are a SLOPE, not two independent readings: before
calling a level "converged", check the earlier probe of the same run, and state the direction. Run 59 had exactly
the two points needed and the 06:07 tick had already seen one of them. (3) A gate's own artefact must be read by a
script that reproduces the PREVIOUS probe's published columns first (this tick's `/tmp/gateA.py` reproduced every
@500 number in E_STATE (h) before it was allowed to report @1000) -- otherwise a reading error and a real change
are indistinguishable.** Companion to G74 (read the gate's artefact at the gate's step) -- G74 says do not date the
evidence wrong, G78 says do not date the CONCLUSION wrong.

## G78.1 -- A DISCOVERED BEHAVIOUR IS NOT CONSOLIDATED: IT DECAYS WHEN THE RUNG THAT PAYS FOR IT IS DILUTED, AND IT TAKES THE READER'S SKILL ON THAT RUNG WITH IT
**[SUPERSEDED 2026-09-29 10:40 -- READ G79 FIRST. The second half of this heading ("takes the reader's skill with it") is WITHDRAWN, and the rung-share explanation below is falsified by the 1-hop/2-hop internal control. The writer's decay is real; its cause is G79(3).]**
Run 59's `--synth-hops-ramp 250` makes hops `U{1..min(10, 1+step//250)}` (`think_train.py:589-593`), so the 1-hop
share of training anchors is **100% (steps 0-249) -> 50% (250-499) -> 33% (500-749) -> 25% (750-999) -> 20% (1000)**.
1-hop content, measured on a FIXED probe anchor set that never changes, reads **1.000 -> (falling) -> 0.562 @500 ->
~0.5 -> 0.333 @1000** beside it. And the decay is not confined to the writer: at 1 hop, `EM(clean)` fell
**0.729 -> 0.417** and `EM(tf)` (the true chain HANDED to the model) fell **0.938 -> 0.479**, with
`EM(tf)-EM(filler)` decaying +0.229 -> +0.062+-0.046. G75 already said an exploration rung must DOMINATE *while
the behaviour is being found*; run 59 adds the harder half: **once found, the behaviour is not banked. Dilute its
rung and it is spent, and the reader's ability to cash that rung is spent with it.** Any curriculum in this family
must therefore hold the discovery rung at a fixed, dominant share for the WHOLE run, not ramp away from it.
**Three alternatives that this correlation (three probe points) does not exclude, kept so the successor is a test
and not a confirmation: (a) `--stop-explore 0.1` still grinding -- weak, the length is pinned at the floor of 3 on
431/432 anchors so there is nothing left to shorten; (b) ordinary RL drift off a base skill; (c) the deep rung
teaching a COMPETING policy "emit any in-map value" -- `pvls_live`'s `in map` column stays 0.77-0.99 across the
whole run while the identification of WHICH value decays, which (c) predicts and (b) does not.**
One unpredicted positive on the same artefact, recorded because it survives the failure: the sampled arm began
emitting MULTI-VALUE thoughts late (`' 78 48 26 </T>'`, `' 38 97 75 30 </T>'`; 17% of eligible RL rows at steps
950-999 against ~0% at 850-949), so run 59's registered claim that a floor of 3 forces a ONE-value thought
"whatever the depth" is not safe. Lead, not result: n is small and an earlier burst at 800-849 did not persist.

## G79 (2026-09-29, 10:07 headless tick, run 59 @1100) -- **G78.1 IS HALF WITHDRAWN. A RUNG'S SHARE EXPLAINS THE *READER*; IT DOES NOT EXPLAIN THE *WRITER*. THE WRITER DIED BECAUSE A CORRECT NOTE HAD NOTHING TO EARN.**
G78.1 was written on TWO probe points (@500, @1000). The third (@1100, same 432 anchors, paired) confirms its
prediction on the writer -- 1-hop `pvls_greedy` 0.562 -> 0.333 -> **0.271**, paired 500-vs-1100 b=18 c=4 p=0.0043 --
and **falsifies its explanation and one of its two observations.**

**(1) THE OBSERVATION THAT IS WITHDRAWN: "it takes the reader's skill with it".** 1-hop `EM(tf)` reads
0.938 -> 0.479 -> **0.833**. The 1000-vs-1100 bounce is paired-significant (b=6 c=23, p=0.0023) and 500-vs-1100 is
n.s. (p=0.18): the @1000 value was a transient dip. Over ALL 432 anchors the model got **better** across the very
interval G78.1 called a decline -- `EM(clean)` 0.132 -> 0.194 -> **0.273** (paired b=23 c=84, p=2.4e-09), `EM(tf)`
0.130 -> 0.197 -> **0.299** (p=1.0e-14). **G78 warned against reading a slope off two points; G78.1 then did it.**
A two-point "decline" in a quantity whose per-band n is 48 is a slope only after the third point agrees, and here
the writer's did and the reader's did not.

**(2) THE EXPLANATION THAT IS FALSIFIED, BY AN INTERNAL CONTROL THAT WAS ALREADY IN THE DATA.** With
`h ~ U{hops..hops_max}` (`synth.py:143`), **1-hop and 2-hop have IDENTICAL share trajectories at every step** --
both 33% at step 500, both 20% at 1000. Share therefore predicts they move together. They do the opposite:
1-hop `EM(clean)` **0.729 -> 0.438** while 2-hop `EM(clean)` **0.146 -> 0.542**. What share DOES explain is that
every band converges toward the competence its CURRENT share supports (1-hop was over-trained by the 100% phase of
steps 0-249 and regressed; 2-hop started at zero share and climbed; by @1100 they meet at ~0.44/0.54). **But share
cannot explain the writer at all: the deep bands GAINED share from zero and their `pvls_greedy` collapsed anyway**
(3-4: 0.442 -> 0.117; 5-7: 0.368 -> 0.069; >=8: 0.365 -> 0.035). One quantity, two different drivers.

**(3) WHAT ACTUALLY KILLED THE WRITER, MEASURED THREE TIMES: A CORRECT NOTE IS WORTH ZERO BEYOND 2 HOPS.**
`EM(tf) - EM(filler)` -- the gain from being HANDED the true chain, which upper-bounds what any writer could earn:
      band   n     step 500         step 1000        step 1100
      1      48    +0.229+-0.061    +0.062+-0.046    +0.333+-0.069
      2      48    -0.104+-0.045    +0.021+-0.047    +0.146+-0.073
      3-4    77    -0.052+-0.025    +0.013+-0.013    -0.065+-0.047
      5-7   144    -0.076+-0.022    -0.028+-0.017    -0.056+-0.029
      >=8   115    -0.035+-0.017    -0.017+-0.017    -0.017+-0.021
Zero or negative at every depth >= 3, at all three checkpoints. **The ceiling on the reward is a property of the
TASK+MODEL, and it is flat zero over most of the diet.** `--synth-hops-ramp 250` drives the hops<=2 share
1.00 (step 200) -> 0.67 (500) -> 0.50 (800) -> 0.40 (1000) -> 0.22 (2000), so the run spends most of its life
optimising a writer on anchors where **no** writer can be rewarded. Beside it, the across-rollout spread `g_sd`
falls 0.350 -> 0.223 -> 0.121 -> 0.101 -> 0.079 -> **0.044** and `g_mean` sits at ~0 from step 600.

**THE RULE: before building a run around a reward, measure the reward's CEILING per stratum with the
teacher-forced control (`EM(tf) - EM(filler)`), and keep the diet inside the strata where that ceiling is
positive.** A stratum with a zero ceiling contributes only noise to the writer's gradient, and it costs nothing
to find this out -- run 59's @500 probe already showed the >=3-hop bands at or below zero, 600 steps before the
gate that stopped the run.

## G79.1 -- **BEST-OF-8 NEVER BEAT CHANCE IN ANY BLOCK OF RUN 59, AND `g_sd` IS LOGGED PRECISELY SO YOU CAN NOTICE THAT**
**[RETRACTED 2026-09-29 14:07 -- READ G82 FIRST. The ratios below are correctly computed and the tick that
read them was right that nobody had; the 1.576 bar they are read against is a NORMAL-theory null and this
reward is bimodal, so the inference "at or below 1.576 therefore noise" is invalid, and in the regime this
project cares about it is BACKWARDS. The table stays; every "therefore" in this entry is withdrawn.]**
`think_train.py:960-962` logs `g_sd` with the comment "best-of-R inflates the winner by ~1.42 sd on pure noise, so
`g_win - g_mean` is only interpretable next to this number". Nobody had read it against the number. Doing so
(pure-noise expectation for R=8 against the ddof=0 sd that `G.std(0)` returns is **1.576**, Monte-Carlo 4e5 draws):
      block        g_mean    g_win    g_sd   (g_win-g_mean)/g_sd
      0-199        -0.4986  -0.0350  0.3504        1.32
      200-399      +0.0358  +0.3087  0.2232        1.22
      400-599      +0.1221  +0.2821  0.1209        1.32
      600-799      -0.0179  +0.1153  0.1009        1.32
      800-999      +0.0019  +0.0919  0.0788        1.14
      1000-1199    -0.0223  +0.0348  0.0436        1.31
**At or below the noise expectation everywhere, including steps 0-399 when 100% of anchors were rewardable.**
So run 59 contains no evidence that RL-from-reward ever moved this writer. The correct notes it did produce early
were the ORACLE's: `--p-or-start 0.9 --p-or-floor 0.5 --p-or-t0 300 --p-or-t1 800` gives p_or 0.9 -> 0.74 @500 ->
0.50 @800+, beside 1-hop content 1.000 (steps 200-249) -> 0.562 @500 -> 0.333 @1000. **Withdrawing a teacher and
calling what is left "discovery" requires showing the selection beat chance first.** Report this ratio every tick
a run is alive; it is one line and it is the difference between "RL is learning slowly" and "RL is not learning".

## G79.2 -- **A COLLAPSING REWARD SPREAD DOES *NOT* PRODUCE A VANISHING GRADIENT HERE: THE ADVANTAGE IS STANDARDISED, SO THE TRAINER AMPLIFIES NOISE TO FULL STRENGTH**
The intuition "`g_sd` -> 0, so the RL term quietly switches itself off" is wrong in this codebase and the error is
dangerous in the safe-sounding direction. `think_train.py:930-932` computes
`mu`/`sig` over the batch and sets `A = clip((adv_paired - mu)/(sig + 1e-6), +-reward_clip)`: **the advantage is
renormalised to unit variance no matter how small the raw spread is.** As `g_sd` falls 0.350 -> 0.044 the signal's
information content decays toward zero while its gradient magnitude is held constant -- a full-strength random walk
on the writer's content. This is exactly what run 59 shows: the FORM stays pinned by `--min-think 3`
(`len_greedy` exactly 3 on 432/432 anchors at @1100) while the CONTENT rots on every hop band, and by @1100 the
note is a net harm at 1 hop (`EM(greedy)-EM(clean)` = -0.125, McNemar b=0 c=6, p=0.031).
**Consequence for any successor: a near-zero `g_sd` must GATE the RL term (skip the update, or floor `sig` on an
absolute scale), not be trusted to fade out on its own.**

## G80 (2026-09-29, 10:07 headless tick) -- **`probe_thinkmode.py` HAS BEEN DEAD SINCE THE `--bridge` EDIT (`NameError: BRIDGE` ON EVERY INVOCATION), AND ITS INLINE `disc_filler` IS HARD-CODED TO LENGTH 10**
Two independent defects in the same instrument, found by trying to run runbook 3d.1 on run 59's final checkpoint.

**(1) IT DID NOT RUN AT ALL.** `BRIDGE` is READ at l.85, l.101 and l.131 but was never ASSIGNED, so every
invocation died instantly with `NameError: name 'BRIDGE' is not defined`. **The committed version at HEAD is
broken too**, so this is not a local edit gone wrong -- the `--bridge` baseline was added and its argument parse
was never written. The last thinkmode numbers in the tree are the r33/r34 ones from 2026-09-20, which dates the
breakage. Fixed this tick with the parse that keeps the old behaviour as the default
(`BRIDGE = "--bridge" in sys.argv`; the probe's own mask is built WITHOUT a bridge, which is what fixes the
default at False), backup `probe_thinkmode.py.bak_pre_bridgefix_20260929_1007`, `py_compile` OK, diff is exactly
the one added definition. **Lesson: a probe that has not been run since a refactor is not an instrument. Runbook
3d.1 calls this probe at every run end; it was silently uncallable for nine days and no tick noticed, because no
tick had run it.**

**(2) ITS BUILT-IN FILLER FLOOR IS NOT LENGTH-MATCHED, WHICH IS THE EXACT CONFOUND G39 WAS WRITTEN ABOUT.**
`probe_thinkmode.py:103` builds the inline control as `fill[:, :10] = SPACE` -- **hard-coded at 10 spaces**,
whatever the writer emits. On run 59 @1100 the own greedy thought on web text runs to the `L=16` cap on every
sequence (`len_greedy=16.000`, all sequences), so the printed `own greedy MINUS filler` compares a 16-token
thought against an 11-token filler and charges the difference to CONTENT. On the first six sequences that gap is
about $-12.8$ against $-4.8$, which would read as a catastrophic content effect and is partly just five more
tokens of occupancy. **The `--pause N` path is the only length-matched control**, because it replaces the thought
everywhere with `min(N, L-1)` spaces + `</T>`. G39 already said to run the probe twice and compare
`disc_greedy(own)` against `disc_greedy(pause)`; what is new here is that **the inline `disc_filler` column looks
like it makes that second run unnecessary, and it does not.** Never quote `own greedy MINUS filler` from a single
json without first checking `len_greedy` against 10.

## G81 (2026-09-29, 12:07 headless tick) -- **THE WEB-TEXT THINKMODE PROBE CANNOT ANSWER ANYTHING ON THE `--web-pause 2` LINEAGE (RUNS 44+): THE SLOT IS UNTRAINED ON WEB TEXT, SO BOTH ARMS OF THE COMPARISON ARE OUT OF DISTRIBUTION**
The 10:07 tick repaired `probe_thinkmode.py` (G80) and ran it on run 59's final checkpoint plus a length-matched
`--pause 16` control. Both landed. **The numbers are large, and they mean nothing about this project's question.**

**THE NUMBERS FIRST, so nobody re-derives them** (`runs/probes/thinkmode_r59_1100.json` and
`..._pause16.json`, run 59 `checkpoints/think59_tail_archive/step_0001100.pt`, `--slot-pos aligned`, n=465,
PAIRED per G26 -- the traversal is deterministic and the two `disc_filler` arrays are element-wise identical,
which is the alignment check):
    own greedy                       -11.447 +- 0.381
    pause16 control (15 SPACE + </T>)  -5.143 +- 0.248
    inline `disc_filler` (10 SPACE + 6 </T>)  -7.979 +- 0.344
    **own - pause16   -6.303 +- 0.324 (t -19.5)**   own - inline filler  -3.468 +- 0.379 (t -9.1)
`len_greedy` histogram: 16 on 424/465 sequences (the `L=16` cap), 8 on 23, 9-12 on 18. On the 23 sequences where
the writer does stop at 8 the gap is still -2.34 +- 0.79.

**WHY IT IS UNINTERPRETABLE.** Run 59 runs `--web-pause 2`, which (`think_train.py:664-667`) sets `pau`, `ssl`
and `rl` ALL to zero on web micros: web text carries **no slot, no reader loss, no writer loss -- pure NTP**.
And run 59 initialises from `checkpoints/vanilla12b_fresh_ladder/step_0022800.pt`, a token-matched VANILLA
ladder checkpoint with no thought slot in its history at all. **So no gradient has ever touched this model's
behaviour in a `<T>` block on web text, in this run or in any ancestor.** The probe inserts one anyway.
What it then measures is what an untrained slot does, and the `examples` (printed as token ids, per G39, this
being a new lineage) say it plainly: on web text the greedy "thought" is not digits at all, it is **fluent
English that continues the context** (e.g. greedy `your own software without paying for it. You can also copy,
modify, and` where the true window begins `free software without paying`). That is just the vanilla model
auto-completing into an unmasked region; the reader then has a confident divergent continuation sitting in
front of the real one. **Do not read -6.3 as a fifth replication of "the written thought is worse than an empty
slot" (runs 25/27/28/31). Those were runs whose writers were TRAINED on web anchors. This one is not.**

**RIDER, CORRECTING G80(2) ON ITS MECHANISM (the conclusion survives, the reason given was wrong).** G80 said
the inline filler undercounts because the own thought occupies "five extra tokens". **There is no occupancy
axis: `layout.blk = L + 1` is FIXED (`think_par.py:47`) and every window row attends the WHOLE block
(`build_full_mask`, l.123), `</T>` or not.** The slot is the same width whatever the writer emits, so length
can only act through WHICH tokens sit there. The inline filler and `--pause 16` differ in content alone --
10 SPACE + 6 `</T>` against 15 SPACE + 1 `</T>` -- and differ by **+2.835 +- 0.162 (t +17.5) in favour of
pause16**: repeated `</T>` is itself expensive to this reader. So the inline column is not merely mismatched,
it is a control biased ~2.8 nats in the harmful direction, and G80's instruction ("never quote `own greedy
MINUS filler` from a single json") stands for a stronger reason than the one it gave.

**OPERATIONAL CONSEQUENCE, so no future tick spends 40 minutes on this again: runbook 3d.1's `probe_thinkmode`
step is NOT APPLICABLE to a run with `--web-pause 2` (i.e. everything from run 44 onward).** For those runs the
held-out instrument is the E1 gate probe (`probe_e1.py` / `agent/gate_read.py`) on the synthetic task, which is
the only surface the slot was trained on. Only run it on a web-trained writer, and if you do, use `--pause N`
matched to that writer's own `len_greedy`, never the inline column.


## G82 (2026-09-29, 14:07 headless tick, device idle) -- **G79.1 IS RETRACTED: 1.576 IS A NORMAL-THEORY NULL, THIS REWARD IS BIMODAL, AND BELOW-1.576 IS WHAT A *MAXIMALLY INFORMATIVE* BEST-OF-8 LOOKS LIKE HERE. THE GATE BUILT ON IT WOULD HAVE STOPPED RUN 57 AND PASSED RUN 58.**
The 10:07 tick read `(g_win - g_mean)/g_sd` against a Monte-Carlo pure-noise expectation of **1.576**, found run 59
at or below it in every block, and concluded "run 59 contains no evidence that RL-from-reward ever moved this
writer". The 12:07 tick then put that test into the run-60 proposal as a pre-registered STOP gate ("mean over the
first 400 steps must exceed 1.576, else stop and report E1 negative"). **Do not register it. It is invalid, and
this tick checked it three ways.**

**(1) THE BAR ITSELF IS RIGHT, FOR NORMAL GAINS, AND THE STATISTIC IS A RATIO OF MEANS.** `think_train.py:962,
1177-1178` log `g_win = mean_a(max_r G)`, `g_sd = mean_a(sd_r G)` (ddof=0, `G.std(0)`) and `g_mean = mean_a(mean_r G)`,
so the logged ratio is `mean_a(max-mean) / mean_a(sd)`. Both terms are linear in a per-anchor scale, so anchor
heterogeneity cancels and the null is the closed form `c/b` with `c = E[max of 8 N(0,1)] = 1.4236` and
`b = E[s_ddof0]/sigma = 0.9027`: **c/b = 1.577**, reproduced by Monte-Carlo (2e5 draws) at 1.5766. The predecessor's
number is correct arithmetic. It is the DISTRIBUTION it assumes that this run violates.

**(2) THE RUN'S OWN LOG REJECTS NORMALITY, AND NOT MARGINALLY.** Run 59 carries `n_anch` = 4 per step (median, max 9).
Simulating the null at that anchor count (2e5 steps of 4 anchors x 8 rollouts of N(0,1)):
      NULL, n_anch=4:      mean 1.576   sd 0.173   P(per-step ratio < 1.0) = 0.000 (0 of 200,000)
      RUN 59, observed:    mean 1.319   sd 0.623   P(ratio < 1.0) = 0.356 (367 of 1032 steps)
      RUN 57, observed:    mean 1.356   sd 0.488   P(ratio < 1.0) = 0.254
A third of run 59's steps sit in a region the null never visits in 200,000 draws. The eight rollout gains are not
normal, so 1.576 is not their null and "below it" means nothing on its own.

**(3) AND THE DIRECTION OF THE ARTEFACT IS THE ONE THAT MATTERS.** `--reward-mode prob --score-answer-only 2` on a
lookup task scores a rollout on whether it emitted the right value: the per-anchor gain is close to two-valued.
Monte-Carlo of exactly that (rollout right w.p. `p` -> gain D, else 0, plus small noise; 4e5 draws, D in {1,3},
ratio insensitive to D):
      p    0.1    0.2    0.3    0.5    0.7    0.9
      ratio 2.20   1.91   1.62   1.14   0.82   0.65        (P(best-of-8 contains a correct rollout) = 0.57 ... 1.000)
**The ratio falls BELOW 1.576 as soon as the rollouts succeed more than ~30% of the time, and it is LOWEST exactly
when best-of-8 is certain to contain a correct note.** A writer that is right 90% of the time reads ~0.65 -- deep in
what G79.1 called noise. Below the bar is evidence of a bimodal reward, and is equally consistent with a very good
writer and a dead one; it cannot distinguish them.

**(4) THE EMPIRICAL CHECK THAT SETTLES IT: THE GATE IS ANTI-CORRELATED WITH THE OUTCOME IT WOULD RULE ON.** Same
window the gate names (steps 0-399), same statistic, the two runs whose writer outcome we actually know:
      run 57  ratio 1.294 +- 0.022  -> **FAIL** the proposed gate  | 1-hop `pvls_greedy` **0.938** @500 (the project's
                                                                     ONLY demonstrated correct self-written note)
      run 58  ratio 1.586 +- 0.017  -> **PASS** the proposed gate  | 1-hop `pvls_greedy` **0.271** @500 (chance; the
                                                                     run that destroyed the discovery, G75)
      run 59  ratio 1.297 +- 0.027  -> FAIL                        | 1-hop `pvls_greedy` 0.562 @500
**Registered as written, this gate stops the successful run at step 400 and reports "E1 negative", and lets the
failed one through.** That is not a conservative null; it is a false-negative machine pointed at the one positive
result the project has.

**WHAT SURVIVES.** (i) G79's central finding is untouched: it comes from the probe (`EM(tf) - EM(filler)` per hop
band), not from this ratio. (ii) G79.2 (the standardised advantage amplifies a collapsing spread) is untouched --
it is a statement about `think_train.py:930-932`, not about a null. (iii) The writer's decline (`pvls_greedy`
0.562 -> 0.333 -> 0.271, paired p=0.0043) is untouched. (iv) The oracle caveat survives AS A CAVEAT: `p_or` 0.9 ->
0.5 does track the content's fall, and we still have no POSITIVE evidence that reward alone moved this writer. What
is withdrawn is the claim that we have evidence AGAINST it. We have no instrument pointing either way.

**WHAT TO USE INSTEAD.** The quantity that empirically separates run 57 from run 58 is the E1 probe's 1-hop
`pvls_greedy` -- i.e. run 59's checks A and B, which need no distributional assumption because they read the note's
content directly. Register those and nothing else. A training-time best-of-8 null would need a max over R SHUFFLED
rollouts to compare against (`g_shuf`/`g_shuf0` are per-anchor single-shuffle columns, not a max over R), which
costs R extra scorings per step -- do not add it on the strength of this episode.

**THE GENERAL RULE, which is the reason this is a numbered gotcha and not a correction.** A textbook null carries a
distributional assumption that the codebase does not enforce. Before gating a run on one, simulate the null AT THE
RUN'S OWN `n_anch` and check the observed spread against it (5 lines of numpy, and here it failed by a factor of
3.6 in sd and 0.356-vs-0.000 in tail mass) -- and then back-test the gate on the archived runs whose outcome you
already know. Both checks are cheap, both were available on 2026-09-29 10:07, and either one alone would have
caught this before it reached `STATUS.md` and `progress.tex`.

## G83 (2026-09-29, 18:07 headless tick, device idle, zero new training) -- **G79 SHARPENED: BEYOND 2 HOPS A *PERFECT* NOTE IS NOT WORTH ZERO, IT IS WORTH SLIGHTLY *NEGATIVE*, AT EVERY CHECKPOINT. RUN 59 SPENT 60-80% OF ITS DIET MILDLY PUNISHING THE BEHAVIOUR IT WAS TRYING TO LEARN.**

G79 established that *"a correct note has nothing to earn beyond 2 hops"*. Read at full n, paired, on all
three archived checkpoints, the sign is consistently the wrong side of zero. `EM(tf) - EM(filler)` -- the
CEILING on what any note can buy, since `tf` teacher-forces the TRUE scratchpad into the slot:

      checkpoint    hops 1-2 (n=96)              hops >= 3 (n=336)
      @500          +0.0625 +- 0.0414 (t +1.5)   **-0.0565 +- 0.0126 (t -4.5)**
      @1000         +0.0417 +- 0.0328 (t +1.3)   -0.0149 +- 0.0099 (t -1.5)
      @1100         **+0.2396 +- 0.0508 (t +4.7)**  **-0.0446 +- 0.0180 (t -2.5)**

Per hop at @1100: +0.333 / +0.146 at 1 / 2 hops, then -0.111 / -0.024 / -0.080 / -0.035 / -0.054 / -0.033 /
+0.000 / -0.021 at 3..10. **Not a decay to zero -- a sign flip at hop 3 that never comes back.** The
mechanism is the one already registered in sec. 2 for a different instrument ("a confident wrong token is
worse than noise"): past 2 hops the note names an intermediate the reader cannot verify, and occupying the
slot with it costs more than leaving the slot blank.

**THE DIET ARITHMETIC THIS MAKES POSSIBLE, AND IT SIZES THE PROPOSED SUCCESSOR.** `h ~ U{1..hops_max}` and
`hops_max = min(--synth-hops-max, 1 + step//250)`, so the expected ceiling per anchor is just the per-hop
table above averaged over the current diet:

      diet                                            E[EM(tf) - EM(filler)]
      hops_max = 3   (run 59 at step 500)                    +0.123
      hops_max = 5   (run 59 at step 1000-1105, where it died) +0.053
      hops_max = 10  (run 59's designed endpoint)             +0.012
      hops_max = 2   (THE PROPOSED SUCCESSOR)                 **+0.240**

**The proposed diet multiplies the reward a correct note can earn by 4.5x over the diet run 59 was on when
it was stopped, and by 20x over the diet it was heading for.** That is the quantitative form of the question
now sitting in `STATUS.md`, and it is the strongest argument for it that the existing data can make.

**A SECOND READING, ENCOURAGING AND EASY TO MISS: THE PRIZE GREW WHILE THE WRITER SHRANK.** The hops-1-2
ceiling went +0.063 (n.s.) @500 -> +0.240 (t +4.7) @1100. The READER got substantially better at cashing a
correct shallow note over exactly the window in which the WRITER stopped producing one. A successor held at
hops 1-2 therefore starts from a strictly better position than run 59 did at step 500, on both counts.
**Caveat, stated so no later tick inflates this:** `EM(tf)` is TEACHER-FORCED -- it is what a PERFECT note
would buy, an upper bound, and run 59's own writer never got near it. G83 sizes the prize; it is not
evidence that any writer can collect it.

## G84 (2026-09-29, 18:07 headless tick, device idle) -- **TWO MORE OF RUNBOOK 3b's DERAIL AXES ARE DEAD ON THE `--thought-vocab digits` / `--reward-mode prob` LINEAGE (RUNS 57/59 AND ANY SUCCESSOR). BOTH FIRED CONTINUOUSLY ON RUN 59 FROM ITS FIRST BLOCK AND NO TICK ACTED ON EITHER -- WHICH WAS CORRECT, AND WAS NOWHERE WRITTEN DOWN.**

G77 retired `reg%` on this lineage and closed with *"three of runbook 3b's live axes are now known to be
artefacts of the thought's alphabet"*. Two more go, and unlike reg% these were never even discussed. They
matter now because the run under discussion with Emanuele (hold the diet at hops 1-2) inherits both, and a
future tick reading 3b literally would stop it inside its first 600 steps for reasons that are the design.

**(1) LENGTH IS DEAD BY CONSTRUCTION.** 3b: *"thought length within 0.5 of `--min-think` for 2 blocks"*.
Run 59 is `--min-think 3 --jit-min 3` with `--stop-explore 0.1`, and a complete value IS SPACE + two digits,
so len 3 is the designed terminal state, not a collapse -- E_STATE's own registered prediction for check A
was *"1-hop `len_g` (prediction ~3)"*. Measured: block `len_mean` 9.3 / 6.2 / **3.4** / **3.2** (status.py
blocks 0-299 / 300-599 / 600-899 / 900-1104), i.e. the clause fired on the last two blocks; held-out
`len_g` is **3.00 / 3.02 / 3.00** at steps 500 / 1000 / 1100 (`e1_*_n48.json`). Same structure as G54's
`--code-size` exemption (`--L 4 --min-think 4` pins length at 4.0 forever). **Do not stop a complete-value-
floor run on length.** What the clause was written to catch here -- the writer emitting a CONSTANT string --
is caught by `pvls_greedy`, per G77.

**(2) ROLLOUT SPREAD IS DEAD BECAUSE `g_sd` IS NOT THE SPREAD OF THE TRAINED REWARD.** 3b: *"rollout spread
< 0.6"*. Read the source: `g_sd = mean_a( G.std(0) )` over eligible anchors (`think_train.py:962,1178`) --
the spread of the NAT-SCALE GAIN `G`. But under `--reward-mode prob` the trained reward is
`REW = PR - len_cost*S0` with `PR = exp(lw) - exp(lc)` (l.820-824), a PROBABILITY-scale quantity. On such a
run `g_sd` measures the dispersion of a diagnostic the trainer does not optimise, on a scale the 0.6 bar was
never calibrated for. Measured on run 59, `g_sd` block means:

      steps       0- 99   100-299   300-599   600-899   900-1105
      g_sd         0.457     0.242     0.149     0.101     0.049

**It is below the 0.6 bar in the very first block and never once rises above it.** Taken literally the clause
derails run 59 at step ~1, and it would derail its successor identically. The *direction* remains readable
(rollouts agreeing more and more, which on a right/wrong reward means agreeing on WRONG), but the BAR is not
transferable. **Re-derive the bar for the run's own reward scale, or gate on `ob0` and the probe instead
(G56(c) made the same ruling for code runs on `g_r0`).**

**THE GENERAL RULE, now on its fifth instance (G54, G56(c), G76, G77, and this).** Runbook 3b was written for
a free-text writer scored in nats. Every clause in it names a statistic whose SCALE depends on the thought's
alphabet, its length floor, or the reward mode. **Before using 3b on a run, walk its five clauses and ask of
each: what does this read on a HEALTHY run of THIS design? If the answer is "it fires", the clause is dead for
that run and must be written down as dead BEFORE the run starts, not discovered by a tick that stops it.**

## G85 (2026-09-29, 18:07 headless tick, NEW step-900 probe, zero device time) -- **THE WRITER'S COLLAPSE NEEDS NO SCHEDULE TO MOVE: ITS STEEPEST 100 STEPS HAPPEN WITH EVERY KNOB FROZEN. THE TEACHER'S WITHDRAWAL IS EXONERATED. AND THE DISCRIMINATOR I PRE-REGISTERED FOR THIS PROBE WAS MIS-SPECIFIED -- IT DOES NOT SEPARATE THE TWO HYPOTHESES AND I AM NOT CLAIMING THAT IT DID.**

Run 59 had three archived checkpoints and the record read 0.562 @500 -> 0.333 @1000 -> 0.271 @1100. This tick
probed **step 900** (`runs/think59_tail/e1_0000900_n48.json`, same seed, same 432 anchors -- verified
identical `hops_per_anchor` across all four files, so every comparison below is PAIRED on the same anchors).
Pre-registration written before the data: `agent/PREREG_r59_900.md`.

### The schedule state of each interval (all read out of `metrics.jsonl`, not assumed)
`--p-or-t0 300 --p-or-t1 800` -> p_or is pinned at its 0.50 floor from step 800 (min==max==0.5000 over
900-1105). `hops_max = min(10, 1 + step//250)` -> 4 over steps 750-999, 5 from 1000. `rl_w` 0.50, `gamma`
0.97, `eps_th` 0.15, `gate_R` 0, `crowd` 0 -- all constant across 900-1105.

      interval     what moved in it                ALL (n=432)                 hops==1 (n=48)
      500 -> 900   p_or 0.74->0.50, hops_max 3->4  -0.134 (b=96 c=38 p=0.0000)  **+0.083 (b=6  c=10 p=0.45, UP)**
      900 ->1000   **NOTHING**                     -0.100 (b=57 c=14 p=0.0000)  **-0.313 (b=17 c=2  p=0.0007)**
      1000->1100   **NOTHING**                     -0.069 (b=48 c=18 p=0.0003)  -0.062 (b=8  c=5  p=0.58)

      levels   ALL      0.417  0.282  0.183  0.113      (@500 @900 @1000 @1100)
               hops==1  0.562  0.646  0.333  0.271
               hops>=3  0.384  0.196  0.131  0.068
      rate, ALL band, per 100 steps:  -0.034 (500-900)  **-0.100 (900-1000)**  -0.069 (1000-1100)

### (1) WHAT IS ESTABLISHED, AND IT IS THE USEFUL PART
**(a) The teacher's withdrawal did not kill the writer.** The 14:07 tick flagged that run 59 confounds the
depth ramp with the oracle withdrawal and told a successor to move one thing. The confound is now BOUNDED
from the data: across steps 500-900, which is the bulk of the withdrawal (p_or 0.74 -> its 0.50 floor), the
1-hop band did not fall -- it read 0.562 -> **0.646**. The collapse begins 100 steps AFTER p_or stops moving
and is steepest where p_or has been constant for 200 steps. **Leave `--p-or-*` alone in any successor; there
is no longer a reason to spend a run isolating it.**
**(b) The steepest decline in the run is in a fully frozen window.** 1-hop 0.646 -> 0.333 over steps
900-1000, b=17 c=2, p=0.0007, with no schedule moving. Whatever drives the collapse is internal to the
optimisation, not a thing the curriculum did to it on that step.
**(c) The bands die in ORDER, deep first, shallow ~400 steps later.** hops>=3 loses half its content over
500-900 (0.384 -> 0.196) while hops==1 is still rising; hops==1 then collapses over 900-1000. A single shared
policy dragged down by the majority of its diet is the natural reading of that ordering, and it is the diet
hypothesis (G79/G83) **with a lag**.
**(d) Length is NOT the mechanism.** `len_g` went 3.07 (@500) -> 3.87 (@900) -> 3.00 (@1000) -> 3.00 (@1100),
so the note did collapse onto the floor in exactly the guilty window -- but split the @900 anchors by their
own length and both halves fall the same: ALL -0.096 (len>3, n=374) vs -0.121 (len==3, n=58); hops1-2 -0.220
vs -0.214. And the anchors that LOST correctness are indistinguishable from those that KEPT it in length
trajectory (3.84->3.00 vs 3.88->3.02). A co-occurrence, not a cause.
**(e) The stop decision stands.** Check A was read at 1000 and got 0.333 against a 0.90 bar. At 900 it would
have read 0.646 -- still a fail. Run 59 was not stopped at an unlucky moment.

### (2) WHAT I GOT WRONG, REGISTERED BEFORE THE DATA AND WRONG ANYWAY (companion to G82)
`PREREG_r59_900.md` registered: *"if the DIET is the driver, `pvls_greedy` should be roughly FLAT from 900 to
1000; if the decay is AUTONOMOUS, 900->1000 should fall about as much as 1000->1100."* **That discriminator is
invalid and the branch it would have selected is not supported.** A constantly-diluted diet exerts a CONSTANT
downward pressure on a shared policy; it does not need to keep deepening to keep doing damage. So "decay
continues while the diet sits still" is exactly what the diet hypothesis predicts too, and the test separates
nothing. I registered a criterion in which one hypothesis was represented by the *derivative* of the
curriculum and the other by its *level*. **A tick reading only the table above could write "the diet is
exonerated, the decay is autonomous". It is not established and I am not claiming it.**
**THE RULE (G82 gave the statistical form of this; this is the causal form): before registering a
discriminator, write down what EACH hypothesis predicts for the quantity, and check the two predictions
actually DIFFER. If a constant cause and an absent cause predict the same curve, the test has no power, and
pre-registering it only launders the ambiguity as a result.**

### (3) WHAT IT MEANS FOR THE RUN AWAITING EMANUELE'S OK
Net: **modestly stronger, with one gate consequence.** The teacher is exonerated (1a), so "leave `--p-or-*`
exactly as run 59 had it" is now supported by measurement rather than by caution; the diet remains the live
suspect, uncontradicted, and G83 sizes its prize at 4.5x. But (1b)+(1c) say the collapse can proceed with the
diet standing still, so **the successor must be judged on whether the shallow band HOLDS over hundreds of
steps, not on whether it starts high** -- run 59 read 0.646 at step 900 and 0.333 one hundred steps later.
Any gate read at a single step would have called that run healthy at 900. (Proposal only; sec. 0000 -- the
gate belongs to Emanuele.)

## G86 (2026-09-29, 18:07 headless tick, step-900 probe + a re-read of all four run-59 checkpoints) -- **THE NOTE IS EXACTLY ONE VALUE AND THE TASK NEEDS A CHAIN. AT 1 HOP IT CAN ONLY RESTATE WHAT THE MODEL ALREADY KNOWS (0 OF 76) AND IS NET DESTRUCTIVE; AT 2 HOPS IT WRITES THE CORRECT *FIRST STEP* ON 58% OF THE PROBLEMS IT CANNOT SOLVE AND THEN STOPS. THE BINDING CONSTRAINT IS LENGTH, NOT THE DIET.**

This supersedes the framing of G78.1/G79/G83 -- not their measurements, their *subject*. Four checkpoints
(500/900/1000/1100), same seed, the SAME 432 anchors, everything paired.

### (1) THE NOTE IS ONE VALUE. LOOK AT IT.
`examples` token ids, step 900 and step 1100, every anchor shown:

      greedy note   [220, d, d, 151668, 151668, ...]        <- SPACE + two digits + `</T>` forever
      true scratch  [23, 15, 220, 19, 22, 220, 19, 17, ...] <- a CHAIN of values separated by 220

`len_g` = 3.00 on **432 of 432** anchors at step 1100 (histogram: len3=432). A value is SPACE + two digits =
3 tokens, so the note holds **exactly one value and cannot hold two.** E_STATE registered this in advance as
run 59's structural failure mode ("one value is not a chain"); this is the direct evidence, and it is the
single most important fact about the run.

### (2) AT 1 HOP THE CHAIN IS LENGTH 1, SO THE NOTE CAN ONLY RESTATE THE ANSWER -- AND IT NEVER SUPPLIES ONE
Split every anchor by whether the model answers it correctly with NO note at all (`EM(clean)`):

      hops==1        P(note has the target | model knows)   P(note has the target | model does NOT)
      @500           0.771 +- 0.071 (n=35)                  **0.000 (n=13)**
      @900           0.775 +- 0.066 (n=40)                  **0.000 (n= 8)**
      @1000          0.800 +- 0.089 (n=20)                  **0.000 (n=28)**
      @1100          0.619 +- 0.106 (n=21)                  **0.000 (n=27)**

**Pooled: 0 of 76, 95% upper bound 0.039.** Its own note rescues 1 of those 76 (bound 0.061), while it
BREAKS **18 of the 116** it could already answer unaided. At 1 hop the note costs 18 and buys 1. There is
nothing to compute at 1 hop -- the "chain" is a single lookup -- so the slot can only echo knowledge already
present, and it echoes it imperfectly.

### (3) AT 2 HOPS THE WRITER IS DOING REAL WORK AND CANNOT FINISH
At 2 hops the scratchpad is [intermediate, answer] and `pvls_g` scores the LEADING value, i.e. the
INTERMEDIATE -- **not the same quantity as at 1 hop, and that is why the conditional table inverts**:

      hops==2        P(note has the correct intermediate | model knows)   | ... | model does NOT)
      @500           0.000 (n= 7)    @900 0.000 (n=10)                    | 0.585 (n=41) / 0.658 (n=38)
      @1000          0.062 (n=16)    @1100 0.115 (n=26)                   | 0.562 (n=32) / 0.455 (n=22)

**Pooled over the problems it CANNOT answer unaided: it writes the correct first step 77 of 133 times
(58%), and is rescued into a correct answer only 9 of 133 (6.8%, bound 0.115).** It knows the first step and
never gets to write the second. That gap -- 58% correct first steps, 6.8% correct answers -- is the length
cap, stated in the run's own numbers. (It breaks 6 of 59 here, so at 2 hops the note is roughly break-even,
buying 9 and costing 6, against purely destructive at 1 hop.)

### (4) CONSEQUENCE: THE "WRITER DECLINE" THAT DROVE FOUR TICKS IS LARGELY AN ARTEFACT AT 1 HOP
Hold the step-900 conditional rates fixed (0.775 / 0.000) and multiply by each checkpoint's OWN `EM(clean)`:

      step        500      900     1000     1100
      predicted  0.565    0.646    0.323    0.339
      observed   0.562    0.646    0.333    0.271

Three of four within 0.010. **The 1-hop writer did not decay; the model's 1-hop competence did** (EM(clean)
0.833 -> 0.417 over steps 900-1000, b=22 c=2, p=0.0000), and the note tracked it because the note is a
readout of it. The 1-hop band was the entire diet early and regressed as its share fell -- the share story
that G79 dismissed for the writer is right for the COMPETENCE, and the writer is downstream of that.

### (5) WHAT THIS DOES TO THE QUESTION IN `STATUS.md`
The pending proposal is "hold the whole diet at 1-2 hops, change nothing else". Against it: **1 hop is dead
weight** (0/76; net destructive), so a uniform 1-2 diet spends half its problems on a band where no note can
help; and the diet change does not touch the length cap, which is what (3) identifies as binding. For it:
the reward at 2 hops is real and G83 sizes it. **The single change with the most measured support is not the
diet, it is letting the note hold TWO values** (the floor and whatever pins length at it). That is a design
decision and it is Emanuele's under sec. 0000 -- it is put to him in `STATUS.md`, not launched.

### (6) INSTRUMENT WARNING, AND IT INVALIDATES CROSS-BAND READS OF GATE A
`pvls_g` is "the leading value of the true scratchpad", which is **the answer at 1 hop and an intermediate at
2+ hops**. So `pvls_greedy` does not measure the same thing in different hop bands and its bands must never
be pooled or compared to each other (`gate_read.py`'s ALL row included). Gate A reads the 1-hop band only,
so the gate itself is safe -- but at 1 hop, per (2), it is asking "does the model already know the answer",
which is not the question the project is trying to answer. **A successor's gate should be written on the
2-hop band, and on RESCUE (`EM(greedy)` among anchors with `EM(clean)`=0), not on `pvls` alone.**

## G87 (2026-09-30, run 63). A reader trained ONLY behind true spans over-trusts the slot, and the writer's
rational reply is to shorten the note -- not to improve it.
`--reader-on-ssl 2` gives the reader nothing but perfect notes to practise on, so it believes the slot
completely. A believed-but-wrong note is then very expensive: run 63's own thoughts sat **5-6 nats BELOW
silence** for 700 steps (`g_r0` -3.2 @300-399, -5.8 @700-799) while sitting **+1.0 ABOVE their own shuffle**
(`ob0`). Structure and usefulness came apart completely. Over steps 900-1030 `len_mean` fell 15.0 -> 5.7, the raw
dumps became `"182 </T>"` -- one content token -- and `g_r0` ROSE to +0.26 while `ob0` FELL to +0.15.
**Read `g_r0` recovery as progress only when `len_mean` is flat or rising; if length is falling, the rise is the
length price coming off.** Third sighting of "a confident wrong token is worse than visible noise" (see the
run-25 ladder, `self16`). Successor lever: `--reader-on-ssl 1` (run 64) so the reader also practises behind the
writer's own sampled notes and learns how far to trust them.

## G88 (2026-09-30). THERE IS NO SINGLE "UNTRAINED BAR". The init's own-minus-filler margin swings 15 nats
across the span-offset schedule, so every gate must be read against the init AT THAT GATE'S OWN OFFSET.
`probe_span.py`, unseen-FineMath val, paired, each model against its own filler:
```
  vanilla step_0022800   o = -15 : own - filler  +6.93 +- 0.85   (span is pure visible past -- free to copy)
  vanilla step_0022800   o =  -7 : own - filler  +1.41 +- 0.88
  vanilla step_0022800   o =  +1 : own - filler  -7.99 +- 0.50   (pure forecast)
```
`SUPERVISOR.md` sec. 00S had been quoting the o=-15 figure as if it were THE bar; it is one point on a steep
curve. Two consequences, both of which bit this tick: (1) comparing a run to the init at a DIFFERENT offset
overstates or understates the verdict by up to 15 nats; (2) comparing a run to **its own earlier self** across
an offset change is confounded for the same reason. Build the init's reading at an offset BEFORE gating a run
there. Files: `runs/probes/span_vanilla22800_o{-15_c,-7,+1}.json`.

## G89 (2026-09-30). A SHORT NOTE IS NOT AUTOMATICALLY A DEAD NOTE -- do not gate on length.
Run 63 @1000 had collapsed to a 2-token note and I registered a `len_mean >= 9` clause in its successor's gate
on the assumption that this was surrender. The held-out probe falsified it the same hour: that 2-token note
reads **own - filler +1.81 +- 0.415 at o=-7, PASS at 4.4 s.e.** The length gate would have stopped a successor
that was winning, and it was removed before the successor reached step 20. (The run still failed, but on the
right instrument: against the init at the SAME offset, +1.81 vs +1.41, i.e. **+0.39 +- 0.97 -- it bought
nothing.** That, not the note's length, is the reason to stop a run.) Related: G84 already retired `len_mean`
and rollout spread as stop axes on the run-59 lineage; this extends the ban to held-out gates.

## G90 (2026-09-30). `keepalive_think.sh` leaks one `plot_gain.py` daemon per run switch.
It starts a plot daemon for whatever run is named in `.watchdog_run` but never reaps the previous run's, and the
daemon keeps its parent `keepalive_think.sh` shell alive with it. Four had accumulated by this tick (runs
60/61/62/63), each redrawing a stopped run's `gain.png` every 5 minutes. Harmless but it competes for CPU with
the host-side samplers and the CPU probes. Kill them by numeric PID at a run switch (`pgrep -f plot_gain.py` in
ONE call, `kill <pids>` in the NEXT -- gotcha G20), or fix the script to kill daemons whose run name is not the
current one.

## G91 (2026-09-30). THE SLOT CHANNEL DECAYS ON BOTH SIDES: read `span - filler`, not just `own - filler`.
Runs 60-64 were all diagnosed as "the writer writes worse notes" and each successor changed a writer- or
reader-training flag to fix the writer. Run 64 shows that frame is wrong. At o=-15, the same probe on the same
132 unseen-FineMath val anchors, each model scored against ITS OWN filler:
```
                    own-filler        span-filler   own/span   own-filler / clean
  init  22800       +6.93 +- 0.85       +15.65        0.45            39%
  run 64 @200       +2.61 +- 0.56       +10.07        0.29            22%
  run 64 @400       +1.84 +- 0.50        +9.01        0.23            16%
```
**`span - filler` is the reader's sensitivity to a PERFECT note -- a quantity the writer cannot influence at
all -- and it falls by a third in 400 steps.** So at least part of every "the note got worse" verdict in this
lineage is the reader becoming deaf to the slot, which no writer-side lever can repair. `own / span` falls too
(0.45 -> 0.23), so the writer is also losing ground; both are real, and only the pair of numbers separates them.
**Always report `span - filler` beside `own - filler`.** If `span - filler` holds and `own/span` falls, it is a
writer problem; if `span - filler` falls, it is the reader, and a writer lever cannot fix it.
Second, quote the margin as a **FRACTION of `clean`** as well as in nats. `clean` (full context minus silence,
the cost of the blinding) is itself shrinking as the reader learns to predict across the hole -- 17.57 -> 11.82
-> 11.43 here -- so the recoverable headroom a note is competing for shrinks with it, and a flat nat margin is
actually an improving fraction. Both forms agree in run 64 (39% -> 22% -> 16%), which is what makes its verdict safe.
Third, a METHOD note that generalised: run 64's G0 failed at step 200 (+2.61 against a +3.5 bar) and the
temptation was to argue the bar was too strict, since the same run had just passed G1 @400 decisively. **A
second probe on the step-400 checkpoint at the SAME offset settled it in 9 minutes of idle CPU** (+1.84, i.e.
decaying) and turned a borderline call into a confirmed one. When a level gate fails narrowly and the run has
another archived checkpoint, probe the TREND before arguing about the level -- the trend is the question the
level was a proxy for. (It also matters that the step-400 checkpoint was still an o=-15 model: the schedule
switches to -13 AT step 400, so no offset change confounded the comparison, per G88.)

## G92 (2026-09-30). PROBE THE EARLIEST CHECKPOINT: "adaptation shock" and "ongoing erosion" look identical at step 200 and have OPPOSITE successors.
Run 64's held-out margin fell +6.93 (init) -> +2.61 (@200) -> +1.84 (@400) and the pre-registered successor, written
into run 65's config as its G0-FAIL branch, was `--rl-w 0.25`: the decay "spans exactly the RL ramp"
(`--rl-start 100 --rl-ramp 200`, full weight at 300), so REINFORCE was the named suspect. **Run 65 carries the
identical RL schedule, and one extra probe killed that hypothesis for 9 minutes of otherwise-idle CPU.** Probing
run 65's step-**100** checkpoint -- same offset o=-15, same 132 unseen-FineMath val anchors -- gives:
```
                    own-filler       span-filler   own/span   clean    own-fil/clean   span-fil/clean
  init  22800       +6.93 +- 0.85      +15.65        0.449     17.57       39.4%           89.0%
  run 65 @100       +4.06 +- 0.61      +11.87        0.339     13.49       30.1%           88.0%
  run 65 @200       +3.84 +- 0.59      +11.67        0.360     13.83       27.8%           84.4%
  run 65 @400       +4.44 +- 0.59      +11.19        0.461     15.07       29.5%           74.3%
  (run 64 @200      +2.61 +- 0.56      +10.07        0.287     11.82       22.1%           85.2%)
  (run 64 @400      +1.84 +- 0.50       +9.02        0.231     11.43       16.1%           78.9%)
```
**At step 100 the RL weight is exactly ZERO (`--rl-start 100`), and 93% of the total drop from the init has
already happened** (-2.87 of -3.09). After RL turns on, the margin RISES (+4.06 -> +4.44) and `own/span` climbs
0.339 -> 0.461. REINFORCE is not what erodes the note channel. What the first 100 steps are is **one-off
adaptation shock** -- the cost of converting a vanilla next-token model to the block-insertion format under
warmup -- and its successor is nothing like the successor to erosion: shock says change the STARTING POINT
(warmup, lr, curriculum, or accept the one-time cost), erosion says change the OBJECTIVE TERM that is on while
the margin falls. **A level read at step 200 cannot tell them apart. Probe step 100 (or the earliest checkpoint
that survives `--keep-checkpoints`) before naming an objective term as the culprit.** This generalises G91's
"probe the trend before arguing about a level": the trend needs its FIRST point, not just two late ones.
Practical note: `--keep-checkpoints 3` had ~15 minutes left on run 65's step-100 checkpoint when this was read.
If a run's early checkpoint matters, copy it out on the tick that launches the run, not the tick that wants it.
SECOND FINDING, and it is why run 65 is the first run in the lineage that is working: **`own/span` at step 400
is 0.461, ABOVE the untrained init's 0.449.** The writer's own note now captures a larger share of an oracle
span's value than the untrained model's note did -- the first writer measure in runs 60-65 to beat the init.
THIRD, a correction to how G91 should be applied: **`span-filler / clean` keeps falling in run 65 (89.0 -> 88.0
-> 84.4 -> 74.3%) while `own/span` RISES.** In run 64 the two fell together, which is what made "the reader is
going deaf" look like the whole story. They have now decoupled. So reader slot-insensitivity is real and still
running, but it does NOT by itself determine whether the writer is learning -- report it as headroom being lost,
and steer on `own/span` and `own-filler`. (It also means the ceiling is still moving away from us: if the reader
held 89% of `clean`, run 65's margin would be materially larger than +4.44.)

## G93 (2026-09-30). A `--span-sched` CHECKPOINT BELONGS TO THE *PREVIOUS* STAGE, so a gate written at "the schedule's offset at step N" silently becomes a TRANSFER test.
`--span-sched "0:-15,400:-13,600:-11,800:-9,1000:-7,..."` switches the offset **at** the listed step. So the
step-1000 checkpoint has trained 200 steps at o=-9 and **zero** at o=-7. Run 65's config used this fact
CORRECTLY for G1b -- "the step-400 checkpoint is still an o=-15 model: the schedule switches to -13 AT step 400,
so it has had ~0 steps at the new offset -- this is what made run 64's 200-vs-400 comparison clean" -- and then,
forty lines later, got it BACKWARDS for G2: "G2 @1000 -- held-out, o=-7 (the schedule's offset at step 1000)".
Both clauses were written in the same sitting by the same tick. The error is easy because the schedule entry
`1000:-7` reads like a label for step 1000 when it is a boundary.
COST IF UNCAUGHT: G2 would have compared weights to the init at an offset the weights had never seen, and a
FAIL would have been read as "the margin dies as the span slides" -- triggering the registered successor (a
slower slide, a ~9-hour run) on a measurement that never tested the claim. This was caught at step 746, before
the checkpoint existed, which is the only time a gate can be amended honestly.
RULE: **name a checkpoint's stage from the schedule entry strictly BELOW its step, never the one equal to it.**
When a gate must live at a stage boundary, register both readings before the data: the transfer test (next
stage's offset) AND the trained-offset test (this stage's offset, plus the same-offset test one stage later),
each against the init at its OWN offset, and say which one may stop the run. For run 65: G2 stayed the
registered gate at o=-7 with an arithmetic bar (+3.54 = init +1.41 + 2 s.e. of the difference), but a G2 fail
alone cannot stop the run -- step 1000 @ o=-9 (bar +3.90) and step 1200 @ o=-7 (bar +3.54) must fail too.
SECOND FINDING FROM THE SAME TICK, and it corrects the intuition five runs were steered by. With the init now
probed at every offset in the schedule (`agent/init_refs.sh`), the slide does NOT get harder because an oracle
note gets weaker -- **the oracle note gets STRONGER as the span moves into the future** (`span-filler` +15.65 at
o=-15 rising monotonically to +18.73 at o=-7). What collapses is the untrained model's ability to produce one:
`own/span` 0.449 -> 0.323 -> 0.264 -> 0.108 -> 0.071 -> -0.403. Knowing the future is worth MORE; guessing it
is what is hard. Two consequences: (a) the init's bar is cheap from o=-9 on, so "beats the init" gets easier
down the slide exactly where the science gets harder -- quote `own/span` too, or a pass will be oversold;
(b) restate a nat bar in oracle-share before trusting it. G2's +3.54 at o=-7 is `own/span` ~= 0.185, and run 65
was holding 0.468 -- which turned a gate that reads intimidating into one the run should clear with room, and
that is a prediction worth recording before the number lands rather than after.
THIRD, an operational bug fixed the same tick: `archive_ckpts.sh` used
`[ -e "$AR/$b" ] || ln ... && echo "... archived $b" >> archive.log`. Shell precedence makes that
`(A || B) && C`, so when the file already existed A was true and the echo ran anyway -- the cron job logged
"archived step_0000500.pt" every 3 minutes for 70 minutes. The hard-link itself was correctly skipped, so the
damage was purely a log that invited a later tick to diagnose archive churn that was not happening. Wrapped in
`if [ ! -e ]; then ... fi`; verified in an isolated tmp tree (links the right files once, skips `*optim*`,
silent on re-run). **A `||`-guard followed by `&&` is not a guard.**
CORRECTION TO THE PARAGRAPH ABOVE, 25 MINUTES LATER AND IN THE SAME TICK (kept rather than edited away, because
the whole point of recording a forecast is that it can fail). The step-800 probe at o=-11 read own-filler
**+2.28 +- 0.53** against +4.39 at step 600 (o=-13), and `own/span` **0.468 -> 0.331**. So "run 65 should clear
G2 with room" was wrong within the hour. **The mistake was treating `own/span` as a property of the writer that
carries across stages. It is not: it is RE-EARNED at every offset**, and run 65 lost 29% of it crossing one
stage where the init loses 18%. Restating a nat bar in oracle share (the useful half of the finding) remains
right; using an oracle share measured at one offset to predict the next is what fails. The extrapolated share
at o=-7 is ~0.167 against the 0.185 bar, i.e. the gate is borderline, not comfortable.
THE DECOMPOSITION IS THE ACTIONABLE PART, and it points the opposite way from run 64: `span-filler / clean` was
FLAT across this stage (72.9% -> 75.0%) while `own/span` fell, so the reader did not go deafer -- the WRITER lost
ground as the span slid. Run 64 failed with both falling together. Same symptom, different half of the machine,
different successor: run 64's diagnosis was a reader lever, this one is the curriculum.
AND A THIRD READING, on the SHAPE of the decay, which is the thing to design the successor from: the loss is not
spread over training time, it is paid **at** the stage change and it grows with the jump -- crossing o=-15 -> -13
cost ~0 (+4.44 -> +4.39), crossing -13 -> -11 cost -2.10 +- 0.80. The span is x[n+o..n+o+14], so those two
crossings take the number of FUTURE tokens inside it from 0 to 2 to 4. **If the cost tracks the jump rather than
the elapsed steps, the lever is the slide's STEP SIZE (+1 per stage), not its TIME per stage (+2 every 400),**
which is what the decision tree had registered. A run spent on the wrong one of those two is ~9 hours.

## G94 (2026-09-30, 12:45 headless tick). **THE PROGRAMME'S PRIMARY BAR, `own_minus_filler`, IS WINNABLE BY MAKING THE *FILLER* WORSE -- AND RUN 65 WON IT THAT WAY, SCORING A `PASS` FIVE MINUTES AFTER IT WAS STOPPED FOR REGISTER COLLAPSE. THE GUARD AGAINST THIS WAS ALREADY IN THE PROBE, IN A COLUMN THE RUNBOOK HAD MISREAD FOR SIX RUNS.**

**The two readings of the same file.** Run 65 step 1200, `probe_span.py` at o=-7, 132 unseen-FineMath val anchors:
```
  own +0.15 +- 0.16 | shuf +0.16 | span +10.63 | filler -4.24 | clean +7.30
  own_minus_filler +4.39 +- 0.30  -> "PASS" (bar +3.54)      own_minus_shuf -0.02 +- 0.03
```
The note is worth nothing over silence (+0.15 +- 0.16) and nothing over ANOTHER ANCHOR'S note
(-0.02 +- 0.03). The entire +4.39 is the 10-SPACE filler having become 4.24 nats WORSE than silence for
these weights. At the same time `reg%` was 50.7 (bar 15), rollout `spread` 0.51 (bar 0.6), `g_sd` 0.06 and
`of   </T` was 55% of all sampled prefixes: a constant register. **The derail rules saw it; the gate did not.**

**Why the ruler moved, and why it moved most in the run that was judged by it.** The `filler` column of every
span probe on disk: the INIT reads ~0 at EVERY offset (+0.17, +0.05, -0.15, +0.27, -0.09, -0.34, +0.08, +0.77,
+1.67 for o=-15..+1), which is exactly why own_minus_filler looked like a clean instrument when G88 built the
untrained bars from it. Run 65's filler: **+0.56 (@200) -> +1.34 (@400) -> +2.88 (@600) -> +2.51 (@800) ->
-1.48 (@1000,o=-9) -> -3.14 (@1000,o=-7) -> -4.24 (@1200,o=-7).** A 7-nat swing, inside every gate number.
It cut BOTH ways: at o=-13 run 65's own was +7.26 +- 0.65 against the init's +5.53 +- 0.82 -- run 65 was
BETTER -- while own_minus_filler said +4.39 vs +5.48 and three ticks wrote down "1.09 +- 1.05 BELOW the init".
The mechanism is run 65's own single flag: `--pause-frac 0.15 -> 0` removed every filler-in-slot anchor from
training, so the model's response to a filler slot is trained on nothing and free to drift. **The flag that
made run 65 look like the first working run of the lineage is the flag that broke the ruler used to say so.**

**The guard was already there and the runbook had it backwards.** `probe_span.py:56` builds `shuf` as
`np.roll(own, 1, axis=0)` -- **another anchor's own note from the same sequence**, i.e. the MISPLACED-NOTE
control. `own_minus_shuf` therefore answers "is this note about THIS continuation, or would any note of mine
do?": paired, cancels the block-insertion price, immune to filler drift AND to the silence baseline improving
(which it did enormously -- `clean` fell 18.2 -> 7.3 as the reader learned to predict across the hole).
SUPERVISOR.md rule (i) said to distrust "the note's own shuffle (ob0 > 0 proves ORDER-STRUCTURE, not
usefulness)" and applied that to this column. **But `ob0 = g_r0 - g_shuf0` is a TRAINING metric that permutes
tokens WITHIN one note (think_train.py:905-925). Two different controls; the caveat belonged only to the
training one, and the probe's is the stronger of the two.** Six runs were gated on the weaker one.

**What the corrected instrument says about runs 60-65: there is no win anywhere.** `own_minus_shuf` at each
checkpoint's TRAINED offset (G93):
```
   offset      init              run 65
    -15    +6.56 +- 0.83   @200 +3.92 +- 0.71 , @400 +4.72 +- 0.70
    -13    +5.09 +- 0.75   @600 +4.62 +- 0.61
    -11    +6.21 +- 1.17   @800 +1.93 +- 0.55
     -9    +3.75 +- 1.14   @1000 +0.18 +- 0.13
     -7    +4.32 +- 1.13   @1200 -0.02 +- 0.03
     -5/-3/-1   +3.62 / +2.39 / +5.30      (never reached)
     +1    +0.53 +- 0.55
```
Below the untrained model at every offset it reached, falling monotonically to zero. **Training did not add
anchor-specific information to the note; it removed ~2 nats of what the untrained model's note already had,
inside the first 200 steps, at the EASIEST offset, while G0 was being recorded as a PASS.** That last fact
also retires the curriculum hypothesis as the NEXT lever: the loss begins where no curriculum applies.

**THE RULES THIS BUYS.**
 1. **Never gate on a difference against a baseline the run is free to move.** Before trusting any paired bar,
    print the baseline's own level and check it against the reference model's. If the baseline has moved more
    than ~1 nat, the bar is void, not merely noisy.
 2. **Quote the decomposition with every verdict** -- `own`, `own_minus_shuf`, `filler`, `span`, `clean` -- not
    the single paired number. `agent/armed_probes_r66.sh` now prints that line into `runs/probes/armed.log`
    automatically so no tick can report the headline alone.
 3. **VALIDITY CLAUSE for any future gate: a reading whose `own` s.e. across anchors is < 0.25, or whose
    `own_minus_shuf` is within 2 s.e. of zero, is a CONSTANT REGISTER and cannot pass, whatever the paired
    number says.** Run 65 @1200 fails both (s.e. 0.16; -0.02 +- 0.03) and "passed" at +4.39.
 4. **When a derail rule and a gate disagree, the derail rule is the one measuring the failure mode.** reg%
    and rollout spread are computed on the policy itself and cannot be gamed by a baseline; a gate written as
    a difference can. Run 65 was stopped at 12:14 on 3b and "passed" its gate at 12:19:56.

## G95 (2026-09-30, 14:07 headless tick). **A FLAG THAT ENTERS THE REWARD AS A PER-ROLLOUT-CONSTANT IS A NO-OP UNDER `--baseline loo`. RUN 66'S STEP-200 WEIGHTS ARE BIT-IDENTICAL TO RUN 65'S -- 311/311 TENSORS -- AND ITS "REPLICATION" GATE R1 SCORED A PASS ON RUN 65'S OWN MODEL.**

Run 66 was launched at 12:23 as run 65 + one flag, `--len-cost 0.02 -> 0.0`. At 13:30 its armed R1 probe at
step 200 reported `own-filler +3.84 +- 0.59, own-shuf +3.92 +- 0.71` -- **identical to run 65's step-200 probe
in every one of five quantities to four decimal places.** The JSONs are byte-equal apart from the `ckpt` field.
That is not a replication and it is not a coincidence:

```
  .venv/bin/python agent/ckpt_same.py checkpoints/think65_nofiller_archive/step_0000200.pt \
                                      checkpoints/think66_nolencost_archive/step_0000200.pt
  -> ckpt_same: IDENTICAL (311 tensors)
```
(The files' md5s differ; the difference is entirely in the `args` metadata blob. Compare MODEL TENSORS, never
file hashes, when asking whether two runs have diverged.)

**Mechanism, and it is exact arithmetic, not an approximation.** `think_train.py:883-889`:
```
  REW  = G - len_cost * S0                       # S0 = per-rollout stop position
  PAIR = (REW.sum(0)[None,:] - REW) / (RO-1)     # leave-one-out baseline
  ADV_R = REW - PAIR
```
`ADV_R[r] = REW[r] - mean_{j!=r} REW[j]`, so **any term that takes the same value for every rollout of an
anchor subtracts out exactly.** While every rollout is at `len 15`, `len_cost * S0` is such a term. Verified
numerically to 8.9e-16. `--len-cost` appears nowhere else in the trainer except the `--baseline greedy` branch
(l.933), which this lineage does not use. So **`--len-cost` is identically inert until the first sub-15
rollout** -- run 65 step 514 -- and run 66 is a bit-exact re-execution of run 65 up to that step.

Three things follow.
 1. **R1 @200 is VOID, not a pass**, and any gate on a checkpoint before the divergence step is void. The
    duplicate-checkpoint guard is now in `agent/armed_probes_r66.sh` v2: it runs `agent/ckpt_same.py` against
    run 65's archive before each probe and, on IDENTICAL, copies run 65's JSON and logs VOID instead of
    spending ~18% of the live run's step rate re-measuring the same weights. The @400 probe was suppressed
    this way.
 2. **It is a cleaner experiment than it was designed to be, by accident.** Run 66 is identical to run 65
    right up to the step the length price first acts, and differs only from there. The step at which
    `ckpt_same.py` first reports DIFFER is the sharpest available H1.
 3. **BEFORE LAUNCHING A ONE-FLAG RUN, ASK WHERE THE FLAG ENTERS THE ADVANTAGE.** If it enters as a
    per-rollout constant, the run is a duplicate and the device time is spent for nothing. The 12:07 tick
    checked that the FLAGS array differed in exactly one line; it did not check that the line could move a
    gradient.

## G96 (2026-09-30, 14:07 headless tick). **THE SAME CANCELLATION KILLS THE REGISTERED NEXT HYPOTHESIS (00S.0 ITEM 7c) BEFORE IT COSTS A RUN -- AND EXPOSES A REAL DEFECT IT WAS POINTING AT: THE ADVANTAGE IS SCALE-INVARIANT, SO REINFORCE PUSHES AS HARD ON NOISE AS ON SIGNAL AND CANNOT STAND DOWN WHEN THE REWARD GOES SILENT.**

**(a) 7(c) is a no-op.** It proposed scoring the reward against a FILLER slot instead of silence, on the
argument that "the reward is own-vs-silence, which at hard offsets is ~0 or negative for every note, so the
between-rollout content differences are a rounding error on the block price." `_score()` (l.851-860) returns
`lw(c) - _base`, and `_base` is the SAME vector for every rollout of an anchor. By G95's identity it cancels
in `ADV_R` exactly (verified: max |ADV difference| 1.8e-15 between a silence and a filler baseline). The
absolute gain level reaches the gradient nowhere else: `elig` (l.935) is logging only (`n_elig`, the dump
file), never a loss weight. **The block price was never in the reward. The reward has always been a pure
between-rollout content comparison, and the G52 "own-vs-silence" framing is a statement about the METRIC `g`,
not about the objective.** Item 7(c) is struck.

**(b) What is actually wrong is one line further on** (l.989-992):
```
  mu  = mean(adv_paired);  sig = sd(adv_paired)
  A   = clip((adv_paired - mu) / (sig + 1e-6), +-reward_clip) * rlf
```
The advantage is **z-scored with no floor, so it is scale-invariant**: `mean|A|` is ~0.8 whether the true
between-rollout spread is 2.0 nats or 0.006. At this lineage's ACTUAL `--reward-clip 7.1` (not the 3.0
default) the clip essentially never binds, so `mean|A|` is not merely "about the same", it is pinned at
`sqrt(2/pi) = 0.798` for ANY spread:
```
   adv sd  2.180 -> mean|A| 0.790  (0% clipped)     adv sd  0.060 -> mean|A| 0.800  (0% clipped)
   adv sd  0.500 -> mean|A| 0.770  (0% clipped)     adv sd  0.006 -> mean|A| 0.812  (0% clipped)
```
**The trainer is by construction incapable of noticing that its reward has gone silent.** It renormalises
whatever is left -- at `g_sd 0.06`, numerical noise -- back to full scale and keeps applying it at `rl_w 0.5`.
This is a mechanism for the absorbing state that needs NO length price, and the evidence for it is already on
disk: a dead gradient would leave the writer FROZEN, but run 65's writer went on moving hard while `g_sd` was
0.06, converging onto `of   </T` at 553 of ~840 prefixes (reg% 50.7). A frozen signal with a moving policy is
the signature of full-scale updates on noise.

**(c) The fix, implemented and tested this tick, default off:** `--adv-sig-floor` (default 0.0), divisor
`max(sig, floor)`. At floor 0 the expression is bit-identical to the old one (`sig` is a sqrt, so
`max(sig,0.0) == sig`; max |A difference| 0.0e+00 over run 65's whole `g_sd` trajectory). At floor 0.5 the
healthy phase is untouched and the RL term fades out exactly over the collapse window:
at `--reward-clip 7.1`, over run 65's own `g_sd` trajectory (using adv sd ~ 1.155 * g_sd, see the caveat):
```
   g_sd 2.18 (adv sd 2.52) -> mean|A| floor0 0.752  floor0.5 0.752
   g_sd 1.08 (adv sd 1.25) -> mean|A| floor0 0.828  floor0.5 0.828
   g_sd 0.33 (adv sd 0.38) -> mean|A| floor0 0.768  floor0.5 **0.472**
   g_sd 0.06 (adv sd 0.07) -> mean|A| floor0 0.782  floor0.5 **0.103**
```
`py_compile` OK; `USE_TORCH_XLA=0 python test_rollouts.py` passes in full (rollout + soft-gate + curriculum +
mixed-anchor), both before and after the instrumentation below. **`sig` was not observable anywhere**, so the
same patch logs `adv_sig` (the divisor's raw input) and `adv_absA` (the mean |A| actually applied) to
`metrics.jsonl` -- without them a floor run cannot be told from a no-floor run. CAVEAT to carry into any config that sets it: **the floor is on the sd of `adv_paired`
(a leave-one-out difference), not on `g_sd`**; for RO=4 the former is the larger by about sqrt(RO/(RO-1)) =
1.15, so a floor chosen off a `g_sd` reading is ~15% tighter than intended. Pick the number from the
`adv_paired` sd or state the conversion.

**(d) The general lesson, which is G95's and is now twice earned in one day.** Two separate hypotheses -- the
length price and the reward baseline -- were both argued from the ABSOLUTE level of the reward, and the
objective only ever sees DIFFERENCES between rollouts of the same anchor. **Before registering a hypothesis
about the reward, write down `ADV_R[r] = REW[r] - mean_{j!=r} REW[j]` and check that the proposed change
survives the subtraction.** Anything that does not is a metric change, not an experiment.

## G97 (2026-09-30 16:07 tick) -- A ONE-FLAG RUN CAN DIVERGE IN WEIGHTS AND STILL NOT MOVE THE VARIABLE THE FLAG WAS CHOSEN TO MOVE. CHECK THE OVERLAY, NOT THE DIVERGENCE FLAG.

G95 gave the guard for the opposite error (a flag that is a per-rollout constant leaves the weights
bit-identical, so nothing moves at all). This is the next failure mode along, and run 66 is the instance.

**The fact, and it is not in dispute.** Run 66 = run 65 + `--len-cost 0.02 -> 0.0`, one flag, same init, same
seed, same schedule. Their weights first *can* differ at step 514 (run 65's first sub-15 rollout); the
duplicate guard independently confirmed DIFFER at step 600 and again at step 800. Matched 50-step bands of
`len_mean`, run 65 / run 66:
```
   500-549  15.00 / 15.00       650-699  14.90 / 14.86
   550-599  14.99 / 15.00       700-749  14.57 / 14.60
   600-649  14.96 / 14.97       750-799  14.27 / 14.21
```
Maximum discrepancy over 285 post-divergence steps: **0.06 tokens.** `len_spec` (the greedy row) is exactly
15.00 in both for all 800 steps. **Removing the only length pressure in the objective changed the note length
by less than the band-to-band noise.** What the flag did change is `g_r0`, by +0.3 to +0.8 over 550-799
(650-699: +5.67 vs +4.85) -- it bought gain, not length.

**Why this is a distinct trap from G95.** The 14:07 guard asks "can this flag move a gradient at all?" and run
66 passes it from step 514 on: the weights genuinely diverge, the runs are genuinely different, every gate
after 514 is a genuine measurement. What the guard cannot tell you is whether the flag moves the SPECIFIC
quantity the hypothesis was about. A flag can be live and still be irrelevant to its own hypothesis.

**The rule.** For a one-flag run whose hypothesis names an observable (here: note length), the read is the
MATCHED-BAND OVERLAY of that observable against the predecessor, not the run's absolute trajectory and not
the fact of divergence. Run it as soon as ~100 post-divergence steps exist -- it is free, it is on disk, and
here it delivered the mechanism verdict ~3 hours before the pre-registered gate could.

**Status of the verdict itself, stated honestly.** The pre-registered H1 band is steps 950-999 (run 65: 6.96),
and run 65's cliff is AT the step-800 stage change (14.27 -> 10.58 -> 7.27 over 800-899), which run 66 had
only just crossed when this was written (step 802, len 12.7 instantaneous). So the overlay above is a strong
PREVIEW pointing at FALSIFIED, not the reading. Recorded now because the overlay is a method that should have
been run at step 620, and because writing a forecast down before its data is the habit this programme keeps
having to re-learn. If run 66 holds >= 12.0 through 950-999 after tracking run 65 to 0.06 tokens for 285
steps, that is a surprise the record should show me failing to predict.

## G98 (2026-09-30 18:07 headless tick). **THE LENGTH PRICE IS NOT WHAT SHORTENS THE NOTE -- IT IS WHAT MAKES THE SHORTENING *ABSORBING*. AND THE MATCHED-BAND OVERLAY OF G97, RUN BEFORE THE REGIME CHANGE IT WAS ABOUT, READ THE OPPOSITE OF THE TRUTH AND I PUBLISHED IT AS "POINTS HARD AT FALSIFIED".**

**The retraction first.** Six hours ago (G97, 16:07) I read run 65 vs run 66 on matched 50-step `len_mean`
bands, found them identical to 0.06 tokens over 285 post-divergence steps, and wrote that the run's one flag
(`--len-cost 0.02 -> 0.0`) was "irrelevant to its own hypothesis", with H1 "pointing hard at FALSIFIED". The
overlay stopped at step 812. The pre-registered H1 band is 950-999. Here is the whole table now that it exists
(run 65 / run 66, 50-step `len_mean`, weights first differ at step 514):

```
   500-549  15.00 / 15.00      800-849  10.58 / 11.63     1050-1099  3.90 /  9.77
   550-599  14.99 / 15.00      850-899   7.27 / 11.56     1100-1149  3.05 /  6.47
   600-649  14.96 / 14.97      900-949   8.20 / 10.25     1150-1199  3.17 /  9.02
   650-699  14.90 / 14.86      950-999   5.71 / 11.17     1200-1242   ---- /  8.00
   700-749  14.57 / 14.60
   750-799  14.27 / 14.21      H1 = len_mean(950-999) = 11.17 against a <8.0 falsify / >=12.0 confirm
                               bar: the REGISTERED DEAD BAND, and 5.5 tokens above the predecessor.
```

The two runs are the same curve for 285 steps and then separate by 5.5 tokens inside the next 50. **The
preview was not early by a little; it was taken on the wrong side of the only regime change that mattered.**

**What the flag actually buys, which is not length-in-the-mean.** Both runs shorten. The difference is the
SHAPE: run 65 falls monotonely to 3.0-3.2 -- its `--min-think 3` floor -- and never leaves. Run 66 falls,
rises, falls, rises: 11.2, 7.8, 9.8, 6.5, 9.0, 8.0, with no trend. The gradient-side reading is far louder
than the length one, and it is the one to quote:

```
   g_sd    950-999   1000-1049  1050-1099  1100-1149  1150-1199     (3b derail bar: 0.60)
   run 65    0.69      0.33       0.32       0.05       0.07        <- writer dead, absorbing state
   run 66    1.84      1.41       2.08       1.22       1.91        <- exploration fully intact
```

**So the mechanism is one step back from where run 66's hypothesis put it.** The note shortens at hard
offsets whether or not there is a length price, because content gain falls when the slot moves toward the
anchor -- that part of run 65's story was never about `--len-cost`, and H1's dead-band verdict is exactly
right to refuse both stories. What the length price adds is a floor-seeking *drift* that is always in the
same direction: once the rollouts are short, the price rewards the short ones again, spread collapses, and
with no spread there is no exploration to climb back out. **A monotone ratchet, not a shortening.** Remove
it and the same downward pressure becomes a fluctuating equilibrium the writer can leave -- which is why run
66 is still writing 8-token notes with `g_sd` 1.9 at step 1200 where run 65 was writing 3-token notes with
`g_sd` 0.07, and why run 66's held-out probes kept improving (R4 @1200 o=-7 `own_minus_shuf` +4.75 +- 0.63
against run 65's -0.02 +- 0.03 at the identical step and offset -- 7.6 s.e.).

**Note what this does NOT say.** It does not say the length price caused run 65's collapse on its own: G96's
scale-invariant advantage (`A = clip((adv-mu)/(sig+1e-6), +-3)`, no floor) is the thing that keeps REINFORCE
pushing at full strength after the reward has gone silent, and it is what turns any dip in spread into a
one-way door. The length price is the *bias* that picks the direction; the missing sd floor is what makes
the door shut. Run 66 removed the first and the door never shut. That is consistent with both, and run 67
(`--adv-sig-floor 0.5`, registered 14:07) is still the test of the second -- but its own null clause now
applies: run 66's `adv_sig` never went silent (min `g_sd` 0.294, 3 steps of 727 below the 0.5-equivalent,
last-50 ~2.28), so there is nothing in run 66 for the floor to have bitten on.

**THE METHOD RULE, amending G97 rather than replacing it.** G97's overlay is still the right instrument and
it is still free. The amendment is *where it has to reach*: **a matched-band overlay is only readable once it
spans the predecessor's known failure region.** Run 65's cliff is AT its step-800 stage change; an overlay
ending at 812 was comparing the two runs only where neither had been stressed yet, which is precisely where a
flag that changes the RESPONSE TO stress cannot show. The general form: **before running the overlay, write
down the step at which the predecessor did the thing you are trying to prevent, and do not report a verdict
from bands that end before it.** If the live run has not reached that step, the honest output is "no reading
yet", not a preview with a direction attached. I attached a direction, and it was the wrong one.

**And a bookkeeping error found while checking this, worth one line.** AMENDMENT A registered H1 as
`len_mean(950-999)` for run 66 but quoted run 65's comparator as "6.96", which is run 65's *900-999* band;
its 950-999 band is 5.71. Nothing turns on it here (run 66 reads 11.17 on the 50-step band and 10.71 on the
100-step band -- dead band either way), but a bar and its comparator computed over different windows is how a
gate silently becomes a different gate. Compute both sides of a bar with the same code path.

## G99 (2026-09-30 20:07 headless tick). **RUN 66 BOUGHT THE NOTE'S *FORM*, NOT ITS *CONTENT*: THE READER LEARNED TO READ A WRITER-SHAPED OBJECT IN THE SLOT (+3.7 NATS FOR *ANY* NOTE, INCLUDING ANOTHER ANCHOR'S), WHILE THE ANCHOR-SPECIFIC PART IS STATISTICALLY UNCHANGED FROM THE UNTRAINED MODEL AT EVERY OFFSET. AND THIS IS *WHY* `own_minus_filler` IS A BROKEN RULER -- G94 SAW THE SYMPTOM; HERE IS THE CAUSE.**

**The finding.** Run 66 versus the untrained init, same 132 unseen-FineMath anchors, every condition
measured in nats **against silence** (an all-`T_CLOSE` slot) within its own model:

```
              o=-7                      o=-5                      o=-3
          r66    init    diff       r66    init    diff       r66    init    diff
 own     +2.47  +1.32  +1.15(1.1)  +0.78  -0.91  +1.68(1.6)  +0.63  -3.93  +4.56(3.8)
 shuf    -2.28  -3.00  +0.72(0.8)  -0.67  -4.53  +3.85(3.9)  -2.67  -6.32  +3.65(3.4)
 filler  -1.91  -0.09  -1.82(3.4)  -3.16  -0.34  -2.82(5.2)  -4.30  +0.08  -4.38(7.1)
 span   +12.45 +18.64  -6.19(4.9) +12.68 +18.95  -6.27(5.2) +13.53 +18.17  -4.64(3.9)
 clean   +9.34 +17.97  -8.63(6.5)  +7.79 +16.89  -9.11(8.0)  +6.10 +15.72  -9.63(9.8)
 own-shuf +4.75 +4.32  +0.43(0.3)  +1.45  +3.62  -2.17(1.7)  +3.30  +2.39  +0.91(0.6)
```
(s.e. of each difference in brackets, as multiples.) **`own` and `shuf` both go UP by roughly the
same amount. `filler`, `span` and `clean` all go DOWN, hard and monotonically. `own_minus_shuf` --
the only anchor-specific quantity, and the programme's bar -- does not move at any offset.**

**What that decomposition says, in one line: the model learned what a note LOOKS like, and learned
nothing about what this particular note SAYS.** A misplaced note (`shuf`, another anchor's own note)
gains almost exactly as much as the right one. Whitespace in the slot goes from free (+0.08 at init)
to a 4.3-nat penalty. The model has become *slot-dependent*: it now expects a writer-shaped object
there and is hurt by anything else.

**This is the correct frame for "run 66 is tied with the untrained model", and it is worse news than
a tie sounds.** A model that learns only the note's format is *predicted* to score exactly a tie on
`own_minus_shuf`, because format is common to `own` and `shuf` and cancels. Run 65 scored BELOW the
init because its writer died; run 66's achievement is **"did not break"**, not "learned to think".
Four consecutive ties is not four near-misses accumulating toward a win -- it is the signature of a
run that never engaged the anchor-specific channel at all.

**`clean` is the other half and nobody was reading it.** `clean` (the score with the window VISIBLE,
vs silence) fell from ~+16-18 to ~+6-9, by 6.5-9.8 s.e. That is the *ceiling*: the total worth of the
hidden information. **The run is closing the gap from the wrong side -- making the hidden window
UNNECESSARY rather than making the note CARRY it.** A note that recovers 54% of a ceiling that
training shrank by 60% is not obviously better than one that recovers 15% of the full ceiling, and no
gate in this programme distinguishes the two. Add `clean` to every probe read from here.

### G99.1 -- THE RULER RULE, and the mechanism behind G94
Every condition in `probe_span.py` is stored as `(condition - base)` where `base` is the **silenced**
slot. `base` is on-distribution for the init and **off-distribution for any trained model**, because
training fills the slot at every step. So **no raw vs-silence column is legible across models** --
not `own`, not `span`, not `clean`. Only a **paired within-model difference** cancels `base`:
`own_minus_shuf = (own-base) - (shuf-base) = own - shuf`.

This retires half of a standing instruction. `agent/armed_probes_r66.sh`'s header names two verdicts,
"`own_minus_shuf` AND own-vs-silence". **The second one is not cross-model legible and must not be
quoted against the init.** (The script is being executed by live pid 1417155 and bash reads a script
incrementally, so it was NOT edited; the correction lives in SUPERVISOR.md 00S.0 and here.)

And it explains G94 rather than merely restating it. G94 found `own_minus_filler` swinging in both
directions and called it broken. The cause: **`filler` is not note-shaped, so its value is itself a
strong monotone function of training** -- run 66's filler runs +0.56, +2.91, +2.83, -0.01, -1.91,
-3.16, **-4.30** across steps 200->1600. `own_minus_filler = own_minus_shuf + (shuf - filler)`, and
that second term is pure training drift. Hence the swing, and hence the **standing prediction**: from
step 1000 on, `own_minus_filler` will OVERSTATE by an ever-growing margin. It already does -- 1400:
+3.94 vs +1.45; 1600: +4.92 vs +3.30.

**The general rule, for any future probe: a cross-model contrast is only legible between two
conditions that are BOTH on-distribution for BOTH models.** Two model-written notes qualify. A note
versus silence, versus whitespace, or versus an oracle copy does not.

## G100 (2026-09-30 22:07 headless tick). **RUN 66 IS STOPPED ON ITS OWN PRE-REGISTERED GATE, AND THE FIVE TIES HAVE A ONE-LINE CAUSE IN THE OBJECTIVE: `ADV_R[r] = REW[r] - mean_{j!=r} REW[j]` CANCELS EVERYTHING CONSTANT ACROSS ROLLOUTS BUT NOTHING GENERIC ACROSS *ANCHORS*, SO THE CHEAPEST GRADIENT AVAILABLE IS THE ONE THAT HELPS *ANY* ANCHOR. ALSO: RUN 66 OVERRAN ITS OWN STOP BY 375 STEPS BECAUSE A GATE READ EVERY 2 h IS NOT A GATE.**

**THE STOP.** R7 @1800 @ o=-1 landed at 20:43 with `own_minus_shuf` **+1.16 +- 0.41** against the
registered `< +2.30 -> STOP at 1800` bar (config.sh AMENDMENT E, written at step 1714 before the
checkpoint existed; comparator init +5.30 +- 1.41). Decomposed: own -0.24 | shuf -1.40 | filler -4.09
| span +13.31 | **clean +3.91**. Run 66 STOPPED at step 2182, 22:15. Two independent grounds agreed:
runbook 3b fired twice on the last block (reg% **33.5** against a bar of 15, rollout spread **0.26**
against 0.60) -- run 65's absorbing state, reached 900 steps later and **with no length price in the
objective at all**, which retires any remaining reading of G98 that made `--len-cost` the cause rather
than the accelerant.

**RUN 66'S FULL HELD-OUT LADDER, EIGHT OFFSETS, ON THE CORRECTED INSTRUMENT.** Each checkpoint at the
offset it was trained at, against the init at the SAME offset, 132 unseen-FineMath val anchors:
```
   step   o    run 66          init            diff (s.e.)     own    filler   clean
    600  -13  +4.32 +- 0.60   +5.09 +- 0.75   -0.77 (0.8)    +8.46   +2.91   18.41
    800  -11  +3.75 +- 0.58   +6.21 +- 1.17   -2.46 (1.9)    +5.60   +2.83   15.84
   1000   -9  +2.55 +- 0.43   +3.75 +- 1.14   -1.20 (1.0)    +2.09   -0.01   10.74
   1200   -7  +4.75 +- 0.63   +4.32 +- 1.13   +0.43 (0.3)    +2.47   -1.91    9.34
   1400   -5  +1.45 +- 0.28   +3.62 +- 1.21   -2.17 (1.7)    +0.78   -3.16    7.79
   1600   -3  +3.30 +- 0.61   +2.39 +- 1.35   +0.91 (0.6)    +0.63   -4.30    6.10
   1800   -1  +1.16 +- 0.41   +5.30 +- 1.41   -4.14 (2.8)    -0.24   -4.09    3.91
```
**Eight offsets, five ties and one 2.8-s.e. loss, no trend, and no win anywhere.** The two columns that
DO move monotonically are `filler` (+2.91 -> -4.09: the model becomes slot-dependent, hurt by anything
not note-shaped) and `clean` (18.41 -> **3.91**: the ceiling the note is supposed to carry shrinks by
79%). Neither is anchor-specific. **Run 66's achievement is "did not break for 1,800 steps"; run 65's
writer died at 1,205. That is a smaller claim than five gate PASSes look like, and it is the whole of
what the run bought.**

**THE CAUSE, AND IT IS ONE LINE OF THE OBJECTIVE.** The reward is
`REW[r] = G[r] - len_cost*S0[r]` (think_train.py:921) with `G[r]` = own-vs-silence-under-the-same-mask,
and the advantage is leave-one-out, `ADV_R[r] = REW[r] - mean_{j!=r} REW[j]`. LOO is doing real work:
it cancels every term CONSTANT ACROSS THE 4 ROLLOUTS -- the block-insertion price, the anchor's
intrinsic difficulty. **What it cannot cancel is a per-rollout quantity that is generic across
ANCHORS.** A note helps its anchor through two routes: (i) it says something about THIS continuation,
or (ii) it is a better *object to have in the slot* -- more note-shaped, more fluent, better register,
priming the reader to expect mathematical text. Route (ii) varies from rollout to rollout, so it
survives LOO and earns gradient, and **it is strictly cheaper to find than route (i).** Nothing in the
objective distinguishes them. Run 66 spent 1,800 steps buying route (ii), and the measurement says so
directly: `own` and `shuf` rose by the SAME ~+3.7 nats over the init (G99), i.e. a note about the WRONG
anchor gained as much as the right one, and `own_minus_shuf` -- the difference, which is exactly route
(i) -- did not move at any offset. **A format-only learner scores a tie by construction, because format
is common to `own` and `shuf` and cancels in the difference we gate on.**

**THE GENERAL LESSON, and it is the one to carry: A GATE THE OBJECTIVE DOES NOT CONTAIN IS A HOPE, NOT
A TARGET.** This programme has gated on `own_minus_shuf` since 00S item 4 (2026-09-30 12:45) and has
never once optimised it. Six runs asked a reward that scores "did this note help here" to produce a
model that wins on "does this note help HERE RATHER THAN ANYWHERE". Those are different quantities and
the gap between them has a name -- it is the whole of what runs 60-66 bought. **Before registering any
further gate, write down the reward, apply the baseline subtraction on paper, and ask whether the
quantity in the gate is a term in what survives.** Compare G95/G96 (a flag that does not survive the
subtraction is a no-op); this is its mirror image -- a GATE that is not in the objective is unpursued.

**THE FIX, AND IT IS RUN 67 (`think67_xanchor`, launched 22:20, smoke passed 22:19).** One flag,
`--xanchor-w 0.0 -> 1.0`, implemented and checked this tick:
`REW[r] = G[r] - w * (gain of the SAME note placed ONE ANCHOR OVER)`, the misplaced-note control built
as `np.roll(content, 1, axis=0)` -- byte for byte how `probe_span.py:56` builds its `shuf` column. At
w=1.0 **the reward IS `own_minus_shuf`: the objective becomes the ruler.** Cost: RO=4 extra scoring
passes, measured at 18.0 s/step against run 66's 14.4 (+24%), ETA ~12 h. Four pre-launch checks, all
passed and all recorded because G95 is what happens without them: (a) the flag SURVIVES the LOO
subtraction -- `GX[r]` varies per rollout, max|ADV(w=1)-ADV(w=0)| = 2.94 on a mock, so it is not a
no-op; (b) the roll is ALIGNED -- `GX[r][a]` == note of anchor a scored at anchor a+1, verified for all
(r,a) against a mock reader, max error 0; (c) it DOES WHAT IT IS FOR -- on a mock where one rollout
writes a fluent generic note and another an anchor-specific one, the raw reward picks the generic and
the differenced reward picks the specific; (d) `--xanchor-w 0.0` is bit-identical (GX stays zeros, the
extra passes are inside the `> 0.0` guard) and `test_rollouts.py` passes in full.
**And a fifth check that only the live trainer could give: at step 0 the new metric
`g_r0 - gx_r0` (training-time `own_minus_shuf`) reads +5.42, against the init's held-out +6.56 at the
same offset.** A broken roll would read ~0. `gx_sd` -- the between-rollout spread of the DIFFERENCED
reward -- reads 2.9-4.3, ABOVE `g_sd`'s 2.0-3.5, so the feared variance collapse of a difference of two
noisy estimates did not happen; X1's 0.60 bar exists anyway, registered before the number was seen.

**THE DEGENERATE SOLUTION IS NAMED IN ADVANCE AND GATED.** `own_minus_shuf` can be maximised from the
wrong end: make the MISPLACED note actively harmful (a confidently wrong, highly specific note damages
its neighbour) rather than make the own note useful. So every run-67 gate carries **Q1** (a pass
requires `own` >= run 66's `own` at the same step and offset, minus 1 s.e.) and **Q3** (a reportable
result requires `clean` >= 8.0 -- run 66 ended at 3.91, having closed the gap from the wrong side by
making the hidden window unnecessary rather than making the note carry it). No gate in this programme
has ever distinguished a note that recovers 54% of a ceiling training shrank by 79% from one that
recovers 15% of the full ceiling. Both qualifiers are in `runs/think67_xanchor/config.sh`, written
before the first checkpoint existed, and both are printed automatically beside every verdict.

## G100.1 -- THE OPS HALF, AND IT COST 1.5 h OF THE ONLY DEVICE ON THIS BOX TODAY. **A GATE THAT IS ONLY *READ* EVERY 2 h IS NOT A GATE. RUN 66 OVERRAN ITS OWN PRE-REGISTERED STOP BY 375 STEPS.**

R7's number was decided in advance at 20:12. The probe landed at **20:43** -- 24 minutes after the
20:07 tick had finished -- and the next tick was at 22:07, so the run trained on for **375 steps (~1.5
h)** past a stop that had already been written down, and spent those steps entering the register
collapse (reg% 1.5 -> 33.5). Nothing about this was a judgement failure: the bar was right, the reading
was right, the enforcement was a cron schedule.

**Fixed, not noted.** `agent/armed_probes_r67.sh` now ENFORCES every bar it reads: on a STOP verdict it
writes the `STOPPED` STATUS and kills the run itself, within ~1 min of the probe landing.
`agent/watch_r67.sh` does the same on a 5-minute loop for the two clauses that live in `metrics.jsonl`
(X1 readability @400, H4 writer-death, continuous from step 600). **The enforcement path was verified
on real data before the run needed it: fed run 66's actual step-1800 probe JSON with run 66's own bar,
it emits `VERDICT_STOP` -- i.e. had it existed at 20:43 it would have stopped run 66 then, and fed a
bar the same file clears it emits `VERDICT_CONTINUE`.** Both directions tested; a stop rule that has
only been tested in the direction that fires is half tested.

**AND THE PLACEMENT LESSON FOR H4, which run 66 supplies at its own expense.** Run 66's H3 was
`len_mean <= 3.5 AND g_sd <= 0.30` on the single band 1750-1799. It read 3.67 / 0.74 and did not fire
-- correctly -- and the run then collapsed over the NEXT 250 steps (2050-2099: len 3.16, g_sd 0.24,
reg% 33.5). **The clause was right in form and 300 steps late in placement: a death watch pinned to one
pre-chosen band only catches a death that is punctual.** Run 67's H4 is the same condition evaluated
CONTINUOUSLY on the most recent complete 50-step band from step 600 on, and read on `gx_sd` (the spread
of the quantity that is now the reward) as well as `len_mean`.

## G101 (2026-10-01 00:35, run 67 at step ~365) -- FIXING THE OBJECTIVE ALSO MADE IT READABLE, AND THE FIRST READING IS BAD

G100 said the gated quantity (`own_minus_shuf`) was never a term in the objective, so the gate was a
hope rather than a target. Run 67's one flag makes it a term: with `--xanchor-w 1.0` the per-anchor
reward IS `G - GX`. **The side effect is the thing to notice. Because the reward is now the gated
quantity, the reward's own LEVEL is legible every step, for free, from metrics already being written:
`D := mean(g_r0 - gx_r0)`.** No earlier run in this programme had any live read on the value of what it
was maximising -- `g_r0` was the reward's level for runs 60-66, but `own_minus_shuf` was the bar, and
those were different numbers. **A correctly specified objective is an instrument; a misspecified one
hides its own failure.** That is the generalisable half of G101 and it holds independently of how run 67
turns out.

**The first 365 steps of that instrument, block 0 (o=-15, a single fixed offset):**

```
  band      0-49   50-99  100-149 150-199 200-249 250-299 300-349
  D       +2.514  +2.447  +2.178  +1.993  +2.128  +1.734  +1.768
  band s.e. 0.27    0.21    0.24    0.16    0.16    0.20    0.16
  slope of D over steps >= 100:  -0.24 per 100 steps  (-0.63 over 260 steps, ~3 band s.e.)
  g_r0  +0.59 -> +2.19   gx_r0  -1.59 -> +0.42   (from the RL trough at 100-149)
```

**D is falling, and it is falling because the misplaced note is improving FASTER than the own note
(+2.01 vs +1.60) -- which is run 66's G100 pathology, appearing in the training metric of the run built
to prevent it.** Making `own_minus_shuf` the reward has not, in the first 265 RL steps, stopped the
writer from buying generic note-quality; it has made the purchase visible while it happens.

**What this is NOT, and the restraint matters as much as the reading.** (1) It is not yet evidence
against the hypothesis. `--rl-start 100` with `--rl-ramp 200` means rl weight only reached its full 0.5
at step 300, so block 0 confounds the ramp with the trend, and no bar may be set from it. (2) It is not
a derail axis: D is **not comparable across `--span-sched` boundaries**, because the task changes every
200 steps -- the identical argument that ruled runbook 3c out for runs 66 and 67. A continuous stop
clause on D's level or slope would repeat a mistake this document already records. The only legitimate
reading is WITHIN a sched block, and the first clean one is 400-599.

**So A is registered as a READING, not a gate** (`runs/think67_xanchor/config.sh` AMENDMENT A): D(400-449)
and D(550-599) reported beside X2's verdict, with `D(550-599) <= 0.00` meaning the reward level has
reached zero (the writer trained toward notes that serve a NEIGHBOUR at least as well as their own --
the degenerate end Q1/Q3 exist to catch, reached from inside the objective) and `D(550-599) <= D(400-449)
- 0.60` registered in advance as the leading explanation for any X2/X3 failure, which then may NOT be
attributed to the hypothesis being false. **No stop, and the reason is a trade rather than an oversight:
block 400-599 completes ~10 min before X2's probe lands and stops the run anyway, and D's within-block
noise has never been measured (per-step sd 1.32). Ten minutes is not worth an untested kill path on a
12-hour test.** This is a deliberate narrow exception to G100.1's "a gate that is only read every 2 h is
not a gate": that rule governs clauses that STOP, and A does not stop anything.

## G101.1 -- DELETING AN OPTIMIZER STATE ARMS THE ORPHAN QUARANTINE AGAINST THAT RUN

The 22:07 tick freed 33 G by deleting the `*.optim.pt` of stopped runs 65 and 66, correctly reasoning
that no scientific claim depends on an optimizer state. This tick extended that to every finished run
and freed **571 G** (77 G -> 649 G free, 97% -> 67%). **What neither tick's reasoning included, and what
must be written down: `watchdog.sh` and `keepalive_think.sh` both QUARANTINE any model `step_*.pt` that
has no `.optim*.pt` beside it** (`mv` to `<ckptdir>/.orphans/`, guarding against a checkpoint killed
mid-write -- gotcha G12). Both act ONLY on the run named in `.watchdog_run`, so the deletion is inert
today. **It becomes destructive the moment `.watchdog_run` is pointed at any run whose optimizer state
has been deleted: within 5 minutes every model checkpoint of that run is moved aside and
`--resume latest` fails.** Consequence, standing: **a run whose `*.optim.pt` have been deleted can never
be resumed or relaunched under its own name; repoint `.watchdog_run` to it only after restoring an
optimizer file or disabling clause 1.** The quarantine is a `mv`, so the damage is recoverable from
`.orphans/`, but the run will have died and been "healed" in a loop first.
`checkpoints/think21_phaseA` was EXCLUDED from the sweep: `benchmark/bench_final_r21.sh:7` blocks until
`step_0007700.optim.pt` exists, so deleting it would hang that script forever rather than fail it.

## G102 (2026-10-01 02:07 headless tick, run 67 at step ~730) -- **G101's ALARM DID NOT SURVIVE FULL RL WEIGHT: D IS HOLDING. AND THE REAL HOLE IS NOT IN D BUT IN THE LADDER -- EVERY GATE STOPS ONLY AT 2 s.e. *BELOW RUN 66*, A BAND RUN 66 ITSELF TRAVERSED FOR 1,800 STEPS AND LOST IN. A RUN THAT MERELY TRACKS ITS OWN CONTROL PASSES EVERY GATE AND TEACHES NOTHING.**

**(a) G101's falling D was the ramp, not the run.** G101 (00:35, step ~365) recorded `D := mean(g_r0 -
gx_r0)` -- the level of the differenced reward itself -- falling at **-0.24 per 100 steps** across block 0,
and named it "run 66's G100 pathology in the training metric of the run built to prevent it". It also
refused to set a bar from that block, because steps 0-299 precede full `--rl-w 0.5` (`--rl-start 100
--rl-ramp 200`). **That refusal was correct and the trend did not survive.** AMENDMENT A's two registered
reads, both inside the single o=-13 block at full rl weight: **D(400-449) +1.714 +- 0.19, D(550-599)
+1.795 +- 0.175** -- above zero and not falling, so A resolves on its third branch ("D is holding, and a
later gate failure IS about the hypothesis"). In the next block (o=-11) D is **rising**: +1.98 / +2.10 /
+2.27 for bands 600 / 650 / 700.
**THE TRANSFERABLE PART: a slope measured across a reward-weight ramp is a slope of the weight, not of the
quantity.** G101 nearly became "the fix does not work" on 265 steps, 200 of which were at partial weight.
The protection was procedural, not clever -- *name the first clean block before reading any trend* -- and it
is the second time in two days that pre-registering the read beat the reading (cf. G95, where the "flag"
was inert for 514 steps).

**(b) X2 @600 passed its bar and moved the WRONG WAY against its own control.** o=-13, step 600:
`own_minus_shuf` **+3.56 +- 0.53** against the registered CONTINUE band (STOP < +2.62, WIN >= +6.59).
Matched against run 66 at the SAME step and offset: **own_minus_shuf -0.76** (run 66 +4.32), **own -1.09**
(+7.37 vs +8.46). Against the init at o=-13 (+5.09 +- 0.75): **-1.53, i.e. 1.7 s.e. BELOW the untrained
model.** Neither difference is individually significant (s.e.(diff) ~0.85-0.92). `clean` **+18.52** (Q3
PASS, and far above run 66's terminal 3.91 -- run 67 is not shrinking the ceiling).
**So putting `own_minus_shuf` INTO the reward has not, after 600 steps, produced any separation from the
run that did not have it.** That is not yet a refutation -- it is one gate, and the differenced reward only
reached full weight at step 300.

**(c) THE LADDER'S HOLE, WHICH IS THE RESULT OF THIS TICK. Every X-bar in run 67's config stops the run
only at 2 s.e. BELOW run 66 at the matched step.** X3's CONTINUE band is +1.35 .. +6.03; X4's is +0.55 ..
+6.04. **Run 66 lived inside exactly these bands for 1,800 healthy steps and ended in a registered LOSS.**
A run that simply tracks run 66 therefore clears every gate in the ladder, survives to FINAL, and is run 66
re-executed at +24% step cost (the price of `--xanchor-w`'s extra scoring passes). **A ladder built only
from "not catastrophically worse than the control" cannot detect "indistinguishable from the control",
which is the single most likely outcome and the one this programme has produced eight times.**
**The hypothesis makes a sharper prediction than the bars do:** reward differencing should put run 67
*ABOVE* run 66 on the gated quantity at matched steps. With per-gate s.e. of 0.6-0.9, the testable content
of that prediction is **the SIGN, not the significance**. Hence AMENDMENT B2: `delta_k = run67 - run66` at
X2/X3/X4, **stop at 1400 if delta(X3) < 0 AND delta(X4) < 0** (p = 1/8 for three negatives; delta(X2) =
-0.76 is already known and disclosed in the amendment, so the clause itself turns on two unknowns, p = 1/4).
It buys ~1,000 steps = **~4.7 h of the only device** and risks killing a working-but-noisy run -- accepted
*because* (a) certified D as holding, so a failure is attributable to the hypothesis rather than to the
instrument or a dead writer.
**STANDING RULE EARNED HERE: when a run's comparator is a PREVIOUS RUN rather than the untrained model,
the gate table needs a clause in the direction the hypothesis PREDICTS, not only a floor in the direction
of disaster. Write the sign test at launch.** Every run 60-67 shipped with disaster floors and no
prediction clause, and that is why seven of them had to be read as ties after the fact instead of being
stopped as ties during.

**(d) A SMALL AMBIGUITY IN A MANDATORY QUALIFIER, RESOLVED BEFORE THE GATE IT AFFECTS (AMENDMENT B1).**
Q1 ("a pass needs `own` >= run 66's own - 1 s.e.") printed **FAIL** at X2 while the headline printed
CONTINUE. Q1's own rationale names the mechanism it guards: *"if own_minus_shuf RISES while own FALLS, the
run is winning by hurting the control."* At X2 both fell together, so the degenerate route is not what
happened. B1 therefore scopes Q1 to fire only when the headline is at or above run 66's matched value AND
`own` is below -- i.e. on a WIN, the only verdict where the headline can rise. **The point is the timing,
not the content: a qualifier whose trigger is discovered to be ambiguous must be disambiguated at a step
where the gate it governs does not yet exist**, or the resolution is indistinguishable from choosing the
answer. X3 was 275 steps away when B1 was written.

**(e) ADDED 02:50, same tick: THE `DRIFT!` TOKEN FIRED ON RUN 67'S STATUS LINE AT STEP 749 AND IT IS A
DEFECT IN THE INDICATOR, NOT A FACT ABOUT THE RUN. RULED DEAD FOR THIS DESIGN, WITH A NULL.**
`monitor_think.py:34-41` computes `drift = roll20(ntp1)[now] - MIN(roll20(ntp1))` over the whole run and
appends ` DRIFT!` when `drift > 0.5` (auto-kill disabled since 2026-09-05 by user directive, so it is a
label only). **The subtrahend is a RUNNING MINIMUM, which can only fall as n grows, so on any stationary
noisy series the statistic is a RATCHET that crosses any fixed threshold eventually.** Run 67's numbers:
per-step ntp1 sd **0.549**, roll20 sd 0.125, roll20 range 1.647-2.270; the min sits at step 529 (1.647)
and "now" is 2.138. **Per-block mean ntp1 is FLAT -- 1.956 (o=-15, steps 0-399) / 1.961 (o=-13, 400-599)
/ 1.961 (o=-11, 600-751) -- and the 50-step roll20 bands oscillate 1.80..2.16 with no trend.**
**THE NULL, because "it looks like noise" is not an argument:** permute the per-step ntp1 series (which
destroys any trend while preserving its marginal distribution) and recompute at the same n -- 2,000 draws
give **mean drift +0.329, median +0.327, and P(drift > 0.5) = 0.10.** Observed: **+0.492.** The flag is
INSIDE the null at a threshold the null itself clears 10% of the time at n=730, and that probability
rises toward 1 with n by construction.
**RULING (G84's rule -- write the dead clauses down before they fire): `DRIFT!` IS NOT A DERAIL AXIS ON
THE SLIDE DESIGN and must not be read as one.** Two independent reasons: (i) the ratchet above, and
(ii) `--span-sched` changes the task every 200 steps, so `ntp1` is not comparable across blocks -- the
same argument that already ruled out runbook 3c for runs 66 and 67 and that A used to refuse a bar on D.
**The base-health instrument is the HELD-OUT MIX CE** (2.775 -> 2.801 over 600 steps = +0.004 per 100,
bar +0.25 total), plus 3b's own `ntp1 > 4.0` absolute clause (live value 1.79-1.93). Both far clear.
The indicator is NOT being patched: `monitor_think.py` is read every ~3 min by the live monitor, the
token is cosmetic, and three runs' STATUS histories would become incomparable for no gain.

## G103 (2026-10-01 04:07 headless tick, run 67 at step ~1100) -- **A PRE-REGISTERED STOP WAS VOIDED BY ITS OWN RULE AND I DID NOT WRITE A SECOND ONE. THE REASON IT IS NOT NEEDED CORRECTS G102(c): X5 WAS ALREADY A SIGN TEST ALL ALONG. AND THE RUN'S REAL NEW FACT IS THAT A MEASURABLY HEALTHIER WRITER MOVED THE GATE BY NOTHING.**

**(a) WHAT X3 SAID.** `armed.log` 2026-10-01 03:44:14, step 1000, o=-9, enforced automatically:
`own_minus_shuf` **+2.98 +- 0.62** against the registered bar STOP < +1.35 -> **VERDICT_CONTINUE**.
Decomposition: `own` +2.73+-0.51 | `filler` +1.59 | `span` +14.46 | **`clean` +13.76** (Q3 PASS).
Q1 PASS (own +2.73 >= run 66's +2.09 - 0.51 = +1.58). Matched against run 66 at the same step and
offset: **`own_minus_shuf` +0.43 (r66 +2.55), `own` +0.64 (r66 +2.09)** -- i.e. the matched delta went
POSITIVE for the first time.

**(b) B2 IS VOID BY ITS OWN TEXT, AND THAT IS THE POINT OF HAVING WRITTEN IT.** B2 (02:30) was
`delta(X3) < 0 AND delta(X4) < 0 -> STOP AT 1400`, with `any one >= 0 voids it`. **delta(X3) = +0.43.**
The clause is dead and the run continues. The three matched deltas are **-0.76, -0.19, +0.43; mean
-0.17** -- textbook no-difference. So B2 was a 1/4 prediction that did not come in, which is what a
prediction is allowed to do.

**(c) THE DECISION THAT COST SOMETHING: NO SECOND SIGN TEST.** The temptation was obvious -- re-run
B2's logic over X4+X5, two fresh unknowns, ~1/4 again, and it would have been defensible on its face.
**Rejected. B2's entire value was that its inputs did not exist when it was written; a second
conjunction composed 20 minutes after the first was voided by its own rule is the second draw from the
same urn, and the practice of re-drawing until one fires is exactly what pre-registration exists to
prevent.** The honest consequence of a voided prediction is that the run runs. **Price: ~6.3 h of the
only device, paid knowingly.** Generalised: *a voided pre-registered stop does not entitle you to a
replacement; it entitles the run to its remaining steps.*

**(d) AND THE REPLACEMENT WAS NOT NEEDED, WHICH CORRECTS G102(c) RATHER THAN WITHDRAWING IT.** G102(c)
read "every X-bar stops run 67 only at 2 s.e. BELOW run 66, a band run 66 traversed for 1,800 steps and
lost in." **True of X2/X3/X4. FALSE of X5. X5's STOP bar is +2.30 and run 66 ACTUALLY SCORED +1.16
THERE** -- that reading IS run 66's terminal R7 failure, at o=-1, the offset where run 66 died. So X5
implicitly demands **delta(X5) >= +1.14**, a *strictly stronger* requirement than B2's `delta >= 0`, and
unlike B2 it is enforced by `agent/armed_probes_r67.sh` within ~1 min. **The ladder's only blind stretch
is X4: 400 steps, ~1.9 h.** The lesson is narrower than G102(c) and sharper: *before inventing a gate,
check whether an inherited bar already encodes the comparison you want -- a bar copied from the run that
FAILED it is a sign test wearing a floor's clothing.*

**(e) AMENDMENT A RESOLVED ON ITS THIRD BRANCH FOR THE SECOND TIME.** Registered read beside X3, inside
the single o=-9 block at full rl weight: **D(800-849) +1.848 +- 0.208, D(950-999) +1.907 +- 0.175** --
`D > 0` and `D(950-999) > D(800-849) - 0.60`. D by 50-step band over 0-1099:
`2.51 2.45 2.18 1.99 2.13 1.73 1.77 1.90 | 1.71 1.91 1.49 1.80 | 1.98 2.10 2.02 1.99 | 1.85 1.91 1.86
1.91 | 1.81 2.48` (`|` = `--span-sched` boundary). Flat-to-rising inside every full-weight block; G101's
-0.24/100 was the `--rl-ramp` (G102(a)). **A's third branch binds for the second time: an X4/X5/FINAL
failure is about THE HYPOTHESIS.**

**(f) THE SUBSTANTIVE RESULT OF THE TICK: RUN 67 IS A MEASURABLY HEALTHIER WRITER THAN RUN 66 AT
MATCHED STEPS AND IT BOUGHT NOTHING AT THE GATE.** Same init, same corpus, same schedule, one flag
apart, identical step numbers:

| band | quantity | run 66 | run 67 | gap |
|---|---|---|---|---|
| 600-699 | `len_mean` | 14.916 | 14.947 | +0.03 |
| 800-899 | `len_mean` | 11.597 | 13.811 | +2.21 |
| 1000-1099 | `len_mean` | **8.801** | **13.164** | **+4.36 (~25 s.e.)** |
| 600-699 | `g_sd` | 2.290 | 2.609 | +0.32 |
| 1000-1099 | `g_sd` | **1.747** | **2.426** | **+0.68 (~7.7 s.e.)** |
| 900-1199 | `reg%` | 0.80 | 1.41 | ~0 in BOTH |
| @1000 o=-9 | probe `own_minus_shuf` | +2.55 | +2.98 | **+0.43 (the null)** |

`len_mean` and `g_sd` are the two axes whose collapse (len -> 3.5, spread -> 0.26) ended run 66, and run
67 is plainly not on run 66's shortening trajectory. **But read it in both directions: `reg%` is ~zero
for BOTH at matched steps -- run 66's blow-up to 33.5 was a post-1900 event -- so NO claim that run 67
avoids the degeneracy is admissible before step 1800.** And the gate did not move regardless.

**(g) WHY (f) MATTERS MORE THAN ANY OF THE GATE READINGS, AND THE SUCCESSOR RULE IT FORCES (config.sh
AMENDMENT C2).** Two explanations have been carried for the eight ties: *the writer dies* (7(f)'s
target: an unfloored z-scored advantage amplifying a silent reward) and *the objective does not contain
the gated quantity* (G100's target, which run 67's `--xanchor-w 1.0` removes by construction). **Run 67
is the first run that can retire BOTH at once, because its objective IS `own_minus_shuf` and its writer
is demonstrably in better shape than run 66's at the same step -- and the matched delta is +0.43 +-
~0.85.** Registered before X4/X5/FINAL exist: **C2a** -- if run 67 reaches >= 1800 with `len_mean > 11.0`
AND `gx_sd > 1.00` on the last 50-step band and the X4/X5 deltas sit within +-1 pooled s.e. of 0, then
both explanations are REFUTED and the successor is NOT 7(f) (a patch for a failure mode this run did not
have, whose expected outcome is a ninth tie); it must move the one dimension runs 60-67 all held fixed,
**what the note is asked to DO and how much room it has** (note length / `--rollouts`, i.e. the
signal-to-noise of the differenced reward against fixed scorer noise), chosen at FINAL from the FINAL
decomposition. **C2b** -- if run 67 collapses before 1800, 7(f) stands unchanged. **C2c** -- on a WIN,
Q1 is mandatory and unweakened (B1). *A decision rule registered before its inputs is not a decision.*

**(h) STANDING RULES EARNED HERE.** (1) A voided pre-registered stop entitles the run to its remaining
steps, not the supervisor to a fresh clause. (2) Before writing a new comparison gate, check the
inherited bars: one copied from a run that FAILED it already encodes the sign test. (3) When two
competing explanations for a repeated null can be killed by the same run, say so BEFORE the run's last
gate, and attach the successor to the branch -- otherwise the successor gets chosen by whatever the
final number happens to be.

## G104 (2026-10-01 06:07 headless tick, run 67 at step ~1485/2400) -- **X4 PASSED AND ITS MATCHED DELTA IS THE LARGEST POSITIVE IN THE LADDER (+1.49, 2.4 s.e.), BUT POOLED OVER ALL FIVE OFFSETS RUN 67 AND RUN 66 ARE THE SAME RUN TO +-0.35 NATS -- AND BOTH SIT ~1.3 NATS *BELOW THE UNTRAINED MODEL*, 2.3-2.5 s.e., WHICH IS THE ONLY COMPARATOR THE PROGRAMME CLAIMS TO CARE ABOUT. POOL THE LADDER; DO NOT READ ITS RUNGS.**

**(a) WHAT LANDED.** `agent/armed_probes_r67.sh` enforced X4 by itself at 05:51:12, 16 min before this
tick woke: `own_minus_shuf` **+2.94 +- 0.56** at o=-5/step 1400 against the registered bar
`STOP < +0.55`, **VERDICT_CONTINUE**; Q1 PASS (own +0.93 vs the +0.24 floor), Q3 PASS (`clean` +10.28
vs the 8.0 floor); decomposed own +0.93 | shuf -2.01 implied | filler -1.04 | span +14.96.
**delta(X4) = +2.94 - 1.45 = +1.49 +- 0.63 = +2.38 s.e. ABOVE run 66** at the same step and offset --
the first matched delta in this programme that is nominally significant in the direction the hypothesis
predicts. B2 was already VOID (G103, on delta(X3) = +0.43); X4 voids it a second time, independently.

**(b) AND THAT IS WHY THE POOLED READ HAD TO BE DONE BEFORE ANYTHING WAS CONCLUDED FROM IT.** Five
matched deltas now exist, all on the same 132 unseen-FineMath val anchors, run 67 minus run 66 at
identical steps and offsets:
```
   step   o     run 67        run 66      delta      s.e.(delta)
    600  -13   +3.56+-0.53   +4.32+-0.60   -0.76        0.80
    800  -11   +3.56+-0.61   +3.75+-0.58   -0.19        0.84
   1000   -9   +2.98+-0.62   +2.55+-0.43   +0.43        0.75
   1200   -7   +3.17+-0.57   +4.75+-0.63   -1.58        0.85
   1400   -5   +2.94+-0.56   +1.45+-0.28   +1.49        0.63
   POOLED (mean of 5, s.e. = sqrt(sum se^2)/5):          -0.122 +- 0.348   z = -0.35
```
**Run 67 and run 66 are the same run to within +-0.35 nats.** One flag apart -- the flag that makes the
reward BE `own_minus_shuf` by construction (G100's whole diagnosis) -- and over 1,400 matched steps at
five offsets the gated quantity moved by nothing, with a confidence interval tight enough that a
half-nat effect would have shown. X4's +2.38 s.e. is 1 rung of 5 whose set averages zero; with five
draws, P(some |z| >= 2.38) ~ 5 x 0.017 = 8%. **It is not a result, and the rung that looks like one is
exactly the rung a ladder read one-at-a-time would have been reported on.**

**(c) THE READING THAT MATTERS, AND IT IS THE PROGRAMME'S OWN STATED BAR.** Beating run 66 was never the
claim; beating the UNTRAINED model is (`own_minus_shuf` is a paired within-model difference, so it IS
legible across models -- G99.1). Against the init at the same five offsets:
```
   o        init         run 66   delta       run 67   delta
  -13   +5.09+-0.75     +4.32    -0.77       +3.56    -1.53
  -11   +6.21+-1.17     +3.75    -2.46       +3.56    -2.65
   -9   +3.75+-1.14     +2.55    -1.20       +2.98    -0.77
   -7   +4.32+-1.13     +4.75    +0.43       +3.17    -1.15
   -5   +3.62+-1.21     +1.45    -2.17       +2.94    -0.68
  POOLED               run 66  -1.234 +- 0.541  z=-2.28  (4/5 negative)
                       run 67  -1.356 +- 0.553  z=-2.45  (5/5 negative)
```
**Run 67 is below the untrained model at every offset it has gated on, pooled 2.45 s.e. below. So is
run 66, 2.28 s.e. below.** This is G93/#5's finding about run 65 (*"training did not ADD anchor-specific
information to the note; it REMOVED what the untrained model's note already had"*) reproduced twice more,
at reduced magnitude, on two further runs and a different objective. **Three runs, nineteen offset
readings, one direction.** The ladder's rungs are run-66-relative floors -- G102(c)'s hole -- and that is
why five CONTINUEs in a row have coexisted with a pooled loss to the init the whole time.

**(d) THE STANDING RULE, AND IT IS THE CHEAPEST ONE THIS PROGRAMME HAS EARNED IN A WEEK.** *A gate ladder
is a multiple-comparison instrument, so pool it before you read any rung of it.* Runs 60-67 shipped
8, 7 and 5 sequential gates and were each read rung-by-rung as "tie, tie, tie, LOSS" or "PASS, PASS,
PASS"; both readings are artifacts of the serial presentation. The pooled statistic (mean of matched
deltas, s.e. = sqrt(sum se^2)/k) costs one line of arithmetic, has 2-3x the power of any rung, and in
every case so far says the same thing the eight separate verdicts did not: **no difference between
consecutive runs, and a consistent deficit to the untrained model.** Quote both pooled numbers --
vs the previous run AND vs the init -- with every future gate verdict, and quote them BEFORE the rung.

**(e) WHAT IT DOES *NOT* SAY, STATED BECAUSE THE TEMPTATION IS TO OVERSHOOT.** The pooled deficit is
~1.3 nats against a 3.5-6.2 nat init baseline, i.e. the trained note still carries most of the
anchor-specific information the untrained one did; this is erosion, not collapse, and the writer is
demonstrably healthier than run 66's (G103's C2 fact: len_mean 13.2 vs 8.8 at matched steps). And the
init's five probes are independent while run 67's five checkpoints are one trajectory, so the pooled
s.e. of the vs-init numbers is understated by an unknown amount -- z=-2.45 should be read as "clearly
and consistently negative", not as a p-value. The matched run67-run66 pooling has no such problem in
the comparator (both are trajectories, the init probes cancel exactly).

**(f) C2a's TRIGGER IS NOT MET AT X4, AND NO CLAUSE WAS TOUCHED.** C2a (G103) needs the X4 and X5 matched
deltas "within +-1 pooled s.e. of 0"; delta(X4) = +1.49 is 2.38 s.e. out, positive. So the registered
refutation of BOTH carried explanations does not fire on present data, and the successor is not yet
forced off 7(f). **Nothing was re-registered to fix this** -- that is G103's second-draw error, and the
honest consequence of a clause whose trigger misses is that the next gate decides. X5 @1800 @ o=-1
already is the decisive test and needs no help: its STOP bar is +2.30 and run 66 scored +1.16 there, so
**X5 CONTINUE == delta(X5) >= +1.14 > 0**, and two positive matched deltas at the two hardest offsets
would be the first non-tie signal runs 60-67 have produced. X5 STOP kills the run inside ~1 min,
unattended. Either way the pooled numbers in (b)/(c) get recomputed with k=6 at that verdict, first.

**(g) HEALTH AT THE TICK, for the record** (step 1485, 17.9 s/step, ETA 4.6 h -> ~10:45): held-out mix
2.775 -> 2.812 over 1400 steps = **+0.0026 per 100** (bar +0.02 per 100 over 300); reg% 4.0 on the last
block (bar 15); band 1450-1484 len_mean **13.47**, g_sd 2.25, gx_sd **2.79** (H4 bars 3.5 / 0.30, 3b's
spread bar 0.60); ntp1 1.97 (3b's absolute bar 4.0); **zero NaN/inf in 1485 rows, zero `NRT_`/Traceback
in any log**; 65 procs; disk 629 G. D = `mean(g_r0 - gx_r0)` by 50-step band, 1200-1484:
`1.78 1.74 1.64 2.17 | 1.38 1.95` (`|` = the o=-5 -> o=-3 span-sched boundary) -- above zero, no trend,
and A's third branch (G102) therefore still binds: an X5/FINAL failure is about the hypothesis.
**The ` DRIFT!` token on the STATUS line is still the G102(e) ratchet (running-minimum subtrahend); it is
cosmetic and auto-kill on it has been off since 2026-09-05 by user directive.**

### G105 -- RUN 67 IS THE NINTH TIE, AND POOLED OVER SEVEN OFFSETS IT IS *THE SAME RUN AS RUN 66 TO +-0.29 NATS*. BOTH SIT ~1.36 NATS BELOW THE UNTRAINED MODEL. THE PROGRAMME HAS NEVER ASKED WHERE THAT DEFICIT COMES FROM, AND RUN 68 ASKS.
*(2026-10-01 08:07 tick. Run 67 STOPPED ITSELF at 07:55, step 1822, on its pre-registered X5 bar:
`own_minus_shuf` +2.01 +- 0.56 at o=-1 vs a registered STOP bar of +2.30. Enforced by
`agent/armed_probes_r68.sh`'s predecessor inside ~1 min, with no supervisor awake -- G100.1's ops fix
working as designed for the second time.)*

**(a) THE POOLED LADDER FIRST, BEFORE ANY RUNG, BECAUSE THAT IS NOW THE RULE (G104(d)).** Seven matched
rungs exist, same 132 unseen-FineMath anchors, same probe, each checkpoint read at the offset it was
trained at:

```
 step   off    run 67        run 66        init          r67-r66        r67-init       r66-init
  600   -13   +3.56+-0.53   +4.32+-0.60   +5.09+-0.75    -0.76+-0.80   -1.53+-0.92   -0.77+-0.96
  800   -11   +3.56+-0.61   +3.75+-0.58   +6.21+-1.17    -0.19+-0.84   -2.65+-1.32   -2.46+-1.31
 1000    -9   +2.98+-0.62   +2.55+-0.43   +3.75+-1.14    +0.43+-0.75   -0.77+-1.30   -1.20+-1.22
 1200    -7   +3.17+-0.57   +4.75+-0.63   +4.32+-1.13    -1.58+-0.85   -1.15+-1.27   +0.43+-1.29
 1400    -5   +2.94+-0.56   +1.45+-0.28   +3.62+-1.21    +1.49+-0.63   -0.68+-1.33   -2.17+-1.24
 1600    -3   +2.82+-0.55   +3.30+-0.61   +2.39+-1.35    -0.48+-0.82   +0.43+-1.46   +0.91+-1.48
 1800    -1   +2.01+-0.56   +1.16+-0.41   +5.30+-1.41    +0.85+-0.69   -3.29+-1.52   -4.14+-1.47
 ------------------------------------------------------------------------------------------------
 POOLED (mean; s.e. = sqrt(sum se^2)/7)                  -0.034+-0.292 -1.377+-0.496 -1.343+-0.488
                                                          z = -0.12     z = -2.77     z = -2.75
                                                                        6/7 NEGATIVE  5/7 NEGATIVE
```

**(b) WHAT THAT TABLE SAYS, IN THE ORDER THAT MATTERS.** (i) **Run 67 and run 66 are the same run.**
One flag apart -- and not a cosmetic flag: `--xanchor-w 1.0` is G100's own correction, the change that
makes the REWARD BE the gated quantity -- over 1,800 matched steps and seven offsets, the gated
quantity moved by **-0.034 +- 0.292 nats**. The interval is tight enough that even a half-nat effect
could not hide in it. (ii) **Both runs are ~1.36 nats BELOW the untrained model**, and the two deficits
agree to 0.03 nats. (iii) `own_minus_shuf` is a paired within-model difference, so it IS legible against
the init (G99.1) -- this is the programme's own stated bar, not a new one.

**(c) SO THE RESULT OF RUNS 65, 66 AND 67 IS NOT "A TIE". IT IS A CONSISTENT LOSS, AND IT IS THE ONLY
EFFECT THIS PROGRAMME HAS EVER MEASURED WITH ITS OWN INSTRUMENT.** Training makes the model WORSE at
using its own note than the untrained model is. Three runs, twenty-one offset readings here plus G93/#5's
run-65 finding, one direction. **The eight "ties" were never ties; they were a loss, reported one rung at
a time against a previous-run floor, which is exactly the hole G102(c) opened and G104 quantified.**
Stated limits, unchanged: ~1.36 nats against a 3.5-6.2 nat baseline is EROSION, not collapse; and the
init's probes are independent while each run's seven checkpoints are one trajectory, so +-0.50 is
optimistic and z=-2.77 means "consistently and clearly negative", not a p-value. The r67-r66 pooling has
no such defect -- the init probes cancel exactly, and that is the number doing the work in (b)(i).

**(d) 7(f) `--adv-sig-floor 0.5` IS RETIRED AS THE SUCCESSOR, ON THE TEST ITS OWN CAVEAT ASKED FOR.** The
02:30 caveat (G102) said: do not launch 7(f) reflexively; settle it at X4/X5 with the data, because 7(f)
fixes how runs DIE and "if run 67 ends in a tie while HEALTHY, the failure mode 7(f) repairs did not
occur in it." The data are now in, and they are better than the caveat hoped for -- a matched A/B on
writer health:

```
  band 1750-1799       len_mean   g_sd   gx_sd        own_minus_shuf @1800
  run 66                 3.67      0.74    --              +1.16 +- 0.41
  run 67                 9.41      1.54   1.84             +2.01 +- 0.56
```

Run 66 died with a writer at 3.67 tokens and spread 0.74 (H4's floor is 3.5/0.30 -- it was *at* the
absorbing state). Run 67 at the identical band had a writer **2.5x longer with 2x the spread**, plainly
alive. **And pooled over seven rungs the two runs scored the same to +-0.29 nats.** So: a 2.5x healthier
writer bought **0.03 +- 0.29** nats. 7(f) does one thing -- it makes the writer healthier still by
flooring the advantage divisor. Its expected effect is therefore bounded by a quantity now measured to
be zero. **This retires 7(f) with DATA rather than with an argument, which is precisely what G103
demanded and what the caveat was written to make possible.** It stays implemented and unrun.

**(e) A RETRACTION OWED FROM THE 04:25 TICK.** That tick wrote that run 67 "is plainly not on run 66's
shortening trajectory." **That was true at step 1100 and it is false at 1800.** Run 67's writer went
len_mean 14.89 (@730) -> 13.47 (@1450-1484) -> **9.41 (@1750-1799)**, and `clean` went +18.52 -> +13.76
-> +10.28 -> **+7.69**, failing Q3's +8.0 ceiling for the first time at X5. Run 67 is on the SAME
trajectory, just slower and ~400 steps later. This does not touch (d) -- (d)'s comparison is at MATCHED
steps, where run 67 was 2.5x healthier -- but the claim as written was wrong and is withdrawn.

**(f) C2a AND C2b BOTH HAVE UNMET TRIGGERS, AND NEITHER WAS STRETCHED TO FIT.** C2a needed step >= 1800
*with* `len_mean > 11.0` on the last 50-step band: the run reached 1800 but read **9.41**, and its
X4/X5 deltas (+1.49, +0.85) are nowhere near "within +-1 pooled s.e. (0.29) of 0". C2b needed a collapse
*before* 1800 (H4, len <= 3.5, reg% >= 15): none fired -- len 9.41, reg% 7.1. **The run died on its gate
at 1800 while moderately healthy, which is a third path C2 did not enumerate.** That is an honest gap in
a clause I wrote, and the rule from G103 applies to my own clauses too: **a clause whose trigger misses
does not get retrofitted.** C2a is NOT invoked below. The successor is chosen on (d)'s route -- the
02:30 caveat, which DID fire and which explicitly instructed this tick to settle the question with the
X4/X5 data in hand.

**(g) AMENDMENT A's THIRD BRANCH STILL BINDS, CHECKED AT THE BOUNDARY AND NOT ACROSS IT.** D =
`mean(g_r0 - gx_r0)` by 50-step band, 1400-1823: `1.38 2.08 1.85 2.27 | 0.66 0.69 0.77 1.49 | 0.21`
(`|` = `--span-sched` boundaries at 1600 and 1800). The halving at 1600 lands EXACTLY on the o=-3 ->
o=-1 stage change, and **inside** the o=-1 block D is 0.66 -> 0.69 -> 0.77 -> 1.49, i.e. **rising** --
the same flat-to-rising-within-block pattern C1 recorded. The 0.21 is 24 steps of a brand-new o=+1
block and reads nothing. So the instrument was not failing, and X5's failure is about THE HYPOTHESIS.
**Reading D across a span-sched boundary would have manufactured a collapse that is not there** -- the
same error G102 forbade for reward-weight ramps, now also earned for curriculum boundaries.

**(h) THE SUCCESSOR, AND WHY IT IS A CONTROL AND NOT A NINTH LEVER. RUN 68 `think68_norl` = RUN 67 +
`--rl-w 0.5 -> 0.0`, ONE FLAG.** Eight runs of lever-guessing have moved the measured quantity by
nothing, while the erosion in (c) has gone unexplained and unattacked for three runs. **Every slide run
60-67 carries BOTH a writer RL gradient AND a reader/SSL slot gradient, and no run has ever separated
them.** `think_train.py:1049` is the writer's entire gradient,
`w3 = imit_w_eff*row_ssl + rl_w_eff*A*row_rl`; run 67 already sets `--imit-w 0.0` (run 63 settled that),
so `--rl-w 0.0` makes **`w3` identically zero: the writer receives no gradient of its own**, while the
reader/window gradient and the oracle-span supervision train exactly as in run 67. That buys the first
decomposition the programme can state:

```
    run68 - init   =  everything EXCEPT the writer's own RL gradient
    run67 - run68  =  the writer's own RL gradient
    and the two sum to run 67's measured -1.377 +- 0.496.
```

Precision, so it is not over-claimed later: **the writer is not frozen** -- it shares every weight with
the reader, so reader/SSL gradients still move the writing policy. The partition above is exactly what
is bought, no more.

**(i) THE RUN-51 OBJECTION, FACED BEFORE LAUNCH INSTEAD OF AFTER.** This POSTMORTEM already records the
rule: *"a control that cannot distinguish anything is not a control"* -- run 51 (`--rl-w 0`, run 50's
registered control) was dropped because run 50's writer was at chance, so `run50 - run51` was `0 - 0`
and 13 h of device bought a number with no content. **That objection does not bite here, numerically:**
run 67's writer is nowhere near chance (`own_minus_shuf` +3.56..+2.01) and the quantity under test is a
**-1.377 +- 0.496** deficit. The contrast `run68 - run67` has s.e. ~0.8 per rung and **~0.30 pooled over
seven**, so it resolves the whole 1.38-nat deficit at ~4.6 s.e. -- **the best-powered contrast this
programme has ever been able to run, and it is only well-powered because of G104(d)'s pooling rule.**
But run 51's failure mode survives in one form: if the reader/SSL machinery trains almost nothing with
the writer's gradient deleted, run 68 just reproduces the init and `run68 - init ~ 0` is trivially true.
**So that is a GATE (V @600), registered before the data rather than discovered after it.**

**(j) RUN 68's GATES ARE POOLED, NOT PER-RUNG -- THIS WEEK'S OWN RULE APPLIED TO ITS FIRST NEW RUN.**
X0 @200 divergence vs run 67 (hard stop, 7(g)); **V @600** validity (held-out mix moved >= 0.010 OR
`span` moved >= 1.00 from run 67's +16.08; fail -> stop, and report it as a finding about the DESIGN);
**P1 @1000** pooled over three rungs, s.e. ~0.46, `|pooled(run68-run67)| >= 0.92` -> answered, stop at
1000 rather than spend 800 steps confirming a settled sign; **P2 @1800** pooled over all seven,
s.e. ~0.30, `>= +0.60` H_RL / `<= -0.60` RL is protective / `|.| < 0.60` **H_READER: the RL term is not
the cause and the erosion lives in the slot/reader machinery every run 60-67 shares and no gate has ever
varied.** **All three P2 branches are informative and each names a different successor** -- that
property is what makes this control worth 8 h when run 51 was not, and it is written down now so it can
be checked later. **No win bar is registered and none may be added**: run 68 has no writer training, so
a high score here would only mean the init was good, which is already known.

**(k) THE DEAD CLAUSES, WRITTEN DOWN FIRST (G84).** On run 68, `H4` and 3b's `reg%`/spread clauses are
**dead by construction**: with `w3 == 0` there is no reward pressure on the writer, so the absorbing
shortening state of runs 65/66/67 cannot occur, and a writer with no gradient cannot degenerate into a
register. If the note shortens anyway that is reader-side drift and it is a **result**, not a derail.
3c stays dead (`--span-sched`). Live, because the reader/SSL loss still trains every weight: held-out
mix <= 3.025 and <= +0.02 per 100 sustained over 300; ntp1 > 4.0; NaN/inf; `NRT_`/Traceback.

---

## G106 (2026-10-01 10:07 tick) -- RUN 68 IS HEALTHY AT STEP 350/1800 AND IT HAS ALREADY ANSWERED A QUESTION FOR FREE, BEFORE ITS FIRST SCIENCE GATE: **THE IN-TRAINING EROSION SURVIVES THE DELETION OF THE WRITER'S ENTIRE RL GRADIENT, AT FULL STRENGTH.** NO ACTION TAKEN, NO BAR CHANGED, NO CLAUSE ADDED.

**(a) OPS FIRST, BECAUSE A GATE THAT FIRES BY ITSELF IS THE ONLY KIND THAT COUNTS. X0 @200 PASSED
UNATTENDED AT 09:23:10**: `run68 X0 @200: weights DIFFER from run 67 -- removing the RL term reaches the
gradient. PASS.` Third consecutive run whose armed daemon did its job without a supervisor present.
Health at step 350, every LIVE clause (G105(k)) checked and every one passing: held-out mix
**2.7749 -> 2.7839 over 300 steps = +0.0030 per 100** (bars: <= 3.025 absolute, <= +0.02 per 100
sustained over 300); `ntp1` roll20 **2.021** (bar > 4.0); **zero NaN/inf in 347 rows**; **zero
`NRT_`/Traceback** in `train.log`; 65 procs; 17.9 s/step, ETA ~17:15; disk 592 G; daemons
`armed_probes_r68.sh` (1719329) and `archive_r68.sh` (1719322) alive, plot daemon re-pointed to run 68,
all five crons present. `len_mean` **15.0 and flat**, `reg%` 1.1 -- as G105(k) predicted by construction,
with `w3 == 0` the absorbing shortening state of runs 65/66/67 does not occur.

**(b) AN IDENTITY CHECK NOBODY ASKED FOR, AND IT CORROBORATES X0 FROM THE OTHER SIDE.** Run 67 sets
`--rl-start 100`, so runs 67 and 68 must be bit-identical for their first 100 steps and must diverge at
exactly step 100. Measured on `D := mean(g_r0 - gx_r0)` step by step: **max |D(run68) - D(run67)| over
steps 0-99 is exactly 0.00e+00**, and the bands differ from step 100 onward. X0 says the weights differ
at 200; this says they differ *starting where the flag starts acting and not one step earlier*. Two
independent confirmations that `--rl-w 0.0` reaches the gradient, which is what killed 514 steps of
run 66 (7(g)).

**(c) THE FINDING. D FALLS IN RUN 68 AT LEAST AS FAST AS IN RUN 67, AND THE TWO TRAJECTORIES ARE THE
SAME TRAJECTORY.** All of this is **inside the single `--span-sched` block 0 (o=-15, steps 0-399)**, so
G105's own new rule -- never read D across a `--span-sched` boundary -- is respected, and only bands
>= 100 carry information (below that the runs are identical by construction, per (b)).
```
   band      run 68 D        run 67 D       run 67 rl_w
    0-49    +2.51+-0.27     +2.51+-0.27     0.0   <- identical by construction
   50-99    +2.45+-0.21     +2.45+-0.21     0.0   <- identical by construction
  100-149   +2.26+-0.21     +2.18+-0.24     ramping
  150-199   +1.91+-0.17     +1.99+-0.16     ramping
  200-249   +1.96+-0.17     +2.13+-0.16     ramping
  250-299   +1.79+-0.21     +1.73+-0.19     ramping
  300-349   +1.47+-0.19     +1.77+-0.16     0.5 (full)
  -------------------------------------------------------
  SLOPE, steps 100-349, weighted per-step fit:
      run 68  -0.339 +- 0.118 per 100      (2.9 s.e. below zero, with NO writer gradient at all)
      run 67  -0.237 +- 0.114 per 100
  MATCHED per-step difference D(run68) - D(run67), steps 100-349:
      -0.081 +- 0.065   (n=250; serially correlated, so this s.e. is optimistic -- read it as
                         "the same trajectory", not as a significance test)
```
**G101's ATTRIBUTION IS CONTRADICTED, AND IT WAS MY OWN.** At 00:35 I read run 67's falling block-0 D
and wrote that the cause was *"the MISPLACED note improved by MORE than the own note -- run 66's G100
pathology in the training metric of the run built to prevent it"*, hedged with *"rl weight only reached
its full 0.5 at step 300, so block 0 confounds the ramp with the trend, and no bar may be set from it."*
**The hedge was right to exist and the confound now resolves the opposite way from what it implied:**
the fall is not the ramp and it is not the reward, because **deleting the reward term entirely leaves the
fall intact and marginally steeper.** The information here is asymmetric and that is worth stating --
run 67's block-0 slope was confounded by its ramp, run 68's has no ramp to be confounded by, so a fall
in run 68 is the clean measurement and the one that carries the inference.

**(d) THE MECHANISM, DECOMPOSED, AND IT IS THE SAME MECHANISM IN BOTH RUNS. THE CONTROLS IMPROVE FASTER
THAN THE OWN NOTE.** D is a difference, so its fall could be the own note getting worse or the
comparator getting better. It is entirely the latter:
```
             run 68 (no writer gradient)        run 67 (full RL)
  band      g_r0    gx_r0   g_shuf0          g_r0    gx_r0   g_shuf0
  100-149   +0.78   -1.48    +0.12           +0.59   -1.59    +0.05
  300-349   +2.59   +1.11    +2.15           +2.19   +0.42    +1.54
  change    +1.81   +2.59    +2.03           +1.60   +2.01    +1.49
```
In **both** runs the model gets better at exploiting a note that belongs to a *neighbouring* anchor
(`gx_r0`), and better at exploiting a *shuffled* note (`g_shuf0`), **faster than it gets better at
exploiting its own** (`g_r0`). The own-note advantage is not destroyed; it is **outrun by a generic
slot-reading skill**. And the writer's reward gradient is not needed for any of it: run 68's `g_shuf0`
rises **+2.03 against run 67's +1.49**, i.e. the no-RL control erodes the own-vs-shuffled margin
*slightly faster*. `read_ce` is flat at ~2.13 in both; `or_g` is flat.

**(e) WHAT THIS IS AND IS NOT, AND A PREDICTION REGISTERED NOW SO IT CANNOT BECOME A STORY LATER.**
It is **not** the gated quantity. The gate reads `own_minus_shuf` on **132 unseen FineMath anchors at the
offset the checkpoint was trained at**; this reads train anchors at o=-15 with span supervision live, and
the training `g_shuf0` is a per-rollout shuffle rather than the probe's construction. Different
distributions, suggestive link, nothing more -- so **no bar moves, no clause is added, V/P1/P2 stand
exactly as registered in G105(j)**, and the second-draw error of G103 is not committed. But a reading
this clean should be allowed to predict, and a prediction written before the data is free:
**on the strength of (c) and (d) I expect `pooled(run68 - run67)` to land in the H_READER band
(`|.| < 0.60`) or negative, and I expect `run68 - init` to show an erosion of the same order as run 67's
-1.377.** If instead P1/P2 come back `>= +0.92 / +0.60` (H_RL), **this entry is the thing that was
wrong**, and the mechanism in (d) will need a reason why it fails to transfer to held-out anchors.
**One honest caveat on the gate that fires next:** V @600's held-out-mix arm is already at +0.0090
against a +0.010 bar at step 300 and rising ~+0.003 per 100, so **V will almost certainly pass on that
arm alone, and a V pass therefore means only "the weights moved" -- it does NOT establish that the
reader/slot machinery learned anything about the slot.** That is a weakness in a gate I wrote, recorded
before it fires rather than after; the `span` arm is the informative one, and the real validity evidence
is the rung-600 diagnostic beside it.

### G107 (2026-10-01 12:07 tick) -- V @600 passes on the arm that means nothing, and G106's "same trajectory" is wrong: the writer's RL term buys SPECIFICITY and pays for it in RAW USEFULNESS

**(a) V @600 PASSED AT 11:37:32, UNATTENDED, EXACTLY THE WAY G106 SAID IN WRITING IT WOULD.** Held-out
mix moved **+0.0179** against a 0.010 bar; `span` **+15.89** against run 67's +16.08, i.e. moved **0.19**
against a 1.00 bar. It passed on the mix arm **alone** and the informative arm did not move. G106 wrote
that down *before* the gate fired, and the payoff is concrete: the number is read as **"the weights
moved"** and not as "the reader learned something about the slot". That is the whole value of naming a
gate's weakness in advance -- the alternative is a tick that reads `V_PASS` and believes it.

**(b) THE RUNG-600 DIAGNOSTIC, NO BAR ATTACHED (G104(d): the enforced read is pooled).**
`own_minus_shuf` **+2.14 +- 0.45** at o=-13; **vs run 67 (matched step and offset) -1.42 +- 0.70**;
**vs the untrained init -2.95 +- 0.88**. `clean +18.48`, `own +10.10`.

**(c) G106's CENTRAL READING IS WRONG AND THIS IS THE CORRECTION.** G106 pooled the matched per-step
`D := g_r0 - gx_r0` difference over steps 100-349, got **-0.081 +- 0.065**, and called the two runs "the
same trajectory". Measured again at step 726, **within each `--span-sched` block and never across one**
(G105's own rule):
```
  MATCHED per-step D(run68) - D(run67)
    block o=-15   steps   0- 399   n=400    -0.092 +- 0.045
    block o=-13   steps 400- 599   n=200    -0.570 +- 0.093
    block o=-11   steps 600- 726   n=127    -1.127 +- 0.144      <- 12x the first block
  MATCHED per-step d(own_minus_shuf)
    o=-15  -0.086 +- 0.035      o=-13  -0.275 +- 0.072      o=-11  -0.436 +- 0.093
  MATCHED per-step d(g_r0), THE ABSOLUTE GAIN -- AND IT GOES THE OTHER WAY
    o=-15  +0.068 +- 0.045      o=-13  +1.911 +- 0.141      o=-11  +2.986 +- 0.272
```
G106 read the flat *start* of a trajectory as the trajectory. The runs had not separated at step 350;
by 726 they plainly have, and the separation is monotone in the curriculum.

**(d) THE FIRST TIME THIS PROGRAMME HAS MEASURED THE WRITER'S RL TERM DOING ANYTHING AT ALL, AND IT
DOES TWO OPPOSITE THINGS.** Deleting the writer's entire RL gradient makes the note **more useful in
absolute terms** (`g_r0` +9.64 at steps 700-799 vs run 67's +4.17; matched `d(g_r0) = +2.99 +- 0.27`)
and **less specific to the anchor it was written for** (`d(own-shuf) = -0.44 +- 0.09`; run 68's
`g_shuf0` reaches +9.39 where run 67's is +4.24). **So the RL term under `--xanchor-w 1.0` buys
SPECIFICITY and pays for it in RAW USEFULNESS.** It is not the classic register pathology: `reg%` 0.9,
`len_mean` 15.0 and flat. It is a note that helps whatever slot it is dropped into.

**(e) WHAT SURVIVES OF G106 AND WHAT DOES NOT.** The mechanism survives: *the controls improve faster
than the own note* is still the named account of the ~1.36-nat erosion. The sentence that dies is
G106's *"the writer's reward gradient is not needed for any of it"* -- it is needed for part of it, and
that part now has a sign and a magnitude.

**(f) G106's PREDICTION WAS HEDGED INTO NEAR-UNFALSIFIABILITY, AND IT IS BEING SCORED AS SUCH RATHER
THAN COLLECTED AS A HIT.** It read: *"I expect `pooled(run68 - run67)` to land in the H_READER band
(`|.| < 0.60`) **or negative**"* -- which is every outcome except H_RL. The only thing it risked was
H_RL; H_RL is indeed being ruled out; **that single bit is all the prediction earned.** New standing
habit, earned here: **a registered prediction names ONE band.** A disjunction covering the complement
of one branch is not a prediction.

**(g) THE P1 CLAUSE IS HONOURED AS WRITTEN, AND THE COST OF HONOURING IT IS STATED.** Rung 600 is
-1.42, so unless rungs 800 and 1000 reverse hard, `agent/armed_probes_r68.sh` will fire
`P1_ANSWERED RL TERM IS PROTECTIVE` and end run 68 at step 1000. It is **not** overridden to harvest
seven rungs (G103 applies to clauses that help me too). The price: `run68 - init` is then pooled over
**3** matched rungs rather than 7, so the decomposition
`(run68 - init) + (run67 - run68) = (run67 - init)` must be reported restricted to those three rungs.

**(h) THE SUCCESSOR, WRITTEN WITH NO P1 DATA IN HAND -- WHICH IS THE ONLY STATE IN WHICH WRITING IT IS
WORTH ANYTHING.** Run 69 `think69_rlw1` = run 67 + **one flag**, `--rl-w 0.5 -> 1.0` (FLAGS diff vs run
67 verified to be exactly one line). It is a **dose-response on the only term in the objective that
contains the gated quantity** -- `REW = G - len_cost*S0 - xanchor_w*GX` (`think_train.py:922`), writer
gradient `rl_w_eff * A * row_rl` (`:1049`) -- so G100 is satisfied squarely: not a gate the objective
lacks, but the weight on the term that has it. Config, armed daemon and archiver all written and
syntax-checked this tick; **launch is CONDITIONAL on branch A** (`P1_ANSWERED RL TERM IS PROTECTIVE` or
`P2 FINAL: the RL term is net PROTECTIVE`) and on H_READER/H_RL it must **not** be launched. Full gate
table in `runs/think69_rlw1/config.sh`; summary in SUPERVISOR 00S.0's 12:07 amendment.
**The weakest link, named before the run rather than after:** the two dose points are rl_w **0.0** and
**0.5**, and **0.0 is degenerate** -- at zero the writer has no gradient at all, so the 0.0 -> 0.5
difference contains *turning the writer's learning on* and not only *more dose*. A function may not be
extrapolated through a point where its mechanism is absent. The arithmetic that motivates run 69
(0.0 -> init-2.95, 0.5 -> init-1.377, slope ~+1.5 nats per 0.5 of dose, so rl_w 1.0 extrapolates to
~parity with the untrained model) is recorded as the **motive** and explicitly **not** as the bar.
**The bar is the sign.**
**7(f) `--adv-sig-floor 0.5` is CONDITIONALLY REOPENED by branch A, with the trigger written down now
so it cannot be retrofitted later.** It was retired at 08:07 today *on data*: "a 2.5x healthier writer
bought 0.03 +- 0.29 nats", i.e. its expected effect was bounded by a quantity **measured to be zero**.
Branch A destroys that premise by measuring the writer's RL term at ~1.4 nats, and 7(f)'s entire
function is to stop that term from dying. So **7(f) becomes run 70 if and only if run 69 hits H4's
absorbing state or fails Q3's `clean >= 8.0` before step 1400**; if run 69 completes 1800 in health,
7(f) stays retired on its own 02:30 caveat (a tie reached while healthy means its failure mode never
occurred). It is not bundled into run 69: one change per run.

### G108 (2026-10-01 14:07) -- run 68 ENDED on its own P1 clause: the writer's RL term is PROTECTIVE. And the decomposition exposes that `own_minus_shuf` has been bought by PUSHING THE CONTROL DOWN.

**(a) THE RESULT, AND IT IS THE PROGRAMME'S FIRST MEASURED CAUSE.** `agent/armed_probes_r68.sh` fired
P1 @1000 at 13:43:53 unattended and ended run 68 at step 1026 by `finish_done`, exactly as the 12:07
tick predicted in writing (it named 13:45-14:00 and the branch). Three matched rungs, same 132 unseen
FineMath anchors, each checkpoint read at the offset it was trained at:
```
 step  off   run69-era comparators        run 68      run 67      run68-run67     run68-init
  600  -13   own-shuf                    +2.14+-0.45  +3.56+-0.53  -1.42+-0.70   -2.95+-0.88
  800  -11   own-shuf                    +2.38+-0.37  +3.56+-0.61  -1.18+-0.71   -3.83+-1.23
 1000   -9   own-shuf                    +1.58+-0.43  +2.98+-0.62  -1.40+-0.75   -2.17+-1.22
 -------------------------------------------------------------------------------------------
 POOLED over 3                                                     -1.334+-0.417 -2.984+-0.646
                                                                    z = -3.20     z = -4.62
```
**Deleting the writer's entire RL gradient makes the erosion WORSE, by 1.33 +- 0.42 nats.** The term is
net protective. H_RL -- "the writer's RL gradient IS the erosion" -- is REFUTED with a sign, and so is
G106's sentence *"the writer's reward gradient is not needed for any of it."*

**(b) THE DECOMPOSITION, REPORTED RESTRICTED TO THE 3 RUNGS P1 BOUGHT (the cost of honouring the clause
per G103, stated rather than quietly dropped).** Over rungs 600/800/1000 only:
```
   (run68 - init)   = -2.984 +- 0.646     everything EXCEPT the writer's own RL gradient
 + (run67 - run68)  = +1.334 +- 0.417     the writer's own RL gradient
 = (run67 - init)   = -1.650 +- 0.690     (z -2.39; the 7-rung figure is -1.377 +- 0.496)
```
So the erosion is **reader/slot-side and the writer's RL term claws back about 45% of it.** The 7-rung
version of the left-hand term does not exist and will not: the run stopped at 1000 as registered.

**(c) THE THING THAT MATTERS MORE THAN (a), AND IT IS A RULER PROBLEM IN THE GATED QUANTITY ITSELF.**
Put the two runs' absolute decompositions side by side -- all already in `runs/probes/armed.log`, none
of it needing a new probe:
```
 rung    own r67   own r68   |  shuf r67  shuf r68  |  span r67  span r68  |  clean r67  clean r68
  600     +7.37    +10.10    |   +3.81     +7.96    |   +16.08    +15.89   |   +18.52     +18.48
  800     +6.53    +13.86    |   +2.97    +11.48    |   +16.20    +20.77   |   +17.54     +21.66
 1000     +2.73    +15.28    |   -0.25    +13.70    |   +14.46    +24.08   |   +13.76     +22.83
 1800     -0.37      (n/a)   |   -2.38      (n/a)   |   +16.52      (n/a)  |    +7.69       (n/a)
```
(`shuf := own - own_minus_shuf`.) At rung 600 the two runs are the SAME on every absolute: own within
2.7, span and clean within 0.2. By rung 1000 they are not the same run at all. **Run 67's own-note gain
falls +7.37 -> +2.73 -> -0.37 while run 68's rises +10.10 -> +15.28.** Run 67's no-note ceiling `clean`
falls +18.52 -> +7.69 (failing Q3 at 1800) while run 68's rises to +22.83.
**And run 67's positive margin is built out of the control going NEGATIVE.** At rung 1000 its
`own_minus_shuf` is **+2.98** while its own note is worth **+2.73** and the shuffled note **-0.25**; at
rung 1800 the margin is **+2.01** while **the own note is worth -0.37, i.e. the note is worse than no
note at all.** Run 68, with no writer gradient, has a smaller margin (+1.58) because **both** arms are
large (+15.28 / +13.70).
**So `--xanchor-w 1.0` with `--rl-w 0.5` does not teach the model to write a better note. It teaches it
to stop being able to use notes, and the own-note arm merely degrades more slowly than the control.**
A margin can be won by lifting the numerator or by destroying the denominator, and this one was the
second. That is the same family of error as G94's broken ruler and it is in the quantity this entire
programme has gated on since run 60.

**(d) WHAT THIS DOES AND DOES NOT DO TO (a).** It does **not** overturn it: `run68-run67` is a matched
contrast and the protective sign stands. What it changes is **what the sign is protecting.** The honest
sentence is: *the writer's RL gradient slows the decay of the own-note arm relative to the control, in a
regime where both arms and the whole reader are decaying.* It is not "the RL term makes the note good".

**(e) RUN 69 IS LAUNCHED ANYWAY, AND WHY THAT IS THE RIGHT CALL RATHER THAN A FLINCH FROM (c).** Branch
A fired as pre-registered, so run 69 `think69_rlw1` (`--rl-w 0.5 -> 1.0`, FLAGS diff verified to be
exactly one line against `runs/think67_xanchor/config.sh`) went live at 14:08; smoke passed with
`rl 1.00`, 65 procs, `--rl-w 1.0` confirmed in the live cmdline, step 11/1800 at 17.9 s/step, ETA
~23:10. Not stopping it is deliberate: (c) is an analysis of the PREVIOUS run's data, and **run 69 is
the experiment that settles (c) decisively.** Doubling the dose doubles whatever mechanism produced
run 67's collapse, so (c) makes a sharp prediction that costs nothing to collect and would otherwise
need a run of its own. Killing a run 7 minutes after launching it, on data that existed before it
launched, is the churn the runbook forbids.

**(f) THE READ THAT MUST ACCOMPANY RUN 69's VERDICT -- REGISTERED NOW, NO BAR MOVED, NO CLAUSE ADDED.**
The W gates in `runs/think69_rlw1/config.sh` stay exactly as written at 12:07. But **no run-69 verdict
may be stated without the two-arm decomposition beside it**, and the numbers are already logged per
rung by `agent/armed_probes_r69.sh` (`own` and `own-shuf`, hence `shuf`), so this needs no code change
to an armed daemon mid-run:
```
  If pooled(run69-run67) > 0 AND the own-note arm is LOWER than run 67's at the matched rungs,
  the dose "worked" by destroying the control and MUST be reported as
  "won on a shrunken ceiling", the same words Q3 uses -- NOT as a dose-response success.
  A W(ii) win at any rung with clean < 8.0 is already VOID by the registered rule.
```

**(g) ONE BAND, NAMED, BECAUSE G107(6) CONVICTED ME OF HEDGING INTO UNFALSIFIABILITY.** Not "or
negative", not a disaster floor -- one band each, and the arithmetic is against me:
```
  pooled(run69-run67) over 7:  I predict  [+0.0, +0.9]  -- positive but SMALLER than the +1.334 the
      0.0->0.5 step bought, because that step included TURNING THE WRITER ON (the degenerate-point
      caveat written at 12:07), so the second half-unit of dose must buy less than the first.
      This lands in the registered SATURATED band or weakly in `confirmed`, and NOT in `harmful`.
  pooled(run69-init) over 7:   I predict  < 0.00  -- i.e. the registered "STILL ERODING" branch.
      -1.650 + at most +1.334 = -0.32 at best, and the win bar is +1.00. **The run's own motivating
      arithmetic does not reach its own win bar.** The 12:07 motive said rl_w 1.0 extrapolates to
      "~init parity"; parity is 0.00, and the bar is +1.00. I am saying so BEFORE the data, not after.
  AND THE MECHANISM PREDICTION FROM (c), which is the one I actually care about:
      run 69's `own`, `span` and `clean` will fall FASTER than run 67's, and I expect Q3
      (`clean >= 8.0`) to FAIL before step 1400 -- which by the 12:07 clause makes 7(f) run 70.
  IF pooled(run69-run67) comes back >= +1.3 WITH `own` and `clean` HOLDING, (c) is what was wrong,
  and the dose-response is real in the direction the programme wanted.
```

## G109 (2026-10-01 16:07 headless tick, run 69 at step ~405/1800) -- **D1 PASSED AND THE IN-TRAINING DOSE-RESPONSE IS MONOTONE OVER THREE DOSE POINTS. BUT THE SAME ROWS SHOW THE DOUBLE DOSE BUYING THAT MARGIN WHILE THE TRUE-SPAN CEILING `or_g` FALLS 0.36 NATS FASTER THAN RUN 67's, 5 s.e. -- WHICH IS G108(3)'s WORRY ARRIVING THROUGH A DIFFERENT DOOR: NOT THE CONTROL BEING WRECKED, THE READER BEING WRECKED. NO BAR ADDED, NO RUN STOPPED; Q3 @600 IS THE REGISTERED GUARD AND IT FIRES IN AN HOUR.**

**(a) D1 @400 PASSED UNATTENDED AT 16:16, INSIDE THE BAND PREDICTED AT 12:07 BEFORE THE RUN EXISTED.**
`armed.log`: matched per-step `D := g_r0 - gx_r0`, run 69 minus run 67, over all 300 steps of the single
`--span-sched` block o=-15 (steps 100-399, G105's rule respected) = **+0.093 +- 0.070**, bar **>= -0.123**.
The bar is not a round number: it is **what run 68 at ZERO dose measured on this exact window**. So the
three dose points read, on one instrument and one window:
```
   rl_w 0.0  ->  -0.123 +- 0.061     (run 68, measured 12:07)
   rl_w 0.5  ->   0.000              (run 67, the reference by construction)
   rl_w 1.0  ->  +0.093 +- 0.070     (run 69, measured 16:16)
```
**Monotone in the dose, with a SHRINKING increment (+0.123 then +0.093).** The shape is the one the
14:07 prediction committed to in writing for the held-out quantity -- "the second half-unit must buy
less than the first". Two caveats, stated with the result and not after it: the two increments are
**not distinguishable** from each other at these s.e., and `D` is a **train-anchor statistic at o=-15**,
not the gated quantity (G106's caveat), which is why D1 is labelled a derail gate in its own daemon.

**(b) THE SAME MATCHED ROWS, READ FOR THE THING 14:07 SAID TO WATCH, AND THEY DO NOT SAY WHAT (3) SAID
THEY WOULD.** G108(3) predicted the dose would win the margin by pushing the **shuffled control** down.
Matched per-step, run 69 minus run 67, inside block o=-15:
```
   d(g_r0)      the own-note arm        100-378  +0.078 +- 0.069   |  300-378  +0.204 +- 0.128
   d(g_shuf0)   the shuffled control    100-378  -0.096 +- 0.053   |  300-378  -0.067 +- 0.111
   d(own-shuf)  the gated margin        100-378  +0.174 +- 0.053   |  300-378  +0.271 +- 0.082
```
**The own arm goes UP and the control barely moves.** On this instrument the double dose is lifting the
numerator, which is the honest direction, and **my own 14:07 mechanism claim is not what is happening
here.** Said plainly rather than quietly dropped.

**(c) AND THE ROW THAT MATTERS, WHICH I ONLY FOUND BECAUSE `orG` MOVED ON THE STATUS LINE (-1.46 ->
-2.95 while run 67 sat at -1.88). `or_g` IS THE TRUE-SPAN GAIN -- THE IN-TRAINING COUSIN OF THE PROBE'S
`clean`/`span` CEILING -- AND IT IS FALLING FASTER IN RUN 69, MONOTONICALLY, AT 5-9 s.e.**
```
   matched d(or_g) = or_g(run69) - or_g(run67), same steps
     steps 100-199   run69 -1.165   run67 -1.005   d  -0.161 +- 0.022   (n=100)
     steps 200-299   run69 -1.785   run67 -1.409   d  -0.376 +- 0.042   (n=100)
     steps 300-399   run69 -2.286   run67 -1.876   d  -0.410 +- 0.063   (n=100, z -6.5)
   d(read_ce)  300-399  +0.033 +- 0.003    the reader behind a TRUE span is measurably worse too
```
**CORRECTED AT 16:25, WITHIN THE SAME TICK, AGAINST MY OWN FIRST NUMBER.** The third band was first
computed as **300-404**, which **crosses the `--span-sched` boundary at step 400** (o=-15 -> o=-13) and
therefore breaks G105's own rule. Restricted strictly to the block it belongs to it is **-0.410 +- 0.063
(z -6.5)**, not -0.355: the boundary crossing had **diluted** the effect, so the finding strengthens.
**AND THE BOUNDARY ITSELF IS A LIVE CAUTION AGAINST (c), WHICH IS WHY IT IS RECORDED HERE AND NOT
OMITTED:** over the first 19 steps of the NEW block o=-13, matched `d(or_g)` is **+0.592 +- 0.091** --
**the sign flips and run 69 is AHEAD.** n=19, and the opening steps of a block are a transient, so this
is **not** read as a result either way. It is a candidate instance of (e)'s own named falsifier
("`d(or_g)` flat over 400-600"), it is not flat, and it is in the *favourable* direction.
**THE 18:07 TICK MUST RE-READ `d(or_g)` OVER THE WHOLE OF BLOCK o=-13 (STEPS 400-599) BEFORE TREATING
(c) AS ESTABLISHED.** On the same strict window `d(own-shuf)` is **+0.235 +- 0.080** and `d(ntp1)` is
**+0.015 +- 0.001**.
**So the double dose buys +0.24 nats of margin and pays -0.41 nats of true-span ceiling, and the ceiling
term is the one with 6.5 s.e. behind it.** That is **G108(3)'s pathology arriving through a different
door**: not "the margin was won by wrecking the control" but **"the margin is being won while the
reader's ability to use a CORRECT note erodes faster"**. Same family -- a ratio improved by damaging
what it is divided by -- and the programme already has the right guard for it, written before this data
existed: **Q3, `clean >= 8.0` at every rung, and a win at a rung with `clean < 8.0` is VOID.**
**Scale caveat, because the two instruments are not interchangeable:** `or_g` is a train-anchor quantity
near -2 nats; the probe's `clean` is a held-out absolute near +18 at rung 600. `or_g` predicts the
**direction** of `clean`'s motion and says nothing about its value. Run 67 itself finished with
`clean +7.69` and FAILED Q3 at 1800.

**(d) WHAT I DID NOT DO, AND WHY IT IS THE WHOLE DECISION OF THIS TICK.** I did **not** stop run 69 and
I did **not** write an `or_g` bar. D1 is the registered derail gate for this window and it **passed**;
(c) has no registered bar, and inventing one at 16:07 out of the data that suggested it is exactly the
retrofit G100 and G103 exist to forbid. The registered guard for (c) is Q3 at rung 600, it is checked
inside the armed daemon, and it fires around **17:15** -- about an hour. **A finding that sharpens a
pre-registered prediction is not a licence to add a clause to a live run.**

**(e) THE PREDICTION, SHARPENED, ONE BAND, AND TIME-STAMPED BEFORE RUNG 600 EXISTS.** From (c)'s slope:
run 69's rung-600 `clean` lands **below run 67's +18.52**, in **[+13, +18]** -- degraded but NOT yet
voiding, because 0.36 nats of train-anchor drift is small against an +18.5 ceiling this early. Q3 fails
**between rungs 1200 and 1600** (14:07 said "before 1400"; this is the same claim with the band narrowed
and no bar moved). Rung-600 `own_minus_shuf` lands **above** run 67's +3.56 by **+0.1 to +0.5**, the
held-out echo of (b)'s +0.27. **FALSIFIER for the whole of (c): rung 600 comes back with `clean >= 18.5`
AND `d(or_g)` flat over steps 400-600 -- then `or_g`'s drift is a block-boundary artefact and (c) is
noise I promoted.**

**(f) HEALTH, CLAUSE BY CLAUSE, ALL PASSING AT STEP 405.** held-out mix **2.793 @300**, drift
**+0.0063 per 100** (bars <= 3.025, <= +0.02/100 over 300); `ntp1` roll20 **1.826** (bar > 4.0) and
matched `d(ntp1) = +0.014 +- 0.001` vs run 67, i.e. **the double dose costs a real but tiny amount of
next-token prediction** -- which is the trade Emanuele authorised, recorded as a price paid rather than
a loss hidden; **H4 far away** (roll50 `len_mean` **15.00** vs bar 3.5, `g_sd` **2.283** vs bar 0.30);
reg% **1.2** (bar 15); **0 NaN/inf in 405 rows**; 0 `NRT_`/Traceback; **65 procs**; both daemons alive;
17.8 s/step, ETA **~23:10**; disk 547 G; 6 crons.

## G110 (2026-10-01 18:07 headless tick, run 69 at step ~755/1800) -- **G109(4)'s MEASUREMENT REPLICATED AT z -7.4 AND ITS CONCLUSION IS DEAD. `or_g` ERODED HARDER OVER THE FULL BLOCK (-0.589 +- 0.080) AND THE HELD-OUT CEILING IT WAS SUPPOSED TO PREDICT WENT *UP* +1.00 NATS. I WROTE THE FALSIFIER AS A CONJUNCTION AND THAT IS WHY THE CLAIM SURVIVED A TEST IT SHOULD HAVE FAILED -- SO I AM RETIRING IT ON THE SINGLE CLAUSE THAT DECIDED IT.**

**(a) THE TEST THE 16:25 ADDENDUM ORDERED, RUN FIRST, AS INSTRUCTED.** The addendum saw `d(or_g) = +0.592 +- 0.091` over the first 19 steps of block `o=-13` and wrote: *"re-read matched `d(or_g)` over the WHOLE of block o=-13 (steps 400-599) BEFORE reading rung 600, and if it stays positive then (4) was a block-position artefact."* Over the whole block:

```
   matched d(or_g) = or_g(run69) - or_g(run67), by --span-sched block (G105: never cross a boundary)
     o=-15   0-399    -0.237 +- 0.021   n=400   z -11.1
     o=-13   400-599  -0.589 +- 0.080   n=200   z  -7.4      <-- the ordered re-read
     o=-11   600-755  -0.378 +- 0.064   n=156   z  -5.9
   within o=-13, 50-step quarters:  -0.487 / -0.287 / -0.431 / -1.153   (all negative)
```

It did **not** stay positive. The +0.592 was a 19-step block-opening transient, exactly as the addendum
flagged, and the full-block erosion (**-0.589**) is *worse* than the -0.410 the addendum reported over
400-499. **So the `or_g` measurement in G109(4) replicates and strengthens for the third block running.**

**(b) AND THE CONCLUSION BUILT ON IT IS REFUTED, BY THE RUNG THAT LANDED 63 MINUTES LATER.** G109(4)
said `or_g` is *"the in-training cousin of the probe's `clean`/`span` ceiling"* and committed, in writing,
that it *"predicts the **direction** of `clean`'s motion"*. Direction predicted: **down**. Rung 600, read
at `o=-13`, the offset it was trained at (G93):

```
   run69 600_o-13   own +9.55+-0.73 | own-shuf +5.01+-0.76 | filler +2.62 | span +17.07 | clean +19.52
   run67 600_o-13   own +7.37+-0.63 | own-shuf +3.56+-0.53 | filler +3.26 | span +16.08 | clean +18.52
                                                                            span +0.99   clean +1.00
```

`clean` went **UP +1.00** and `span` **UP +0.99** while the train-anchor cousin fell 0.59 nats at 7.4 s.e.
**The bridge from `or_g` to the held-out ceiling is broken in the direction that matters.** Q3
(`clean >= 8.0`) passes with 11.5 nats of headroom, not marginally.

**(c) THE PROCESS FAULT, WHICH IS THE PART WORTH KEEPING.** G109(6)'s falsifier was
*"rung 600 returns `clean >= 18.5` **AND** `d(or_g)` flat over 400-600"*. Clause one **fired** (+19.52);
clause two **did not** (z -7.4). A conjunction, so the formal falsifier is not met -- and **hiding behind
that would be the retrofit G100/G103 exist to forbid, pointed the other way.** The conjunction was
wrong when I wrote it: the two clauses test *different* things. `d(or_g)` flat would have said *the
measurement was noise*; `clean >= 18.5` says *the measurement was real and the inference from it was
wrong*. Only the second was ever load-bearing for the claim. **Rule: when a falsifier conjoins a clause
about the measurement with a clause about the inference, it cannot kill the inference -- split them.**
**G109(4) is RETIRED as a ceiling claim.** What survives it is narrow and stated as such: at double RL
weight the *train-anchor* true-span gain erodes monotonically faster than at half weight, and that fact
now has **no demonstrated consequence** for held-out ability.

**(d) THIS IS G94/G108's LESSON ARRIVING ONCE MORE, AND I WALKED INTO IT WITH THE LESSON WRITTEN DOWN.**
`or_g` and `read_ce` are train anchors measured *with span supervision live*; `clean`/`span` are
held-out absolutes. G93, G105 and G106's own caveat all say train anchors are not the instrument. I
recorded the scale caveat in G109(4) and then reasoned from the quantity anyway. The matched
`d(read_ce)` tells the same story and should be read the same way -- real, monotone, and **not** evidence
about the reader's held-out ability:
```
   d(read_ce)   o=-15 +0.016 +- 0.002  |  o=-13 +0.039 +- 0.003  |  o=-11 +0.048 +- 0.004
```

**(e) BOTH OF G109(6)'s BANDS MISSED, BOTH HIGH, AND THAT IS A BIAS AND NOT TWO ACCIDENTS.**
```
   predicted rung-600 clean          [+13, +18]              actual +19.52   MISS, above
   predicted rung-600 own-shuf       run67 +3.56, +0.1..+0.5 actual +5.01    MISS, above (+1.45)
```
I have now under-predicted run 69 twice in the same direction in one tick's distance. The bands below
are widened for it, and the bias is named so the next tick discounts my centre and not just my width.

**(f) THE RUNG ITSELF, WITH THE ONLY COMPARATOR THE PROGRAMME CLAIMS TO CARE ABOUT (G104).**
```
   run69 600_o-13  vs run 67 (matched step+offset):  +1.45 +- 0.92    (r67 +3.56)
   run69 600_o-13  vs the UNTRAINED init:            -0.08 +- 1.07    (init +5.09)
     for scale: run67 @600 vs init -1.53 | run68 @600 vs init -2.95 +- 0.88
     pooled over 7: run67 -1.377 +- 0.496 | pooled over 3: run68 -2.984 +- 0.646
```
**-0.08 +- 1.07 is the closest any run in this programme has come to the untrained model at this rung** --
and it is **one rung of seven**, the registered read is **pooled at 1800** (W(ii), bar >= +1.00 at 2 s.e.),
and 1.07 nats of s.e. covers both "ahead" and "a nat behind". It is **not** a win and must not be quoted
as one. The registered words for landing in [0.00, +1.00) are *"REACHES the untrained model, does NOT
beat it"*, and even that is not yet earned.

**(g) THE LIVE BLOCK HAS TURNED AGAINST THE DOSE ON THE IN-TRAINING MARGIN, AND THIS IS LOGGED BEFORE
RUNG 800 EXISTS.** The gated quantity's train-anchor shadow, matched, by block:
```
   d(own-shuf)   o=-15   100-378  +0.174 +- 0.053   (G109(3))
                 o=-13   400-599  +0.197 +- 0.076   z +2.6
                 o=-11   600-755  -0.237 +- 0.087   z -2.7      <-- SIGN FLIP
   absolute, o=-11:  run69 +0.443   run67 +0.680
   both arms collapsing together there: d(g_r0) -2.103 +- 0.140 | d(g_shuf0) -1.881 +- 0.132
   writer shortening faster: d(len_mean) o=-11 -0.365 +- 0.043 (run69 14.38 vs run67 14.87; both from 15.00)
```
No bar is attached to this and none may be (G103): the registered read is **P1 @1000, pooled over rungs
600/800/1000, stop only at <= -0.92**. With rung 600 at **+1.45**, P1 can only stop if rungs 800 and 1000
average about **-2.1** each.

**(h) PREDICTIONS, WIDENED FOR (e), TIME-STAMPED 18:07 BEFORE RUNG 800 (~18:30) EXISTS.**
`d(own-shuf)` vs run 67 at **rung 800** in **[-1.0, +2.0]** -- I expect it *below* +1.45 because of (g),
and I decline to commit to a sign. **P1 @1000 CONTINUES** (pooled > -0.92). **W(ii) @1800 pooled over 7
in [-1.5, +1.0]**, centre "REACHES, does not beat". **Q3 holds at every remaining rung** -- `clean` has
11.5 nats of headroom and is rising, so the mechanism that voided run 67 at 1800 is not in play here.
**FALSIFIER, SINGLE-CLAUSE THIS TIME (per (c)): rung 800 returns `d(own-shuf)` vs run 67 below -1.0 ->
the dose advantage was confined to the two deepest offsets and (g)'s train-anchor flip led the held-out
read rather than lagging it.**

**(i) NO DEVICE ACTION, AND NOT FOR WANT OF A CANDIDATE.** Every health clause passes on measurement,
not impression: held-out mix **2.8087** absolute (bar 3.025) drifting **+0.0029 per 100** over steps
400-700 (bar +0.02) and *decelerating*; H4 roll50 `len_mean` **14.10** / `g_sd` **2.332** (bars 3.5 /
0.30); `ntp1` roll20 **2.084** (bar 4.0); zero `NaN`/`inf`, zero `NRT_`/`Traceback`; 64 ranks, 65 workers,
17.8 s/step, ETA ~23:20; 544 G free. Runbook 3b/3c/3d: nothing triggered. Run 70 `think70_frozen`
(FROZEN READER) is armed and its launcher is waiting on run 69's STATUS.

## G111 (2026-10-01 20:07 headless tick, run 69 at step ~1140/1800) -- **MY 18:07 FALSIFIER FIRED AT RUNG 800 (-1.98 +- 0.76, BAR -1.0), SO G110's RETIREMENT OF G109(4) IS ITSELF WITHDRAWN: THE HELD-OUT CEILING FELL -3.81 AND -2.80 NATS BELOW RUN 67 AT THE NEXT TWO RUNGS. I RETIRED A CLAIM ON ONE RUNG IN THE SAME TICK IN WHICH I REFUSED TO CROWN A WIN ON ONE RUNG.**

**(a) WHAT LANDED UNATTENDED, AND THE REGISTERED GATE SAID CONTINUE.** `armed_probes_r69.sh` fired three
rungs without a supervisor present. All numbers from `runs/probes/armed.log`, matched step+offset:
```
 rung  off | own-shuf  |  vs run 67      | vs the UNTRAINED init | own   | filler | span   | clean  | Q3
  600  -13 | +5.01+-.76| +1.45 +- 0.92   | -0.08 +- 1.07         | +9.55 | +2.62  | +17.07 | +19.52 | PASS
  800  -11 | +1.58+-.45| -1.98 +- 0.76   | -4.63 +- 1.25         | +0.96 | +0.67  | +11.41 | +13.73 | PASS
 1000   -9 | +2.22+-.56| -0.76 +- 0.84   | -1.53 +- 1.27         | +0.63 | -0.66  | +10.54 | +10.96 | PASS
  POOLED k=3   run69 - run67 = -0.431 +- 0.486 (z -0.89)   run69 - init = -2.081 +- 0.693 (z -3.00)
```
`P1_CONTINUE` at 19:35:47: pooled is **not** <= -0.92, so the run goes to 1800 for all seven rungs. On
the registered W bands as they stand at k=3: **W(i) SATURATED** (|-0.431| < 0.60), **W(ii) STILL ERODING**
(< 0.00). Both are three-rung previews of a seven-rung read and neither is a verdict.

**(b) THE SINGLE-CLAUSE FALSIFIER I WROTE AT 18:07 FIRED, AND IT IS THE FIRST FALSIFIER IN THIS
PROGRAMME THAT KILLED THE THING IT WAS AIMED AT.** It read, in full: *"rung 800 returns `d(own-shuf)` vs
run 67 below -1.0 -> the dose advantage was confined to the two deepest offsets and (g)'s train-anchor
flip led the held-out read rather than lagging it."* Rung 800 returned **-1.98 +- 0.76**. **So I am
obliged to the consequence and I take it: the dose advantage was confined to o=-15 and o=-13, and the
train anchor LED the held-out read.** The lead is datable, which is what makes it a lead and not a
coincidence: the flip was measurable in-training over steps 600-799 and the rung that confirmed it is the
rung at **step 800**, probed at 18:30 -- after G110 was written at 18:07 asserting the opposite.

**(c) AND THEREFORE G110's CENTRAL ACT -- RETIRING G109(4) AS A CEILING CLAIM -- IS WITHDRAWN. THE
MATCHED CEILING TABLE, WHICH IS THE QUANTITY G109(4) COMMITTED ITS DIRECTION TO:**
```
 rung  off |  clean r69  clean r67   d(clean) |  span r69  span r67   d(span)
  600  -13 |   +19.52     +18.52      +1.00   |   +17.07    +16.08     +0.99     <- the rung G110 decided on
  800  -11 |   +13.73     +17.54      -3.81   |   +11.41    +16.20     -4.79
 1000   -9 |   +10.96     +13.76      -2.80   |   +10.54    +14.46     -3.92
```
G109(4) said the faster in-training erosion of `or_g` predicts the **direction** of the held-out
ceiling's motion. Over three rungs it is **right twice and wrong once, and the two it is right about are
three to five times larger than the one it is wrong about.** Rung 600 was the outlier, and rung 600 was
the entire evidential basis of the retirement. **G109(4) stands, with its scale caveat intact: `or_g` is
a train anchor measured with span supervision live, so it predicts a SIGN and a RANKING, never a
magnitude.**

**(d) THE PROCESS FAULT, AND IT IS THE REASON THIS G EXISTS RATHER THAN A ONE-LINE CORRECTION.** G110
did two things with the same n=1 and held them to opposite standards:
 - on the **favourable** number (-0.08 vs init at rung 600) it wrote *"IT IS ONE RUNG OF SEVEN ... I am
   declining to use the registered [0.00,+1.00) wording on a single rung"* -- correct, and G104(d)'s rule;
 - on the **unfavourable** number (clean +1.00 at the same rung 600) it **retired a standing claim
   outright.**
Same rung, same n, opposite thresholds. **NEW STANDING RULE: G104(d)'s ONE-RUNG EMBARGO BINDS
RETIREMENTS EXACTLY AS IT BINDS WINS. A RUNG THAT IS TOO THIN TO ESTABLISH A CLAIM IS TOO THIN TO RETIRE
ONE.** The asymmetry survived its own tick because it pointed at self-criticism, and self-criticism is
the direction nobody audits -- which makes it a *better* hiding place for a sample-size error than
boasting, not a worse one. **Scepticism aimed at my own claim is still a claim and takes the same n.**

**(e) WHAT SURVIVES G110 UNTOUCHED, AND IT IS THE PART THAT PAID FOR TODAY'S READ.** G110(c): *never
conjoin a measurement clause with an inference clause in a falsifier; split them.* That rule is about the
CONSTRUCTION of a test and is independent of whether the claim under test was true. It is precisely why
this tick could be decided in one line: the 18:07 falsifier was single-clause, one number met it, and
there was no conjunction to shelter behind. **The rule was right and the claim it was used to kill was
wrong, and both of those sentences are allowed to be true at once.**

**(f) THE "ONE-DIRECTIONAL HIGH BIAS" I CONVICTED MYSELF OF AT 18:07 IS REFUTED BY THE NEXT TWO
OBSERVATIONS, BOTH LOW.** Running tally of bands, 16:07 onward:
```
  16:07  rung-600 clean       [+13, +18]      -> +19.52   MISS HIGH
  16:07  rung-600 own-shuf    +0.1 .. +0.5    -> +1.45    MISS HIGH
  18:07  rung-800 d(own-shuf) [-1.0, +2.0]    -> -1.98    MISS LOW
  18:07  W(ii) pooled         [-1.5, +1.0]    -> -2.081 at k=3   MISS LOW (3 of 7 rungs in)
  18:07  P1 @1000 CONTINUES                   -> CONTINUE   HIT
```
**So 18:07's diagnosis ("one-directional, named as a bias, not luck; the next tick should discount my
centre, not just my width") was itself an inference from n=2, and it is wrong in the same way (d) is.**
The defensible reading of two same-direction misses is **my intervals are too NARROW**, which is a
statement about width and needs no direction at all. **STANDING RULE: DO NOT NAME A DIRECTIONAL BIAS
FROM FEWER THAN FOUR BANDS. Two misses buy a wider interval, not a signed correction.**

**(g) THE TRAIN-ANCHOR TABLE, EXTENDED TO FIVE BLOCKS, EVERY BLOCK READ INSIDE ITS OWN `--span-sched`
SEGMENT (G105).** Matched per-step, run 69 minus run 67:
```
    block     off   n  |  d(own-shuf)         |  d(or_g)             |  d(read_ce)    |  d(len_mean)
    0- 399    -15  400 |  +0.127+-0.039 z +3.3|  -0.237+-0.021 z-11.1|  +0.016+-0.001 |  -0.001+-0.001
  400- 599    -13  200 |  +0.197+-0.076 z +2.6|  -0.589+-0.080 z -7.4|  +0.039+-0.003 |  -0.005+-0.003
  600- 799    -11  200 |  -0.238+-0.074 z -3.2|  -0.547+-0.071 z -7.7|  +0.046+-0.003 |  -0.586+-0.072
  800- 999     -9  200 |  -0.412+-0.082 z -5.0|  -0.892+-0.103 z -8.7|  +0.025+-0.004 |  -2.090+-0.136
 1000-1199     -7  141 |  -0.386+-0.086 z -4.5|  -1.259+-0.108 z-11.7|  +0.035+-0.003 |  -1.962+-0.161
```
Three things, and only the first is established. (i) `d(own-shuf)` flipped sign at o=-11 and has **stayed**
flipped for three consecutive blocks -- that is the measurement (b) acted on. (ii) `d(or_g)` is monotone
worse across all five blocks, -0.237 -> -1.259, now at z -11.7. (iii) The writer is **shortening ~2 tokens
faster than run 67** from block o=-11 onward, which is when everything else turned; the note got shorter
and the ceiling fell in the same block. **(iii) IS A CORRELATION ACROSS TWO BLOCKS AND I ATTACH NO BAR TO
IT (G103)** -- I name it because it is the obvious mechanism candidate and because naming it now is the
only way a later tick can be surprised by it.

**(h) A READING CORRECTION ON reg%, BECAUSE THE DERAIL CLAUSE WOULD HAVE BEEN READ OFF A RESOLVED
TRANSIENT.** `agent/status.py` bins reg% in **300-step** blocks and printed **12.0** for block 900-1126 --
into the neighbourhood of 3b's *"> 8% and rising"* clause. Decomposed per 100 steps, with n printed:
```
   800- 899  n= 956  reg% 10.0  uniq% 82.6
   900- 999  n=1016  reg% 24.6  uniq% 57.2   <- '544 numb' x136, a real lock-in
  1000-1099  n= 859  reg%  2.0  uniq% 93.8   <- gone, unaided, inside 100 steps
  1100-1133  n= 287  reg%  2.4  uniq% 95.8
```
The 12.0 is one spike smeared over 300 steps; the live value is **2.4**. **And run 67, the matched run,
did the same thing at steps 300-399 (reg% 10.6, `'ing the '` x123) and was back to 0.8 by step 500** --
so a single-block prefix lock-in that dissolves is design-normal for this family, with precedent in the
very run this one is matched against. **RULE: BEFORE STOPPING ON reg%, DECOMPOSE THE BLOCK INTO 100-STEP
BINS AND PRINT n. A 300-STEP MEAN CANNOT TELL "COLLAPSING NOW" FROM "SPIKED 200 STEPS AGO AND
RECOVERED".** No bar moved: 12.0 is not > 15%, and it had been above 8% for one block, not three, so the
clause never actually fired -- I am recording the resolution so that the NEXT tick, which will see a
300-block mean still carrying this spike, does not stop a healthy run on it.

**(i) HEALTH, EVERY CLAUSE ON A MEASUREMENT.** held-out mix **2.8105** absolute (bar 3.025), drifting
**+0.0011 per 100** over steps 800-1100 (bar +0.02) and still decelerating; H4 roll50 `len_mean` **10.610**
and `g_sd` **2.383** against bars **<= 3.5 AND <= 0.30** -- H4 needs BOTH and `g_sd` is 8x its bar, so H4 is
not close despite the length fall in (g)(iii); `ntp1` roll20 **1.666** (bar 4.0); rollout spread 2.38 (bar
0.6); **0** NaN/inf in 1140 metric rows, **0** `NRT_`/`Traceback` in `train.log`; 64 ranks / 65 workers,
17.4 s/step, **ETA ~23:28**; 1556 GB mem, **536 G** disk; 6 crons; `archive_r69.sh`, `armed_probes_r69.sh`,
`keepalive_think.sh` and run 70's launcher (pid 1935711) all alive. **Runbook 3b/3c/3d: nothing triggered.**

**(j) THE DECISION IS TO CONTINUE, AND THE CASE AGAINST IT DESERVES STATING.** Run 69 is behind the
untrained model by **-2.081 +- 0.693** pooled over three rungs, its ceiling is eroding faster than run
67's, and its own 14:07 arithmetic already conceded it cannot reach W(ii)'s +1.00. A stop now would be
**me inventing a bar from unfavourable data at step 1140**, which is the retrofit G100/G103 exist to
forbid, pointed in the direction that happens to feel humble. **W(i) -- the dose-response, which is the
entire scientific purpose of run 69 -- is genuinely undecided at -0.431 +- 0.486 and needs all seven
rungs.** So: no device action. Run 70 `think70_frozen` is armed and launches itself at run 69's STATUS.

**(k) PREDICTIONS, TIME-STAMPED 20:07 BEFORE RUNG 1200 (~20:35) EXISTS, WIDENED PER (f) AND SIGNED WHERE
I HAVE A MECHANISM.** Run 67's remaining rungs, for the reader: `clean` 1200 **+13.23**, 1400 **+10.28**,
1600 **+9.31**, 1800 **+7.69**; `own-shuf` +3.17 / +2.94 / +2.82 / +2.01.
 - **rung 1200 `d(clean)` vs run 67 NEGATIVE, in [-5.0, -0.5]** -- signed, because (c) now has two
   concordant rungs and (g)(ii) is monotone at z -11.7.
 - **rung 1200 `d(own-shuf)` vs run 67 in [-3.0, +0.5]**, centre ~-1.0.
 - **Q3 `clean >= 8.0` PASSES at rung 1200 and FIRST FAILS at rung 1400.** Arithmetic, stated so it can
   be checked: run 67 @1400 is +10.28 and run 69 is running 2.8-3.8 nats below it.
 - **W(i) @1800, pooled over 7, in [-1.5, -0.2], centre ~-0.75 -> the registered verdict is `<= -0.60`
   NET HARMFUL: `rl_w 0.5` IS A MAXIMUM AND THE DOSE-RESPONSE IS NON-MONOTONE** (0.0 -> 0.5 bought
   +1.334; 0.5 -> 1.0 gives it back). This reverses 14:07's [+0.0, +0.9] and I am saying so with four
   rungs still to come.
 - **W(ii) @1800 pooled over 7 in [-3.0, -1.0]: STILL ERODING,** not REACHES.
 - **FALSIFIERS, SPLIT PER (e). MEASUREMENT: rung 1200 returns `d(clean)` >= 0.00** -> the ceiling
   erosion is not monotone, rung 600 was not an outlier, and (c) must be withdrawn as (c) withdrew G110.
   **INFERENCE: W(i) @1800 pooled over 7 comes in >= +0.60** -> the dose HELPS, (b)'s "confined to the
   two deepest offsets" was a mid-run dip, and the non-monotone-dose claim in (k) dies.

**(l) ONE BOUNDARY QUESTION, ANSWERED NOW SO IT CANNOT BE USED LATER.** Run 69's config makes 7(f)
(`--adv-sig-floor 0.5`) run 70 *"IF AND ONLY IF run 69 hits H4's absorbing state, or fails Q3's
`clean >= 8.0`, BEFORE step 1400."* (k) predicts Q3 fails **at** rung 1400, which is not *before* step
1400 -- and the whole trigger is **superseded** by Emanuele's 16:20 decision installing
`think70_frozen` as run 70. **BOTH ROUTES ARE CLOSED: DO NOT LAUNCH 7(f) OFF A BOUNDARY READING.** If
Q3 fails at 1400 or 1600, the consequence is the registered one and only that one -- a win at that rung
would be VOID -- and run 69 continues to 1800, because Q3 voids wins and run 69 is not winning.

## G112 (2026-10-01 22:07, headless tick, run 69 `think69_rlw1` live at step 1507/1800) -- RUNGS 1200 AND 1400 CAME BACK **POSITIVE**, THE 20:07 SIGNED PREDICTION MISSED HIGH, AND W(i) IS NOW **DETERMINED BY ARITHMETIC** TO BE `SATURATED`. THE DOSE DID NOT MOVE THE MEAN; IT MULTIPLIED THE RUNG-TO-RUNG DISPERSION BY 4.7x.

**(a) THE TWO RUNGS, VERBATIM FROM `runs/probes/armed.log`.** Both PASS their own-filler gate and both PASS Q3:
```
 rung off | own-filler | own-shuf  | vs run 67     | vs INIT       | own   | filler | span   | clean  | Q3
 1200  -7 | +3.24+-.52 | +4.30+-.54| +1.13 +- 0.79 | -0.02 +- 1.25 | +1.46 | -1.78  | +12.35 | +10.19 | PASS
 1400  -5 | +1.72+-.54 | +3.32+-.53| +0.38 +- 0.77 | -0.30 +- 1.32 | +0.75 | -0.97  | +15.26 | +10.40 | PASS
```
**POOLED k=5: run69 - run67 = +0.044 +- 0.366 (z +0.12). run69 - init = -1.312 +- 0.553 (z -2.37).**
The vs-init pooled improved from -2.081 (k=3) to -1.312 (k=5), and **at the two newest rungs run 69 is at
parity with the untrained model** (-0.02 +- 1.25, -0.30 +- 1.32). All seven rungs and W @1800 stand.

**(b) THE 20:07 BAND TALLY. THE MEASUREMENT FALSIFIER DID NOT FIRE; THE SIGNED PREDICTION AND THE Q3
PREDICTION BOTH DID.**
```
  20:07 rung-1200 d(clean)    NEGATIVE [-5.0,-0.5] -> -3.04          HIT   (so G111(c) STANDS)
  20:07 rung-1200 d(own-shuf) [-3.0,+0.5] centre -1.0 -> +1.13       MISS HIGH
  20:07 Q3 PASSES @1200                            -> PASS +10.19    HIT
  20:07 Q3 FIRST FAILS @1400                       -> PASS +10.40    MISS
```
`d(clean)` over five rungs: 600 **+1.00**, 800 **-3.81**, 1000 **-2.80**, 1200 **-3.04**, 1400 **+0.12**.
The 1200 band was a hit, so G111(c) is not withdrawn -- but **at rung 1400 run 69's ceiling is now ABOVE
run 67's** (+10.40 vs +10.28), which is the second of five rungs on the other side of zero. "Rung 600 was
the outlier" survives at 1200 and is weaker at 1400; it is a 2-of-5 claim and I am labelling it as one.

**(c) THE DIRECTIONAL-BIAS QUESTION G110 OPENED AND G111 DEFERRED IS NOW ANSWERABLE, AND THE ANSWER IS
"NO BIAS, WIDTH".** G111(f) set the rule: *no directional bias from fewer than four bands.* There are now
nine scored bands across four ticks -- **3 MISS HIGH (16:07 x2, 22:07 x1), 2 MISS LOW (18:07 x2), 4 HIT.**
A 3-2 split is no bias in any direction, and **the misses are large: +1.13 against a ceiling of +0.5, and
-1.98 against a floor of -1.0.** The defect is width and (g) below measures exactly how much.

**(d) W(i) IS NOW DETERMINED BY ARITHMETIC, NOT BY PREDICTION, AND I AM REGISTERING THAT BEFORE RUNGS 1600
AND 1800 EXIST.** Five rungs sum to **+0.220**. The registered bars are pooled-7 `>= +0.60` (CONFIRMED) and
`<= -0.60` (NET HARMFUL). So:
```
  CONFIRMED requires  d(1600) + d(1800) >= +3.98   (i.e. ~+2.0 each)
  HARMFUL   requires  d(1600) + d(1800) <= -4.42   (i.e. ~-2.2 each)
  observed range of the five landed rungs: -1.98 .. +1.45
```
**Both routes require the final two rungs to fall outside the entire observed range, in the same direction,
together.** Over a 5x4 grid of scenarios spanning d(1600), d(1800) in [-1.0,+2.0] -- wider than anything
observed -- **19 of 20 cells return `SATURATED`**; the single exception is (+2.0, +2.0), which returns
+0.603, three thousandths past the bar. **THE REGISTERED VERDICT WILL BE `W(i) SATURATED`.** Its registered
consequence is already written in `runs/think69_rlw1/config.sh`: *"the term is NECESSARY AND NOT SUFFICIENT,
and the programme must go to the reader (run 70)"* -- and run 70 `think70_frozen` is the frozen-READER run,
already armed. **Nothing needs redesigning; the programme's next step was correct before this read.**

**(e) THE RIGHT DESCRIPTION OF RUN 69 IS AN EXCURSION, NOT A FLIP -- AND THE "MONOTONE" ANCHOR G111 LEANED
ON TURNED TWO BLOCKS AFTER IT WAS ASSERTED.** The matched per-step train-anchor table, extended by two
blocks (each block inside its own `--span-sched` segment, G105):
```
    block     off   n |  d(own-shuf)         |  d(or_g)             | d(read_ce)    | d(len_mean)
    0- 399    -15  400| +0.127+-0.039 z +3.3 | -0.237+-0.021 z-11.1 | +0.016+-0.001 | -0.001+-0.001
  400- 599    -13  200| +0.197+-0.076 z +2.6 | -0.589+-0.080 z -7.4 | +0.039+-0.003 | -0.005+-0.003
  600- 799    -11  200| -0.238+-0.074 z -3.2 | -0.547+-0.071 z -7.7 | +0.046+-0.003 | -0.586+-0.072
  800- 999     -9  200| -0.412+-0.082 z -5.0 | -0.892+-0.103 z -8.7 | +0.025+-0.004 | -2.090+-0.136
 1000-1199     -7  200| -0.342+-0.075 z -4.6 | -1.141+-0.089 z-12.8 | +0.030+-0.003 | -2.213+-0.121
 1200-1399     -5  200| -0.187+-0.079 z -2.4 | -0.955+-0.080 z-11.9 | +0.049+-0.003 | +0.255+-0.118
 1400-1599     -3  124| -0.314+-0.091 z -3.5 | -0.590+-0.090 z -6.5 | +0.046+-0.004 | -0.459+-0.216
```
Three things in it: (i) `d(or_g)` is **NOT monotone** -- it bottomed at -1.141 (block 1000-1199) and has
recovered to **-0.590**, and G111(k) signed its rung-1200 prediction on *"a z -11.7 monotone anchor"*
which turned within two blocks of the sentence; (ii) `d(len_mean)` went -2.213 -> **+0.255** -> -0.459, so
the "writer shortens ~2 tokens faster" effect G111(g) named was a transient of blocks 800-1199 and not a
trend; (iii) **the whole shape is an EXCURSION over roughly steps 700-1200 with a recovery after 1200**, on
the held-out rungs (-1.98, -0.76, then +1.13, +0.38) and on three train anchors at once. `SATURATED` at
k=7 is **the excursion and the recovery cancelling**, which is a different physical claim from "the dose
does nothing" and must be reported as the former.

**(f) THE TRAIN ANCHOR DOES NOT PREDICT THE HELD-OUT RUNG'S SIGN. 3 OF 5. THAT IS CHANCE.**
```
  block 400- 599 +0.197  ->  rung  600 +1.45   AGREE
  block 600- 799 -0.238  ->  rung  800 -1.98   AGREE
  block 800- 999 -0.412  ->  rung 1000 -0.76   AGREE
  block 1000-1199 -0.342 ->  rung 1200 +1.13   DISAGREE
  block 1200-1399 -0.187 ->  rung 1400 +0.38   DISAGREE
```
G111(b) claimed the programme's first case of a training-time number predicting a held-out one before the
fact. **It held for one rung and has failed the next two.** The z-scores are large (-2.4 to -5.0) only
because n = 200 **serially correlated** steps, and a tight interval around the wrong sign is worse than no
interval. **NEW STANDING RULE (G112(f), SUPERVISOR 00S.0): NEVER DERIVE A HELD-OUT RUNG'S SIGN FROM A
TRAIN ANCHOR.** Train anchors may gate a derail (D1 did its job) and may never be the basis of a signed
held-out prediction. This is G106's reporting caveat extended to prediction, where it was being evaded.

**(g) THE MEASUREMENT THAT MAKES SENSE OF ALL OF IT: THE DOSE DID NOT MOVE THE MEAN, IT MULTIPLIED THE
RUNG-TO-RUNG DISPERSION BY 4.7x.** On the five matched rungs, held-out `own_minus_shuf` LEVELS:
```
  run 69 (rl_w 1.0): 5.01 1.58 2.22 4.30 3.32   mean +3.286   sd across rungs 1.418
  run 67 (rl_w 0.5): 3.56 3.56 2.98 3.17 2.94   mean +3.242   sd across rungs 0.303
                                   d(mean) +0.044        sd ratio 4.68x   F 21.9 (df 4,4)
```
Per-rung probe s.e. is ~0.57 in BOTH runs, so the dispersion gap is not an instrument difference.
**Doubling `--rl-w` left the mean held-out own-note margin where it was (+0.044 +- 0.366) and roughly
quintupled its variance across rungs** -- the signature of REINFORCE weight, which scales the gradient's
variance and its mean together. **CAVEAT, NAMED BEFORE IT CAN BE USED AGAINST ME:** the seven rungs are one
trajectory and the probes at different rungs likely share sequences, so the per-rung s.e.s are not
independent and the F is descriptive, not a test. Run 67's observed sd (0.303) sitting BELOW its own
per-rung s.e. (0.57) is direct evidence of that correlation.
**THE CONSEQUENCE FOR THE GATE, AND THE BAR IS NOT MOVED (G103).** The empirical s.e. of the pooled mean is
**1.418/sqrt(5) = 0.634 against the quoted 0.366 (1.73x)**; vs init, 1.954/sqrt(5) = 0.874 against 0.553
(1.58x). At k=7 the empirical s.e. is **~0.535 against the gate's assumed ~0.30**. So **W(i)'s +-0.60
SATURATED band is about 1.1 empirical s.e. wide: `SATURATED` is the verdict noise alone would produce.**
`+-0.60` STANDS EXACTLY AS REGISTERED. What is registered now, before the data, is the **reporting**
requirement: **W(i) must be printed with the empirical s.e. beside the quoted one, and a SATURATED verdict
must be reported as "no effect was measured", never as "the effect was measured to be zero".**

**(h) 7(f) `--adv-sig-floor 0.5` IS NOW DEAD BY DATA AND NOT ONLY BY SUPERSESSION.** Its trigger was *"run
69 hits H4's absorbing state, or fails Q3's `clean >= 8.0`, BEFORE step 1400"*. **Step 1400 has passed with
Q3 PASS (+10.40) and H4 nowhere** (`g_sd` 2.574 against its <= 0.30 bar, 8.6x). The condition is
unsatisfiable and the clause is closed by arithmetic. G111(l) closed both routes by argument; this closes
them by measurement, which is the stronger closure.

**(i) HEALTH, EVERY CLAUSE ON A MEASUREMENT.** held-out mix **2.8134** absolute (bar 3.025), **+0.001 per
100** over steps 1100-1400 (bar +0.02), **+0.038** above the run's start (bar +0.25); `ntp1` roll20
**1.920**, block mean 1.99 (bar 4.0); H4 roll50 `len_mean` **11.774** and `g_sd` **2.574** against **<= 3.5
AND <= 0.30** -- `g_sd` is 8.6x its bar; rollout spread 2.574 (3b bar 0.6); length 11.8 vs `--min-think` 3
(3b bar: within 0.5); **0** NaN/inf in 1507 rows, **0** `NRT_`/`Traceback`; 64 ranks / **65 workers**;
17.0 s/step; **530 G** disk, 1555 GB mem; 6 crons; `archive_r69.sh`, `armed_probes_r69.sh`,
`keepalive_think.sh`, run 70's launcher (pid 1935711) all alive.
**reg%, DECOMPOSED PER G111(g) BEFORE ACTING ON IT:** `status.py`'s 300-block printed 9.3 for 900-1199 and
8.6 for the 4-step tail. Per 100 steps with n: **1100-1199 1.5** (n=848), **1200-1299 2.3** (n=755),
**1300-1399 1.7** (n=845), **1400-1499 2.5** (n=766), 1500+ 4.9 on **n=82** (eight steps -- top-3 share of
82 samples is inflated by construction). The 900-999 spike G111 predicted this tick would see is confirmed
resolved. **Nothing above 2.5 on any full bin. G111(g)'s rule worked exactly as written.**

**(j) 3c LITERALLY READS "FIRE" THIS TICK AND MUST NOT BE ACTED ON -- AND HERE IS THE PROOF, NOT THE
ARGUMENT.** Runbook 3c: *after >= 1,500 steps, if `g_r0` and `gSp` have not improved by >= 0.3 nats over
their first-300 values, stop.* Run 69 is at step 1507 and reads `g_r0` 1.432 -> **-1.329** and `g_spec`
2.555 -> **-0.876**: both far WORSE, so the clause fires on its face. **Run 69's config declared 3c DEAD
BEFORE LAUNCH** (`--span-sched` is on, so the in-training gain is measured at a different, harder offset
every block and is not comparable across the run). The proof that this was a design fact and not a
convenience: **run 67 -- which completed all 1800 steps in health and is this run's matched control --
reads `g_r0` -2.196 and `g_spec` -3.091 on 3c's own test.** 3c would have killed the control. **This is
the sixth instance of G84's rule (walk all of 3b/3c before launch and write the dead clauses down) and the
first where the pre-launch declaration actually prevented a wrongful stop.**

**(k) DECISION: CONTINUE. NO DEVICE ACTION, DELIBERATELY.** Run 69 is healthy, every registered gate is
self-enforcing inside ~1 min of each checkpoint, and the two remaining rungs plus W @1800 land before the
next tick. **Run 70's chain was verified end to end this tick rather than assumed** (see the journal entry
for the seven checks).

**(l) PREDICTIONS, TIME-STAMPED 22:07 BEFORE RUNGS 1600 (~22:45) AND 1800 (~23:50) EXIST. BANDS FROM THE
EMPIRICAL sd OF (g) -- +-1.4 PER RUNG -- AND UNSIGNED WHERE (f) FORBIDS SIGNING.** Run 67's two remaining
rungs: `own-shuf` 1600 **+2.82**, 1800 **+2.01**; `clean` **+9.31**, **+7.69**. The init references the
enforcer will use: 1600 o=-3 **+2.39 +- 1.35**, 1800 o=-1 **+5.30 +- 1.41**.
 - **rung 1600 `d(own-shuf)` in [-2.5, +2.5], UNSIGNED.** I have a 3-of-5 anchor and a 4.7x dispersion;
   neither licenses a centre. Saying "unsigned" is the whole content of (f).
 - **rung 1800 `d(own-shuf)` in [-2.5, +2.5], UNSIGNED.**
 - **Q3 PASSES at 1600** (`clean` band [+7.0, +11.5], bar 8.0). **Q3 at 1800 is a coin flip** (band
   [+5.5, +10.0]; run 67 failed it there at +7.69) **and a failure voids nothing, because W(ii) will not
   be met and Q3 voids wins.**
 - **W(i) @1800 = `SATURATED`, pooled-7 in [-0.30, +0.45].** Determined, per (d).
 - **W(ii) @1800 = `STILL ERODING`, pooled-7 in [-2.0, -0.5], centre ~-1.2.** SIGNED, and the mechanism is
   named: five rungs sum to **-6.560**, so REACHES (>= 0.00) needs the last two to sum to **+6.56**, and
   the init's reading at o=-1 is **+5.30**, the highest of all seven offsets, so rung 1800 contributes a
   large negative almost regardless of what run 69 does.
 - **FALSIFIERS, SPLIT PER G111(e) -- MEASUREMENT CLAUSE AND INFERENCE CLAUSE NEVER CONJOINED.**
   **MEASUREMENT: W(i) pooled-7 lands outside [-0.60, +0.60]** -> (d)'s arithmetic determination was wrong,
   a rung fell outside the entire observed range, and the "19 of 20 cells" reasoning must be withdrawn.
   **INFERENCE: W(ii) pooled-7 >= 0.00** -> (l)'s signed erosion claim dies and run 69 REACHES the
   untrained model, which would be the programme's best result to date.
**NEXT GATES, ALL SELF-ENFORCING: rung 1600 ~22:45, rung 1800 + W ~23:50, run 69 DONE ~23:30, run 70
`think70_frozen` auto-launched ~23:32.** The 00:07 tick must, in this order: read W from
`runs/probes/armed.log` against (l); **verify run 70's smoke -- `--frozen-reader 1` will be executing for
the first time in this project's history**, and 00I says fix it and relaunch run 70, never fall back to
7(f); then `main.tex`, untouched since 2026-09-23 and the only debt left.

## G113 (2026-10-02 00:07 headless tick, run 69 `think69_rlw1` **DONE** at 23:47, run 70 `think70_frozen` **LIVE** at step ~79/2400) -- **ALL SIX OF THE 22:07 BANDS LANDED INSIDE, INCLUDING BOTH SIGNED ONES, AND THE RUN STILL ENDED BEHIND THE UNTRAINED MODEL. THE HEADLINE VARIANCE NUMBER I PUBLISHED AT 22:07 (4.7x) IS 2.3x ON THE COMPLETE LADDER AND I AM CORRECTING IT MYSELF. AND RUN 69's OWN GATE ENFORCER KILLED RUN 70 TWO SECONDS AFTER ITS LAST PROBE.**

**(a) THE REGISTERED VERDICTS, AS THE ENFORCER PRINTED THEM, WITH BOTH STANDARD ERRORS.**
```
  rung off | r69 own-shuf | vs run 67     | vs INIT       | clean | Q3
   600 -13 | +5.01        | +1.45 +- 0.92 | -0.08 +- 1.07 | 19.52 | PASS
   800 -11 | +1.58        | -1.98 +- 0.76 | -4.63 +- 1.25 | 13.73 | PASS
  1000  -9 | +2.22        | -0.76 +- 0.84 | -1.53 +- 1.27 | 10.96 | PASS
  1200  -7 | +4.30        | +1.13 +- 0.79 | -0.02 +- 1.25 | 10.19 | PASS
  1400  -5 | +3.32        | +0.38 +- 0.77 | -0.30 +- 1.32 | 10.40 | PASS
  1600  -3 | +3.19        | +0.37 +- 0.78 | +0.80 +- 1.46 |  8.90 | PASS
  1800  -1 | +2.15        | +0.14 +- 0.79 | -3.15 +- 1.52 |  7.68 | **FAIL**
  W(i)  pooled k=7  run69 - run67 = +0.103 +- 0.306 quoted / **+-0.439 empirical** -> SATURATED
  W(ii) pooled k=7  run69 - init  = -1.274 +- 0.496 quoted / **+-0.741 empirical** -> STILL ERODING
```
**W(i) = `SATURATED`. Report it as "NO EFFECT WAS MEASURED", never as "measured to be zero"** -- the
registered +-0.60 band is **1.37 empirical s.e.** wide, so it is the band noise alone would fill. The
registered consequence stands as written before launch: `rl_w` 0.0 -> 0.5 bought the margin (run 68:
-1.334 +- 0.417 without it), 0.5 -> 1.0 buys nothing; **the term is NECESSARY AND NOT SUFFICIENT and the
programme goes to the reader.** That run is `think70_frozen` and it was already armed.
**W(ii) = `STILL ERODING` as registered, and the empirical s.e. makes the honest sentence narrower than
the verdict word: at +-0.741 the pooled -1.274 is z -1.72, not z -2.57. Run 69 is BEHIND the untrained
init and is NOT a demonstrated erosion.** Q3 failed at rung 1800 (`clean` +7.68 < 8.0), which voids a win
at that rung; there was no win to void. **Nothing in this project has yet beaten the untrained model.**

**(b) SIX BANDS, SIX HITS -- AND THAT IS NOT A LICENCE, IT IS THE FIRST TICK THAT EARNED ITS WIDTHS.**
`d(own-shuf)` @1600 **+0.37** and @1800 **+0.14**, both inside the unsigned [-2.5,+2.5]; Q3 PASS @1600
(`clean` +8.90 inside [+7.0,+11.5]); Q3 FAIL @1800 (+7.68 inside the [+5.5,+10.0] coin-flip band, called
as a coin flip in writing); **W(i) +0.103 inside [-0.30,+0.45]**; **W(ii) -1.274 inside [-2.0,-0.5],
centre predicted -1.2.** Both MEASUREMENT and INFERENCE falsifiers were given their own clause per
G111(e) and **neither fired.** The nine-band tally G112(2) opened is now **15 bands: 3 MISS HIGH, 2 MISS
LOW, 10 HIT.** The lesson is the one G112(4) wrote: the defect was WIDTH, and the fix was to widen from
the run's own dispersion instead of from a feeling.

**(c) I AM CORRECTING MY OWN HEADLINE NUMBER FROM 22:07. THE DISPERSION RATIO IS 2.34x, NOT 4.68x.**
`own_minus_shuf` LEVELS, now over all seven rungs rather than the five that existed at 22:07:
```
  run 69 (rl_w 1.0): 5.01 1.58 2.22 4.30 3.32 3.19 2.15   mean +3.109  sd 1.234
  run 67 (rl_w 0.5): 3.56 3.56 2.98 3.17 2.94 2.82 2.01   mean +3.006  sd 0.528
                 k=5 (22:07): sd 1.419 / 0.303 = 4.68x    k=7: 1.234 / 0.528 = **2.34x**, F 5.5 (df 6,6)
```
The last two rungs are where run 67's own ladder spreads out (2.82, 2.01 after five readings inside
2.94--3.56), so the ratio fell by more than half when the denominator finally moved. **The qualitative
finding SURVIVES -- doubling the REINFORCE coefficient left the mean at +0.10 and roughly doubled the
rung-to-rung scatter, which is what a coefficient on a score-function gradient does -- but the number I
put in STATUS.md and in SUPERVISOR.md as "4.7x" was computed on a partial ladder and is wrong by 2x.**
G112 named the right caveat (one trajectory, correlated probes, descriptive not a test) and still
published a ratio two ticks before its denominator was complete. **NEW RULE G113(c): A RATIO OF TWO
DISPERSIONS MAY NOT BE QUOTED UNTIL BOTH LADDERS ARE COMPLETE. A partial-ladder sd is not an estimate of
the ladder's sd -- it is an estimate of the sd of the rungs that happened to land first, and rungs land
in curriculum order, which is the opposite of random.** The empirical s.e. in (a) uses k=7 and is the
number to carry.

**(d) THE INFRASTRUCTURE DEFECT, AND IT COST REAL STEPS: A FINISHED RUN'S ENFORCER KILLED ITS
SUCCESSOR.** Timeline, from three logs that agree: 23:42:41 `launch_r70_after_r69.sh` sees run 69's
STATUS go DONE and launches run 70; 23:45:01 run 70's smoke exits 0 and its training starts; **23:47:18
run 69's enforcer finishes its rung-1800 probe and runs `pkill -f "think_train.py --corpus"` -- an
UNSCOPED pattern -- and at 23:47:20 run 70's 64 ranks take SIGTERM** (`train.log`: `SignalException:
Process 2047703 got signal: 15`); 23:47:25 run 70's own supervisor logs `training exit: 1` and
`crashed -- retry 1/3 (resume latest)`; 23:47:56 it relaunches and has been healthy since. **Cost: three
steps, and one of three retries spent on a kill that was not run 70's fault.** The retry counter does
**not** reset on progress, so run 70 has 2 left for 2,320 steps (the keepalive is the second net and
demonstrably works -- `watchdog.log` 23:45:02 "no workers -- keepalive relaunches within 10 min").
**FIXED, not just recorded:** `agent/armed_probes_frozen.sh` now kills on
`think_train[.]py .*runs/$NAME/metrics` -- every trainer carries `--metrics-file runs/<name>/metrics.jsonl`,
which is unique per run -- with a logged widening fallback so that an ENFORCED STOP can never leave its
own run alive. Verified by `pgrep`: the scoped pattern matches **65** processes for `think70_frozen` and
**0** for `think69_rlw1` (backup `.bak_pre_scopedkill_20261002_0007`). The running enforcer keeps the old
inode and is unaffected (pid 1995770 alive); the fix reaches run 71 via `launch_next`.
**THE GENERAL RULE, G113(d): AN AUTOMATION THAT KILLS MUST NAME ITS VICTIM. Every `pkill` in this repo
that can fire while another run is live must match a per-run string, and the per-run string we have is
the metrics path.** This is the first time an agent-written script damaged a live run, and it was caught
only because the damage happened to be logged in the victim's own `supervisor.log`.

**(e) `or_g` DOES NOT EXIST IN RUN 70, BY CONSTRUCTION, AND SEVEN TICKS OF G109--G112 REASONING RAN ON
IT.** Run 70 sets `--ssl-frac 0.0`, so there are no supervised/oracle anchors: `n_or` summed over the
whole run is **0** and `or_g` is printed as `+0.00` at every step. The train-anchor "ceiling cousin" that
G109 raised, G110 retired, G111 withdrew the retirement of, and G112 found non-monotone **is unavailable
for this run**. The only ceiling read run 70 has is the probe's absolute `clean` at gate time, and
`n_sm` = 168 confirms the frozen-reader silence baseline is the arm actually being computed. **Do not
open a `d(or_g)` paragraph on run 70: it would be a paragraph about a column of zeros.**

**(f) THE FROZEN READER's "NEVER EXECUTED" RISK IS RETIRED, AT RUNTIME AND NOT BY READING THE DIFF.**
G112(8) could verify the code and not the execution. Now: `train.log:968` prints **"FROZEN READER: notes
are scored by a frozen copy of the init"** in the LIVE process (line 100 was the smoke, so the flag has
executed twice independently); smoke exit 0; 83 metric rows with **0** NaN/inf; `reader_bb` is
`deepcopy(backbone)` taken at `think_train.py:511` **before** `gradient_checkpointing_enable` and before
`build_optimizer` at `:523`, is not a submodule of `model`, so `model.parameters()` cannot contain it, and
every parameter is `requires_grad_(False)` with `.eval()` at `:529-531`. **The reader cannot move.**
**ONE CONSEQUENCE TO STATE BEFORE THE FIRST GATE, SO IT CANNOT BE CLAIMED AFTERWARDS: a frozen reader
fixes the SILENCE and FILLER baselines; it does NOT fix the SHUFFLE control**, which is a permutation of
the writer's own note and therefore moves with the writer. At steps 0-79 both arms are rising together
(`g_r0` +2.45 -> +2.84, `g_shuf0` +1.67 -> +1.92). **G108's exploit is therefore still available in run 70
through `own_minus_shuf`, and run 70's gates are correctly written on `own_minus_filler` instead. Read
run 70 on filler and on the absolute arms; never crown it on own-minus-shuffle.**

**(g) A LATENT FRAGILITY, LOGGED AND NOT PATCHED.** If a run exhausts its 3 retries, `run_think.sh`
writes `FAILED` and the keepalive revives it from the latest checkpoint -- but the enforcer's `dead()`
greps `^(DONE|STOPPED|FAILED)`, so it would see `FAILED`, exit "run ended before step N", and **disarm
every remaining gate on a run that is still training.** Not patched because run 70's enforcer is already
running from the old inode and cannot be changed under it; the tick that sees a `FAILED`-then-revived run
must **re-arm the enforcer by hand**.

**(h) THE PAPER DEBT IS PAID DOWN, NOT DEFERRED AGAIN.** `overleaf/iclr2027/main.tex` had been untouched
since 2026-09-23 and documented the project only through runs 37--39, while claiming "no run is live" and
"thirty-eight runs". Added, with every number traced to a probe file or a STATUS line: **`sec:synthetic`
(runs 40--59)** -- the synthetic-task escape, where the reader provably can use a note (run 52 EM
+0.450, McNemar p=6.8e-17; run 55 +0.510, p=5e-21) and the writer still emitted nothing
(`scr_pv_g` 0.000), and where the one discovery that happened (runs 56/57, 0.938 at one hop) died under a
uniform hop mixture (0.271, then 0.333 and declining) -- and **`sec:slide` (runs 60--69)** -- the slide,
the reader-moves-the-ruler instrument failure (runs 61/62: true span +15.82 -> +9.40 on the same 132
anchors; `clean` 17.6 -> 7.7), the 67/68/69 decomposition with (a)'s numbers and (c)'s corrected
dispersion, and run 70's registered gates with its pre-data prediction. The closing paragraph no longer
says no run is live. **Compiles: `pdflatex` twice, rc=0, 67 pages, 0 undefined references** (backup
`main.tex.bak_pre_runs40to69_20261002_0007`).

**(i) DECISION: CONTINUE. NO DEVICE ACTION.** Run 70 is at step ~79/2400, 15.5 s/step, ETA **~10:10**;
held-out mix **2.7749** against a bar of +-0.25 around 2.775; `ntp1` roll20 **1.91** (bar 4.0); 0 NaN;
65 workers / 64 ranks; 522 G free; all daemons and 6 crons alive; enforcer armed with run 71 as the
G1/G2 successor. **G1 @400 lands ~01:35, G2 @1000 ~04:15, TRANSFER @1800 ~07:40, FINAL @2400 ~10:10.**

**(j) THE ONE THING THE 08:07 TICK OWES, AND WHY IT IS NOT DONE NOW.** `armed_probes_frozen.sh` launches
a successor **only** on a G1/G2 failure. If run 70 runs to 2400 in health, nothing is armed and the
device idles from ~10:10 until a tick notices. **The 08:07 tick must write the successor config and arm a
launcher**, because by then G1, G2 and all three TRANSFER reads at `o=+1` exist, and the queue in
SUPERVISOR 00I branches on exactly those: FINAL/TRANSFER above -4.00 -> unfreeze the reader with clean
NTP only; compression learned but the forecast flat at the untrained -7.99 -> longer notes (`--L 32`);
nothing moving even with copy-teaching -> report the negative to Emanuele. **Pre-committing the branch now,
at step 79 with zero run-70 evidence, would be choosing a design before the data that decides it** --
which is the one discipline this whole programme is built on. Two ticks (08:07 and 10:07) get the chance
before the device can idle.

## G114 (2026-10-02 02:07 headless tick, run 70 `think70_frozen` **LIVE and HEALTHY** at step ~510/2400) -- **THE 00:07 FIX KILLED THE THING IT WAS FIXING. EDITING A RUNNING BASH SCRIPT IN PLACE DESTROYED RUN 70's GATE ENFORCER AT 01:38:13, AND G1 @400 WENT UNREAD FOR 29 MINUTES WITH EVERY LATER GATE SILENTLY DISARMED. NO SCIENCE WAS LOST; THE REGISTERED CHAIN IS RE-ARMED AND RUNNING. I READ NO G1 NUMBER THIS TICK BECAUSE THERE WAS NONE TO READ.**

**(1) WHAT HAPPENED, FROM THE THREE ARTEFACTS THAT PROVE IT.** The 00:07 tick applied the G113(d)
scoped-kill fix to `agent/armed_probes_frozen.sh` and recorded in the journal: *"The running enforcer
holds the old inode and is unaffected."* **That sentence was wrong on both halves, and it cost a gate.**
```
  stat -c '%i %s'  agent/armed_probes_frozen.sh                      -> 18343229  5758   mtime 00:14:51
  stat -c '%i %s'  ...sh.bak_pre_scopedkill_20261002_0007             -> 18343364  4817
```
The **backup** carries the new inode, because `cp` created it; the **original path was rewritten in
place** and kept inode 18343229 -- the very inode the live enforcer (started 23:42:41) had open. And the
inode was never the point: **bash does not read a script into memory. It reads incrementally and seeks by
BYTE OFFSET.** The file grew **4817 -> 5758 bytes (+941)** under a process holding an offset into the old
layout. At **01:38:13** the enforcer finished `grab 0000400` (`archived step_0000400` -- its last healthy
act), went back for the next chunk of script text, and read shifted bytes:
```
  agent/armed_probes_frozen.sh: line 94: syntax error near unexpected token `0'
  agent/armed_probes_frozen.sh: line 94: `  if .venv/bin/python -c "import sys; sys.exit(0 if ...
```
**`bash -n` on that file passes today and passed at 00:07.** The line is valid on disk. The error exists
only inside a process reading a file that moved under it -- which is why a file-level check could never
have caught this, and why the failure surfaced **84 minutes after the edit**, at the first moment the
interpreter needed new text.

**(2) THE BLAST RADIUS, STATED EXACTLY.** The enforcer died *between* archiving step 400 and evaluating
it, so **`gate 0000400 -15 9.00 G1` never ran** (`ls runs/probes/ | grep think70` returned the log and
nothing else at 02:07 -- **zero probe files for run 70**), the **TRANSFER @400 o=+1** read never ran, and
**G2 @1000, TRANSFER @1000/@1800 and FINAL @2400 were all disarmed**. The run itself never noticed: 65
workers, 15.3 s/step, uninterrupted. **The dangerous half is not the missing number, it is the missing
consequence.** A G1 failure was registered to `stop_run` and `launch_next`; with the enforcer dead, a run
that had already failed its easiest gate would have ground on to ~10:10 and **burned ~8 h of device time
that the registered design says it should not have had**. The 29 min of unread gate is the cheap part.

**(3) THE PATTERN I HAVE TO NAME: TWO CONSECUTIVE TICKS WHERE THE MOST IMPORTANT EVENT WAS AN AUTOMATION
DEFECT, AND BOTH DEFECTS WERE INTRODUCED BY A FIX.** G113(d) repaired an unscoped `pkill` that had killed
run 70, and **in the same edit disarmed every gate on run 70.** The repair was correct; the method of
applying it was not. A fix applied to a live automation **is a change to a running system**, and it was
verified the way one verifies a file (`bash -n`, a diff, a backup) rather than the way one verifies a
system (is the process still alive, and is it still making progress). **Two commands -- `ps -p 1995770`
and `tail runs/probes/armed_think70_frozen.log` -- run at any point after 00:14:51 would have caught
this; `ps` would have caught it at 01:38 and a tail would have shown the syntax error in plain text.**
The 00:07 tick ran neither, because it believed an argument about inodes instead of looking.

**(4) THE RULES. G114(a): NEVER EDIT A SCRIPT THAT IS CURRENTLY BEING EXECUTED.** Not with `>`, not with
a heredoc, not with an editor that preserves the inode. bash's lazy byte-offset read means any
length-changing edit corrupts the interpreter **at an unpredictable later time**, reporting a syntax
error on a line that is valid on disk -- the single most misleading diagnostic in this repo's history.
Edit a copy; launch the copy.
**G114(b): A LONG-RUNNING AUTOMATION IS LAUNCHED FROM AN IMMUTABLE SNAPSHOT.**
`agent/running/<script>_<run>_<stamp>.sh`, `chmod 444`. The file in `agent/` becomes a **template for the
next launch and is never the thing executing**, which is what makes it safe to edit mid-run. This is now
structural, not advisory: `launch_next` snapshots before launching the successor's enforcer.
**G114(c): AFTER ANY EDIT NEAR A LIVE AUTOMATION, VERIFY THE PROCESS, NOT THE FILE.** `ps -p <pid>` plus
a log line newer than the edit's mtime. `bash -n` is necessary and **not sufficient** -- it is a statement
about bytes on disk, and the failure lives in a file descriptor.
**G114(d): A GATE WHOSE CHECKPOINT HAS LANDED BUT WHICH HAS LOGGED NO READING IS A DEFECT, NOT A DELAY --
AND EVERY TICK CHECKS IT FIRST.** Run 70's G1 was due ~01:35 and at 02:07 had no probe file at all. The
check is one `ps` and one `tail`, it is now the tick's opening move, and it is the only reason this was
found 29 min late rather than at 08:07 with three gates missing.

**(5) THE DESIGN FLAW UNDERNEATH, LOGGED WITH A DELIBERATE DECISION NOT TO PATCH IT TONIGHT.** The
enforcer is careful about a probe that returns nothing (line 84: *"probe produced no number -- NOT
enforcing; the supervisor must read it"*) -- but **a dead enforcer fails SILENT**, and nothing in the repo
watches it. `keepalive_think.sh` and the watchdog watch **training**; they have no notion of a gate. The
honest fix is a liveness check, and **I am not writing one tonight, for a reason I want on the record:
`keepalive_think.sh` is running right now (pid 2151723), so editing it is precisely the sin in G114(a)**,
and a brand-new daemon to watch a daemon buys at most a 2-hour latency improvement over the tick check in
G114(d) while adding a fresh failure surface to the exact subsystem that just failed twice in two ticks.
**G114(d) captures nearly all the value for zero new code.** If a future tick wants the daemon, it must
write it as a NEW file and launch it from a snapshot -- never by editing the live keepalive.

**(6) ONE NARROW GAP THAT A LATER TICK MUST NOT TRIP OVER.** The enforcer now live (pid 2168108) is my
02:07 snapshot, so it carries the **OLD** `launch_next`: if G1 or G2 fails, it will launch run 71's
enforcer **from the template path** `agent/armed_probes_frozen.sh`, not from a snapshot. So: **before
editing `agent/armed_probes_frozen.sh`, run `pgrep -af 'bash agent/armed_probes_frozen\.sh'` and refuse
to edit if it returns anything** -- the check I ran before tonight's edit, which returned NONE. From run
72 onward the snapshot path is inherited automatically and the gap closes itself.

**(7) WHAT IS TRUE OF RUN 70 RIGHT NOW, ALL MEASURED THIS TICK.** Step ~510/2400, **15.3 s/step, ETA
~8.0 h (~10:10)**, 65 workers, `ntp1` roll20 **2.300** (3b bar 4.0), **0 NaN/inf in 514 rows**, 494 G
free, 1562 G RAM avail. Held-out mix **2.7749 -> 2.8329 over 400 steps**; the absolute bar is 3.025
(**margin 0.192**) and the rate bar is +0.02 per 100 over 300: measured **+0.0180** (0->300) and
**+0.0179** (100->400) -- **INSIDE, but this is 6x the erosion rate of runs 67/68/69 (+0.003 per 100) and
the closest any run has come to 3b's rate clause.** It is decelerating, which is why no action is taken:
per-100 segments run **+0.0044, +0.0247, +0.0249, +0.0040**, so the 300-windows are carrying the steep
200-300 stretch and the latest stretch is nearly flat. **REGISTERED NOW: if the 300-window rate exceeds
+0.02 per 100 at any later tick, 3b fires and the run stops -- I am not going to widen that bar after
seeing the data.** The single `Traceback` in `train.log` is at line 915, the torchrun `SignalException:
signal 15` from the 23:47 kill, i.e. **before** the live process's frozen-reader print at line 968 --
the known, already-documented event, not a new one. ` DRIFT!` on the STATUS line is still the G102(e)
cosmetic ratchet (+0.520 against a 0.5 trigger).

**(8) DECISION: CONTINUE, NO DEVICE ACTION, CHAIN RESTORED.** Re-armed at **02:09:40** from the read-only
snapshot (`armed for think70_frozen (successor: think71_frozen_imit)` logged; pid 2168108 verified alive
at +1:43 while the template was edited underneath it -- **the rule demonstrating itself**). It re-enters
at `grab 0000400`, finds the archived checkpoint, and runs the **full registered chain from G1 forward**
with the original bars untouched: G1 @400 o=-15 >= +9.00, TRANSFER @400 o=+1, G2 @1000 o=-9 >= +3.70,
TRANSFER @1000/@1800, FINAL @2400 at 32 seqs. **No bar moved, no clause added, nothing redesigned.**

**(9) THE 00:07 PREDICTIONS STAND UNCHANGED AND UNREAD -- WHICH IS THE ONLY HONEST THING TO SAY.** G1
@400 PASSES, band [+9.0,+14.0]; G2 @1000 PASSES NARROWLY, band [+3.5,+8.0]; TRANSFER @400 at o=+1 stays
below -4.00, band [-8.0,-4.0]; FINAL @2400 PROGRESS not win, band [-6.0,-1.0]. Untrained references
(o=-15 and o=-9 are 10 val seqs, **n=132**; o=+1 is 32 seqs, n=424): o=-15 **+6.93+-0.85**, o=-9
**+1.71+-0.91**, o=+1 **-7.99+-0.50**. **Tally stays 15 bands: 10
HIT, 3 MISS HIGH, 2 MISS LOW** -- it does not move tonight, because a tick that loses its instrument does
not get to score itself. **The 04:07 tick reads G1 against this list. G1 is now due ~02:4x, not ~01:35.**

**(10) A SECOND DEFECT FOUND WHILE THE G1 PROBE RAN, AND IT POINTS THE DANGEROUS WAY: BOTH FROZEN-READER
CONFIGS CITE THE WRONG UNTRAINED REFERENCE AT o=-15, UNDERSTATING IT BY 4.8 NATS.**
`runs/think70_frozen/config.sh:22` and `runs/think71_frozen_imit/config.sh:22` both read
*"G1 @400 ... own_minus_filler >= +9.00 **(init +2.1)**"*, and line 26 *"G2 @1000 ... >= +3.70 **(init
+2.0)**"*. Measured from the reference probes this tick:
```
  span_vanilla22800_o-15_c.json  own +7.099  filler +0.173  ->  own_minus_filler +6.925 +-0.848  n=132
  span_vanilla22800_o-9.json     own +1.977  filler +0.268  ->  own_minus_filler +1.709 +-0.915  n=132
  span_vanilla22800_o+1.json     own -6.314  filler +1.674  ->  own_minus_filler -7.988 +-0.500  n=424
```
At o=-9 the config's "+2.0" is close to the init's **`own`** mean (+1.977), so the comments appear to quote
`own` rather than the paired `own_minus_filler` -- but even that reading fails at o=-15, where `own` is
**+7.099**. **"+2.1" matches nothing at this offset.** **Why this is not pedantry:** the first three
per-sequence reads of the live G1 probe are `own` +4.05 / +3.87 / +3.67 against `filler` -0.06 / +0.37 /
-0.14, i.e. own-filler **~ +3.8**. Against the config's "+2.1" that looks like **the writer beating the
untrained init -- the programme's first win.** Against the true **+6.93** it is **~3.1 nats BEHIND the
untrained init.** A tick that trusted the parenthetical would have announced precisely the false win this
programme exists to prevent, and it would have been the **fourth** instrument failure in the series (G94
filler drift, G108 shuffle exploit, G112 partial ladder, now this). **The gate's ENFORCEMENT is unaffected:**
the enforcer compares against the literal +9.00 bar, never against the comment. Only the *interpretation*
was at risk -- which is exactly how the previous three got through.
**RULE G114(e): A PARENTHETICAL "(init X)" IN A CONFIG IS NOT A MEASUREMENT.** Every init reference must be
re-read from its probe JSON at the moment it is used, and **the statistic must be named** (`own` vs
`own_minus_filler` is a 5-nat difference here). **I am NOT editing the two configs:** run 70 is live and its
config is its pre-registration document, so editing it mid-run is the same class of sin as G114(a). They are
marked stale here and in 00S.0; the authoritative numbers are the three JSON files above.
Also corrected, mine from 00:07 and repeated by me earlier in this entry before I checked: **only o=+1 has
n=424.** o=-15 and o=-9 are 10 val seqs, **n=132**. The 2026-09-30 journal had this right ("10 val seqs,
n=132-133; o=+1 32 seqs n=424") and the 00:07 tick generalised 424 to all three.

## G115 (2026-10-02 04:07 headless tick; run 70 `think70_frozen` STOPPED 02:20 on G1, run 71 `think71_frozen_imit` LIVE at step ~440/2400 and **G1 PASS**) -- **THE FIRST TIME ANY RUN HAS BEATEN THE UNTRAINED MODEL (+13.23 vs +6.93 AT o=-15), AND A SECOND, BETTER RESULT UNDERNEATH IT: AT THE FORECAST OFFSET BOTH TRAINED RUNS NOW WRITE ANCHOR-SPECIFIC NOTES AND THE UNTRAINED MODEL DOES NOT. ALSO: WE KILLED RUN 70 ON A DIP, AND I CAN PROVE IT.**

### (1) The instrument is sound, and this is the first tick that could verify it from the data rather than the code.
Across all four o=+1 probes -- untrained, run 70 @400, run 70 @500, run 71 @400 -- the frozen-reader baselines are
**identical to the decimal**: `filler +1.05`, `span +14.78`, `clean +11.03` (n=134). At o=-15 likewise:
`filler +0.1735`, `span +15.8194`, `clean +17.5714` (n=132), matching `span_vanilla22800_o-15_c.json` exactly.
G94 (filler drift) and G108 (`shuf` wrecked to win `own_minus_shuf`) cannot recur on `own_minus_filler` with the
reader frozen. **The baselines are now literally constants.** This is what runs 60-69 never had.

### (2) G1 @400: run 71 `own_minus_filler = +13.232 +- 0.961` at o=-15, bar +9.00, untrained +6.925 +- 0.848.
Delta vs the untrained init **+6.31 +- 1.28, z = +4.9.** First beat in the programme. **Say what it is**: at
o=-15 the span the note stands for is ALREADY VISIBLE in the writer's context, and run 71 is trained with
`--imit-w 1.0` directly on that span. The capability demonstrated is **supervised copying of visible text**.
`own +13.41` against `span +15.82` -- the writer recovers 85% of inserting the true passage. Dumps confirm it:
at step 400 the greedy note is ` feed one adult, you will most likely have to feed one adult will most` against
the oracle ` need two handfuls of grain to feed one adult, you will most likely need`.
**The bar's provenance, recorded because it looks like luck and is**: +9.00 was derived from the WRONG
"(init +2.1)" parenthetical (G114(e)). The true init is +6.925 +- 0.848; the paired s.e. of the difference is
hypot(0.848, 0.591) = 1.034, so "beats the init at 2 s.e." = **+8.99**. The mis-derived bar is numerically the
right bar. It was not moved in either direction and it is not being retro-justified -- it is being recorded.

### (3) THE RESULT THAT MATTERS, AND IT IS NOT THE HEADLINE ONE. `own_minus_shuf` AT o=+1.
All four rows below are the SAME 134 held-out items and the SAME frozen reader.

| model | own | shuf | own-filler | own-shuf |
|---|---|---|---|---|
| untrained init | -6.48 +-1.03 | -7.43 +-0.89 | **-7.53 +-1.02** | **+0.95 +-1.07** |
| run 70 @400 | -1.41 +-0.56 | -2.03 +-0.52 | -2.46 +-0.54 | +0.62 +-0.54 |
| run 70 @500 | -1.97 +-0.68 | -4.48 +-0.66 | -3.02 +-0.66 | +2.51 +-0.65 |
| run 71 @400 | -0.66 +-0.62 | -5.06 +-0.80 | **-1.71 +-0.62** | **+4.40 +-0.98** |

vs the untrained init: run 71 `own_minus_filler` **+5.82 +- 1.20 (z +4.9)**, `own_minus_shuf` **+3.45 +- 1.45 (z +2.4)**.
**The untrained model's forecast note is not anchor-specific at all** (+0.95 +- 1.07, indistinguishable from
zero): a note written for a DIFFERENT anchor serves the reader just as well. **Run 71's is** (+4.40 +- 0.98).
So the writer has begun to say something about the span it has NOT seen. And `own_minus_filler` is still
**-1.71**: a blank note still beats it. **Both facts at once have one reading -- the note is RELEVANT BUT WRONG
OFTEN ENOUGH THAT A CONFIDENT WRONG GUESS COSTS THE READER MORE THAN SILENCE. The missing ingredient is
accuracy, not relevance.** Mechanism: copy-teaching at o=-15 teaches "emit the 16 tokens the span covers"; at
o=+1 those tokens are the FUTURE, so the same policy emits the model's own continuation -- relevant by
construction, and wrong whenever the continuation is wrong. **Caveat stated with the claim: at step 400 run 71
had only ever trained at o=-15. No writer has yet been TRAINED at o=+1 and measured. Run 71's ladder reaches
+1 at step 1800.**

### (4) RULE G115(a): `own_minus_shuf` AT o=+1 IS THE PROGRAMME'S EARLY INDICATOR; `own_minus_filler` REMAINS THE WIN CONDITION.
`own_minus_filler` conflates two things -- is the note relevant, and is it right. `own_minus_shuf` isolates
relevance and is immune to the blandness confound in (5). Untrained reference, measure every claim against it:
**+0.95 +- 1.07 (n=134)**. Both are reported, always, and neither alone is a verdict.

### (5) RULE G115(b): "PROGRESS = above -4.00" IS RETIRED. IT WAS REACHABLE BY WRITING NOTHING USEFUL.
Runs 70/71's configs define PROGRESS at o=+1 as `own_minus_filler > -4.00`, "half the init's deficit closed".
But the init's note is actively HARMFUL (own -6.48 vs filler +1.05), so **a writer that learns only to be BLAND
walks own up toward filler and books "PROGRESS" having transmitted nothing** -- the ceiling of harmlessness is
own = filler, i.e. `own_minus_filler = 0`, scoring as maximal progress with zero information. Registered in
run 72's config before any run-72 data. **At o=+1 only `own_minus_filler > 0` is a win; everything short of it
is reported as harm reduction, in those words.** Note this criticism does NOT retract (3): (3) rests on
`own_minus_shuf`, which blandness cannot buy -- a bland note is equally bland for every anchor.

### (6) WE STOPPED RUN 70 AT THE BOTTOM OF A DIP, AND THE STEP-500 CHECKPOINT PROVES IT.
G1 @400 read `+4.475 +- 0.591` and the enforcer stopped the run at step 558. This tick probed the step-500
checkpoint the shutdown left on disk:

      o=-15 own_minus_filler:   step 400  +4.475 +- 0.591   (4.53 below the bar, 2.45 below the init, z -2.37)
                                step 500  +6.910 +- 0.700   (2.09 below the bar, 0.02 below the init, z -0.01)
      in-training greedy g_spec: 350-399 +4.33 | 450-499 +5.49 | 500-549 +5.74

**+2.44 per 100 steps on the probe statistic.** Linear extrapolation crosses +9.00 near step 590. The stop was
CORRECT BY THE REGISTERED RULE and is not overturned; **the rule was wrong.** A gate that reads ONE checkpoint
inherits that checkpoint's noise and its position in a transient.
**RULE G115(c): A GATE THAT STOPS A RUN MUST READ THE SAME STATISTIC AT TWO CHECKPOINTS AND STOP ONLY IF BOTH
FAIL *AND* THE SLOPE IS FLAT OR FALLING.** Implemented in `agent/armed_probes_trend.sh` for run 72: a G2 failure
at 1000 triggers a re-read at 1100 at the same offset, and stops only if both are below the bar and the slope is
under +1.0 per 100 -- **less than half the +2.4 per 100 run 70 was actually doing**, so the clause would have
spared run 70 without being generous. Bar unchanged at +3.70.

### (7) RULE G115(d): READ THE TARGET-OFFSET NUMBER BEFORE A PROXY GATE CAN DESTROY IT.
`armed_probes_frozen.sh` evaluates the COMPRESSION gate first and calls `stop_run` on failure, so the `o=+1`
TRANSFER read that follows it never executes. That is literally what happened to run 70: G1 failed, the run
died, and the forecast number -- the programme's actual target, and as (3) shows **the most informative number
of the night** -- had to be recovered by hand two hours later from an archived checkpoint. In
`armed_probes_trend.sh` the o=+1 read precedes every gate. Independently, `agent/transfer_curve.sh` (armed this
tick, pid 2286606, CPU-only) reads run 71 at o=+1 at steps 600/1200/2000 **regardless of whether a gate stops
the run**, giving a 7-point forecast curve with the enforcer's 400/1000/1800/2400.

### (8) A CALIBRATOR THAT MAKES THE NEXT PREDICTION CHEAP, AND A DOCSTRING CORRECTED.
`probe_span.py` calls `gen.generate(..., spec_row=1, ...)` and reads `z[1]`, and `think_par.py:313` makes row
`spec_row` the **argmax** row. So the probe's `own` is the **GREEDY** note -- its train-side twin is `g_spec`,
NOT `g_r0` (a sampled row). I briefly mis-read `ga = SimpleNamespace(temp=1.0, ...)` as evidence the probe
sampled; `temp` only touches the non-spec rows, which the probe discards. The docstring ("the model's own
greedy thought") is right. **Calibration: run 70 block 350-399 `g_spec` +4.332 -> probe @400 +4.475, offset
+0.14.** Applied to run 71 (`g_spec` 350-399 = +10.561) it predicted +10.7 against an actual +13.23 -- useful
to ~2-3 nats, no better, and biased LOW for an imitation-trained writer (train anchors are in-distribution).

### (9) SCORING, AND THE BIAS IS NOW TOO CLEAR TO CALL NOISE.
Six bands resolved (the 00:07 pair and four registered in `agent/predictions/pred_20261002_0407.md`, written
with `ls` verifying none of the JSONs existed):

      R70 G1 @400 o=-15      [+9.0,+14.0]  -> +4.475   MISS LOW
      R70 TRANSFER @400 o=+1 [-8.0,-4.0]   -> -2.46    MISS HIGH
      R70 @500 o=-15         [+5.0,+7.0]   -> +6.91    HIT
      R70 @500 o=+1          [-4.0,-0.5]   -> -3.02    HIT
      R71 G1 @400 o=-15      [+8.5,+12.5]  -> +13.23   MISS HIGH
      R71 TRANSFER @400 o=+1 [-11.0,-6.0]  -> -1.71    MISS HIGH

**Tally 15 -> 21 bands: 12 HIT, 6 MISS HIGH, 3 MISS LOW.** Every miss tonight is the same error twice over:
**too pessimistic wherever supervision or transfer was involved (three, all the outcome landing above my band)
and too optimistic about reward alone (one, below).** The R71 TRANSFER band is the worst of my career here --
I predicted [-11.0,-6.0] and got -1.71, after having ALREADY READ run 70's -2.46 forty minutes earlier. I had
the number that should have moved that band and did not use it. **RULE G115(e): WHEN A BAND IS WRITTEN AFTER A
RELATED NUMBER HAS LANDED, STATE IN THE BAND WHICH NUMBER WAS IN HAND -- a band that ignores evidence already
on disk is not a forecast, it is an anchor.** The transfer-curve bands (@600 [-6.0,-1.0], @1200 [-5.0,+1.0],
@2000 [-4.0,+2.0], registered 04:19) carry the same defect and I expect them to miss high; **they stand.**

## G116 (2026-10-02 06:07 headless tick; run 71 `think71_frozen_imit` LIVE and HEALTHY at step ~805/2400, ETA ~13:10) -- **THE FORECAST-OFFSET SIGNAL GOT STRONGER ON A SECOND INDEPENDENT CHECKPOINT (own_minus_shuf +4.40 -> +4.77, own_minus_filler -1.71 -> -1.24), MY 04:19 BAND HIT, AND I CHECKED THE CONTROL'S SOURCE CODE INSTEAD OF TRUSTING EITHER OF OUR TWO CONTRADICTORY DESCRIPTIONS OF IT. ONE OF THEM WAS WRONG. SEPARATELY: RUN 71's G2 GATE WAS STILL THE OLD SINGLE-CHECKPOINT KIND THAT KILLED RUN 70, AND IT IS NOW G115(c)/(d)-CONFORMING WITH EVERY BAR UNTOUCHED.**

**(1) THE NUMBER OF THE TICK: THE o=+1 SIGNAL IS NOT A ONE-CHECKPOINT ACCIDENT.** `agent/transfer_curve.sh`
(pid 2286606, armed 04:19 and deliberately independent of every gate) read run 71's step-600 checkpoint at the
forecast offset at 05:18:58. Same 134 held-out items, same frozen reader (`vanilla12b_fresh_ladder/step_0022800.pt`),
same probe:

| ckpt | own | shuf | filler | span | own_minus_filler | own_minus_shuf |
|---|---|---|---|---|---|---|
| untrained (n=134) | -6.31* | -6.84* | +1.67* | +15.65* | **-7.525 +- 1.020** | **+0.951 +- 1.070** |
| run 71 @400 | -0.66 | -5.06 | +1.05 | +14.78 | **-1.708 +- 0.624** | **+4.401 +- 0.983** |
| run 71 @600 | -0.19 | -4.97 | +1.05 | +14.78 | **-1.239 +- 0.636** | **+4.774 +- 0.902** |

(*absolute means from the n=424 file `span_vanilla22800_o+1.json`; the paired columns for the untrained row are
from `span_vanilla22800_o+1_n10.json`, **n=134**, which is the matched-n comparison. The n=424 file's paired
values are own_minus_filler **-7.988 +- 0.500** and own_minus_shuf **+0.528 +- 0.548** -- quote whichever, but
name which, per G114(e).)

Margins over the untrained at matched n=134: own_minus_shuf **+3.450 +- 1.454 (z 2.37)** at step 400 ->
**+3.823 +- 1.396 (z 2.74)** at step 600. own_minus_filler **+5.817 +- 1.196 (z 4.9)** -> **+6.286 +- 1.203
(z 5.2)**. Both improved on an independently-armed second checkpoint. **The absolute own_minus_filler is still
NEGATIVE (-1.24): a blank note still beats the writer's note, so by G115(b) this is HARM REDUCTION, not a win,
and I will keep using those words.** Slope on own_minus_filler at o=+1 is **+0.235 per 100 steps** over 400->600.

**(2) I READ THE CONTROL'S SOURCE BECAUSE OUR OWN TWO DESCRIPTIONS OF IT CONTRADICT EACH OTHER, AND ONE IS WRONG.**
G113(4b) calls `shuf` *"a permutation of the writer's own note"*; the 04:07 STATUS calls it *"another passage's
note"*. Those are different instruments and only one of them licenses the claim in (1). `probe_span.py:63` is
`"shuf": np.roll(own, 1, axis=0)` with `own` of shape `(layout.N, L)` -- **axis 0 is the ANCHOR axis**, so `shuf`
is **the note this same writer wrote for the NEIGHBOURING ANCHOR of the same sequence**, not a token permutation.
**STATUS was right; G113(4b)'s mechanism sentence was wrong** (its *conclusion* -- that the baseline moves with
the writer, so a frozen reader does not fix this control -- survives, because it is still the writer's own note).
This matters in our favour and I did not expect that: anchors are `--stride 64` apart, so the competitor note was
written for text 64 tokens away **in the same document**. `shuf` therefore controls for topic and for
general note quality, and beating it by +4.77 nats means the note is specific to **its own 16-token span**, not
merely on-topic. **RULE G116(a): BEFORE A CONTROL CARRIES A HEADLINE, READ THE LINE OF CODE THAT BUILDS IT.**
Four instrument failures (G94, G108, G112, G114(e)) were all cheaper to prevent than to discover.

**(3) A CALIBRATION I HAD NOT THOUGHT TO MAKE, AND IT IS THE MOST FAVOURABLE HONEST FRAMING AVAILABLE.** The
untrained model's own_minus_shuf by offset (all frozen-reader probes already on disk, n=132-134):
o=-15 **+6.560+-0.826**, o=-13 **+5.087+-0.746**, o=-11 **+6.210+-1.173**, o=-9 **+3.749+-1.137**,
o=-7 **+4.323+-1.128**, o=-5 **+3.621+-1.213**, o=-3 **+2.389+-1.348**, o=-1 **+5.302+-1.411**,
o=+1 **+0.951+-1.070**. The untrained writer is anchor-specific at every offset where the span is VISIBLE and
collapses to zero exactly at the forecast offset. **Run 71's +4.774 at o=+1 sits level with the untrained
model's o=-7 (+4.323) and above its o=-9 (+3.749): at the offset where the text is not yet written, run 71's
note is as specific to its own span as the untrained model's note is when the span is already in context.**
That is a statement about relevance only. Accuracy is still the missing ingredient (item 1's negative absolute).

**(4) RUN 71's G2 GATE WAS STILL THE INSTRUMENT THAT KILLED RUN 70, AND IT IS NOW CONFORMING. NO BAR MOVED.**
G115(c) and G115(d) were written at 04:07 but run 71's enforcer was armed at 02:21 and predates both: its
`gate 0001000 -9 3.70 G2` (`armed_probes_frozen.sh:108`) stops the run on a **single** checkpoint, and it
reads o=-9 **before** o=+1, so a stop would again have destroyed the forecast number. G114(a) forbade editing
that file (a bash process was executing it, exactly the 01:38:13 corruption), so the change went into a **new
file**, `agent/armed_probes_frozen_g115c.sh`: the o=+1 read moves FIRST (G115(d)), and a below-bar G2 triggers a
second read at step 1100 and stops only if that is **also** below +3.70 **and** not rising (G115(c)).
**Every bar is byte-identical: G2 >= +3.70 at o=-9, FINAL @2400 o=+1 at 32 seqs.** Old enforcer pid 2169913
killed 06:11, new pid **2300030** launched from `chmod 444` snapshot
`agent/running/armed_probes_frozen_g115c_think71_frozen_imit_20261002_061121.sh`, verified by `ps` plus a
06:11:21 log line postdating the 06:10:28 file mtime (G114(c)); 65 workers before and after.
**ONE DEVIATION I AM NAMING RATHER THAN HIDING:** G115(c) permits a stop when the slope is *flat*; my clause
continues on exactly-flat (`v2 >= v1`), and run 72's `armed_probes_trend.sh` is stricter still (it needs
`slope >= +1.0 per 100` to continue). Run 71's is the most permissive of the three **deliberately**: it is the
only run that has ever beaten the untrained model and its ladder does not reach the target offset until 1800.
With probe s.e. ~0.9 the difference is unobservable in practice.

**(5) HEALTH, AGAINST THE BAR REGISTERED AT 02:07 AND NOT WIDENED.** Held-out mix 2.7749 -> 2.8188 -> 2.8436 ->
2.8157 -> 2.8046 -> 2.8086 -> 2.8071 -> 2.8099 (steps 0..700). 300-window rate 400->700 = **+0.0017 per 100**
(bar +0.02); absolute max **2.8436** against 3.025 (margin 0.181). **The curve flattened**: the +0.0180 per 100
that was 6x runs 67/68/69 at 02:07 has decayed to a tenth of that. 0 NaN/inf in 805 rows, ntp1 block mean
**1.931** (bar 4.0), reg% **0.7** (bar 15), spread **2.51** (bar 0.6), len 14.8 (min-think 3), 65 workers,
1546 GB RAM free, 452 GB disk. **`or_g` is -70 to -140 and IS NOT AN AXIS**: run 71 carries `--bridge 15`, and
the sec. 3b bridge exemption (G52) applies; with a FROZEN reader the window-masked CE is far more negative than
the "-11 to -13" that exemption quotes for a learning reader, which is expected and enters no loss.

**(6) SCORING: ONE BAND, AND IT HIT -- BARELY, AT THE EDGE I SAID I WOULD MISS.** The 04:19 transfer-curve band
for @600 o=+1 was **[-6.0,-1.0]** against **-1.239**: **HIT**, 0.24 nats inside the top edge, and I had written
at 04:07 that I expected these bands to miss high. **Tally 21 -> 22 bands: 13 HIT, 6 MISS HIGH, 3 MISS LOW.**
The pessimism bias is still live -- the band hit only because it was five nats wide. The @1200 **[-5.0,+1.0]**
and @2000 **[-4.0,+2.0]** bands stand. Six new bands for 1000/1100/1800 were registered at 06:12 in
`agent/predictions/pred_20261002_0607.md`, `ls`-verified against the non-existence of their JSONs, and each one
NAMES the landed numbers that informed it (G115(e)) and is deliberately shifted UP for the standing bias.

## G117 (2026-10-02 08:07 headless tick; run 71 `think71_frozen_imit` LIVE and HEALTHY at step ~1250/2400, ETA ~13:10) -- **BOTH GATES PASSED AND THE BEST FORECAST-OFFSET NUMBER YET LANDED (-0.65, `own` POSITIVE FOR THE FIRST TIME). I ALSO READ THE RAW NOTES, FOUND THE WRITER COPYING THE TEXT IMMEDIATELY BEFORE THE WINDOW, REGISTERED A BAND SAYING THAT MECHANISM EXPLAINS OUR HEADLINE -- AND THEN BUILT THE CONTROL AND **REJECTED MY OWN NULL AT z=5.3: THE NOTE BEATS A VERBATIM COPY OF THE TEXT IT JUST READ BY +3.28 +- 0.62, AND ADJACENCY ACCOUNTS FOR NONE OF THE +4.11.** SEPARATELY, "THE SIGNAL IS STRENGTHENING" IS WITHDRAWN: ON FOUR POINTS IT IS FLAT.**

**(1) LIVENESS FIRST (G114(d)). ALL THREE AUTOMATIONS ALIVE, EVERY LANDED CHECKPOINT READ.**
Gate enforcer pid 2300030, transfer curve pid 2286606, successor watcher pid 2287957; 65 workers;
`.watchdog_run` = think71_frozen_imit. Steps 1000, 1100, 1200 had all landed and **1000 and 1200 both had
logged readings**; 1100 was correctly never probed (the two-checkpoint branch did not fire). No defect.

**(2) BOTH GATES PASSED, AND THE REVISED G2 BEHAVED EXACTLY AS DESIGNED.**
`TRANSFER @1000 o=+1` was read FIRST at 07:12:17 (G115(d)) -- own_minus_filler **-1.615 +- 0.791** -- and
only then `G2 @1000 o=-9` at 07:20:33: **+12.549 +- 0.96** against the +3.70 bar, **PASS**, untrained
+1.709 +- 0.915. So the single-checkpoint-vs-trend question never arose. **It still mattered that I fixed
it at 06:11**: had G2 failed, the old enforcer would have killed the run before the o=+1 number existed.

**(3) THE FOUR-POINT o=+1 CURVE, MATCHED n=134, SAME FROZEN READER. THE @1200 POINT IS THE BEST YET.**
| step | own | shuf | filler | own_minus_filler | own_minus_shuf |
|------|-----|------|--------|------------------|----------------|
| untrained | -6.479 | -7.430 | +1.046 | **-7.525 +- 1.020** | **+0.951 +- 1.070** |
| 400  | -0.662 | -5.064 | +1.046 | -1.708 +- 0.624 | +4.401 +- 0.983 |
| 600  | -0.193 | -4.967 | +1.046 | -1.239 +- 0.636 | +4.774 +- 0.902 |
| 1000 | -0.569 | -5.310 | +1.046 | -1.615 +- 0.791 | +4.741 +- 1.072 |
| 1200 | **+0.398** | -3.714 | +1.046 | **-0.648 +- 0.742** | **+4.112 +- 0.772** |

`span` +14.78, `clean` +11.03, `filler` +1.046 are frozen-reader constants and identical in all four files,
which is the standing proof that the reference really is fixed. **At step 1200 `own` is POSITIVE for the
first time: the note no longer damages the reader in absolute terms. It is still below a blank note
(+1.046), so this is HARM REDUCTION, NOT A WIN (G115(b)), and that is the first thing to say about it.**

**(4) I AM WITHDRAWING MY OWN 06:07 HEADLINE. "THE SIGNAL STRENGTHENED" WAS A TWO-POINT LINE.**
OLS over the four points: own_minus_filler **+0.087 per 100** (slowly up), own_minus_shuf **-0.031 per 100
-- FLAT**. The "+0.235 per 100" I registered at 04:19 and repeated at 06:07 came from exactly two points
(400, 600) and has not held. The honest statement: **the relevance signal appeared within the first 400
steps and has been flat at about +4.5 ever since; what is creeping up, slowly, is the harm number.**
**NEW RULE G117(b): A RATE COMPUTED FROM TWO POINTS IS NOT A RATE. STATE THE NUMBER OF POINTS BEHIND
EVERY SLOPE I REPORT, IN THE SAME SENTENCE AS THE SLOPE.** Three of my last four slope claims were n=2.

**(5) THE IMPORTANT PART OF THIS TICK: I READ THE ACTUAL NOTES, AND THEY ARE COPIES OF THE PRECEDING TEXT.**
From `dumps_eligible.jsonl` at step 1180 (ladder at o=-7), thought vs the window it is supposed to help:
 - thought `" want to borrow.\n2. Input the payment term: Enter the time in"` | window `" interest rate:
   Enter the interest rate for the loan. This is usually expressed as an annual percentage rate (APR)."`
 - thought `" get a better interest rate.\n• When your financial goals are right, but"` | window
   `" Negotiate: Don't be afraid to negotiate with lenders to get a better deal."`
 - thought `" calculator allows you to compare multiple loan options together. Therefore, these tools optim"`
**The note is a near-verbatim continuation of the text immediately BEFORE the window** -- and because six
hand-picked examples are not evidence, I measured it over **all 1177 eligible notes from steps 1180-1267**:
the note's words overlap the 40 words BEFORE the anchor **60.6%** of the time versus **38.2%** for the 20
words AFTER it, and **50% of notes share a verbatim run of >= 4 words with the preceding text against 9%
with the following text** (mean longest shared run 3.61 words before vs 1.73 after). At steps 0-120 those
figures were 15% and 2% (runs 2.13 vs 1.14), **so copying is a behaviour the run ACQUIRED: it more than
tripled over training.** `--imit-w 1.0` teaches precisely this, and at a visible offset it is why G1/G2
read +13.2 / +12.5. **At o=+1 the span is
invisible, so "copy the nearest visible text" is the only strategy the writer has -- AND THAT MECHANISM
PREDICTS own_minus_shuf > 0 ON ITS OWN, WITH ZERO FORECASTING**, because `shuf` is the note for the anchor
64 tokens EARLIER (G116(2)) and therefore echoes text further from this window. **Our best number may be
measuring adjacency, not anticipation.**

**(6) SO I BUILT THE CONTROL THAT DECIDES IT, AND REGISTERED THE BAND BEFORE RUNNING IT.**
New file `pre-training/train/probe_span_echo.py` (`py_compile` clean, `chmod 444`) adds one condition:
**`echo` = the L-1 = 15 tokens ending immediately before the span, verbatim, pasted in as the note** --
the locality null made into a baseline. It reports `own_minus_echo`, `echo_minus_shuf`, `echo_minus_filler`
and how many echo windows straddle an EOD. The `ok` mask is byte-identical to `probe_span.py`'s, so n=134
and `own`/`shuf`/`span`/`filler` must reproduce the landed @1200 values exactly -- a built-in consistency
check on the new file. **`probe_span.py` ITSELF IS NOT EDITED** (verified byte-identical, `git diff` empty):
two live automations are calling it right now and G114(a) forbids changing an instrument under a running
process. Running on step 1200 at o=+1, pid 2317105.
**MY REGISTERED BAND (`agent/predictions/pred_20261002_0807.md`, written before launch, `ls`-verified):
own_minus_echo in [-3.0, +1.0]; P(own beats echo at > 2 s.e.) = 0.25. I expected our headline to be
explained away.**

**(6b) THE LOCALITY NULL IS REJECTED, DECISIVELY, AND I WAS WRONG. THIS IS THE STRONGEST RESULT THE
PROGRAMME HAS PRODUCED.** Step 1200, o=+1, n=134, same frozen reader:
| content in the note slot | nats vs silence |
|---|---|
| `span` -- the TRUE next 15 tokens | **+14.78 +- 0.95** |
| `filler` -- 10 spaces (a blank note) | +1.05 +- 0.44 |
| **`own` -- the writer's note** | **+0.40 +- 0.60** |
| **`echo` -- the 15 tokens it JUST SAW, verbatim** | **-2.88 +- 0.53** |
| `shuf` -- its note for the neighbouring anchor | -3.71 +- 0.80 |

 - **own_minus_echo = +3.28 +- 0.62 (z = 5.3).** The writer's note is worth over three nats MORE than a
   verbatim copy of the text it just read. **Copying is not what it is doing at the forecast offset.**
 - **echo_minus_shuf = +0.83 +- 0.93 -- INDISTINGUISHABLE FROM ZERO.** Adjacency buys at most a fifth of
   the +4.11 `own_minus_shuf`, and that fifth is not statistically there. **The deflationary explanation I
   proposed in item (5) accounts for essentially none of our headline number.**
 - **echo_minus_filler = -3.93 +- 0.72.** Pasting raw recent text into the slot ACTIVELY HARMS the reader,
   while the true future passage in the SAME FORMAT (`span`, also L-1 tokens + `</T>`) is worth +14.78. So
   the slot is not confused by raw text -- the reader is sensitive to WHICH text. That contrast is itself
   the cleanest demonstration we have that the instrument measures content relevance and not formatting.
 - **INSTRUMENT CHECK PASSED.** The new file reproduced every landed @1200 value to the printed digit:
   own +0.40, shuf -3.71, span +14.78, filler +1.05, clean +11.03, own_minus_filler -0.65, own_minus_shuf
   +4.11. The `echo` number can therefore be read straight against the curve in item (3). 2 of 134 echo
   windows straddled an EOD -- immaterial.
 - **WHAT IS NOW ESTABLISHED, STATED WITH THE DEFLATION FIRST.** A blank note still beats the writer's note
   (**own_minus_filler -0.65 +- 0.74**, i.e. parity within one s.e.), so by G115(b) this is STILL HARM
   REDUCTION AND NOT A WIN. But the content of that note is now positively characterised: it is **not**
   blandness (blandness cannot buy own_minus_shuf, which is equally bland for every anchor), it is **not**
   adjacency (echo_minus_shuf ~ 0), and it is **not** document-level gist (G116(2): `shuf` holds the
   document and the topic fixed). **It is anchor-specific content about 16 tokens the writer has not seen,
   worth +4.11 +- 0.77 over the same writer's note for a neighbouring anchor and +3.28 +- 0.62 over the
   text it just read.** What it still lacks is enough accuracy to beat saying nothing.
**NEW RULE G117(a): WHEN A CONTROL DIFFERS FROM THE TREATMENT ON AN INCIDENTAL DIMENSION, BUILD THE
CONTROL THAT MATCHES ON THAT DIMENSION BEFORE THE RESULT CARRIES A HEADLINE.** `shuf` differs from `own`
in two ways at once -- which anchor it was written for AND how far its source text is from the window --
and we attributed the whole effect to the first. This is the fifth instrument problem of the programme
(G94, G108, G112, G114(e)) and the second I have caught before publishing it rather than after.

**(7) THE BLANDNESS WATCH IS NOW A LIVE, REGISTERED QUESTION.** Between 1000 and 1200 `own` rose +0.97
but `shuf` rose **+1.60** -- the control improved MORE than the treatment, which is exactly the signature
G115(b) warned about: a writer that buys harm reduction by becoming BLAND walks `own` toward `filler`
while `own_minus_shuf` decays toward zero. The difference is -0.63 +- ~1.3, well inside noise, so this is
**not a claim**; it is registered as band 8 (**NO, p~0.6**) to be settled by the 1400/1600 reads.

**(8) DEVICE ACTION: I CLOSED THE 600-STEP HOLE IN FRONT OF THE DECISIVE READ.** The o=+1 curve had
400/600/1000/1200 and then **nothing until 1800** -- and 1800 is the first checkpoint ever TRAINED at the
forecast offset, the read this whole line turns on. With a 600-step hole in front of it **we could not
have told a step change at 1800 from a dip, which is the exact instrument failure that cost us run 70**
(G115(6)). New file `agent/transfer_curve_gap.sh` (`bash -n` clean), a second concurrent instance of the
curve with its own log, armed for **1400 (ladder -3) and 1600 (ladder -1)**: steps that collide with
neither the enforcer's 400/1000/1800/2400 nor the first curve's 600/1200/2000. Launched 08:14:02 from
`chmod 444` snapshot `agent/running/transfer_curve_gap_think71_frozen_imit_20261002_081402.sh`, verified
by `ps` plus a log line postdating the template mtime (G114(c)). Probes are CPU-only (`USE_TORCH_XLA=0`,
24 threads, load 71 of 192 cores); **workers were 65 before and 65 after every launch this tick.**
`agent/transfer_curve.sh` verified byte-identical and untouched.

**(9) RUN 72's AUTO-LAUNCH VERIFIED NOW RATHER THAN AT 13:10.** `checkpoints/think72_frozen_resume/`
holds `step_0000500.pt` + `step_0000500.optim.pt` (3.0 GB + 6.0 GB, staged 04:23); `run_think.sh:113`
adds `--resume latest` whenever that directory is non-empty, and `think_train.py:588` takes `start_step`
from the checkpoint, so the span schedule -- keyed to ABSOLUTE steps -- resumes correctly at 500 and runs
1900 further steps to 2400. **A staging defect found now is free; found at 13:10 it is a failed smoke and
an idle device.** No defect. Run 72 stands as armed.

**(10) HEALTH, AGAINST THE 02:07 BAR, NOT WIDENED.** Held-out mix 800->1100 = 2.8090 -> 2.8143, 300-window
rate **+0.0018 per 100** (bar +0.02, 4 points) -- the 02:07 scare is gone; absolute max 2.8436 vs 3.025
(margin 0.181). **0 NaN/inf and 0 Traceback in 1232 metric rows**, ntp1 1.828, reg% 0.5-2.4, spread 3.11,
len 15.0, 65 workers, 446 G free. No derail clause fires. **No futility clause fires**: run 71's only
registered bars are G1 and G2 and both PASSED, the o=+1 harm number is at its best value, and inventing a
stop now would repeat run 70 exactly. **Run 71 continues to 2400.** `or_g` -31 to -40 is still not an axis
(sec. 3b G52 bridge exemption, `--bridge 15`, enters no loss).

**(11) SCORING: SIX BANDS RESOLVED, FIVE HIT, AND THE ONE MISS IS INSTRUCTIVE.** G2 @1000 [+8.0,+14.0]
-> +12.549 **HIT**; two-checkpoint branch fires NO -> did not fire **HIT**; own_minus_shuf @1000
[+3.5,+7.5] -> +4.741 **HIT**; held-out rate [-0.005,+0.010] -> +0.0018 **HIT**; the 04:19 band @1200
[-5.0,+1.0] -> -0.648 **HIT** (and I had predicted in writing that this one would miss high -- it did not).
**MISS: TRANSFER @1000 o=+1 [-1.6,+0.4] -> -1.615, LOW BY 0.015.** I will not round that into a hit: it
is a miss, and its direction matters -- it is the first time I have been too OPTIMISTIC on a transfer
band, because at 06:12 I shifted it up to correct the standing pessimism. **The correction overshot, so
bands 4-7 today are centred on the measured points and are NOT shifted up again.**
**Tally 22 -> 28: 18 HIT, 6 MISS HIGH, 4 MISS LOW.**

**(12) AND THEN THE LOCALITY BANDS RESOLVED WITHIN THE SAME TICK -- ALL THREE MISSED.** own_minus_echo
[-3.0,+1.0] -> **+3.28: MISS HIGH by 2.28**, and I had put P(own beats echo at >2 s.e.) at 0.25 when it
beat it at 5.3 s.e.; `echo` absolute [-2.0,+3.0] -> **-2.88: MISS LOW by 0.88**; echo_minus_shuf
[+1.5,+7.0] -> **+0.83: MISS LOW by 0.67**. **Tally 28 -> 31: 18 HIT, 7 MISS HIGH, 6 MISS LOW.**
The lesson is the standing one, reasserting itself in the same tick in which I claimed to have corrected
for it: **I am systematically too pessimistic about what the writer can do, and I was wrong three times in
a row inside twenty minutes about a mechanism I had just talked myself into.** G117(a) still stands as a
rule -- building the matched control was the right move and it is the reason we now know what the +4.11
is NOT. The rule earns its keep whether the null survives or dies.

---

## G118 (2026-10-02 10:07 headless tick; run 71 `think71_frozen_imit` LIVE and HEALTHY at step ~1690/2400, ETA ~13:10) -- **THE TWO POINTS THAT CLOSED THE 600-STEP HOLE LANDED AND ALL FIVE OPEN BANDS HIT. ON SIX POINTS THE HARM NUMBER IS NOW RISING AT z=2.59 -- BUT MOST OR ALL OF THAT SLOPE IS THE TRAIN/TEST OFFSET GAP CLOSING, AND THAT MECHANISM SATURATES AT STEP 1800, WHICH IS A FALSIFIABLE PREDICTION ABOUT THE NEXT THREE READS. SEPARATELY: THE WRITER HAS LARGELY STOPPED COPYING (VERBATIM-RUN SHARE 44% -> 15%) WHILE ITS OVERLAP WITH THE FUTURE ROSE, AND RUN 73 IS THE PROGRAMME'S FIRST WARM RESUME, WHICH IS A DOCUMENTED OOM RISK I INSURED BEFORE IT COULD IDLE THE DEVICE.**

**(1) G114(d) LIVENESS FIRST. FIVE PROCESSES EXPECTED, FOUR FOUND, AND THE MISSING TWO WERE BOTH CORRECT.**
Enforcer 2300030 alive; transfer curve 2286606 alive; successor **2328320** alive (the NEW r71->r73 handover
armed by the 09:25 interactive session); **gap curve 2317442 GONE -- because it FINISHED**, logging
`transfer curve (GAP) complete` at 10:03:29 after both of its reads; **old 71->72 waiter 2287957 GONE --
retired on purpose at 09:25.** 65 workers, `.watchdog_run` = think71_frozen_imit, 6 crons alive. Every
landed checkpoint has a logged reading: 1400 and 1600 both read. No defect. `status.py` needs
`.venv/bin/python` (it imports numpy); the bare `python3` in the runbook's instruction fails.

**(2) THE o=+1 CURVE IS NOW SIX POINTS AND THE 600-STEP HOLE IN FRONT OF THE DECISIVE READ IS CLOSED.**
Matched n=134, same frozen reader, `filler` +1.046 / `span` +14.78 / `clean` +11.03 identical in all six
files (the standing proof the reference really is fixed):

| step | ladder | own | shuf | own_minus_filler | own_minus_shuf |
|------|--------|-----|------|------------------|----------------|
| untrained | -- | -6.479 | -7.430 | **-7.525 +- 1.020** | **+0.951 +- 1.070** |
| 400  | -13 | -0.662 | -5.064 | -1.708 +- 0.624 | +4.401 +- 0.983 |
| 600  | -11 | -0.193 | -4.967 | -1.239 +- 0.636 | +4.774 +- 0.902 |
| 1000 | -7  | -0.569 | -5.310 | -1.615 +- 0.791 | +4.741 +- 1.072 |
| 1200 | -5  | +0.398 | -3.714 | -0.648 +- 0.742 | +4.112 +- 0.772 |
| 1400 | -3  | **+0.602** | -4.430 | **-0.444 +- 0.746** | **+5.030** |
| 1600 | -1  | +0.283 | -4.780 | -0.763 +- 0.901 | **+5.060** |

**DEFLATION FIRST, AS ALWAYS (G115(b)): a blank note still beats the writer at every one of the six points,
so this is HARM REDUCTION, NOT A WIN.** What is new and solid: **`own` is positive at 1200/1400/1600, three
checkpoints running** -- the note helps the reader in absolute terms, it just helps less than saying nothing.

**(3) ON SIX POINTS THE HARM NUMBER IS RISING AT z=2.59, AND own_minus_shuf IS STILL FLAT (G117(b): POINT
COUNTS STATED).** OLS, **six points**: own_minus_filler **+0.0903 +- 0.0349 per 100 (z 2.59)**, residual sd
0.361; own_minus_shuf **+0.0357 +- 0.0356 per 100 (z 1.00) -- FLAT**, which also retires the sign of my 08:07
four-point estimate (-0.031) without changing its verdict. The filler-relative fit crosses zero at step
**2218**, i.e. before run 71's own final read. **At 08:07 I withdrew "the signal is strengthening" because on
four points it was flat; on six points the HARM number, and only the harm number, is no longer flat.**

**(4) BUT I WILL NOT SELL THAT SLOPE, BECAUSE ITS MOST LIKELY CAUSE DIES AT STEP 1800. G118(a): WHEN A
CURVE'S x-AXIS ALSO MOVES THE TASK, THE SLOPE IS CONFOUNDED WITH THE SCHEDULE -- NAME THE CONFOUND IN THE
SAME SENTENCE AS THE SLOPE.** Across those six points the TRAINING offset marched -13 -> -1 while the PROBE
offset stayed at +1, so part of +0.09/100 is simply the train/test gap closing. **The ladder holds at +1 from
step 1800, so that mechanism saturates there and extrapolation past it is unjustified.** This is testable,
not rhetorical, and it is registered as **band 12: is @2400 own_minus_filler more than 1 s.e. above @1800's?
NO, p~0.6.** If flat, the gain was the schedule and **more compute at the forecast setting cannot buy the
win** -- which is precisely the claim run 73 `think73_hold` (3000 extra steps all at o=+1) was armed to test,
and the discriminator now exists *before* run 73 starts rather than after it finishes.

**(5) ALL FIVE OPEN BANDS HIT, INCLUDING THE ONE I CARED ABOUT.** @1400 own_minus_filler [-2.2,+0.3] ->
**-0.444 HIT**; @1400 own_minus_shuf [+2.8,+5.8] -> **+5.030 HIT**; @1600 own_minus_filler [-1.8,+0.8] ->
**-0.763 HIT**; @1600 own_minus_shuf [+2.5,+5.8] -> **+5.060 HIT**; and **band 8, THE BLANDNESS
DISCRIMINATOR (registered NO, p~0.6) -> HIT**: own_minus_filler went -0.648 -> -0.444 -> -0.763 (**did not
rise**) while own_minus_shuf went +4.112 -> +5.030 -> **+5.060, its highest value in the programme**. The
blandness signature is own_minus_filler rising as own_minus_shuf falls; **we got the opposite, so the writer
is becoming MORE anchor-specific, not blander.** **Tally 31 -> 36: 23 HIT, 7 MISS HIGH, 6 MISS LOW.**

**(6) THE WRITER HAS LARGELY STOPPED COPYING, MEASURED OVER ~1200 NOTES PER WINDOW, NOT SIX EXAMPLES.**
Word overlap of the note with the 40 words BEFORE the anchor vs the 20 AFTER, and the share of notes sharing
a verbatim >=4-word run with each side (`dumps_eligible.jsonl`, `src=rl`):

| steps | ladder | overlap BEFORE | overlap AFTER | >=4-run BEFORE | >=4-run AFTER | run ratio |
|-------|--------|----------------|---------------|----------------|---------------|-----------|
| 0-120     | -15 | 32.3% | 17.1% | 11.7% | 1.6%  | 7.3x |
| 1180-1267 | -5  | 46.6% | 24.8% | **44.0%** | 5.7%  | 7.7x |
| 1550-1700 | -1  | 36.1% | 28.2% | **15.1%** | **11.5%** | **1.3x** |

**Copying PEAKED mid-ladder and has collapsed by o=-1, while overlap with the FUTURE rose 5.7% -> 11.5%.**
**HONESTY ABOUT THE INSTRUMENT: these are MY numbers from MY implementation and they do NOT reproduce the
absolute levels I quoted at 08:07 (60.6%/38.2%, 50%/9%) -- a different normalisation, and I did not keep the
08:07 code. Only the within-implementation comparison across step ranges is used, which is the valid one.
G118(b): AN AD-HOC MEASUREMENT WORTH QUOTING TWICE IS WORTH SAVING AS A FILE THE FIRST TIME -- AND I
OBEYED IT IN THE SAME TICK: it is now `agent/copy_rate.py <run> [lo:hi ...]` (`py_compile` clean), which
reproduces the 0-120 and 1180-1267 rows above to the printed digit. Quote that file, never a fresh
reimplementation.** One wrinkle it exposed at once: the 1550-1700 window was still FILLING when I first
read it, so re-running 12 min later gives **n 1201 -> 1565 and 15.1%/11.5% (ratio 1.3x) -> 13.7%/11.7%
(ratio 1.2x)** -- substance unchanged, direction if anything stronger, but **a step range whose upper
bound is ahead of the current step is not a fixed sample: quote the range WITH its n.**
**DEFLATION: at o=-5 the span is still partly visible, so copying the preceding text IS the task there; part
of this fall is the target moving rather than the writer improving.** A representative step-1662 note shows
what replaced copying -- context ends `...two numbers 9 and 2. 9`, note `9 and 2. For example, 9 % 2 =  `,
window begins `%2 is 1 because when 9 is divided by 2` -- part copy, part genuine continuation.

**(7) THE DEVICE ACTION: THE ECHO CONTROL NOW TRAVELS WITH THE HEADLINE INSTEAD OF TRAILING IT.** At 08:07
the locality control existed at ONE checkpoint, 1200, trained with the span only 5 tokens out of sight. New
`agent/echo_curve.sh` (`bash -n` clean), launched 10:15:00 as pid **2336939** from `chmod 444` snapshot
`agent/running/echo_curve_think71_frozen_imit_20261002_101500.sh`, verified by `ps` **plus a log line
postdating the template mtime 10:12:05** (G114(c)): it runs `probe_span_echo.py` at **1800 (10 seqs) and
2400 (32 seqs)** -- the first checkpoints ever TRAINED at o=+1, exactly where "copy the nearest visible
text" becomes a WRONG strategy and where the null I rejected could come back to life. It reads only the
checkpoints the enforcer has **already archived** and **waits for the enforcer's own `TRANSFER @N o=+1` log
line before starting** (90-min cap so a read is never lost), so the decisive read never contends with it for
CPU. probe_span.py untouched -- verified this tick by mtime **2026-10-01 16:17:07**, i.e. before run 71 even launched (G114(a)). The log formatter was dry-run against
the landed @1200 JSON and reproduced every printed digit before launch. **Workers 65 before and after.**
**A FREE INSTRUMENT CHECK FALLS OUT OF THIS: `echo` does not depend on the writer at all** -- it is the
verbatim pre-span tokens scored by the frozen reader -- **so it is a constant like `span`/`filler`/`clean`
and must reproduce -2.879 at 1800 and 2400. If it moves, the control is not writer-independent and the
own_minus_echo bands are void** (registered as band 15).

**(8) THE SECOND DEVICE ACTION, AND IT IS OPERATIONAL: RUN 73 IS THIS PROGRAMME'S FIRST WARM RESUME AND
`train.py:280` SAYS THAT IS HOW YOU LOSE A NIGHT.** Run 73 `think73_hold` resumes run 71's step-2400 model
**and Adam state**; `pre-training/train/train.py:280` documents a **2026-08-28 history in which warm resumes
OOMed at the FIRST POST-RESUME CHECKPOINT SAVE through three different save implementations**, because
restoring the optimizer eagerly gives the process a device-allocation layout unlike a fresh run's, and only
the fresh layout is proven to fit the per-step transients AND a save. For run 73 that save is **step 2500,
~27 min after a ~13:10 launch**; `run_think.sh` would retry with `--resume latest` -- warm again -- fail the
same way three times, write FAILED, and **idle a trn2.48xlarge until a tick noticed (up to 14:07).** The
documented fix `--resume-model-only 1` is wired and costs ~50 of 3000 steps of Adam momentum (1.7%).
**I registered the contingency in `runs/think73_hold/config.sh` BEFORE any run-73 data existed** (backup
`.bak_pre_contingency_20261002_1015`; `diff` of the sourced `MAX_STEPS|TITLE|FLAGS` proves the executable
content is **byte-identical** -- a comment only), then armed `agent/resume_oom_guard.sh` (`bash -n` clean,
helpers unit-tested on real and missing files, the config patch dry-run on a copy with the FLAGS expansion
checked), pid **2337419**, snapshot `agent/running/resume_oom_guard_think73_hold_20261002_101808.sh`.
**It is deliberately hard to fire: `.watchdog_run` must name run 73, `supervisor.log` must show `retry 2/3`
(TWO failures), the max step must still be <= 2550 (a crash at step 4000 means the warm resume WORKED and is
the keepalive's business), and FLAGS must not already carry the flag. It intervenes at most once, verifies
its own stop by G22 (metrics static) before relaunching, and logs the error text whatever it turns out to
be.** **G118(c): BEFORE THE FIRST EXECUTION OF ANY CODE PATH THIS PROJECT HAS NEVER RUN, GREP THE SOURCE
FOR THE COMMENT THAT SAYS WHY THE FLAG EXISTS.** **AND THE OPEN ITEM A LATER TICK MUST NOT DROP: run 72
`think72_frozen_resume` IS ALSO A WARM RESUME (from step 500) and it launches after run 73, ~24 h away --
the tick that sees run 72 start must arm a guard for it with `agent/resume_oom_guard.sh think72_frozen_resume 500`.**
I did not arm one now on purpose: it would wait 24 h, stand down, and leave a false sense of cover.

**(9) FUTILITY CLAUSE 3c IS A DEAD AXIS ON A LADDER RUN, AND IT WOULD FIRE RIGHT NOW IF READ LITERALLY.**
Sec. 3c stops a run if after 1,500 steps `g_r0` and `gSp` have not improved by >= 0.3 over their
first-300-step values. Run 71 is at 1690 with `g_r0` **+5.21 (0-299) -> +2.68 (1500-1648)** and `gSp`
**+7.15 -> +3.71** -- *worse* by 2.5 and 3.4, so a tick reading 3c literally would **kill run 71 with 710
steps to go and destroy the 1800/2000/2400 reads the whole programme turns on.** It must not. Under
`--span-sched` the TASK gets harder every 200 steps by design: the in-training gain per rung recovers and
plateaus within ~100 steps (o=-11 9.05/8.72/9.49/9.61; o=-9 7.19/7.88/7.86/7.73; o=-7 6.55/7.26/7.23/6.60;
o=-5 4.16/4.28/4.98/4.91), so a fixed-step comparison of `g_r0` compares **different tasks**. This is the
sixth instance of the same general rule (G54/G56(c)/G76/G77/G84): **walk all of 3b AND 3c before a launch
and write down which clauses are dead for that design.** On a ladder run futility must be judged *within* a
rung, not across the schedule -- and run 71 has **no remaining stopping bar** anyway (both gates passed).

**(10) HEALTH, AGAINST THE 02:07 BAR, NOT WIDENED.** Held-out mix 1300 -> 1600 = 2.8161 -> 2.8185,
**+0.0008 per 100** (bar +0.02, **4 points**); absolute max **2.8185** vs 3.025 (margin 0.207, widening).
**0 NaN/inf in 1690 rows, 0 Traceback and 0 `NRT_` in train.log**, 65 workers, 441 G free, reg% 0.6
(bar 15), spread 3.32 (bar 0.6), len 15.0 (bar 3.5). `status.py` printed `DRIFT!` on a single-step ntp1
2.156 against roll20 2.182 -- the derail bar is a **block mean > 4.0** and the block reads 2.019, so no
action (same non-event as 08:07). `orG` -45 is not an axis (sec. 3b G52 bridge exemption, `--bridge 15`).
No derail clause fires; no futility clause fires (see (9)).

**(11) RUN 73's STAGING VERIFIED NOW, NOT AT 13:10 -- THREE THINGS THAT WOULD EACH HAVE COST THE NIGHT.**
(a) `--lr-horizon` is **not** dead: argparse at `think_train.py:74` and consumed at `train.py:67`
(`horizon = getattr(args,"lr_horizon",None) or args.max_steps`; `if step >= horizon: return args.min_lr`),
so run 73 holds **5e-6 flat** from 2400 instead of the cosine re-stretch jumping to ~8.1e-6 -- I checked
this because a grep of think_train.py alone showed the flag parsed and never used, which is exactly what a
broken flag looks like. (b) The frozen reader really is the INIT: `copy.deepcopy(backbone)` at
`think_train.py:518` precedes the resume at `:594`. (c) `--resume-model-only` is wired at `train.py:280`,
which is what made (8) possible. `agent/armed_probes_hold.sh` exists and is `bash -n` clean, and the
successor's seeding/launch logic reads correctly for both branches (2400 reached -> run 73; ended early ->
run 72).

**(12) WHAT THE NEXT TICK IS FOR.** **TRANSFER @1800 o=+1 (~11:1x) is the first checkpoint ever TRAINED at
the forecast offset and the read this line turns on** (band 9 [-1.0,+2.0], own_minus_shuf band 10
[+3.8,+6.5]), immediately followed by **ECHO @1800** (bands 14/15). Then @2000 (~12:1x), **FINAL @2400 32
seqs (~13:1x; band 13 [-1.0,+1.3], P(WIN) = 0.12)** and **ECHO @2400** (band 16, P(own beats echo at
>3 s.e.) = 0.85 -- I put it at 0.25 last time and was wrong by 5 s.e.). Then run 73 self-launches and
**band 12, the saturation discriminator, decides whether more compute at the forecast setting can buy
anything at all.** All bands: `agent/predictions/pred_20261002_1007.md`.

---

## G119 (2026-10-02 12:07 headless tick; run 71 `think71_frozen_imit` LIVE and HEALTHY at step ~2126/2400, ETA ~13:31) -- **THE SIX-POINT SLOPE I REFUSED TO SELL AT 10:07 IS DEAD ON EIGHT POINTS (z 2.59 -> z 0.35), AND THE IN-TRAINING SIGNAL SAYS WHY: AT THE FORECAST RUNG THE RUN IS BUYING BLOCK PRICE (g_shuf0 RISING AT z 5.1) AND NOT CONTENT (ob0 FLAT AT z 0.2). SEPARATELY, I FOUND THAT EVERY COPY-RATE NUMBER THIS PROGRAMME HAS EVER QUOTED CAME FROM A GAIN-SELECTED DUMP WHOSE SELECTION RATE FELL FROM 71% TO 1.6% AT STEP 1800, VOIDED MY OWN BAND BECAUSE OF IT, BUILT THE UNSELECTED INSTRUMENT -- AND THE UNSELECTED NUMBERS ARE BETTER: THE SHARE OF NOTES CARRYING A VERBATIM >=4-WORD RUN OF THE *FUTURE* DOUBLED, 16.4% -> 32.8%, ON MATCHED ANCHORS. A THIRD OF THE NOTES ANTICIPATE REAL FUTURE TEXT AND THE READER STILL PREFERS SILENCE.**

**(1) THE SLOPE DIED EXACTLY WHERE I SAID IT WOULD, AND THAT IS THE RESULT.** At 10:07 I had six points of
the o=+1 transfer curve rising at **+0.0903 +- 0.0349 per 100 (z 2.59)**, a fit crossing zero at step 2218 --
and I wrote in the same sentence that the TRAINING offset had marched -13 -> -1 toward the probe's +1 over
those six points, that this confound saturates at step 1800 where the ladder stops moving, and that
extrapolating past 1800 was not warranted. The two reads that landed since (@1800 **-1.430 +- 0.900**, @2000
**-1.714 +- 0.998**) take the fit to **+0.0130 +- 0.0365 per 100 (z 0.35) on eight points**: flat. The
six-point fit predicted **-0.20** at step 2000 and the measurement is **-1.714**, 1.5 s.e. below it.
**G119(a): WHEN YOU NAME A CONFOUND IN THE SAME SENTENCE AS A SLOPE AND REGISTER THE STEP AT WHICH IT DIES,
THE PROGRAMME GETS A FREE FALSIFICATION TWO HOURS LATER. THAT IS WHAT THE DISCIPLINE IS FOR.**

**(2) AND THE IN-TRAINING INSTRUMENT SAYS WHAT THE FORECAST RUNG IS ACTUALLY BUYING.** New
`agent/rung_trend.py` (G118(b): the rung-recovery numbers were quoted inline to justify run 73 at 09:25 and
re-derived inline to doubt it at 12:07, so they are a file now), 50-step bins, on
`ob0 = g_r0 - g_shuf0` -- the same note shuffled, so the order-free part subtracted:

| rung | bins | **ob0 slope /100** | ob0 mean | g_r0 slope /100 | **g_shuf0 slope /100** |
|------|------|--------------------|----------|-----------------|------------------------|
| o=-5 | 4 | **+0.624 +- 0.182** | +5.21 | +0.590 +- 0.200 | -- |
| o=-3 | 4 | **+0.398 +- 0.230** | +5.43 | +0.716 +- 0.137 | -- |
| o=-1 | 4 | **+0.613 +- 0.268** (z 2.29) | +4.24 | +0.790 +- 0.349 | +0.177 +- 0.217 (z 0.82) |
| **o=+1** | **6** | **+0.041 +- 0.171 (z 0.24)** | **+1.99** | +0.415 +- 0.202 (z 2.06) | **+0.374 +- 0.073 (z 5.10)** |

Every rung before the forecast one recovers its anchor-specific signal at +0.40 to +0.62 nats per 100 steps.
The forecast rung recovers **nothing** in 300 steps -- while `g_r0` rises at z 2.06 with `g_shuf0` rising
at **z 5.10** underneath it. **The run is learning what the block of 15 extra positions is worth, not what
the note says.** That is `--code-size` run 37's disease (G56(c): never gate on `g_r0` alone) in a run nobody
thought was at risk of it, and it is the cleanest instance yet because the control moves FASTER than the
treatment. **G119(b): A GAIN AXIS THAT INCLUDES A PRICE TERM IS NOT A LEARNING CURVE. PLOT THE PAIRED
DIFFERENCE AGAINST ITS OWN CONTROL, AND IF THE CONTROL IS MOVING FASTER, THAT IS THE STORY.**

**(3) THE COPY-RATE INSTRUMENT WAS BROKEN AND I BROKE MY OWN BAND ON IT.** Band 17 (registered 10:14) asked
for the verbatim-run copy rate at steps 1850-2000. It reads 14.9% BEFORE / 21.3% AFTER, ratio 0.7x, which
would have scored HIT / MISS HIGH / YES. **It is VOID.** `agent/copy_rate.py` reads
`runs/<run>/dumps_eligible.jsonl`, which `think_train.py:1004` writes only where
`elig = rlf * (gain > distill_eps)` -- **the notes that BEAT the baseline** -- and that selection rate is in
metrics.jsonl all along:

| steps | ladder | **sel%** | ov BEFORE | ov AFTER | >=4 BEF | >=4 AFT | ratio |
|-------|--------|------|-----------|----------|---------|---------|-------|
| 0:120 | -15 | **71.4%** | 32.3% | 17.1% | 11.7% | 1.6% | 7.5x |
| 1180:1267 | -5 | **72.5%** | 46.6% | 24.8% | 44.0% | 5.7% | 7.8x |
| 1550:1700 | -1 | **54.3%** | 35.2% | 28.6% | 13.7% | 11.7% | 1.2x |
| 1850:2000 | **+1** | **1.6%** | 29.5% | 39.3% | 14.9% | 21.3% | 0.7x |

The rate collapsed at step 1800 because at o=+1 the reward baseline costs ~11 nats, so almost nothing clears
zero. **The 1850-2000 row is the best 1.6% of notes; the rows it was to be compared with are 54-72% of
notes.** The first three are within 2x of each other, so G118(6)'s claim -- copying peaked mid-ladder and
collapsed by o=-1 -- survives. `dumps.jsonl` is no escape: it keeps `argmax(gain)` of one micro-batch, which
is *more* selected. `copy_rate.py` now prints `sel%` per window and prints
`*** NOT COMPARABLE (G119) ***` when two windows differ by more than 2x. **G119(c): THE SELECTION RATE OF A
SELECTED SAMPLE IS PART OF THE MEASUREMENT. PRINT IT NEXT TO n, OR THE COMPARISON IS NOT A COMPARISON --
AND A RATE THAT MOVES WITH THE INDEPENDENT VARIABLE IS THE WORST CASE, BECAUSE THE ARTEFACT WILL LOOK LIKE
THE EFFECT.** Four ticks quoted copy rates off this file before anyone looked at `n_elig/n_anch`.

**(4) SO I BUILT THE UNSELECTED INSTRUMENT, AND IT IS THE BEST NEWS OF THE DAY.** `agent/note_dump.py`
(new, `py_compile` clean) generates notes at **every valid anchor** of the held-out blocks off a checkpoint,
with `probe_span.py`'s generation path copied line for line -- same layout, same `set_writer_delay(0, O-1)`,
same sampling args, same `default_rng(61)`, same VAL shard, same greedy row -- so the notes are exactly the
population behind the probe's `own` column: **n = 134, sel% 100**. It is a SEPARATE FILE on purpose:
`probe_span.py` is the decisive instrument, its mtime (2026-10-01 16:17:07, before run 71 launched) is quoted
as provenance, and a live enforcer re-reads it for the @2400 gate in 45 minutes. Both checkpoints read **at
o=+1**, so the target does not move between them:

| ckpt (read at o=+1) | n | sel% | ov BEFORE | ov AFTER | **>=4 BEF** | **>=4 AFT** | ratio | run BEF | run AFT |
|---------------------|---|------|-----------|----------|---------|---------|-------|---------|---------|
| step 1200 | 134 | 100% | 43.0% | 34.8% | **22.4%** | **16.4%** | **1.4x** | 2.11 | 1.86 |
| step 2000 | 134 | 100% | 35.7% | **47.2%** | **17.9%** | **32.8%** | **0.5x** | 1.91 | **2.81** |

**The share of notes carrying a verbatim >= 4-word run of the text AFTER the anchor doubled, 16.4% ->
32.8%, while the BEFORE share fell -- unselected, matched anchors, fixed offset.** A step-2000 note,
verbatim: the context ends `...frictional forces at the bases of A and B`, the note is `, will obviously be
directed towards left (newton's 3rd law`, and the window begins ` will obviously be directed towards left,
as u say.` That is a correct forecast of the next clause, nearly word for word.

**(5) AND THAT IS THE PROGRAMME'S PROBLEM STATED PROPERLY FOR THE FIRST TIME.** @2000 own_minus_filler is
**-1.714**: **a third of the notes anticipate real future text and the frozen reader still prefers a blank
note.** The slot is not the problem -- `span`, the TRUE 15 tokens in the SAME slot at the SAME positions,
is worth **+14.78** nats. So the value of a note must be steeply non-linear in how much of the span it
actually contains, and a note that is one clause right and the rest wrong apparently costs more than
silence. **That is now a pre-registered, one-scoring-pass experiment (band 19): split the 134 anchors at
@2400 by whether the note carries a verbatim >= 4-word run of the future and difference the per-anchor `own`
gain, WITH minus WITHOUT -- registered [+0.5, +4.0] nats, P(positive) = 0.85. If it is ZERO, the reader does
not pay for correct forecast content at all and the frozen-reader route is dead.** `agent/note_value.py`,
next tick, after the decisive read.

**(6) RUN 73's PREMISE IS NOW DOUBTFUL BEFORE RUN 73 STARTS, SO I REGISTERED THE CHEAP EARLY TEST AND LEFT
THE RUN ALONE.** Run 73 (the hold: run 71 continued at o=+1 for 3000 more steps) is the owner's explicit
instruction from 09:25 today and it is the registered test of exactly the question (2) just answered badly.
One rung of 300 steps is not grounds to cancel a 13-hour run the owner asked for -- but it is ample grounds
to let it prove itself fast. `runs/think73_hold/config.sh` now carries, **written before run 73 has a single
row of metrics** and with the executable content `diff`-verified byte-identical (FLAGS md5 match,
backup `.bak_pre_obgate_20261002_1207`): **at step 3000 only, stop for futility if BOTH (i) run 73's ob0
slope over 2400-2999 is < +0.15 per 100 -- a QUARTER of the weakest earlier rung -- and its last bin is not
>= +0.5 above run 71's o=+1 mean of +1.98, AND (ii) the enforcer's READ @3000 is <= BASE.** An AND of two
independent instruments, so neither can kill a working run alone; it adds a stop and removes none.
Registered probabilities: (i) 0.65, (ii) 0.55, the AND **0.45**.

**(7) AND RUN 73 WOULD HAVE SHOWN A +11-NAT IMPROVEMENT AT STEP 2400 FOR NO SCIENTIFIC REASON WHATSOEVER.**
`think_train.py` was edited at **11:24 today** (mtime verified against run 71's 02:21 launch) so that
`if _cur_bridge > 0 or args.frozen_reader:` at l.898 makes a frozen-reader run baseline the reward on
SILENCE UNDER THE SAME MASK at every offset. Run 71 is running the **old** source, where at o=+1 (bridge 0)
the baseline fell back to the CLEAN rows -- ~11 nats higher -- which is why its `g_r0` reads -11 from step
1800. **Run 73 resumes run 71 at the SAME o=+1 setting from the NEW source, so its `g`, `g_r0`, `g_shuf0`,
`gSp`, `g_buf`, `g_win` and `eligRate` all step up by ~11 nats at step 2400 as an artefact of the edit.**
The gradient is unchanged, and that is proved rather than assumed: `PAIR = (REW.sum(0) - REW)/(RO-1)` is the
leave-one-out mean over the rollouts **of the same anchor** (l.954) and `ADV_R = REW - PAIR`, so a per-anchor
additive constant cancels exactly before `mu`, `sig` and `A` are formed (l.1056-1068); `elig` (l.1004) feeds
only a counter, the eligible dump and the printed `el` field, never `w3`, and `--distill-w 0.0` here.
**G119(d): WHEN A RUN RESUMES ANOTHER RUN, DIFF THE SOURCE BETWEEN THE TWO LAUNCH TIMES AND WRITE DOWN WHICH
LOGGED COLUMNS ARE NO LONGER COMPARABLE -- BEFORE THE SECOND RUN PRODUCES ITS FIRST BLOCK.** Across this
boundary the comparable axes are **ob0, imCE, ntp1, held-out mix and the probe reads**; `g_r0` and
`eligRate` are not. Registered as **band 21, an instrument check on my own source reading**: run 73's
`g_r0` over 2400-2499 must land in [-3.0, +4.0] with `ob0` continuous to within 1 nat -- **if `g_r0` stays
near -11 I have read the edit wrong, note (A) in config.sh must be retracted and band 20 re-derived.**
Also operational: the extra silence pass now runs at o=+1 too, so expect ~15.5 s/step instead of run 71's
14.95 s/step at that offset (measured per rung: o=-3 16.12s, o=-1 15.44s, **o=+1 14.95s** -- the rung that
skipped the pass) -> 3000 steps is ~12.9 h, ETA ~02:30.

**(8) ELIGRATE 0.022 AND g -11 ARE THE SAME ARTEFACT AND NEITHER IS A DERAIL.** `status.py` prints
`eligRate 0.695 -> 0.022` and `g +2.71 -> -10.31`, which looks like a run falling apart at step 1800. Both
are the 11-nat baseline, measured at the one place it bites, and `elig` enters no loss (7). `ob0 +1.96` and
`g_sd 3.4-3.8` say the signal is intact. **3b walked clause by clause: held-out mix 1700->2000
2.820 -> 2.821 = +0.0003 per 100 (bar +0.02), absolute max 2.821 vs 3.025 (margin 0.204); reg% 5.8 in the
last block (bar 15, and `> 8 and rising for 3 blocks` needs three -- this is one, watch it); spread 3.54
(bar 0.6); len 15.0 (bar 3.5); ntp1 block mean 1.98 (bar 4.0); 0 NaN/inf in 2126 rows, 0 Traceback, 0
`NRT_`.** 65 workers, 1501 GB free, 432 G disk. No derail clause fires. Futility is judged within the rung
per 00S.0 item 4 (G118(8)) and run 71 has no remaining stopping bar: both gates passed, @1800/@2400 are
report-only.

**(9) BANDS. 36 -> 40: 24 HIT, 7 MISS HIGH, 8 MISS LOW** (band 17 void, band 15 an instrument check).
Band 9 (@1800 own_minus_filler [-1.0,+2.0]) **MISS LOW at -1.430**; band 10 (own_minus_shuf [+3.8,+6.5])
**HIT at +3.822 -- by 0.022, which is luck, not calibration**; band 11 (@2000 own_minus_shuf [+3.5,+6.5])
**HIT at +3.781**; the 04:19 @2000 own_minus_filler band [-4.0,+2.0] **HIT at -1.714 (a 6-nat band earns
nothing)**; band 14 (own_minus_echo @1800 [+2.5,+4.5]) **MISS LOW at +2.496 -- by four thousandths, because
I called the band "nearly forced with `own` in [0.2,1.0]" and `own` came in at -0.384**; band 15, the
instrument check, **HIT: `echo` reproduced -2.8799 against -2.8799 at step 1200, identical to four decimals,
so it is writer-independent as claimed and bands 14/16 are live.** **Both misses are LOW and at the same
place: `own` at the forecast offset is lower than I keep guessing.** Bands 12, 13 and 16 resolve on the
@2400 reads in ~1.5 h; on the eight-point fit I would now write band 13 as [-2.2,+0.3] with P(WIN) 0.04
instead of [-1.0,+1.3] with 0.12, and **I am not rewriting it -- if @2400 lands below -1.0 that is a third
consecutive MISS LOW and it goes in the tally.** All bands: `agent/predictions/pred_20261002_1207.md`.

**(10) WHAT THE NEXT TICK IS FOR.** **FINAL @2400, 32 seqs (~13:4x; band 13) and ECHO @2400 (bands 15/16)**
-- the decisive reads, from the enforcer 2300030 and the echo curve 2336939, both alive. Then **band 12, the
saturation discriminator**. Then **run 73 self-launches** (successor 2328320; its resume guard 2337419 is
armed) and **band 21 checks my source reading inside its first 100 steps**. Then, in order: run
`agent/note_dump.py` on step_0002400 for **band 18**, write `agent/note_value.py` for **band 19 -- the
slot-vs-content test, which is the most informative thing left on the table** -- and at run 73's step 3000
read **band 20**, the futility clause. **OPEN ITEM, DO NOT DROP (carried from G118(7)): run 72 is also a
warm resume (step 500) and launches after run 73, ~24 h out; the tick that sees it start must run
`agent/resume_oom_guard.sh think72_frozen_resume 500` from a snapshot.**

## G120 (2026-10-02 14:07 headless tick; run 73 `think73_hold` crash-looped 24 min after launch and is LIVE again at step ~2430 under the registered fallback) -- **THE PROBE'S s.e. IS AN ANCHOR s.e. ON CLUSTERED DATA, AND WHEN THE SEQUENCE IS MADE THE UNIT, ONE SEQUENCE OUT OF TEN TURNS OUT TO HAVE CARRIED EVERY POINT OF THE EIGHT-POINT CURVE THIS PROGRAMME HAS BEEN STEERING BY: DROP IT AND THE FORECAST RUNG IS FALLING AT z -2.66 INSTEAD OF FLAT. THE DECISIVE 32-SEQUENCE READ LANDED AT own_minus_filler -4.08 (-4.07 +- 0.75 CLUSTERED), A THIRD CONSECUTIVE MISS LOW. SEPARATELY THE WARM-RESUME OOM I INSURED AGAINST AT 10:07 HAPPENED EXACTLY AS WRITTEN AND MY OWN GUARD STOOD DOWN INSIDE THE CRASH LOOP BECAUSE ITS CHECK GREPPED THE CONFIG FILE, WHICH CONTAINS THE COMMENT THAT REGISTERS THE FIX.**

**(1) LIVENESS, AND THE ONE THING THAT WAS WRONG WAS THE RUN ITSELF.** Four of the five expected processes
alive (echo curve **2336939**, resume guard **2337419**, new run-73 enforcer **2369521**, new successor
**2369541**); the run-71 enforcer **2300030** and successor **2328320** are gone **because both completed** --
the enforcer logged `FINAL @2400` at 13:52 and the successor logged `LAUNCHED think73_hold` at 13:38. But
`runs/think73_hold/train.log` carried **`NRT_RESOURCE in nrt_tensor_allocate` / `AllocBuffer: Not enough
Neuron memory on core 0 for size=25559040`** twice -- 14:01:52 **rank 25** and 14:09:18 **rank 13** -- and
`supervisor.log` read `crashed -- retry 2/3`. The run had been up 24 minutes and had made 35 steps.
**G120(a): `status.py` said `training (64 ranks)` and showed a fresh step line while the run was in its
second crash. A STATUS LINE AND A WORKER COUNT ARE NOT HEALTH; GREP `train.log` FOR `Traceback|NRT_` AND
`supervisor.log` FOR `crashed -- retry` ON EVERY TICK OF A RUN LESS THAN TWO HOURS OLD.**

**(2) THE 10:07 INSURANCE WAS RIGHT ABOUT THE FAILURE AND MY GUARD DISARMED ITSELF ON ITS OWN COMMENT.**
`agent/resume_oom_guard.sh` matched its signature at 14:10:10 -- correctly, all four conditions -- and then
logged **`ABORT: FLAGS already carries --resume-model-only`** and exited. It does not. Its check was
`grep -q 'resume-model-only' "$CFG"`, and `$CFG` contains, at line 29, **the contingency note the guard
itself asked a tick to register**: *"REGISTERED FALLBACK: if that happens, `agent/resume_oom_guard.sh`
appends `--resume-model-only 1`"*. The guard read its own documentation as proof that its own fix was
already applied, and stood down inside a live crash loop. Verified by hand before acting: the flag appears
in that file at line 29 only, inside a comment; the `FLAGS` array (lines 95-112) did not carry it.
**G120(b): A CONFIG FILE DOCUMENTS FLAGS IT DOES NOT CARRY. NEVER ASK A CONFIG QUESTION WITH `grep` ON THE
FILE -- SOURCE IT AND ASK THE ARRAY.** Fixed: `has_flag()` in `agent/resume_oom_guard.sh` sources `$CFG` and
greps `printf '%s\n' "${FLAGS[@]}"` with `-qxF`, and it replaces BOTH grep sites -- the abort clause and the
`patch did not take` verification, where the same false positive would have made the guard believe a failed
patch had succeeded. Tested in both directions on the real files: the new predicate says HASNOT on the
pre-patch config where the old grep said MATCHED, and HAS on the patched one. Backup
`.bak_pre_flagsgrep_20261002_1414`, `bash -n` clean.

**(3) THE FALLBACK WAS APPLIED BY HAND AND IT WORKED, AND IT IS THE FAILURE `train.py:280` DESCRIBES.**
Stopped the loop (3 `run_think.sh` supervisors + torchrun by numeric PID, G20), verified the stop the only
trustworthy way (G22: max step in `metrics.jsonl` **2435 -> 2435** over 75 s, 0 `think_train` procs, 0
`run_think`), patched `FLAGS` with `--resume-model-only 1` (backup `.bak_pre_resumemodelonly_20261002_1412`;
`bash -n` clean; sourced check `n_flags=121`, no comment words leaked into the array), relaunched.
`train.log` now reads **`resumed MODEL ONLY from .../step_0002400.pt at step 2400 (cold optimizer)`**, 65
workers, **15.3-15.5 s/step** -- which is the 15.5 the 12:07 tick predicted -- and **zero** new allocation
failures. Note the mechanism detail worth keeping: **neither crash was at a checkpoint save** (`--ckpt-every
100`, first post-resume save is 2500), so the 2026-08-28 story in `train.py:280` is slightly wider than its
own wording -- a warm optimizer restore breaks the allocation layout for the **per-step transients**, not
only for the save. A 25 MB allocation failing on a 1.5 TB machine is fragmentation, not shortage.
**G120(c): AT 12B / 64 RANKS, EVERY WARM RESUME GETS `--resume-model-only 1` FROM THE START.** Applied
pre-emptively to **`runs/think72_frozen_resume/config.sh`** (also a warm resume, off run 70's step 500),
before run 72 has launched, with the evidence written into the file (backup
`.bak_pre_resumemodelonly_20261002_1418`, `bash -n` clean, sourced `n_flags=119`). That also retires most of
the OPEN ITEM carried since G118(7): run 72 no longer needs the OOM guard for this failure.

**(4) AND NOW THE SCIENCE, WHICH IS WORSE. THE DECISIVE 32-SEQUENCE READ: own_minus_filler -4.08.**
Run 71 @2400, o=+1, frozen init reader, **n=424 anchors / 32 sequences**: own **-2.40**, shuf -5.20,
span **+15.65**, filler +1.67, clean +8.80, **own_minus_filler -4.077 +- 0.468 (anchor-level)**, verdict
FAIL. Band 13 was **[-1.0,+1.3]**: **MISS LOW by 3.1 nats, and the third consecutive MISS LOW at this
offset.** Band 12 (the saturation discriminator: is @2400 more than ~0.75 above @1800's -1.430? registered
**NO**) **HITS**: on the matched 10-sequence population @2400 reads **-1.966** -- *below* @1800 -- and on 32
it reads -4.08. **The forecast rung saturated at 1800 exactly as the 10:07 caveat predicted, and then went
backwards.**

**(5) THE INSTRUMENT FINDING THAT RE-READS EVERY NUMBER IN THIS LEDGER: THE s.e. IS CLUSTERED.**
`probe_span.py` pools every valid anchor of every sequence and divides by `sqrt(n_anchors)`. **Anchors inside
one 1024-token block share the text, the topic and the writer's state -- the unit of randomness is the
SEQUENCE, not the anchor.** New **`agent/clustered_se.py`** (`py_compile` clean) recovers the per-sequence
means from a probe's own running-mean log (cumulative n and cumulative mean difference out exactly, so no
re-run and no new forward pass) and reports sequence-clustered s.e. On the @2400 read: **-4.07 +- 0.75
clustered against the printed +-0.468 -- the printed s.e. is understated 1.61x**, and the between-sequence sd
of `own` is **3.12 nats**. Deliberately a separate file, not an edit to `probe_span.py`: that probe is the
decisive instrument, its mtime is quoted as provenance and a live enforcer re-reads it between gates (the
same rule the 12:07 tick followed for `note_dump.py`).

**(6) AND CLUSTERING EXPOSED THAT ONE SEQUENCE OUT OF TEN HAS BEEN CARRYING THE WHOLE CURVE.**
Every point of the eight-point o=+1 curve -- the curve that justified the forecast rung and informed run 73
-- was a **10-sequence** read. Per-sequence `own - filler` at @2400 across all 32: thirty are between -9.7
and +0.7, and **sequence #9 reads +11.74**. Sequence #9 is inside the ten. Its contribution grows along the
curve: **+8.49 @1000, +14.26 @1200, +14.12 @1400, +15.60 @1600, +14.34 @1800**. The consequence, from
`agent/clustered_se.py --curve`:
| o=+1 own_minus_filler curve | slope per 100 | z | mean level |
|---|---|---|---|
| all 10 sequences (8 points, clustered s.e. 0.96-2.32) | **+0.0128 +- 0.0365** | +0.35 | -1.20 |
| **sequence #9 dropped** (8 points, n_seq=9) | **-0.0571 +- 0.0214** | **-2.66** | **-2.62** |
**On nine of ten sequences the forecast rung is getting WORSE at z -2.66. The flat-to-rising curve was one
sequence out of ten.** And the 10-sequence prefix is optimistic as a population too: it reads **-1.97**
where the full 32 read **-5.04** on the other 22, a **+3.07 nat** subsample bias.
**G120(d): A TEN-SEQUENCE PROBE READ IS NOT EVIDENCE. QUOTE `n_seq`, QUOTE THE SEQUENCE-CLUSTERED s.e., AND
BEFORE BELIEVING ANY CURVE, DROP THE MOST EXTREME SEQUENCE AND REFIT.** This is the same error as G119(c)
one level down: there the population was selected by the reward, here the uncertainty was computed as if
correlated draws were independent. Both made an artefact look like the effect.

**(7) THE ONE CURVE STILL GOING THE RIGHT WAY WENT RIGHT AGAIN -- AND THAT IS NOW THE WHOLE PUZZLE.**
Band 18 **HIT on both clauses**: the UNSELECTED share of notes carrying a verbatim >=4-word run of the
**future**, every valid anchor off the @2400 checkpoint (`agent/note_dump.py`, n=134, sel% 100), reads
**35.8%** against the band **[25%,48%]**, with BEFORE:AFTER ratio **0.5x** (band: < 1.0). The series is
**16.4% @1200 -> 32.8% @2000 -> 35.8% @2400**, with the BEFORE share falling (43.0% -> 35.7% -> 34.6%).
**So at @2400 more than a third of the notes visibly anticipate the real continuation, the true 15 tokens in
the same slot are worth +15.65, and the reader still scores the notes 4 nats BELOW ten spaces.** Those three
facts cannot all be comfortable. Band 19 (`agent/note_value.py`, written and running this tick) is the
experiment that separates them: split the anchors by whether the note carries the future and difference the
per-anchor gain, with `filler` and `span` printed per group as the anchor-difficulty controls.

**(8) DEVICE DECISION: RUN 73 RUNS TO ITS REGISTERED STEP-3000 GATE, AND I DID NOT INVENT A STOP.**
Three instruments now contradict run 73's premise -- the clustered leave-one-sequence-out curve (-0.0571,
z -2.66), the in-training ob0 at o=+1 (flat, +0.041 +- 0.171, with the `g_shuf0` control rising at z 5.10,
G119), and the decisive read (-4.07 +- 0.75 clustered, 5.4 clustered s.e. below the bar). But the owner asked
for this compute explicitly at 09:25, the step-3000 futility clause was registered in `config.sh` **before
run 73 had a row of metrics**, it is an AND of two independent instruments, and it is only ~570 steps (~2.5 h)
away. **The new evidence makes that gate MORE likely to fire, so waiting costs 2.5 h of device time and buys
a pre-registered kill instead of a post-hoc one.** Health against the 3b bars, clause by clause: held-out mix
**2.8222 @2400 vs 2.7749 at start** (bar start+0.25 = 3.025, margin 0.203) rising **+0.0004 per 100** over
2100-2400 (bar +0.02); reg% **2.6** (bar 15); spread `g_sd` **3.69** (bar 0.6); len 14.6 (bar: within 0.5 of
min-think 3); ntp1 block mean **2.11** (bar 4.0); **0 NaN/inf in 50 rows**; 65 workers, 1301 GB RAM, 421 G
disk. `orG` -39 is not an axis (G52). No derail clause fires.

**(9) BAND 21 CHECKED MY OWN SOURCE READING AND IT WAS RIGHT.** Registered at 12:07: if the 11:24 edit to
`think_train.py:898` really makes a frozen-reader run baseline the reward on silence at **every** offset,
then run 73's `g_r0` must step up ~11 nats from run 71's and `ob0` must be continuous. Run 71's last three
rows: `g_r0` **-10.49, -10.99, -7.85**. Run 73's first 50 rows at the same offset with the same flags:
`g_r0` mean **-2.696**, inside the registered **[-3.0,+4.0]**, and **ob0 +1.713 against run 71's +1.97 at
o=+1 -- continuous to 0.26 nats** (bar: no jump > 1 nat). **HIT.** So `config.sh` note (A) stands: on run 73
the comparable axes are `ob0`, `imCE`, `ntp1`, held-out mix and the probe reads, and `g_r0`, `g`, `gSp`,
`g_buf`, `g_win` and `eligRate` are not comparable across the run 71/73 boundary. Deflation: the registered
window was 2400-2499 and these are 50 rows of it (2400-2415, the warm attempt). The discriminator is 8 nats
wide, so n=50 settles it; the full window completes next tick under the cold optimizer.

**(10) WHAT THE NEXT TICK IS FOR.** **Band 19 (`runs/probes/notevalue_think71_frozen_imit_0002400_o+1.json`,
running since 14:19, ~10 min) is the most informative thing on the table: if (own - filler) WITH minus
WITHOUT is ZERO, the frozen reader does not pay for correct forecast content at all and the frozen-reader
route is dead however good the writer gets -- and the writer is demonstrably getting better (7).** Then
**band 16, own_minus_echo @2400** (echo curve 2336939, at seq 27/32 at 14:22, lands ~14:30) -- **and read it
with `agent/clustered_se.py`, not the printed s.e.** Then **band 20 at run 73 step 3000 (~16:4x)**: read the
ob0 slope with `agent/rung_trend.py` and the enforcer's @3000 READ, and **also report that read clustered and
leave-sequence-9-out, which is band 25 -- registered, not a change to the gate.** Bands 22-24 (the fallback
holds past step 2550; the discriminator if it does not; the cold-optimizer ntp1 price) resolve next tick too.
**OPEN ITEM, NOW SMALLER: run 72 carries `--resume-model-only 1` from the start (3), so the tick that sees it
launch no longer needs the OOM guard -- but it must still confirm the flag took and that run 72's first
checkpoint save at step 600 survives.**

### G120, CONTINUED -- THE THREE READS THAT LANDED AFTER THE ENTRY ABOVE WAS WRITTEN

**(11) BAND 19 ANSWERED, AND IT IS THE BEST RESULT THIS PROGRAMME HAS PRODUCED: THE READER DOES PAY FOR
CORRECT FORECAST CONTENT, AND IT PAYS ABOUT HALF THE AVAILABLE HEADROOM.** `agent/note_value.py` on run 71
@2400, o=+1, frozen init reader, **UNSELECTED (every valid anchor), n=134, 10 sequences**. Provenance check
first: its aggregate reads **own -0.92 / filler +1.05 / span +14.78 / shuf -5.63, own-filler -1.97** --
`probe_span.py`'s matched 10-seq read is **-1.966**. The populations are the same to three decimals, so this
is the probe's own `own` column split anchor by anchor. Splitting the 134 anchors by whether the note carries
a verbatim >=4-word run of the FUTURE (the criterion IMPORTED from `agent/copy_rate.py`, not reimplemented):
| | WITH the future (n=48, 35.8%) | WITHOUT (n=86) | contrast |
|---|---|---|---|
| `own` | **+2.91 +- 1.15** | **-3.06 +- 1.23** | **+5.96 +- 1.69** |
| **`own - filler` (the number to quote)** | **+4.05 +- 1.31** | **-5.32 +- 1.26** | **+9.37 +- 1.81** |
| CONTROL `filler` | -1.14 +- 0.87 | +2.27 +- 0.44 | **-3.41 +- 0.98** |
| CONTROL `span` (the ceiling) | +7.02 +- 1.11 | +19.11 +- 1.10 | **-12.09 +- 1.56** |
| CONTROL `shuf` | -4.47 +- 1.47 | -6.28 +- 1.24 | +1.82 +- 1.92 (n.s.) |
Band 19 was **[+0.5,+4.0]**: **MISS HIGH at +9.37**, and the direction (P(positive) 0.85) was right.
**THE CONTROLS RUN AGAINST THE FINDING, WHICH IS THE STRONGEST FORM IT COULD TAKE.** I pre-registered the
worry that notes containing the future would sit at anchors where ANY content scores higher. The opposite is
true: at WITH anchors `filler` is worth **3.4 nats LESS** and the TRUE span is worth **12.1 nats less**
(+7.02 against +19.11) -- these are the anchors where the reader already half-knows the continuation, so
there is less headroom -- **and the model's own note scores HIGHER there anyway.** Normalised by the headroom
actually available (`span - filler`): **WITH captures +4.05 of +8.16 = 50%; WITHOUT captures -5.32 of
+16.84 = -32%.** A note that contains the future buys half of what the true future would buy. A note that
does not is worth 5.3 nats LESS THAN TEN SPACES.

**(12) AND IT SURVIVES THE CLUSTERING RULE I WROTE ONE HOUR EARLIER -- 8 OF 9 SEQUENCES POSITIVE.** The
WITH/WITHOUT split is observational, so G120(d) binds here too: if the notes carrying the future piled into
one document the whole contrast would be that document. It does not. Per-sequence `own-filler` contrast:
**+5.79, +4.78, +4.65, -10.02, +9.64, +20.01, +22.17, +4.34, (seq 8 has no WITH anchors), +4.75**.
**Clustered (sequence = unit): +7.35 +- 3.16, n_seq=9, zero excluded at 2.3 s.e.; sign agreement 8/9;
LEAVE-ONE-SEQUENCE-OUT range [+5.49, +9.52].** The two extremes rest on tiny within-sequence n (seq 6's
+22.17 comes from ONE WITH anchor, seq 3's -10.02 from two), and dropping either leaves the result intact.
**AND IT EXPLAINS THE OUTLIER OF G120(6): sequence #9, the +11.74 document that carried the entire
eight-point curve, is the document where 13 of 14 notes CONTAIN THE FUTURE** (every other sequence: 0 to 7
of ~13). So band 26 **HITS** -- #9 is again the extreme on `own` -- and the reason is not arithmetic: it is
the one document the writer actually forecasts. **Two independent instruments, one mechanism. The leave-one-out
correction in G120(6) was still the right thing to do, and now we know what it was correcting for.**

**(13) WHAT THAT MAKES THE PROGRAMME'S NEXT MOVE, WITH THE PRIZE PRICED.** The aggregate -1.97 is not a
failure of the slot, the positions or the reader: it is **64% of notes that are wrong costing -5.32 each**
dragging down 36% that are right earning +4.05. **If the writer emitted filler instead of a wrong note, the
aggregate own-filler would be 0.358 x (+4.05) + 0.642 x 0 = +1.45 nats -- ABOVE THE FILLER FLOOR, WHICH IS
THE SUCCESS BAR (sec. 1).** That is the first quantified route to clearing the bar this programme has had.
The honest deflation, stated with it: the split is OBSERVATIONAL, not randomised -- this shows the reader
pays for content that happens to be correct, not that making the writer more accurate transfers one-for-one
-- and the abstention route needs the writer to know WHEN IT DOES NOT KNOW, which is the hard part and which
runs 32/33's abstaining lineage already found ways to do badly (G39: a writer that loses its stop action
loses nats to length alone, and whitespace price is steeply superlinear). But the prize is now measured:
**~3.4 nats available from abstention alone, at the current 35.8% hit rate**, and the hit rate is the one
curve in the programme that is rising (16.4% -> 32.8% -> 35.8%).

**(14) BAND 16 FAILED AND IT IS THE SAME SUBSAMPLE STORY AGAIN.** Echo @2400, 32 seqs: **own_minus_echo
+0.589 +- 0.486 anchor-level, +0.59 +- 0.54 clustered**. Band 16 was **[+2.0,+5.0]** with P(> 0 at > 3 s.e.)
= 0.85: **MISS LOW, and the 3-s.e. clause fails too.** Matched 10-seq it reads +1.96 against @1800's +2.50 (a
modest fall), but on the 22 sequences never measured before it reads **-0.04**: *on unseen documents the
model's own note is worth exactly what an echo of the preceding text is worth.* Dropping the extreme sequence
(#2, +7.60) leaves +0.35. **Consistent with (11)-(13) rather than against them**: the 64% of notes that do not
contain the future are worth about what an echo is worth, and the 36% that do carry the whole value.
**Band 24 HIT**: ntp1 over the first 50 cold-optimizer rows **+1.989** against the band **< 2.6**, and
slightly BELOW the warm resume's +2.073 -- the documented ~50-step Adam-momentum price cost nothing
measurable on ntp1, and `ob0` was continuous across the restart (+2.085 cold vs +1.730 warm).
**Tally 44 -> 47: 29 HIT, 8 MISS HIGH, 10 MISS LOW.**

**(15) AND THE INTERVENTION IN (3) SILENTLY DISARMED RUN 73's GATES FOR 45 MINUTES.** The final sweep found
only ONE of the two run-73 helpers alive. `runs/probes/armed_think73_hold.log` says why: **`14:12:08 run
ended before step 3000`**. The enforcer's `grab()` polls `dead()`, which is
`grep -qE '^(DONE|STOPPED|FAILED)' "$RUN/STATUS"` -- and **the terminal STATUS it read was the one THIS TICK
wrote at 14:12 to stop the crash loop.** The run came back at 14:14; its enforcer did not. For 45 minutes
run 73 was training with **no instrument for its @3000 futility gate and no enforcement of its own
registered two-consecutive-reads stop** -- and nothing would have reported that except a liveness count,
because the run itself looked perfect. The successor survived the same STATUS write only because it has an
explicit revival check (`14:14:13 think73_hold revived (STATUS no longer terminal); still waiting`).
**G120(e): WRITING A TERMINAL STATUS TO STOP A RUN KILLS EVERY ARMED WATCHER OF THAT RUN. AFTER ANY
STOP-AND-RELAUNCH, RE-ARM THE ENFORCERS AND VERIFY THEM BY PID -- AND COUNT THE HELPERS BEFORE AND AFTER THE
INTERVENTION, NOT ONLY AT THE START OF THE TICK.** Re-armed at 14:57:01 from a **byte-identical** snapshot
(`agent/running/armed_probes_hold_think73_hold_20261002_1457_rearm.sh`, md5 `08b61cc5d07bf7273cbc446b26dd6c6d`,
`diff` empty against the 13:38 snapshot, G114(b)); pid 2591274 verified alive and logging. It re-reads BASE,
which is deterministic (seed 61, same VAL shard), so **-1.966 gets a free reproduction check** too.
This is the third instance in one tick of the same error class -- *asking a stateful file for the current
state*: the guard grepped a config that documents the flag (G120(b)), my own wait-loop matched two historical
`crashed -- retry` lines, and here a watcher read a STATUS field that an intervention had just written.

---

## G121 (2026-10-02 16:07 tick) -- THE PROGRAMME'S QUESTION CHANGES: NOT "CAN THE WRITER FORECAST?" BUT "WILL IT SHUT UP WHEN IT CANNOT?". RUN 74 DESIGNED, REGISTERED AND ARMED. AND THE 14:07 ARITHMETIC THAT MOTIVATED IT WAS RIGHT BY LUCK -- THE MECHANISM IT IMPLIES WOULD HAVE BEEN BREAK-EVEN BY CONSTRUCTION.

**(1) THE +1.45 WAS RIGHT AND ITS DERIVATION WAS WRONG, AND THE DIFFERENCE PICKS THE MECHANISM.**
The 14:07 tick wrote: *"Emit filler instead of a wrong note and the aggregate is 0.358x(+4.05) = +1.45."*
That applies the `own-filler` CONTRAST to the retained subset. But `filler` is a CONTROL COLUMN computed on
all 134 anchors and **it does not move when the writer's behaviour changes**, so the aggregate must be
rebuilt from `own` and differenced against the unchanged control. Band 19's own numbers, run 71 @2400, o=+1:

| group | n | own | filler | own-filler |
|---|---|---|---|---|
| WITH a verbatim >=4-word run of the future | 48 (35.8%) | **+2.91** | -1.14 | **+4.05** |
| WITHOUT | 86 | **-3.05** | +2.27 | **-5.32** |
| aggregate | 134 | **-0.92** | **+1.05** | **-1.97** |

Perfect abstention on the 86 WITHOUT anchors, **abstaining to the FILLER**:
`own' = (48*2.91 + 86*2.27)/134 = +2.50`, control unchanged `+1.05` -> **own_minus_filler = +1.45.** Same
number, by the right route. Perfect abstention **to SILENCE** (an empty note): that anchor's block row is
written all-`T_CLOSE` by `_mix` (think_train.py:885-890), which is **byte-identical to the `sil` row the
frozen reader baselines on** (l.897-903), so its gain is **exactly 0.0** -- and filler is worth **+1.05
ABOVE silence** for this reader. So `own' = (48*2.91 + 0)/134 = +1.04` and **own_minus_filler = -0.01:
BREAK-EVEN BY CONSTRUCTION, no matter how perfectly the writer judges itself.**
**G121(a): WHEN AN ARITHMETIC SAYS A BEHAVIOUR CHANGE WOULD PAY, RECOMPUTE THE TREATMENT COLUMN AND LEAVE
THE CONTROL ALONE. A CONTRAST IS NOT A PER-ANCHOR VALUE.** Family: G33/G34/G35/G39/G119(c)/G120(d) -- a
quantity measured under one population quoted as if it held under another. Here the lucky part is that the
headline number survived; the unlucky part is that the obvious mechanism it suggested is the dead one.

**(2) AND THE SIGN OF THE GATE NOW CARRIES THE CLAIM -- THE FIRST TIME IN THIS PROGRAMME.**
Under RANDOM (non-selective) abstention at rate q the mixture keeps its shape:
`own_minus_filler(q) = (1-q)*(-1.97) + q*0 = -1.97*(1-q) <= 0` for every q, with equality only at q=1
(total muteness, runs 32/33's registered collapse). **So own_minus_filler > 0 cannot be reached by
abstaining MORE; only by abstaining SELECTIVELY.** No length artefact, no shortening, no collapse to filler
and no muteness can produce a positive read. Every previous gate in this programme needed a matched control
to be interpretable (G39's whole lesson); this one needs only its sign.

**(3) A SIXTY-SECOND MEASUREMENT KILLED THE ONE-FLAG VERSION OF THE EXPERIMENT BEFORE IT COST ANY DEVICE.**
The cheapest conceivable priced abstention already exists in the trainer and is banned by one flag: `</T>`
at thought position 0 is legal under `--min-think 0` (think_par.py:230,296), scores exactly 0.0 (see (1)),
and takes **one token** of policy gradient because `stop_pos` returns 0 for an immediate close and
`row_rl = (ar <= s0)`. A one-flag run, no code, no new graph. **The question nobody had asked was whether
the action is REACHABLE.** New `agent/abstain_feas.py` answers it off a checkpoint with no training and no
gate, using `gen.generate`'s existing `collect_logits` hook to read the raw position-0 logits BEFORE any ban
(run 73 step_0002800, o=+1, n_anch=134, n_seq=10):

| token at position 0 | mean p | median p | median vocab rank | inside `--top-k-thought 512` |
|---|---|---|---|---|
| `</T>` | 1.80e-04 | **1.09e-09** | **62,270** | **3.0%** (2.9% +- 1.6% clustered) |
| SPACE (220) | **3.90e-02** | 2.60e-04 | -- | -- |

**0.018 expected empty notes per 100 sampled rollouts -- about one per 5,500.** The flag would have made the
action legal and unreachable and the run would have measured nothing for ten hours.
**G121(b): BEFORE GIVING A POLICY A NEW ACTION, MEASURE WHETHER ITS OWN DISTRIBUTION CAN PROPOSE IT.
A LEGAL ACTION AT RANK 62,270 IS NOT AN ACTION.** This is the cheap-instrument-first discipline paying for
itself a second time (G120's `clustered_se.py` reconstructed every past read with no forward pass; this cost
one minute of CPU and saved a run).
The same measurement sized the second flag: p(SPACE|t=0) is **BIMODAL ACROSS DOCUMENTS** (per-sequence means
**3.2e-5 .. 1.57e-1**), so on half the anchors the on-policy rate is ~1e-4 and **a discrimination cannot be
learned from a sample that covers only one side of it** -- hence `--abstain-explore 0.05`, forcing SPACE at
position 0 on 5% of sampled rows, the same device as run 57's `--stop-explore` scoped to t=0. Declared bias:
the forced token is credited with `log p(SPACE)` under the advantage, i.e. epsilon-greedy REINFORCE.

**(4) RUN 74 `think74_abstain`, AND WHY IT IS THE OPPOSITE OF RUN 32 RATHER THAN A REPEAT OF IT.**
Run 32 delivered abstention through the **CE TARGET** (`--abstain-mode freq` wrote SPACE into the imitation
target at ~45% of positions): "emit SPACE" became a *supervised* action, the writer over-generalised, and
`own_space_rate` went 0.83/0.88/0.74 against its own init's 0.026 -- mute (G39/G40). Run 74 moves **no CE
target at all**. Two flags, both default-off with a literal no-op off path (every new rng draw is inside
`if args.abstain_explore > 0`, so runs 70-73 and any keepalive relaunch are bit-identical under the patched
source; backup `think_train.py.bak_pre_abstainfill_20261002_1607`, **19 executable lines added, 0 removed,
0 modified**, `diff`-verified, and the numpy logic unit-tested standalone before launch):
  * `--abstain-fill 1` -- SPACE at position 0 replaces the WHOLE note with the canonical filler, built byte
    for byte as l.966-968 builds it for `--pause-frac`, which is **also `probe_span.py`'s `filler` column**.
    An abstaining rollout therefore earns THE CONTROL'S OWN REWARD, and is **exactly length-matched to the
    control by construction -- G39's confound cannot arise here.**
  * the one line that separates this from runs 32/33: an anchor whose selected rollout abstained takes
    policy gradient at **position 0 only** (`row_rl = (ar == 0)`), so the writer trains **the decision** and
    is never trained to emit ten spaces token by token. Abstention here *reinforces* the stop action's
    neighbourhood instead of destroying it.
  * `abst` (already a logged, unused column on this lineage) carries the realized rate and is the collapse
    detector.

**(5) THE 3b WALK FOUND A TRAP, AND THIS IS THE SIXTH TIME THAT RULE HAS EARNED ITS KEEP (G54/G56(c)/G76/G77/G84).**
Written into `runs/think74_abstain/config.sh` before launch. The dead axes: **reg%** (a writer that learns
this action concentrates its position-0 prefix on SPACE *by design*; 3b's run-32 qualifier already governs),
and -- the trap -- **"thought length within 0.5 of `--min-think` for 2 blocks"**. An abstaining anchor's
content row **IS** the 10-space filler, so `stop_pos` returns 10 and **`len_mean` RISES toward 10 as the
writer goes mute** -- the opposite direction from runs 32/33, where muteness appeared as length collapse.
**The clause that caught muteness on the last abstaining lineage points the wrong way on this one.** The
collapse detector is `abst`, bars registered: mechanism engaged iff `abst` in [0.02,0.90] at resume+400;
mute collapse if > 0.90. `g_sd` stays live but is predicted to RISE (an abstaining rollout scores ~filler, a
wrong content rollout ~5 nats below -- the spread IS the signal).

**(6) DEVICE DECISION: RUN 72 IS NO LONGER NEXT; RUN 74 IS. RUN 73 STILL RUNS TO ITS REGISTERED GATE.**
Killed the run-72 successor (pid 2369541, `ps`-verified gone) and armed `agent/successor_r73_to_r74.sh`
(pid 2605664, logging, immutable snapshot md5 `b120ae45671256b7132aca1dd3b97b1b`) plus the new
`agent/armed_probes_abstain.sh`. Run 72 asks whether copy-teaching was doing the work; band 19 has now
priced where the value in that line actually is, and the one thing going the right way is the content share
under imitation (16.4% -> 32.8% -> 35.8%). Its config is untouched and it remains launchable.
**Run 73 was NOT stopped early a second time.** Its registered @3000 clause is ~25 min away and the
evidence makes it MORE likely to fire: read on its own registered terms (100-step bins, `RUNG_BIN_W=100`),
ob0 over 2400-2899 has OLS slope **+0.006 +- 0.086 per 100 (z +0.07, 5 bins), mean +1.85**, against a bar of
**< +0.15** -- flat. Waiting buys a pre-registered kill instead of a post-hoc one.

**(7) TWO OPERATIONAL FIXES, BOTH FOUND BY TESTING A LINE BEFORE ARMING IT RATHER THAN AFTER.**
(a) `ls ... | sort -t_ -k2 | tail -1` -- the idiom for "the latest checkpoint" -- **returned the step-2500
archive copy while step_0002800.pt sat in the live directory**, because splitting a path on `_` compares
`"hold/step"`, identical for every candidate. Both the successor and the enforcer now sort numerically on
the extracted step. **G121(c): A SORT OVER PATHS IS NOT A SORT OVER STEPS, AND IT FAILS SILENTLY WITH A
PLAUSIBLE FILE.** (b) Run 74 resumes run 73, and `load_checkpoint` takes `start_step` from `ck["step"]`, not
from the filename -- so its end step is **not knowable while its config is being written.** Rather than
leave MAX_STEPS unregistered, the config states the FORMULA (seed step + 2400), every gate is an OFFSET from
the resume step, the successor patches the number and **verifies it from the SOURCED array, not the file**
(G120(b)), and the placeholder is *below* any possible resume step so a failed patch makes the run exit
immediately instead of training something unregistered -- fail loud, not quiet. `--lr-horizon 1` pins the LR
flat at 5e-6 with no number to patch (`lr_at` returns `min_lr` for any step >= horizon).

## G122 (2026-10-02 18:07 tick) -- WE STOPPED THE ONE RUN THAT WAS WORKING, ON A PROXY, USING AN UNPAIRED S.E.

**(a) PAIR READS THAT SHARE ANCHORS. THE UNPAIRED S.E. HID A z=3.6 EFFECT INSIDE "NOISE".**
run 71 @2400 and run 73 @3000 were probed on the same 32 held-out sequences, the same 424 anchors and the
same frozen reader. The `filler` column reads **+1.67 +- 0.40 in both** -- and filler is checkpoint-
independent by construction, so that identity is the *proof* the anchor sets match, not an assumption. The
levels are -4.07 +- 0.75 and -3.08 +- 0.77; compared the way the record compared them -- two levels, each
with its own s.e. -- the +0.99 difference carries a naive s.e. of 1.07, **z 0.9, indistinguishable from
noise**, and that is why three ticks called the forecast rung "flat to falling". But the between-sequence sd
is 3.1-3.7 and it is *the same sequences*, so it cancels:

| comparison | own_minus_filler | s.e. | z |
|---|---|---|---|
| run 71 @2400 (level, clustered) | -4.07 | 0.75 | -- |
| run 73 @3000 (level, clustered) | -3.08 | 0.77 | -- |
| difference, treated as two independent levels | +0.99 | 1.07 | +0.9 |
| **difference, PAIRED per sequence (n_seq=32)** | **+0.990** | **0.274** | **+3.61** |
| paired, dropping the one +12 outlier sequence #9 | +1.005 | 0.283 | +3.56 |
| paired, 10-sequence prefix | +0.965 | 0.506 | +1.91 |

25 of 32 sequences improved. **Pairing cuts the s.e. 3.9x.** This is the exact mirror of G120: there,
ignoring clustering *invented* significance; here, ignoring pairing *destroyed* it. Both errors came from
using a variance that did not match the unit of randomness in the question being asked -- the question
"did this trajectory move?" has the *sequence-level difference* as its unit, not the sequence-level level.
**G122(a): BEFORE COMPARING TWO READS, ASK WHETHER THEY SHARE THEIR ANCHORS. IF THEY DO, THE COMPARISON IS
PAIRED AND THE UNPAIRED S.E. IS THE WRONG INSTRUMENT BY A FACTOR OF FOUR.**
A useful by-product: the 10-sequence prefix's notorious +3.07 nat optimism is a **level** bias only -- on the
*delta* the prefix agrees (+0.965 vs +0.990). Cheap reads on the prefix are fine for measuring change.

**(b) THE PROXY THAT LICENSED THE STOP WAS BLIND TO A ONE-NAT MOVE IN THE DECISIVE INSTRUMENT.**
Band 20's stop rule for run 73 was an AND over (i) the `ob0` trend and (ii) the probe read. Clause (i) fired
exactly as predicted -- `ob0` over 2400-2999 in 100-step bins is **-0.003 +- 0.057 per 100, z -0.06**, flat
to three decimals (band 35, p 0.80, HIT). Clause (ii) **missed**: the probe went *up* a nat. `ob0` is
sampled rollouts on *training* text scored by the *live* reader; the probe is the greedy note on *held-out*
text scored by the *frozen init* reader. Over 600 steps the second improved by +0.99 +- 0.27 while the first
did not move at all. The programme's own runbook calls `probe_span.py` the decisive instrument; the stop
rule nevertheless gave half a vote to a proxy that disagrees with it in sign. **G122(b): A PROXY DOES NOT
GET A VOTE ON STOPPING A RUN. IF IT DISAGREES WITH THE DECISIVE INSTRUMENT, THE PROXY IS THE THING THAT
NEEDS EXPLAINING.** `ob0` is explicitly retired as a gate in `runs/think79_forecast_long/config.sh`.

**(c) WHAT IT COST, STATED PLAINLY.** The forecast note was improving at ~1 nat / 600 steps when run 73 was
stopped at 3000 with 2400 steps of its own registered schedule left. It is still 3.08 nats *worse* than a
blank note, so nothing is "bought" yet -- but at the measured rate the sign flips after ~1900 more steps,
and that is the only trajectory in this programme that has ever had a measured positive slope on the
decisive instrument. The device then spent 16:07-17:00 designing run 74 (abstention) and 17:00-18:07 on the
hindsight chain, both new mechanisms, while the one line that was working sat stopped. **The pattern worth
naming: this programme's ticks are much better at inventing the next mechanism than at finishing the last
one.** `runs/think79_forecast_long/config.sh` -- run 73 continued 3000 -> 5400, one change (`--sync-grads
1`), gates paired against run 73 @3000 -- is written and deliberately NOT armed: run 76 is Emanuele's own
idea and holds the device. It is the next experiment when run 76 ends.

**(d) THE PRE-SEED PATTERN HAS A LATENT SELF-DESTRUCT, NEVER HIT ONLY BECAUSE RUN 74 WAS RETIRED.**
`agent/successor_r73_to_r74.sh` seeds the successor's checkpoint dir with the **model only**, correctly
noting that `--resume-model-only 1` never loads the optimizer (G120(c)). But `keepalive_think.sh` and
`watchdog.sh` both quarantine any `step_*.pt` older than 10 minutes with **no `step_*.optim*.pt` beside
it** -- a rule written for checkpoints killed mid-write (G12). A model-only seed matches that rule exactly,
ten minutes in. The live run survives (its weights are in memory); the damage is on the **first crash before
the first new checkpoint**: the seed is in `.orphans/`, `--resume latest` finds nothing, `run_think.sh` sees
an empty ckpt dir, runs a smoke, and then trains from `--init-from` **at step 0 with every schedule reset**
-- an unregistered run that looks perfectly healthy in every log. `agent/launch_r79.sh` seeds **both** files
(hard links: instant, no extra disk, and a quarantine or a prune only moves run 79's own link) and says why
in a comment, so nobody "cleans up" the optimizer file it never loads. **G122(d): A FILE YOU DO NOT READ CAN
STILL BE LOAD-BEARING. CHECK WHAT THE JANITORS DELETE.**

## G123 (2026-10-02 20:07 tick) -- THE HOST HAS A WALL TOO, AND TWO RUNS WITH DIFFERENT NAMES CAN BE THE SAME MODEL

**(a) RUN 76 DIED OF HOST RAM, AND THE FIX FOR THE DEVICE OOM IS WHAT KILLED IT.** Emanuele's
hindsight->foresight run failed three smokes in 2h12m and only the first two were understood at the time:
#1 (17:55) the batched reader-scoring graph `[8,2158]` failed to LOAD (NRT_RESOURCE); #2 (18:30) completed
step 0 on device and OOM'd at step 1 in generation on **device HBM**; #3 (19:46) **SIGKILL on rank 20**.
Signal 9 is not a device failure. `dmesg`'s last OOM dump: **64 python ranks holding 1968 GB of the host's
2048 GB, mean 30.7 GB/rank, the rank the kernel picked at 35.9 GB**, and **no compiler process alive at that
instant** -- so 30.7 GB/rank is `think_hindsight.py`'s own steady-state host footprint, not a compile
artefact. Measured 45 minutes later on the same box with run 79 training at 64 ranks: **7.1 GB/rank, 456 GB
total, 1524 GB free.** The hindsight path costs **4.3x** the host RAM per rank, and the ceiling at 64 ranks
is 2048/64 = **32 GB/rank**: run 76 was **4% over a wall this programme had never measured, and nothing
watches it.** Three mechanisms watch device HBM, held-out CE, worker counts, step times, checkpoint
orphans and disk; none watches `MemAvailable`.
**The trap is in the order of the fixes.** Smoke #2's bigger chunks fit the host and died on the device.
Smoke #3's smaller chunks -- gen-chunk 8->4, score-batch 2->1, mb-blocks 10->6, the right fix for a device
OOM -- fit the device and died on the host, because smaller chunks mean MORE sequential host-side
iterations holding MORE intermediates at once. **The two walls push in opposite directions, so "shrink
again" could not have converged, and each attempt cost 20-75 minutes of compile to find out.**
`runs/think76r_hindsight32/config.sh` changes the **device mapping instead of the experiment**: 32 ranks x
GA 2 = **65,536 tokens/step, identical to run 76's registered 64 x 1 x 1024**; the FLAGS array is
byte-identical (71 tokens); grad-accum was read in the code before the config was written
(`think_hindsight.py:501` micro loop wrapping the whole pass, `:477` loss/GA, `:642` one sync per step,
`:439` per-micro data, `:662` tokens counted as GA*T*world). HBM per rank is unchanged (one core per rank
either way), so smoke #2's device fix still holds. **G123(a): WHEN A SHRINK FIXES ONE OOM AND CAUSES
ANOTHER, STOP SHRINKING AND CHANGE THE MAPPING. AND MEASURE THE HOST, NOT ONLY THE DEVICE.**

**(b) RUNS 65 AND 66 WERE THE SAME MODEL, TO THE BIT, AT THE STEP WHERE THE RECORD COMPARED THEM.** A
by-product of the band-55 sweep: two probe logs, `span_r65_200_o-15.log` and `span_r66_200_o-15.log`, taken
six hours apart on 2026-09-30 from **different** checkpoints (`think65_nofiller_archive/step_0000200.pt`
and `think66_nolencost_archive/step_0000200.pt`), agree in **every column of every one of their 10
sequences to two decimals**. They are not the same file. The weights are the same model: the parameter sum
over 311 tensors is **223704.01193565433 for both**, and over the first six tensors the two runs are
identical to 8 decimals at steps **200, 400 and 500** and differ from step **600** on. The two configs
differ in exactly one value -- `--len-cost 0.02` -> `0.0` -- and that flag did not enter the loss for 500
steps. The 2026-09-30 record called run 66's @200 read a **"replication"** of run 65 and even wrote "the
flag should be nearly inert this early". It was not *nearly* inert, it was **exactly** inert: the read could
not have disagreed, so its agreement was worth nothing, and a guaranteed agreement read as confirmation is
worse than no read -- it is 55 minutes of CPU spent buying confidence that was manufactured.
**G123(b): BEFORE BELIEVING THAT TWO RUNS DIFFER, CHECKSUM THEIR PARAMETERS AT THE STEP YOU ARE READING.
Two runs are two runs only after that test.** It is six lines of CPU
(`torch.load(..., map_location="cpu")`, sum a few tensors) and it would have caught this on 2026-09-30.

**(c) THE BAND THAT MISSED WAS WORTH MORE THAN THE BANDS THAT HIT.** Band 55 predicted the paired s.e. is
`< 0.5x` the unpaired one across the whole record. `agent/pairing_sweep.py` over the 64 parseable probe
logs: 46 distinct anchor sets, 5 shared by 2+ reads, **76 pairs**; ratio median **0.54x** (bar missed by
0.04), range 0.09x-1.00x, below 0.5x in **36/76**. But **9 of the 76 pairs change their 2-s.e. verdict when
paired**, and they are not random: the pair that produced run 79 (0.26x, z 0.9 -> 3.6) and run 71 @1000 at
o=-9 against the untrained init (0.32x, z 2.7 -> 8.5). **Pairing is worth about 2x on a typical pair and 4x
on the pairs whose verdict the programme actually used.** G122(a) is upheld and quantified, and the
sweep found G123(b) on the way past. **G123(c): A BAND THAT MISSES ITS BAR CAN STILL SETTLE ITS QUESTION --
SCORE THE BAR, THEN READ WHAT THE MEASUREMENT FOUND.**

**(d) A FALLBACK THAT CORRECTLY STANDS DOWN LEAVES NOTHING BEHIND IT.** The 18:07 fallback fired its
trigger at 18:30:07 on smoke #2's failure, waited its 3-minute grace, saw at 18:33:07 that an interactive
owner had relaunched, and **stood down -- exactly as designed**. Then smoke #3 failed at 19:46 and *nothing
was armed*: keepalive refuses a failed smoke, `watchdog.sh` logged "NEEDS A HUMAN" six times between 19:50
and 20:10, and the gate script refused to chain run 77 past a FAILED run. All three correct, and the device
sat with **8 of 64 cores busy on a 4-step diagnostic from 19:46 until 20:17**. A one-shot fallback is a
fallback for one failure. **G123(d): A GUARD THAT DISARMS ITSELF WHEN IT DEFERS MUST RE-ARM, OR IT IS A
GUARD FOR THE FIRST FAILURE ONLY.** Registered as band 64.

## G124 (2026-10-02 22:07 tick) -- A SWEEP THAT CANNOT FIND THE CASE YOU ALREADY KNOW IS NOT EVIDENCE OF ABSENCE

**(a) THE FIRST VERSION OF THE BAND-66 SWEEP RETURNED "NO PAIR IS IDENTICAL" AND IT HAD EXCLUDED THE ONE
PAIR THAT IS.** G123(b) found runs 65 and 66 bit-identical at the step the record compared them. Band 66
asked the obvious follow-up: is any *other* pair in this record the same model at a step where it was
compared? `agent/band66_pairs.py` enumerated every pair of runs sharing an `INIT` with checkpoints archived
at a common step, ranked them by how many flags their configs differ in, tested the 12 closest, and
reported **"no candidate pair is identical at any shared archived step"**. The number is true and the
sentence is worthless, because **runs 65 and 66 were not in the 12.** The flag diff was computed by regex
over `config.sh`'s *text*, and every config in this programme quotes flags in its own prose and comments
(`--rl-w` appears 7 times in run 69's config, mostly in sentences), so the 65/66 pair scored as a >3-flag
difference and sorted out of the candidate set. Sourcing the `FLAGS` array instead -- what a launcher
actually passes -- makes it a **1-token** difference (`--len-cost 0.02` -> `0.0`, 113 tokens each, `diff`
otherwise empty) and sorts it to the front.
**The fix was not the point; the ordering was.** The instrument had already produced a clean negative
result before anyone checked whether it could produce the known positive. Had the 65/66 case not existed in
the record from two hours earlier, that negative would have been believed.
**G124(a): RUN THE KNOWN POSITIVE THROUGH THE INSTRUMENT BEFORE YOU BELIEVE ITS NEGATIVE. A SWEEP IS
CALIBRATED BY THE CASE IT MUST FIND, NOT BY THE CASES IT REPORTS.** This is the same discipline that
`agent/paired_delta.py` was held to at 20:34 -- self-tested against +0.990 +- 0.274 before it was armed --
applied to a tool that was *not* gating a run and therefore got checked last instead of first.

**(b) BAND 66 MISSES, AND THE REASON IS A PROPERTY OF THE RECORD WORTH KEEPING.** With the sourced-FLAGS
instrument, the record holds **15 same-`INIT` pairs differing in <= 4 `FLAGS` tokens with a shared archived
step**, seven of them single-token. Exactly **one** is identical -- runs 65/66, already known, reproduced
here to the digit (`223704.01193565433` at 200, `223244.5070654871` at 400, `222731.14174582151` at 500,
divergent at 600). Band 66 (p 0.45) **MISSES**.
The reason is mechanical and it tells you where to look next time. The other six single-token levers all
enter the computation on the first step they are in force: `--rl-w` 0.5/0.0/1.0 (runs 67/68/69, with
`--rl-start 100`, and the earliest step both runs archived is 200 -- already past it), `--pause-frac`
0.15 -> 0.0 (64/65), `--min-think` 2 -> 3 (57/59), `--synth-hops-ramp` 0 -> 250 (58/59). `--len-cost` is the
only lever in this record **gated by a condition that did not occur**: the penalty multiplies a length
overrun that never happened for 500 steps, so the flag was not "nearly inert", it was *exactly* inert.
**G124(b): THE DANGEROUS FLAG IS NOT THE SMALL ONE, IT IS THE CONDITIONAL ONE.** A lever whose effect is
multiplied by an event that may not occur yet is the only kind that can make two runs bit-identical, and it
is invisible in a config diff. Before an A/B on such a flag, check that its condition fires.

**(c) THE PRUNE THAT MADE THE SWEEP POSSIBLE, AND THE COST OF NOT SEEING IT FIRST.** Checkpoints here are
3.0 GB and the disk serves ~125 MB/s under a live run, so the naive sweep -- checksum all 193 archived
checkpoints, then compare -- needs **580 GB of reads, about 80 minutes**, and it was killed at 25 minutes
having produced nothing (stdout to a file is block-buffered, so a long read loop reports *zero* progress
until it exits: a second reason it looked broken). **Divergence is monotone in training: once two runs
differ at step N they differ at every step after N.** So a pair is tested at its earliest shared archived
step and walked forward only while it still matches -- which kills 14 of 15 pairs on their first comparison.
The answer came from **37 checkpoints, ~110 GB**, cached in `runs/probes/band66/cache.json` so the full
256-pair version is resumable from where it stopped.
**G124(c): BEFORE A SWEEP, ASK WHICH COMPARISON CAN END A CANDIDATE EARLIEST.**

**(d) THE HOST-RAM GAP IS NOW INSTRUMENTED, AND THE 20:07 FOOTPRINT REPRODUCES FROM A SECOND INSTRUMENT.**
G123(a) ended with an unclosed gap: nothing in this programme *acts* on host RAM. `agent/status.py:12` and
`watchdog.sh:100` both **print** `MemAvailable` and no mechanism reads either number, so a host OOM arrives
as signal 9 with no traceback and the diagnosis has to be reconstructed from `dmesg` afterwards -- if the
dump is still in the ring buffer. `agent/host_ram_sentinel.sh` (armed 22:34 from the immutable snapshot
`agent/running/host_ram_sentinel_20261002_223435.sh`, md5 `c3dfc20376ada12af7ebaaacce2fe718`, pid 3190558)
samples every 60 s, keeps high-water marks of total and worst-rank RSS, and on crossing 300 GB (WARN) or
150 GB (CRIT) of headroom writes a per-rank table **while the ranks are still alive**, so the peak is
attributable to a phase instead of inferred from the kernel's victim. All four paths were tested before
arming by moving the bars across the live box: WARN fires, CRIT escalates, recovery logs once, and at
healthy headroom it writes one heartbeat an hour and nothing else.
**It does not stop a run, by design.** No live run has a pre-registered host-RAM stop clause, and stopping
one on an unregistered proxy reading is precisely what G122(b) charges against `ob0`. It records; a tick
decides.
Its first sample is also an independent check on G123(a)'s central number: **67 python processes, 488 GB
total, mean 7.3 GB/rank, worst 13.2 GB** against the 20:07 `dmesg`-derived **7.1 GB/rank** -- two different
instruments, the same footprint. The worst rank at 13.2 GB (1.8x the mean, rank 0 carrying the data loader)
is new information and the right number for sizing: at 32 ranks a hindsight rank scaled the same way implies
a peak well above the per-rank mean, which is exactly what run 76R's 50 GB rank-0 bar (00S.0 item 3) exists
to catch.

**(e) A MIS-ATTRIBUTION, CORRECTED BY BINNING THE WHOLE RUN INSTEAD OF ITS TAIL.** Mid-tick, run 79's step
time read 19.4-20.2 s against the 16.8-17.0 s of ten steps earlier, the band-66 sweep was reading 125 MB/s
off the same disk, and the obvious conclusion -- the sweep is taxing the live run ~15% -- was drawn and
acted on (the sweep was stopped). It was wrong. Step times **did not recover** after the reads stopped, and
binned over the whole run in 50-step blocks they are **17.1 / 18.0 / 16.5 / 17.4 / 16.5 / 17.5 / 21.4 /
17.6 / 17.2** -- the slowest block, 21.4 s, is **3300-3349, which predates the sweep by 40 minutes**. The
"rise" was inside the run's own 16.5-21.4 s band. **G124(e): A TREND READ OFF THE LAST TEN SAMPLES IS NOT A
TREND. BIN THE WHOLE SERIES BEFORE YOU BLAME SOMETHING -- ESPECIALLY WHEN YOU HAVE A SUSPECT IN HAND.** The
same error in the other direction is G122(a): a statistic computed on the wrong window, believed because it
agreed with a story already in mind.

**(f) THE SUPERLATIVE THAT DID NOT SURVIVE ITS OWN CHECK, AND THE SHAPE THAT DID.** Run 79's held-out CE
falls at every scored step (2.8275 -> 2.8203 -> 2.8170 -> 2.8154 -> 2.8141, steps 3000-3400) and the first
draft of `STATUS.md` called it **"the first time the thinking objective has cost nothing in next-token
prediction while training."** A sweep of all 43 runs with >= 4 scored points killed that: **nine runs have an
improving 3-point stretch and four beat run 79's -0.0121** (think38_hash -0.0880, think39_past -0.0838,
think71_frozen_imit -0.0347, think32_abstain -0.0218).
The claim that survives is about **duration and provenance, not magnitude**: run 79's is **5 consecutive
falling points, the longest unbroken improving stretch in the record**, and it is the only one that is not a
model settling after a shock it caused itself. Runs 38/39 fell from a 3.2565 fresh-init base over their first
300 steps. **think71_frozen_imit is the one that matters: 2.7749 -> 2.8188 -> 2.8436 in 200 steps, back down
to 2.8054 by 400 -- a steeper fall than run 79's -- and then monotonically UP for 2000 more steps to 2.8222,
never returning.** Run 79 is falling from a 3000-step-old resume with no shock to recover from, which is the
better sign; think71 is the reason it is a sign and not a result, and the reason band 67 is priced 0.60.
**G124(f): A SUPERLATIVE IS A QUERY, NOT A SENTENCE. "FIRST TIME" AND "BEST EVER" ARE CLAIMS ABOUT THE WHOLE
RECORD, SO THEY MUST BE RUN AGAINST THE WHOLE RECORD BEFORE THEY GO IN THE OWNER'S CHANNEL** -- here it was
one 15-line sweep over files already on disk, and it turned a false headline into a true and more specific
one. The same instinct that made G124(a) necessary (believing a negative from an uncalibrated tool) produces
false positives too when the tool is a nice-sounding phrase.

## G125 (2026-10-03 00:07 tick) -- A THREE-POINT FIT THAT REPRODUCES AN INDEPENDENT COUNT TO THE DIGIT CAN STILL HAVE THE WRONG SLOPE

Run 79's rank 0 was carrying **29.7 GB against a 7.55 GB mean over the other 64 ranks**, and the host-RAM
sentinel's first two heartbeats had it at **13.2 GB (22:34) and 24.1 GB (23:34)** -- read naively, a rank
leaking ~10 GB/h on a box that died of host RAM three times last week.

**(a) THE FIRST QUESTION WAS NOT "HOW FAST" BUT "WHAT SHAPE", AND IT COST 2 MINUTES.** A leak and a
staircase project to completely different places, so sampling `/proc/<rank0>/status` every 10 s answered
the shape before any rate was estimated: **29.700 -> 29.701 GB over 8 steps and 2.5 minutes, flat to 1 MB.**
Not a per-step leak. `train.py:222-230` then names the only rank-0-exclusive allocation in the trainer:
`cpu_sd = _to_cpu(model.state_dict())`, a full fp32 host copy of the model built **by the master alone**
every 100 steps, `torch.save`d, then `del`eted. Python frees it; glibc keeps the arena.
**THE RULE: ASK WHETHER A GROWING NUMBER IS SMOOTH OR DISCRETE BEFORE ASKING HOW FAST IT GROWS.** Three
samples spaced an hour apart cannot tell those apart, and the two of them differ by a factor the decision
depends on. The cheap sample is the one at the cadence of the suspected *event*, not of the monitor.

**(b) THE INSTRUMENT COULD NOT ANSWER THE BAND THAT HAD BEEN REGISTERED AGAINST IT.** Band 73 (22:07) asked
whether **the worst rank stays rank 0** for every sample. `host_ram_sentinel.sh`'s `rank_table` printed
`worst=24.1GB` -- a value with **no owner**: nothing in the log said which process it was, so the band was
unscoreable by the only tool watching it, and the 22:07 tick's confident "and it is rank 0" was in fact
carried over from a `dmesg` reconstruction, not measured. Patched this tick to emit `wpid` / `rank0` /
`worst_is_rank0`, where rank 0 is resolved as **torchrun's first child by ppid, never by pid ordering**,
and tested against the known answer (`wpid=3115817 rank0=3115817 worst_is_rank0=yes`) before arming.
**THE RULE, AND IT IS G124(a) AIMED FORWARD INSTEAD OF BACK: WHEN A BAND IS WRITTEN, NAME THE FIELD IN THE
LOG THAT WILL SCORE IT. IF NO FIELD EXISTS, THE BAND IS PROSE.** G124(a) says calibrate a tool against the
case it must find; this says check the tool *emits the quantity* at all -- the cheaper half, skipped more often.

**(c) THEN THE FITTED SLOPE WAS REFUTED BY THE EVENT IT PREDICTED, AND THE FIT HAD LOOKED EXCELLENT.**
Three points gave `rank0_GB = 7.5 + 3.17 x n_checkpoints`, and that line did something a fit is not
supposed to be able to do by luck: it returned **7.0 saves** where an **independent counter in a different
file** (`grep -c 'copying model to host'`) said **exactly 7**. On that basis the next save was
PRE-REGISTERED, in the probe log, 10 minutes before it happened: **jump +2.8..3.6 GB to 32.5..33.5 GB**,
p 0.85, with the falsifier written beside it.
**Observed: 29.701 -> 35.309 GB, a jump of +5.608 GB. MISS, outside the band, in the direction of worse.**
The shape was right and the slope was wrong: 7 saves had produced only ~4 increments of ~5.6 GB, because
**the ratchet does not fire on every save** -- when the freed hole still fits the next 3 GB request, glibc
reuses it and RSS does not move. The average over 7 saves (3.17) and the increment when it fires (5.6) are
different numbers, and a 3-point fit sees only the average.
**THE RULE: AN AGGREGATE FIT CANNOT RECOVER AN EVENT SIZE WHEN THE EVENT DOES NOT FIRE EVERY TIME.** The
agreement with `ckpts_seen` felt like independent confirmation and was not: that counter constrains **how
many opportunities** there were, never **how many fired**, so it could not have disagreed with any slope --
which makes it the same manufactured agreement G123(b) charges against the runs-65/66 "replication".
**A CONFIRMATION THAT COULD NOT HAVE COME OUT OTHERWISE IS NOT A CONFIRMATION.**

**(d) WHAT THE NUMBER MEANS FOR THE DEVICE -- AND THE CONFOUND IT PLANTS IN A BAND NOT YET READ.** Run 79 is
**safe and was never at risk**: 16 saves remain to step 5400, at the ~3.2 GB/save *average* that is **+51 GB
-> ~86 GB on rank 0** against **1468 GB free**. No device action was taken or warranted.
The real casualty is a reading that has not happened yet. **Band 61 asks whether run 76R keeps rank-0 RSS
below 50 GB at every `[mem]` phase, as the test of whether the 32-rank resize bought back the margin
hindsight needs.** Run 76R's rank 0 starts near **30.7 GB** and will ratchet ~3.2 GB per save on *average*:
the 50 GB bar is crossed after **~6 saves, i.e. ~600 steps**, by `torch.save` on the master and **nothing to
do with hindsight at all.** Had this been found after the fact, a checkpointing artefact would have been
scored as the cost of the hindsight path -- the exact confound shape of G122(b) (`ob0`) and G123(b)
(runs 65/66): a number that moves for a reason outside the question being asked.
**THE RULE: A PRE-REGISTERED BAR IS ONLY AS GOOD AS THE LIST OF THINGS THAT CAN MOVE IT. WHEN A NEW
MECHANISM IS FOUND THAT MOVES A BAR ON AN UNLAUNCHED RUN, RE-PRICE THE BAND BEFORE THE RUN, NOT AFTER.**
Band 61 is re-read this tick as two numbers (baseline at the first `[mem]`, and slope per save) instead of one
threshold, and the one-line fix (`malloc_trim(0)` after `del cpu_sd`) is deliberately **NOT** applied yet:
the successor is armed to launch run 76R, and a silent second change to a run whose whole purpose is to
survive host RAM would cost the one-change-per-run discipline for a run 79 that is in no danger.

**(e) AND THE TREND-FROM-FEW-SAMPLES ERROR REPEATED ITSELF ONE TICK AFTER BEING WRITTEN DOWN.** Band 72
(22:07) predicted run 79's `eligRate` "continues to fall (it has gone 0.380 -> 0.347)" to below 0.30 by step
5000, p 0.55. Binned over the whole run in 200-step blocks the series is **0.349 / 0.376 / 0.381 / 0.413 --
monotonically RISING.** The "0.380 -> 0.347" was two instantaneous readings off a status line whose own
block table disagreed with it on the same screen. **G124(e) was written in the 22:07 tick and violated in
the same tick's prediction table**, which is the useful part: a lesson recorded in the post-mortem is not yet
a habit in the forecast. **THE RULE: A BAND ABOUT A DIRECTION MUST QUOTE THE BINNED SERIES IT IS EXTRAPOLATING,
IN THE BAND ITSELF.** Then the disagreement with the block table is visible at writing time, not two ticks later.

## G126 (2026-10-03 02:07 tick) -- A GATE WHOSE BASELINE IS FILLER CAN BE PASSED BY BECOMING FILLER

Run 79's pre-registered PROGRESS gate fired at step 4000 -- paired delta **+2.930 +- 0.517, z 5.66,
28/32 sequences up**, the level climbing **-3.08 -> -2.73 -> -0.15** -- and the run was **stopped at 4188
anyway**, because in the same 300 steps **54% of the model's own sampled thoughts had become one repeated
digit**.

**(a) THE STRUCTURAL ERROR, AND IT IS IN THE GATE, NOT THE RUN.** The gate's statistic is
`own_minus_filler`: the reader's CE with the model's note, minus its CE with a filler note. **A note that
becomes filler scores exactly 0 on that statistic.** So the entire interval `(-3.08, 0)` is reachable by a
writer that learns nothing and merely stops saying anything, and a *paired delta* over that interval --
the most careful version of the measurement, the one whose s.e. we cut 3.4x by pairing, the one we built
two ticks of machinery for -- **has collapse as its single most likely cause.** Observed level: **-0.15 +-
0.61**, i.e. within a quarter s.e. of the value collapse predicts exactly.
**WIN (level > +3 s.e.) is the only clause a collapse cannot fake**, because filler cannot beat filler.
It was never met. **RULE: A GATE ON A DIFFERENCE FROM A DEGENERATE BASELINE MUST BE PAIRED WITH A
COLLAPSE MONITOR, OR IT IS A DETECTOR FOR COLLAPSE WEARING THE NAME OF THE HYPOTHESIS.** Write the gate on
the level against a NON-degenerate reference the writer cannot become, or read the writer's own output.

**(b) THE INSTRUMENT THAT EXISTED FOR THIS UNDER-REPORTED IT 4x, AND THE REASON IS GENERAL.** Runbook 3b
derails on `reg%` -- status.py's top-3 **8-character-prefix** share over **300-step** blocks. It printed
**12.7**, under its own 15% bar, at the moment the true degenerate share was **54%**. Both halves of the
instrument failed:
  - **the prefix.** The attractor wore variable short prefixes -- ` 22222222222222 `, ` 42653222222222 `,
    ` 50012222222222 `, ` 10222222222222 ` are **one** degenerate thought and **four** distinct 8-char
    prefixes. A prefix counter splits one mode into hundreds and reports diversity.
  - **the block.** 300 steps averaged the onset away; and `uniq%` read 67% while *two thirds of those
    "unique" thoughts were the same string with a different first character*.
Replaced by `agent/collapse_check.py` and a new **`col%`** column in status.py: the share of thoughts whose
**most common single character is >= 70% of the thought**. **It hard-codes no character**, so all-0s,
all-1s, all-2s and all-whitespace score alike -- which mattered immediately, because the attractor
**MIGRATED**: all-0s at 3600, ` 11111111111111 ` at 3800, ` 22222222222222 ` from 3900. The honest series
is **0.4/0.3/0.4/1.1/0.9/1.5/2.4/6.1/24.2/34.0/53.7/53.9** by 100-step bin. It was already **24% at step
3800** -- 388 steps before the stop, and *before* the @4000 read the gate passed.

**(c) THE REWARD WAS PUSHING THE OTHER WAY, WHICH KILLS THE OBVIOUS EXPLANATION.** The easy story is
reward hacking. It is wrong. Compared **within the same anchor's four rollouts** -- siblings, so anchor
difficulty cancels -- the degenerate rollout earns **-2.59 +- 0.41 LESS** gain than its non-degenerate
siblings (z **-6.25**, n=352 mixed groups), and the penalty **grows**: -3.18 / -1.25 / -2.08 / -4.22 across
100-step bins. **RL is actively suppressing this thought and losing.** Something else -- the imitation term
at weight 1.0, or the averaged gradient of the newly-synced 64x batch -- is overriding it.
The between-group number (-2.36) happens to agree here, but it is the confounded comparison and would have
agreed for the wrong reason; **sibling-vs-sibling is the only split that answers "is the reward selecting
this?"** and it is now a column of the tool.

**(d) WHAT I REFUSED TO CONCLUDE.** Run 79 changed exactly one flag from run 73 (`--sync-grads 1`) and
collapsed where run 73 did not, and the mechanism is plausible: 64 privately-diverging writers, each
collapsing to its own attractor, look like an ensemble; one writer following their averaged gradient
collapses to the marginal mode. **That is one run, and "the single changed flag did it" is available for
free after any single-run failure** -- the same shape as G123(b) and G125(c), an explanation that could not
have failed to fit. **No band was written on it.** Band 86 instead asks a *different* run with the same two
ingredients (run 76R: averaged gradient by construction at `think_hindsight.py:643`, `--imit-w 1.0`) to
collapse as well, at p 0.45.

**(e) AND THE CONSOLATION PRIZE IS THE WARNING.** Band 81 predicted that one sequence in 32 would carry
>=40% of the paired delta, as at @3500 where dropping one sequence took +0.350 to +0.117. **MISS**: the
extreme sequence carried **10.2%** and **28 of 32** sequences moved up. The effect was **broad, robust to
leverage, and highly significant** -- every property this programme has spent nine ticks learning to
demand -- **and it was a collapse.** Breadth, pairing and z are protections against *noise*. They are no
protection at all against measuring the wrong quantity.

## G156 (2026-10-07 02:07 tick) -- A GATE IS JUDGED AGAINST ITS BREAK-EVEN PRECISION, AND THAT NUMBER IS COMPUTED BEFORE ANY GATE IS EVALUATED

For the whole of this programme we quoted gates in **nats** and asked whether they crossed zero. That hid the
quantity that decides the question. A gate speaking on set S buys
`P(help|S)*E[g|help] + (1-P(help|S))*E[g|hurt]`, which is zero at
**`precision* = -E[g|hurt] / (E[g|help] - E[g|hurt])`**.
On run 95's deployable arm that bar is **0.756-0.773** at every rung and **0.731-0.813** across an 8x sweep in
thought length (bootstrap CIs all overlapping), because `E[g|hurt]` (-3.1 to -3.5 nats) barely depends on L.
The best precision any decision-time signal achieves -- including the run's own trained heads and a
cross-validated logistic combination of every feature -- is **0.467**, and that cell is the maximum of a
64-cell table. **Compute the bar first. A gate whose attainable precision is half its break-even precision
cannot be rescued by threshold tuning, by longer thoughts, or by more training**, and quoting its nats alone
makes it look like a near miss that another run might close.
Corollary that cost this project months: `oracle-bestG` (perfect gate AND perfect selection over G) is a
headline that needs two oracles. The deployable prize -- perfect gate, one sample -- is only **+0.55 to +1.04
nats/seq**, and it is NEGATIVE if you speak at half the triggered positions.
Tooling: `agent/decision_signal.py` / `agent/decision_signal.sh` print the bar beside the achieved precision
for every ranker at every volume. Run them on every rung.

## G157 (2026-10-07 02:07 tick) -- A BCE LOSS IS REPORTED AGAINST ITS BASE-RATE FLOOR `H(mean(y))`, NEVER AS A RAW LEVEL OR A RAW TREND

Run 95's `gate_pre_bce` falls with z -5.34 over 620 steps. That looks like a head learning. It is not:
BCE is linear in the label, so the best CONSTANT predictor is `p = mean(y)` at loss `H(mean(y))`, and the
label's own mean was drifting down (`helps` 0.1210 -> 0.1148), moving the floor 0.3689 -> 0.3564.
**94.6% of the fall in the loss is the fall in the base rate. The head's excess over the floor is -0.0024 --
0.67% -- and it gained +0.00071 nats of discrimination in 406 steps.** This is also the whole explanation of
`head_pre>0.3 speaks 0.00`: a head that outputs the base rate 0.115 everywhere can never cross an absolute
0.3 threshold, and it was initialised at bias -0.8 (p 0.3100), just above it.
Rules: (a) subtract `H(mean(y))`, computed from the run's own rows, before reporting any BCE level or trend;
(b) read the label's definition and mean OUT OF THE TRAINER (`think_bestof.py:364-371`), never assume which
metric key it corresponds to; (c) a head that collapses to the base rate may still RANK -- run 95's
`gate_pre` is the best single ranker of any feature (AUC 0.574/0.612/0.566) -- so report AUC and BCE-excess
separately, and deploy such a head by RANK, not by an absolute threshold.
