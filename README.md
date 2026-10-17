# Useful Thoughts on Trainium

Controlled experiments on whether self-generated intermediate text improves
language-model capability. This repository contains the experiment specification,
implementation, checks, and research context for a pretrained-model comparison
on AWS Trainium.

## Initial experiment (stopped on 2026-10-17)

Start two copies of released **Qwen3-1.7B-Base** from identical weights and give
them the same ordered mathematical documents:

- **C — continuation:** ordinary next-token training.
- **R — thought training:** sample four prefix-only thoughts, score how each
  changes the likelihood of the true next token, and apply a signed
  leave-one-out policy update to all four samples.

The initial recipe uses a 1,024-action cap, one-token reward, temperature 0.7,
and no reader, teacher, or ordinary-language-model auxiliary loss in R.
The writer never receives the true continuation.

Evaluate both trained models with direct, empty-format and thought-enabled
inference. The primary comparison is **R-thought minus C-thought**; also report
**R-direct minus C-direct**, **R-thought minus R-empty**, and each trained arm
against unchanged base weights. Direct inference permits ordinary calculations.
The first study targeted 256 terminal updates in each arm for two matched seeds,
with a fixed step64 development milestone. Retain failures in evaluation
denominators and report training and inference costs.

## Current status

**The initial R study has been stopped. A short writer-only warmstart now tests
whether useful selected candidates can become useful fresh notes.** Both C seeds
finished256 updates. R1234 stopped cleanly after update63 and saved its state;
R5678 was not launched. The original primary R/C comparison is incomplete.

The key baseline finding is a large **interface penalty**: on fixed DEV512,
raw K1 log likelihood averages−1.118 nats, empty thought format−7.361, and an
own sampled thought−11.203. Own versus unrelated thought is−.284 nats with a
95% conditional interval spanning zero. This motivates testing the interface
before spending more training compute. The [full B0 audit](validation/local_dev_B0_20261017.json)
includes all controls, horizons and strata. It has **one candidate per item**;
K16 means16 future tokens, not a best-of16 thought oracle.

Current work measures reader interfaces, candidate availability on TRAIN/DEV,
and actual pipeline stage costs.
Oracle gains will be reported separately from ordinary one-sample utility.
Official final tests remain on hold. See [current queue](NEXT_STEPS.md).

The first **DEV32 reader pilot** is complete (eight fixed documents per B0
surprisal quartile, original one-sample notes, FP32; no training):

| Reader interface | Empty−raw, nats | Own−empty, nats | Own−other, nats |
|---|---:|---:|---:|
| Original |−6.425 |−3.469 |−0.043 |
| No header |−6.652 |−3.765 |−0.382 |
| Repeat last32 prefix tokens after close |−1.080 |−1.906 |−0.432 |
| Repeat last64 |−1.412 |−2.271 |−0.624 |
| No header + repeat64 |−1.331 |−2.264 |−0.609 |

Re-anchoring reduces the interface penalty, but own thoughts still hurt on
average. With repeat32, own−empty has a95% paired document interval
[−2.679,−1.193]; own−other spans zero. Own also trails raw repeated-prefix
control by1.627 nats. These are exploratory reader interventions, not useful
thoughts or a deployed-policy improvement. The [complete pilot report](validation/interface_tax_dev32_20261017.json)
includes K16, all matched controls and quartiles. [Plot](validation/figures/interface_tax_dev32_20261017.svg).
All896 scores succeeded; original-format token likelihoods reproduce B0 to
0.0000382 nats. Original outputs, selected inputs and fixed first4 examples
are saved in a verified2MB EBS archive. The subsequent G16 search and replay
profiler are complete; [current diagnostic state](validation/new_direction_diagnostics_20261017.json).

A second [DEV32 reader probe](validation/prompt_interface_dev32_20261017.json)
found published RLP instruction wording still carries a6.223-nat empty-format
penalty. Plain newline boundaries plus repeat32 reduce it to1.085, but own
thoughts still trail same-format empty by1.175 nats. A [weight inspection](validation/B0_marker_embeddings_20261017.json)
found exactly identical input rows for the released think/end-think/chat boundary
tokens; correcting the published delimiter therefore produces identical reader
scores. This identifies an initialization detail, not a complete explanation.
A separate [DEV8 support check](validation/reader_support_dev8_20261017.json)
rules out undefined output probability as the measured tax's main source:
renormalization changes the original empty penalty by only0.000549 nats.

The [frozen TRAIN scan](validation/anchors_train1024_20261017.json) found237
high-surprisal anchors among1024 prospectively ordered real FineMath documents;
the first128 fed the completed2048-candidate TRAIN search. Future-based
training-label selection is reported explicitly.
No new training improvement has been established.

The [note-before-prefix probe](validation/prepend_reader_dev32_20261017.json)
provides a cleaner boundary: empty−raw is+.009 nats atK1 and+.032 atK16.
Original G1 content still hurts over16 future tokens: own−empty−1.783,
95% paired interval[−3.041,−.740]. This repairs format cost without establishing
useful thoughts. The [completed TRAIN rescore](validation/candidate_prepend_train128_20261017.json)
reuses all2048 saved candidates. Across128 documents, prepend K16 mean16−empty
is−1.072, while future-selected best16−empty is+1.026 [+.841,+1.226].
Max16-own beats max16-unrelated by+.535 [+.325,+.749] and max16-shuffled
by+.273 [+.088,+.466]. This is conditional TRAIN oracle evidence, not fresh
one-sample utility. A strict nonempty winner rule, also requiring a raw-text
win and all three same-index content-control wins, retains77 training labels.

The [genuine DEV32/G16 oracle](validation/candidate_oracle_dev32_cap256_20261017.json)
is complete. At cap256/reanchor32, ordinary mean16−empty is−1.612 K1;
future-selected best16 is+.330 [+.051,+.684]. K16 best16 is+.395
[−.418,+1.289]. All max-own versus max-unrelated/shuffled intervals span zero.
Oracle selection sees the continuation and does not establish deployed utility.
The [original TRAIN corpus](validation/candidate_corpus_train128_20261017.json)
and [fresh DEV64/G1 baseline](validation/warmstart_dev64_B0_20261017.json)
also remain negative on average. A separate64-update FP32 writer-only run is
now starting from B0 on those77 labels. It supervises note+canonical close,
with no true continuation in the writer input or loss. The raw control uses
the same selected documents/order and16 continuation targets. The primary
development check is fresh G1 notes on DEV64 with the unchanged B0 prepend
reader; training-label gains do not qualify this check.

The [full-gradient padding profile](validation/replay_padding_20261017.json)
preserves reader/signed-loss gradients while reducing isolated real-note warm
replay from~4.24s to~.46s. This is not measured end-to-end training throughput.
A separate local Qwen3-4B teacher pilot is running for explicitly disclosed
prefix-only/hindsight TRAIN supervision. Both initial device next-logp matrices
match native CPU references within.000447 nats; first cached/replay audits also
pass (max.0000315). Generation remains in progress. Base weights and identities
stay separate.

Fixed128-question GSM8K **TRAIN development** results:

| Mode | B0 /128 | C1234@64 /128 | C1234@256 /128 | C256−B0 pp,95% conditional interval |
|---|---:|---:|---:|---:|
| Direct |113 |111 |109 |−3.13 [−8.59,+1.58] |
| Empty format |107 |108 |109 |+1.56 [−3.13,+6.25] |
| One thought |63 |60 |91 |+21.88 [+11.72,+32.03] |

The thought-mode gain has substantial format and resource qualifications.
C256 thought still trails its direct/empty modes by18 answers and B0 direct by22.
Of B0's33 thought-mode format-error cases,26 become correct under C256. Every
C256 thought reaches the1024-action cap, versus4 caps for B0; mean item wall is91
versus23 seconds. Answer truncations are14 in both. All failures remain counted.
These are actual cold/cache/shared-host costs, not a causal throughput comparison.

The [complete terminal-C comparison](validation/held_dev_C_seed1234_step256_20261017.json)
retains all9 conditions, paired intervals, original failures and token/latency
costs. Intervals use5000 paired document draws, conditional on the checkpoints;
they are not training-seed uncertainty. Both64 and256 were fixed prospectively;
terminal256 remains primary. The [C64 report](validation/held_dev_C_seed1234_step64_20261017.json)
and every original output remain. [Verified backup and next release](validation/held_dev_C_seed1234_step256_backup_20261017.json)
records the54-file EBS archive. The later C5678@64 held evaluation and local
C1234@64 evaluation were interrupted on owner stop; their partial outputs remain.

The unchanged B0 **local DEV512** audit is complete: all512 sampled notes and
9216 likelihood rows, no execution failures. Mean K1 log likelihood was−1.118
for the raw prefix,−7.361 for empty format and−11.203 with its own note. Own−raw
is−10.085 nats (95% conditional interval[−10.517,−9.682]); own−empty is−3.842.
The [full local audit](validation/local_dev_B0_20261017.json) retains all6 absolute
conditions, K1/K4/K16, content contrasts and prospective token/surprisal strata.
One B0-derived membership map is frozen for all later readers. [Execution costs](validation/local_dev_B0_execution_20261017.json)
and the [verified backup/next release](validation/local_dev_B0_backup_20261017.json)
retain all outputs and fixed-order examples. This is the starting local baseline;
the old R/C comparison is incomplete, and official test outcomes remain uninspected.

- Both C runs consumed32,768 unique TRAIN records /16.81M source tokens and exited
  normally. All254 steps3–256 average10.056s (seed1234) and10.030s (seed5678).
  Verified EBS128/256 and all original NVMe64/128/256 states remain. Execution and
  backup reports: [C1234](validation/science_C_seed1234_execution_20261017.json),
  [C1234 backup](validation/science_C_seed1234_backup_20261017.json),
  [C5678](validation/science_C_seed5678_execution_20261017.json),
  [C5678 backup](validation/science_C_seed5678_backup_20261017.json).
- R1234 independently started from B0 and stopped after63 updates. Its
  [stop/execution record](validation/science_R_seed1234_stopped_20261017.json)
  retains all63 metrics and the verified recoverable step63 checkpoint. The
  [first32-update record](validation/science_R_seed1234_first32_20261017.json) and
  [execution plot](validation/figures/science_R_seed1234_first32_20261017.svg)
  retain every elapsed observation and changing generation workload. TRAIN
  likelihoods and numerical checks do not establish useful deployed thoughts.
- Actual released Qwen3-1.7B-Base assets, audited FineMath329535 TRAIN/18360 DEV/
  18239 TEST records, the [fixed256 specification](validation/science_256_frozen_20261017.json),
  [qualification](validation/science_256_qualification_20261017.json) and
  [resource projection](validation/science_256_resources_20261017.json) are pinned.
  FP32 signed gradients, full-cap replay and checkpoint recovery pass their
  declared criteria. The original coordinate v1 FAIL and separately prospective
  held-seed v2 derivative PASS remain preserved. R
  [execution](validation/r_fp32_ram_execution_20261017.json) and
  [recovery](validation/r_fp32_ram_resume_20261017.json) evidence is separate from
  scientific capability. Training is serial; no two-R speed multiplier is claimed.

See [STATUS.md](STATUS.md) for exact live ownership/commands and
[DECISIONS.md](DECISIONS.md) for material choices. Official final performance
scores remain uncomputed; earlier overlap screening used benchmark questions.

## Code and plan

| Path | Purpose |
|---|---|
| [configs/weekend.json](configs/weekend.json) | Current proposed experiment, including FP32 precision |
| [docs/WEEKEND_PLAN.md](docs/WEEKEND_PLAN.md) | Scientific design and interpretation |
| [train/neuron_driver.py](train/neuron_driver.py) | Bounded qualification and guarded C/R driver |
| [train/neuron_backend.py](train/neuron_backend.py) | Causal cache, generation, scoring, and policy replay |
| [train/evaluate.py](train/evaluate.py) | Question-only, sharded math evaluation |
| [train/evaluate_likelihood.py](train/evaluate_likelihood.py) | ARC and full-target LAMBADA likelihood |
| [train/evaluate_local.py](train/evaluate_local.py) | Saved prefix-only notes, content controls, and crossed readers |
| [weekend_thinking/](weekend_thinking/) | Objective, data, checkpoint, and analysis primitives |
| [tests/](tests/) | CPU correctness checks; these do not certify hardware |
| [context/](context/) | Research context and evidence limits |
| [legacy/](legacy/) | Preserved historical source and results; not active launch instructions |

## Running checks

Use a compatible, already working Neuron environment for accelerator work.
Instance paths in `scripts/env.sh` are local conveniences; adapt them to your
own installation. Model weights and corpus acquisition are explicit actions.

```bash
bash scripts/run_checks.sh
```

The CPU checks require PyTorch, NumPy, pytest and a compatible Transformers
installation, and disable accidental XLA initialization. Accelerator qualification,
actual distributed updates, full-step timing, and checkpoint resumption are
separate requirements. Read [START_HERE.md](START_HERE.md) and
[docs/NEURON_IMPLEMENTATION.md](docs/NEURON_IMPLEMENTATION.md) before launching.

Host configuration, credentials, downloaded datasets, weights, compiler caches,
raw traces, and checkpoints remain outside Git under ignored local paths.
Small validation reports describe their provenance and limits.
Verified local checkpoint copies are described in
[CHECKPOINT_BACKUP.md](docs/CHECKPOINT_BACKUP.md).

Original authorship and third-party notices are preserved; see
[NOTICE.md](NOTICE.md). Public availability does not assign a new license to
historical material.
