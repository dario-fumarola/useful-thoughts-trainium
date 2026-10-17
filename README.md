# Useful Thoughts on Trainium

Controlled experiments on whether self-generated intermediate text improves
language-model capability. This repository contains the experiment specification,
implementation, checks, and research context for a pretrained-model comparison
on AWS Trainium.

## The experiment

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
The first study targets 256 terminal updates in each arm for two matched seeds,
with a fixed step64 development milestone. Retain failures in evaluation
denominators and report training and inference costs.

## Current status

**The first trained development check is complete. Both C seeds finished256
updates; R1234 is running. The primary R/C comparison remains pending.**

On the fixed128-question GSM8K **TRAIN development** subset, C1234 at the
prospectively fixed step64 has no clear accuracy change from unchanged B0:

| Inference mode | B0 correct /128 | C64 correct /128 | C−B0 percentage points,95% conditional interval |
|---|---:|---:|---:|
| Direct |113 |111 |−1.56 [−6.25,+3.13] |
| Empty format |107 |108 |+0.78 [−4.69,+6.25] |
| One thought |63 |60 |−2.34 [−11.72,+7.03] |

Every failure stays in the denominator. Intervals use5000 paired document
bootstrap draws, conditional on these checkpoints; they are not training-seed
uncertainty. Thought inference used326 versus124 mean thought actions and38
versus23 seconds/item (C64 versusB0). Thought caps increased32 versus4; answer
truncations36 versus14. These are measured costs under actual cache/shared-host
conditions, not a causal throughput benchmark. Fixed terminal256 remains primary;
no checkpoint or prompt was selected using this development result.

The [complete comparison](validation/held_dev_C_seed1234_step64_20261017.json)
retains all modes, errors, caps, token/latency distributions and provenance.
All53 raw output/control files and fixed-order examples have verified EBS copies;
see [backup and next release](validation/held_dev_C_seed1234_step64_backup_20261017.json).
C1234@256 evaluation and the B0 local DEV512 audit are now running on separate
idle devices while R1234 continues through its fixed256 updates.

- Both C runs consumed32,768 unique TRAIN records /16.81M source tokens and exited
  normally. All254 steps3–256 average10.056s (seed1234) and10.030s (seed5678).
  Verified EBS128/256 and all original NVMe64/128/256 states remain. Execution and
  backup reports: [C1234](validation/science_C_seed1234_execution_20261017.json),
  [C1234 backup](validation/science_C_seed1234_backup_20261017.json),
  [C5678](validation/science_C_seed5678_execution_20261017.json),
  [C5678 backup](validation/science_C_seed5678_backup_20261017.json).
- R1234 independently started from B0 and reached41/256 at18:30UTC. The
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
