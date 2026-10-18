# Useful Thoughts on Trainium

Controlled experiments on whether self-generated intermediate text improves
language-model capability. This repository contains the experiment specification,
implementation, checks, and research context for a pretrained-model comparison
on AWS Trainium.

## Current finding

**Useful self-generated thoughts have not been established.** The latest bounded
probe uses unchanged Qwen3-1.7B-Base on 32 GSM8K TRAIN development questions:

| Inference | Strict accuracy | Canonical-number log likelihood |
|---|---:|---:|
| Ordinary calculation, cap256 | 24/32 (75.0%) | −2.739 |
| Generate then discard note, reader cap192 | 24/32 (75.0%) | −2.819 |
| Sampled note cap64 + reader cap192 | 86/128 paths (67.2%) | −2.978 |

Four notes per question estimate a **single sampled-note policy** by averaging
within each question; there is no best-of-four selection or voting. The accuracy
difference from ordinary calculation is −7.81 percentage points, paired 95%
interval [−18.75,+2.34]. Notes also fail to improve numeric likelihood. They beat
shuffled and unrelated text, but that does not establish improvement over silence.
This is exploratory reuse of known TRAIN questions, not blinded confirmation.
Equal maximum token budgets do not imply equal FLOPs or latency.
[Full result, controls, failures, costs and provenance](validation/gsm8k_self_note_train32_20261018.json).

All 128 notes, 416 fresh reader attempts, 416 likelihood rows and 32 reused exact
baseline cells are retained. Independent audits reproduce all means and paired
intervals. A manual audit finds final-answer claims, arithmetic mistakes, wrong
quantity interpretations and unfinished notes; those annotations do not filter
any result. Raw evidence and fixed-order examples have verified local EBS copies.

The next experiment prepares verifiable calculation boundaries from real FineMath
TRAIN/DEV documents. A note computes a proper subexpression using only the past;
the natural source continuation remains the target. This is explicit external
arithmetic supervision, distinct from spontaneous self-discovery. Eligibility and
selection precede model scoring. The first scan produced1,304 exact equations,
but fixed-example review found three answers already visible through currency
or option punctuation. Its original data is preserved; a corrected v2 scan is
running, and no model was scored on v1. No further long RL run is active.
[Current queue](NEXT_STEPS.md) · [Exact work state](STATUS.md).

## What led here

- **Initial R study stopped.** Inserting the original thought format carried a
  large initial penalty: B0 DEV512 K1 raw −1.118, empty −7.361, own −11.203 nats.
  Own versus unrelated content was indistinguishable from zero. R1234 stopped
  after 63 updates; R5678 never launched. Both old C runs finished 256 updates.
  The original primary R/C comparison is incomplete.
  [B0 audit](validation/local_dev_B0_20261017.json) ·
  [R execution and preserved checkpoint](validation/science_R_seed1234_stopped_20261017.json).
- **Repairing format did not establish content utility.** Prepending notes
  reduced empty-format tax to approximately zero. On fixed TRAIN128, a
  future-selected best-of-16 tail gained +1.026 nats over empty, while the
  ordinary mean lost 1.072 nats. Oracle selection sees the target.
  [Interface probe](validation/prepend_reader_dev32_20261017.json) ·
  [TRAIN oracle](validation/candidate_prepend_train128_20261017.json) ·
  [DEV oracle](validation/candidate_oracle_dev32_cap256_20261017.json).
- **The first learning comparison was negative.** A 64-update note-cloning arm
  and a raw future16 CE control shared 77 documents and 512 ordered occurrences,
  with unequal labels and compute. Fresh DEV64 notes still hurt the unchanged
  reader: K16 own−empty B0 −1.101, raw −0.956, cloning −0.649 nats. The
  cloning-minus-control change was +0.307 [−0.209,+0.908]. Both terminal states
  are fully backed up. Reader adaptation also failed.
  [Three-arm result](validation/warmstart_dev64_three_arm_20261018.json) ·
  [Reader crosses](validation/warmstart_reader_cross_dev64_20261018.json).
- **Downstream checks also failed.** GSM8K TRAIN128 direct/empty/thought correct
  counts were B0 113/107/63, cloning 92/92/61, raw CE 56/112/76. Every thought
  mode lost to its own empty control. Raw direct had 59 format errors, so its
  score does not isolate arithmetic ability.
  [All nine conditions](validation/warmstart_gsm8k_train128_terminal64_20261018.json).
- **External partial calculations show limited assistance.** A separate TRAIN32
  pilot scored arithmetic-verified manual notes 30/32, empty 27/32, corrupted
  notes 5/32. All three added strict successes repaired answer formatting or
  avoided truncation; none corrected a wrong parsed number. Numeric likelihood
  improved +0.945 [0.258,1.591] nats. One manual note missed a purchasing
  discount despite correct arithmetic; its frozen note and score are unchanged.
  Local 4B notes included final answers. These are qualifications on assistance,
  not evidence of useful student computation.
  [Pilot](validation/gsm8k_calculation_train32_20261018.json) ·
  [Exhaustive posthoc mechanism panel and correction](validation/gsm8k_calculation_train32_mechanism_20261018.json).

All official final tests remain unused. Failures stay in denominators, controls
are reported alongside raw performance, and oracle results remain separate from
ordinary generated-note utility. Historical implementation and results remain
available in Git and the linked reports.

## Code and plan

| Path | Purpose |
|---|---|
| [configs/weekend.json](configs/weekend.json) | Original stopped experiment configuration |
| [docs/WEEKEND_PLAN.md](docs/WEEKEND_PLAN.md) | Original scientific design and interpretation |
| [train/calculation_boundaries.py](train/calculation_boundaries.py) | Current exact-arithmetic source curator |
| [train/calculation_boundary_probe.py](train/calculation_boundary_probe.py) | Current controlled reader feasibility probe |
| [train/neuron_driver.py](train/neuron_driver.py) | Historical C/R qualification and driver |
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
