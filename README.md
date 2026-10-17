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

**Both scientific C runs completed 256 updates; R1234 is running. FP32 qualification
and the fixed specification are complete; no trained C/R comparison is available.**

- Released model weights and tokenizer are pinned and loaded.
- Each scientific C run consumed 32,768 unique TRAIN records and 16.81M source
  tokens, completed all 256 updates and exited normally. Full-step means after
  two startup steps were 10.056s (seed1234) and 10.030s (seed5678). Both retain
  verified EBS copies of terminal256 and preceding128, plus all NVMe originals.
  See [C1234 execution](validation/science_C_seed1234_execution_20261017.json),
  [C1234 backup](validation/science_C_seed1234_backup_20261017.json),
  [C5678 execution](validation/science_C_seed5678_execution_20261017.json) and
  [C5678 backup](validation/science_C_seed5678_backup_20261017.json).
- R1234 independently started from released B0 and reached step26/256 at
  15:37UTC. The B0 local DEV512 audit and fixed held128 C1234@64 suite have
  started on separate idle devices; no trained evaluation result is complete.
  See [R progress](validation/science_R_seed1234_progress_20261017.json) and
  [development starts](validation/science_development_start_20261017.json).
- The audited FineMath slice contains **329,535 train**, **18,360 dev**, and
  **18,239 test** document-level records.
- Full FP32 cached generation agrees with replay on the exact long trajectory
  that failed BF16 qualification; the fixed tolerance was not relaxed.
- Full-cap FP32 backward now fits and passes signed CPU reference checks after
  correcting an integer upper-bound conversion in sequence normalization.
  The original failures and unchanged tolerances are preserved.
- The 32-core continuation run completed 12 updates, including ten steady steps
  averaging about 10 seconds. Full-model checkpoint recovery now passes, including exact restored state and all 32 RNG states. The
  original CPU per-weight comparison remains failed. A separate, prospectively
  fixed full-gradient comparison against FP64 passes on a new held seed.
- R completed14 updates, all ten fixed timings5–14, and strict restored-state/
  stochastic-tape/full-update recovery. Median full-step time is11.75min, P90
  14.49min; shared-host overlaps remain included. Sampled two-R execution was
  observed; steady simultaneous throughput was not measured. See the
  [execution](validation/r_fp32_ram_execution_20261017.json) and
  [recovery](validation/r_fp32_ram_resume_20261017.json) reports.
- Qualification ended with511/512 empty notes and negative raw utility. Passing
  numerical checks does not establish useful thoughts.
- The [first256-update specification](validation/science_256_frozen_20261017.json)
  fixes two matched seeds with independently fresh-B0 C/R arms. Training will be
  serial from production memory headroom; idle cores may serve development.
  [Recorded qualification](validation/science_256_qualification_20261017.json) and
  [resource projection](validation/science_256_resources_20261017.json) are separate
  from actual per-arm launch releases.
- The fixed128-item TRAIN development baseline scores113/128 direct,107/128
  empty-format and63/128 with a thought. Failures stay counted; this is not a
  training effect or official-test result. See
  [the B0 report](validation/held_dev_B0_20261017.json).

See [STATUS.md](STATUS.md) for active work and [DECISIONS.md](DECISIONS.md) for
material choices. CPU checks and historical results are distinguished from
current accelerator evidence.

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
