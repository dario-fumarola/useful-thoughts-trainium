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

Evaluate both trained models with direct and thought-enabled inference. The
primary comparison is **R-thought minus C-thought**; also report
**R-direct minus C-direct** and **R-thought minus R-direct**. Keep unchanged
base weights as a reference, use fixed terminal checkpoints, retain failures
in evaluation denominators, and report training and inference costs.

## Current status

**Hardware qualification is in progress; no primary scientific result is claimed.**

- Released model weights and tokenizer are pinned and loaded.
- The audited FineMath slice contains **329,535 train**, **18,360 dev**, and
  **18,239 test** document-level records.
- Full FP32 cached generation agrees with replay on the exact long trajectory
  that failed BF16 qualification; the fixed tolerance was not relaxed.
- Full-cap FP32 backward now fits and passes signed CPU reference checks after
  correcting an integer upper-bound conversion in sequence normalization.
  The original failures and unchanged tolerances are preserved.
- The 32-core continuation run completed 12 updates, including ten steady steps
  averaging about 10 seconds. Checkpoint backup is verified. A restore-order RNG
  repair passes its hardware test; full-model recovery is being checked. A strict
  CPU per-value discrepancy is being investigated with a higher-precision reference.
- Development-only math inference runs, including failed outputs, are retained.

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
