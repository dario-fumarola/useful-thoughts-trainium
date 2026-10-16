# FP32 ARC-Challenge and LAMBADA evaluation

`train/evaluate_likelihood.py` uses the released model loader and ordinary-causal
backend. Model/trunk/head computation is FP32. It supports direct likelihood and
one greedy prefix-only thought, with the same settings for C/R. No candidate or
gold target enters thought generation; one note is fixed before any candidate is
scored. Source handling and generation reuse the main evaluator's asset hashes,
strict own-checkpoint loader, fixed-shape greedy decoder, and frozen-protocol gate.

## Exact task definitions

- **ARC-Challenge:** preserve the legacy context `Question: {question}\nAnswer:`
  and one added leading space before each original candidate text. Preserve
  source choice order and map `answerKey` through the original labels, including
  numeric labels. Report both maximum sum token log likelihood and maximum mean
  token log likelihood. The normalization denominator is **token count**, not
  characters/bytes; exact ties select the first source choice.
- **LAMBADA:** preserve the legacy `text.rsplit(" ", 1)` context/final-word split,
  with a leading space before the final word. Report the sum likelihood of every
  target subtoken and exact accuracy requiring every teacher-forced greedy token
  to match. This corrects the archived scorer's first-subtoken-only accuracy;
  those old accuracy values are not directly comparable for multi-token words.
- Encode context and leading-space continuations separately without added
  special tokens, matching the legacy task definitions. Retain the last declared
  maximum context tokens (default 768), logging truncation. Targets exceeding the
  fixed future width (default 128) become reported errors, never truncated labels.
- The thought reader uses header+context+open+actual content+close before the
  candidate, with contiguous real positions and no additional answer suffix.
  The writer masks tokenizer-undefined model rows; reader likelihood and LAMBADA
  greedy-on-gold-prefix predictions use the raw released vocabulary.

The common scoring helper right-pads complete context/continuation sequences to a
fixed width. It gathers all K prediction positions, computes the head in bounded
chunks, masks every non-target future slot, and transfers only selected logps and
argmax IDs. It does not inherit a K=1 training horizon for evaluation.

Every expected item receives an `ok` or `error` outcome. Errors remain accuracy
failures. Mean likelihood is explicitly labeled as conditional on successful
execution, with successful counts and failures reported alongside it. Per-item
IDs, raw/normalized predictions, all continuation token IDs/logps, thought
actions/content/stop type, model/checkpoint hashes, and timing are saved locally.

## Data boundary and commands

The pinned revisions are ARC
`210d026faf9955653af8916fad021475a3f00453` and LAMBADA
`900124bf3b8235c6daf21033af9948b3f07346c4`. Development preparation verifies the
exact pinned ARC Parquet hash and exports separate question/choice and gold files:

```bash
source scripts/env.sh
USE_TORCH_XLA=0 python -m train.evaluate_likelihood prepare-arc \
  --parquet local/data/benchmark_exclusion/sources/allenai--ai2_arc/ARC-Challenge/train-00000-of-00001.parquet \
  --split train --output local/secondary_tasks/arc_train
```

That output already exists; do not overwrite it. Qualification ARC files may use
train/validation only. LAMBADA qualification accepts explicitly synthetic rows;
it does not access the official test set. Inputs contain `item_id`, `task`,
`split`, `context`, and ARC's public `choices`. Gold files contain the matching
`item_id` and `gold` candidate index or LAMBADA `target` string.

The two-item ARC training qualification subset is in
`local/secondary_tasks/arc_qualification/`. The two synthetic LAMBADA examples are
in `local/secondary_tasks/lambada_synthetic/`; their targets have one and three
Qwen tokens. A completed CPU run can be reproduced into a new output directory:

```bash
USE_TORCH_XLA=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -m train.evaluate_likelihood run \
  --task lambada --task-revision synthetic-lambada-v1 \
  --task-file local/secondary_tasks/lambada_synthetic/questions.jsonl \
  --gold-file local/secondary_tasks/lambada_synthetic/gold.jsonl \
  --model-dir local/assets/qwen3-1.7b-base --format local/resolved_format.json \
  --mode direct --max-context-tokens 64 --future-width 32 --thought-cap 8 \
  --device cpu --qualification --output local/qualification/secondary-lambada-cpu-NEW
```

Use `one_thought` for the note mode. Eight actions is a declared CPU diagnostic,
not the final 1,024-action evaluation condition. Accelerator runs require explicit
core coordination with the active trainers; none were launched by this work.

Official test inputs require `--frozen-protocol` matching the exact task/gold
hashes, item IDs, tokenizer/config, numerical settings, format, and evaluator
source hashes. Gold is parsed only after this gate. Final task projection and
the exact frozen secondary protocols must be prepared after the primary plan is
locked, before official test scoring. `prepare-arc` deliberately only prepares
development inputs; no final test answers or outcomes were inspected here.

## Qualification evidence

All eight semantic CPU tests passed: full-logit/all-token parity, unequal target
lengths/padding, raw-versus-normalized ARC disagreement, multi-subtoken LAMBADA
accuracy, unchanged writer inputs under candidate/gold changes, gold/final-split
boundaries, retained error denominators, and canonical label projection.

Released Qwen3-1.7B-Base FP32 CPU execution completed two ARC TRAIN direct rows,
two synthetic LAMBADA direct rows, and the same two synthetic rows with one
greedy eight-action note. All six executions returned `ok`. These are plumbing
checks, not benchmark estimates. The
[CPU receipt](../validation/secondary_likelihood_cpu_20261016.json) binds run
reports and output hashes. No official test item or accelerator was evaluated.

The full CPU suite passed 145 tests plus the toy smoke check; logs are in
`local/checks/tests.vNYBQ7/`. Accelerator scoring/generation timing and the final
thought cap remain unqualified by this work.
