# Evaluation protocol

## Lock before final testing

Primary task: official GSM8K test, pinned revision, one generated path. Before
final testing, record task revision, item-ID hashes, exact prompts/token IDs,
answer extraction, decoding policy, maximum tokens, stop rules, and terminal
checkpoint rule. Calibrate formatting only on designated development/training
examples; do not use test answers to choose prompts or checkpoints.

`train/evaluate.py` has run fixed TRAIN-only direct/thought/empty-format inference
on the real accelerator. Final checkpoint, task, parser, and dependency bindings
are described in [EVALUATION_INTEGRITY.md](EVALUATION_INTEGRITY.md). Secondary and
local evaluators need their own execution evidence; this is not a completed final
benchmark suite.

## Comparisons

Primary: R-thought minus C-thought, same inference policy and cap.
Required secondary: R-direct minus C-direct; R-thought minus R-direct.
Include B0 as the unchanged starting reference and C-thought minus C-direct as
context. Test seeds separately before any clearly labeled aggregate summary.

Direct inference has no inserted thought phase. Thought inference generates one
thought then the answer, with no selection based on the gold answer. Mode-specific
formatting is itself an intervention; matched empty-format/content controls help
separate that cost from thought utility. Use the same prompt for C/R within a mode.
Do not hardcode an answer-dependent continuation into the prompt.

A suggested starting prompt asks for the final result as `#### <number>`. Any
few-shot examples must come from the allowed development/training split and their
IDs must be recorded. No answer-format success is guaranteed for a base model.
Freeze the final choice before testing rather than switching parsers when a
result disappoints. Report format errors and truncation frequency separately.

## File contract

Gold file, visible only to the scoring process:

```json
{"item_id":"task:test:0001","question":"...","answer":"...\n#### 42"}
```

Generation output, one row for EVERY expected item:

```json
{"item_id":"task:test:0001","generated_text":"#### 42","status":"ok","n_generated_tokens":3}
```

Allowed statuses are `ok`, `truncated`, `error`, and `format_error`. Failure rows
must still exist. The numeric scorer counts errors/truncations as failures and
refuses mismatched item sets. It does not silently select an earlier number from
the reasoning or repair invalid outputs. Decimal forms are normalized; fractions
and expression evaluation are not supported by this parser. If another parser
is scientifically warranted, declare it before final tests as a new protocol.

Use `scripts/score_numeric_answers.py` to create scored rows. Use
`scripts/summarize_downstream.py` for the fixed four-condition contrast set.
One score per item is expected; repeated samples must be aggregated by item first.
The confidence interval is conditional on the checkpoints, not uncertainty over
new training seeds or all modeling decisions.

## Local content audit

Fresh documents: raw silence, empty format, own thought, length-matched filler,
unrelated thought, and shuffled thought. Report both absolute evidence arms.
K=1 is primary for this training recipe; K=4/16 are horizon-generalization checks.
Keep reference and generated token counts separate. Independent local branches
are not a measurement of a full online sequence with accumulated thoughts.

On a smaller development subset, cross initial/trained writers and readers with
fixed prefixes. This localizes changes but is not a universal portability gate.
A supplied true target is a plumbing positive control, not semantic reasoning.

## Resource accounting

Save C at the data-matched point. A resource-matched C continuation is secondary
when feasible. Count all generation/scoring work and actual core-time; repeated
raw documents are visits, not new data. Do not call equal documents equal FLOPs
or call an untuned weak C run an optimal conventional-training baseline.
