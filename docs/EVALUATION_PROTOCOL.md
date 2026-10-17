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

Every final GSM8K launch must explicitly pass
`--compute fp32 --head-precision fp32`; the CLI's compute default remains BF16.
Lock these flags in the exact protocol. Local and secondary evaluators use FP32.

## Comparisons

Primary: R-thought minus C-thought, same inference policy and cap.
Required secondary: R-direct minus C-direct; R-thought minus R-direct;
R-thought minus R-empty-format. Include B0 in all three modes and each trained
arm minus B0 under the same policy, plus C-thought minus C-direct as context.
A same-policy gain caused only by C degradation is not an improvement above B0.
Report seeds separately before any clearly labeled aggregate summary.

Direct inference permits ordinary calculations and has no inserted thought phase.
Thought inference generates one thought then the answer, with no selection based
on the gold answer. Mode-specific
formatting is itself an intervention; matched empty-format/content controls help
separate that cost from thought utility. Use the same prompt for C/R within a mode.
Do not hardcode an answer-dependent continuation into the prompt.

A suggested starting prompt asks for the final result as `#### <number>`. Any
few-shot examples must come from the allowed development/training split and their
IDs must be recorded. No answer-format success is guaranteed for a base model.
Freeze the final choice before testing rather than switching parsers when a
result disappoints. Report format errors and truncation frequency separately.

The first-study held-development protocol is frozen in
[`validation/held_dev_protocol_20261017.json`](../validation/held_dev_protocol_20261017.json):
128 deterministic GSM8K TRAIN questions excluding all previous demonstrations
and interface items. Question membership was fixed before reading its TRAIN gold
column or scoring models. Use B0 once and both seeds' C/R checkpoints at fixed 64
and prospective terminal 256, each in direct, empty_format and one_thought mode.
The unchanged four-shot prompt, strict parser, greedy policy, FP32 compute/head,
1024 thought cap and 256 answer cap are bound there. Retain tokens, latency,
errors and truncations. The existing CLI needs `--qualification` to accept these
TRAIN IDs; this flag does not turn development scores into hardware qualification
or final-test evidence. Step64 never selects the primary checkpoint.

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
