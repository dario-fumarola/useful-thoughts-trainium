# Held-out local content audit

`train/evaluate_local.py` separates prefix-only note generation from FP32 reader
scoring. The same saved notes can be read by B0, C, or R without resampling or
selecting by a future token. This supports the smaller crossed writer/reader
development audit as well as same-model held-out evaluation.

The existing audited selection is
`local/data/finemath/fixed-local-evaluation-v1.ids.json`: first 512 dev and first
2,048 test records in the production file's established global hash order.
The complete selection and record-file hashes are validated before sharding.
`--split dev|test` chooses the list; `--shard-index i --num-shards n` takes
`global_ids[i::n]`. Neither the selection nor donor mapping depends on shard count.

The primary writer draws **one** independent note per document at temperature
0.7, with the recipe's 1,024-action cap and tokenizer-defined vocabulary mask.
Each document has a recorded RNG seed derived from `--seed` and its ID, so shard
order does not change draws. `--policy greedy` is a separate diagnostic. Actual
first stops are retained in action IDs/log probabilities, removed from content,
and rendered with the fixed close delimiter. Capped notes get no invented stop.

Example commands, after the final receipt exists:

```bash
source scripts/env.sh
NEURON_RT_VISIBLE_CORES=<allocated-core> python -m train.evaluate_local generate \
  --model-dir local/assets/qwen3-1.7b-base \
  --checkpoint <writer-terminal-checkpoint>/model.pt --arm thought --model-seed 1234 \
  --records local/data/finemath/production-500000.audit-v2.records.jsonl \
  --selection local/data/finemath/fixed-local-evaluation-v1.ids.json --split dev \
  --format local/resolved_format.json --policy sample --seed 1729 \
  --frozen-protocol <evaluation-freeze-v2.json> --output <new-writer-bundle>

NEURON_RT_VISIBLE_CORES=<allocated-core> python -m train.evaluate_local score \
  --model-dir local/assets/qwen3-1.7b-base \
  --checkpoint <reader-terminal-checkpoint>/model.pt --arm thought --model-seed 1234 \
  --records local/data/finemath/production-500000.audit-v2.records.jsonl \
  --selection local/data/finemath/fixed-local-evaluation-v1.ids.json --split dev \
  --format local/resolved_format.json --notes <all-writer-shard-bundles> \
  --frozen-protocol <evaluation-freeze-v2.json> --output <new-reader-scores>
```

Use `--arm B0` with no checkpoint or model seed for the unchanged base. `--seed`
is the note RNG seed; `--model-seed` is the trained checkpoint's seed. They have
different meanings. A crossed reader only changes the reader checkpoint/identity
and output path. All writer shard bundles must cover the global selection once,
with identical generation protocols and checkpoint identities.
The writer identity is explicit in the reader protocol and every score row;
global summaries reject mixed writer/reader/protocol tags. Crossed readers are
flagged as diagnostics and do not populate the same-model primary field.

Before scientific scoring, the exact protocol and both writer/reader identities
must pass the shared final-evaluation gates. Every target ID is restricted to the
scoring phase. Controls are constructed from the complete saved note bundle and
fixed RNG before target IDs are accessed. The six reader conditions are:

| Condition | Reader content |
| --- | --- |
| raw | Original prefix alone |
| empty | Fixed header/open/close, no content |
| own | The one saved note |
| filler | Fixed space-token pattern, repeated to recipient content length |
| shuffled | Fixed-seed permutation of the recipient's content tokens |
| other | Next global selected document's content, repeated/trimmed to recipient length |

An empty donor uses filler. Donor IDs, original lengths, exact token lists, and
length-match flags are saved in `controls.jsonl`. Filler, shuffled, and other
controls match the recipient's content length exactly. These are diagnostic
content controls; an untrained space filler is not an optimal trained pause
baseline, and shuffled notes need not remain grammatical.

Every document produces all six conditions at K=1/4/16. One causal all-16-token
forward per condition supplies the nested horizons, with raw full-vocabulary
reader likelihood. Rows retain absolute evidence, per-target log probabilities,
gains against raw, first target token ID/text, and failures. First-token classes
can therefore be reported without choosing new anchors. K=1 improvement alone
does not establish better mathematical reasoning or longer-horizon prediction.

Generation failures remain explicit error notes. Raw/empty readers can still
run; controls requiring a failed recipient or donor are errors. No numerical
likelihood is invented for an execution failure: counts remain complete and
means are labeled conditional on successful execution. The report explicitly
marks an incomplete audit as unable to establish unconditional primary utility.

CPU qualification accepts only TRAIN/dev selections and may declare smaller
caps/counts. Nine tiny-model tests verify all-token/full-logit parity, stochastic
shard invariance, exact control lengths/donors, complete failure rows, prefix-only
generation, bundle integrity, the final gate before record loading, and an actual
save/load/generate/score CLI round trip with a different reader checkpoint.
This is implementation evidence; released-model accelerator execution and final
test results remain separate requirements.

Released Qwen3-1.7B-Base FP32 CPU execution also completed two already-fixed DEV
items with one sampled eight-action note each, followed by all six reader
conditions at K=1/4/16: 36 successful scores, zero errors. This is a short-cap
plumbing qualification, not the 1,024-action scientific condition or an estimate
of thought usefulness. Raw notes/scores and their report hashes are retained in
`local/qualification/local-audit-cpu-{notes,scores}-001/`; the source revision for
this execution is `29a20a0`, before the subsequent host-read/provenance hardening.

Before the first scientific pair, the remaining substantive items are the actual
global experiment/evaluation freeze, immutable-source qualification bridge,
complete target-shape timing/resumption, and an accelerator local-reader smoke
check. The declared sequence-length-normalized policy loss remains a surrogate;
local self-reader gain, especially at whitespace K=1 targets, is not itself a
downstream capability result. Primary GSM8K C/R thought comparison, direct
contrasts, B0, empty-format costs, separate seeds, and actual resource use remain
required alongside this audit.

The summary includes the predeclared own-minus-raw/empty/filler/shuffled/other
document bootstrap contrasts at each horizon, with fixed 5,000 draws and seed 17.
A contrast gets no CI when either complete paired condition has an error; the
implementation never silently removes failed documents. Shard summaries describe
their shard only. For a global summary, concatenate all score rows and call
`summarize(rows, global_selection_ids)`; strict coverage rejects any missing,
duplicated, or extra item/condition/horizon.
