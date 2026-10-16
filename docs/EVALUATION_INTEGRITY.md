# Final evaluation artifact bindings

Qualification uses only GSM8K TRAIN, ARC TRAIN/validation, synthetic LAMBADA,
or local dev items. It bypasses final-model gates and establishes no test result.
Final evaluators require **both** an exact protocol match and a verified final
model identity. A protocol-only receipt or an arbitrary `model.pt` is insufficient.

`train/evaluation_integrity.py` is the common implementation. Its receipt schema is:

```json
{
  "schema": "evaluation-freeze-v2",
  "status": "frozen",
  "frozen_utc": "<UTC time before final test access>",
  "training_spec": "<whole self-hashed scripts/freeze_experiment.py result as an object>",
  "protocols": ["<exact proposed evaluator protocol objects>"],
  "models": ["<exact final_model_identity(...) objects>"],
  "scoring_manifests": ["<canonical numeric-scoring-v1 objects>"]
}
```

Freeze the training recipe and terminal exposure before training. After training,
lock the resulting terminal checkpoint inventory and evaluation protocols before
official test generation/scoring. These are separate operations: a terminal
checkpoint content hash cannot be known before training. Record their actual
times; these validators do not certify that a claimed historical timestamp is true.

`final_model_identity(checkpoint, assets, training_spec, arm=..., seed=...)`
constructs an inventory entry, using `asset_provenance(model_dir)` for `assets`.
For `vanilla`/`thought`, `checkpoint` must be the terminal `model.pt` in a completed
`neuron-v1` checkpoint. It verifies the frozen specification, terminal step,
topology, arm, seed, training order, source exposure and every manifest-listed
model/optimizer file's size and SHA256. Training order is recomputed from record
headers without parsing held-out futures. The final load also checks the model
dictionary's internal step/run/spec before applying weights. B0 requires no
checkpoint or training seed and matches the frozen released asset manifest.

Final commands use `--arm B0|vanilla|thought` and `--seed <training seed>`
(no seed for B0). The local evaluator calls the latter `--model-seed`, keeping its
stochastic note seed separate. Crossed writer/reader evaluations retain distinct
model identities. A saved writer bundle must belong to the same final inventory.

All final protocols bind `evaluation_dependencies()`: evaluator entrypoints,
their imported decoder/scoring helpers, backend, core, parser/statistics,
pretrained loader, runtime compatibility helper, artifact/order utilities and
canonical numeric projection code. Thus changing `evaluate.py` invalidates a
likelihood protocol that imports its decoder, even if the likelihood entrypoint
itself did not change. No final freeze should be constructed while any of these
files is still being edited.

After training configuration freeze, this command prepares canonical GSM8K test
questions and separate gold without printing answer values:

```bash
source scripts/env.sh
python scripts/prepare_numeric_scoring.py \
  --frozen-experiment local/frozen-experiment.json \
  --canonical-source local/data/benchmark_exclusion/sources/openai--gsm8k/main/test-00000-of-00001.parquet \
  --output local/evaluation/canonical-gsm8k-test
```

Its input must match pinned revision
`740312add88f781978c0658806c59bc2815b9866` and the exact raw Parquet hash. Add
the resulting `scoring-manifest.json` object to the final evaluation receipt.
The numeric scorer requires `--frozen-protocol` and `--scoring-manifest` and
rechecks the canonical projection, gold hash, parser/dependency map and original
complete generation run receipts before parsing/scoring gold. Repeat
`--predictions <original shard>/predictions.jsonl` for multiple shards; all must
use the same frozen model and protocol and cover every canonical item exactly
once. Failed/truncated outputs remain in the denominator. TRAIN-only checks now
require explicit `--qualification`; this flag rejects official test IDs.

The tests use tiny model-file and Parquet fixtures. They establish CPU control
flow and artifact rejection, **not** accelerator qualification or final outcomes.
No official test gold was projected or evaluated while implementing this change.
