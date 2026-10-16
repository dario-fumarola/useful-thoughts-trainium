# Short execution queue

Mark an item done only with an actual artifact/result. This is the active queue;
historical journal instructions are not. Stages are dependencies, not a reason
to idle: data preparation and evaluator work can proceed while compilation runs.

| Stage | Deliverable | Completion condition |
|---|---|---|
| P0 | Instance/assets inventory and explicit deadline | Existing environment identified; local paths resolved; model/data/checkpoint availability not guessed |
| P1 | Ordinary-causal model load and one-step forward | Exact released weights, tokenizer/ties, targets, and CPU/device evidence agree within declared tolerances |
| P2 | Cached generation + scoring + policy replay | Prefix-only generation; fixed shapes; correct stopping; sampled/replayed logps agree; no full-vocabulary host loop |
| P3 | C/R micro-step and distributed update | Reference parity; one reducer; correct accumulation; rank-distinct examples/RNG; finite master-weight updates |
| P4 | Resume + complete timing + evaluator | Same continuation within declared tolerance; ≥10 complete timing steps; a real downstream item runs on accelerator |
| P5 | Frozen first C/R pair | Model/data/format/source hashes; enough records; fixed terminal steps; fresh initial weights; actual deadline respected |
| P6 | Confirmation and complete evaluation | Unchanged second seed if feasible; C/R direct and thought results; local controls; actual resource accounting |
| P7 | Recoverable handoff | Verified checkpoints and destination; short result table; uncertainty; next experiment dictated by evidence |

Primary module to implement: `train/neuron_driver.py`.
Reusable semantics: `weekend_thinking/core.py` and `reference.py`.
Runtime/cache examples: named original source in `docs/NEURON_IMPLEMENTATION.md`.

If P1–P4 do not qualify within the initial engineering allowance, report the exact
failure and do only verified baseline/evaluation or short diagnostic work. A long
unverified run is not a substitute for completing the missing implementation.
