# Short execution queue

Mark an item done only with an actual artifact/result. This is the active queue;
historical journal instructions are not. Stages are dependencies, not a reason
to idle: data preparation and evaluator work can proceed while compilation runs.

| Stage | Deliverable | Completion condition |
|---|---|---|
| P0 | Assets resolved | Inventory/model/data complete; choose a finite planning horizon before freeze, using the owner's weeks-long allowance |
| P1 | Released forward complete | Correct weights/ties/fingerprints and CPU/device forward pass |
| P2 | Full-cap FP32 backward pending | Long cache/replay passes, including the exact old BF16 failure; require finite positive backward |
| P3 | 32-rank C/R qualification active | Tiny mean/update/resume passes; C32 compiling, R32 gated on P2; require real full-model master updates |
| P4 | Resume and timing pending | Full-model resume and ≥10 steady updates per arm; primary accelerator TRAIN inference complete, secondary/local execution still required |
| P5 | Frozen first C/R pair | Model/data/format/source hashes; enough records; fixed terminal steps; fresh initial weights; actual deadline respected |
| P6 | Confirmation and complete evaluation | Unchanged second seed if feasible; C/R direct and thought results; local controls; actual resource accounting |
| P7 | Recoverable handoff | Verified checkpoints and destination; short result table; uncertainty; next experiment dictated by evidence |

Active driver: `train/neuron_driver.py`; use immutable source snapshots for running
jobs. CPU semantics remain in `weekend_thinking/core.py` and `reference.py`.
The immediate dependency is P2 backward, then full C/R updates, resume, and timing.

If P1–P4 do not qualify within the initial engineering allowance, report the exact
failure and do only verified baseline/evaluation or short diagnostic work. A long
unverified run is not a substitute for completing the missing implementation.
