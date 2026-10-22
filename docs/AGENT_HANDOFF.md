# Overnight handoff — October 22, 2026, 11:38 UTC

The owner asked for continuous work through **08:00 Eastern / 12:00 UTC**. Keep working until then. The fixed scientific program, bounded diagnosis and public artifacts are complete. Only the final time-scoped usage snapshot and deadline handoff remain.

[Results](OVERNIGHT_RESULTS_20261022.md) · [All twelve models](MODELS_20261022.md) · [Current status](../STATUS.md).

## What is complete

- Ordinary next-token CPT improves fresh FineMath prediction by **0.014481 [0.013903, 0.015058] nats/token**. All four fixed settings finish before DEV chooses cosine / peak LR 1e-5. English retention regresses.
- Matched computation supervision loses to same-data ordinary training in both source orders: **−0.000944 / −0.000908 nats/token**, negative intervals. All six models improve on B0; the auxiliary criterion fails. A standard CPU check on 1,024 fixed documents preserves the negative comparison in both orders.
- Both fixed source-mixture orders improve fresh Math (**+0.011346 / +0.011204**) and English (**+0.008550 / +0.008756 nats/token**), with positive intervals. Their declared 50/50 target beats the qualified all-math reference. All-math remains better on mathematics alone.
- All twelve models are evaluated on common panels. This is supplemental, without refitting or selecting another checkpoint.
- All six numerical archives and the mixture CPU supplement pass anonymous download, checksums and NumPy replay. All twelve public model assets restore exact BF16 values and pass standard HF load/forward checks. An isolated CPU-only environment also restores and uses the selected model without Neuron or XLA; the Trainium stack is unchanged.
- The fixed first-batch TRAIN-only gradient diagnosis completes **330 CPU forward/backward calls**, zero updates and zero accelerator calls. Source / format-auxiliary / trace-auxiliary norms are **1.411 / 36.763 / 2.267**. Trace is nearly orthogonal and slightly positively aligned with source, weakening a simple initial anti-alignment account. The adaptive-direction calculation is a surrogate, not measured learning or a unique explanation for the completed 128-update result. [All values and limits](../validation/source_auxiliary_gradient_20261022/REPORT.md).

The research conclusion is a healthy ordinary-training reference, a negative result for this supervised auxiliary recipe, and a positive ordinary source-mixture result. No autonomous thought discovery or internalization is established.

## Costs, readiness and retained state

[Resource reconciliation](../validation/overnight_resource_summary_20261022.json) totals **56.899836 assigned chip-hours**, within every separate ceiling. All **85 native attempts** are closed; all eight nonzero exits are earlier baseline qualifications and remain recorded. Twelve models finish **1,536 updates** and **213,700,608 original target exposures** across **51,200 unique training documents** shared across arms. Assigned reservations are not the whole-node invoice.

All **2,074 CPU tests**, smoke and compilation pass; production numerical code has not changed since that check. At 11:40, all twelve original model files and all 96 optimizer-rank files are rehashed: **108 of 108 match their original COMPLETE manifests**. The 298,697,673,828-byte read-only scan takes 54.6 seconds. See `validation/trained_model_native_state_integrity_20261022.json`. At 11:35, 133 current local links pass. Latest pushed result commit before this handoff: `feaf961`.

Full native state paths are cataloged in [the state registry](../validation/trained_model_artifacts_20261022/local-state-registry.json). Model copies are on existing NVMe; optimizer snapshots remain on local RAM/NVMe. **Do not power off the node or discard snapshots.** Public weights and tokenizer are verified remotely, but local optimizer states are not a remote backup. Floors stay **35 GiB free storage / 512 GiB available host memory**.

## Last deadline actions

1. At approximately **11:58 UTC**, run `source scripts/env.sh`, then `USE_TORCH_XLA=0 python local/diagnostics/finalize_node_usage_20261022.py`. It writes `validation/overnight_node_usage_20261022`, preserving compressed original telemetry, the analysis window and missing observations. Preview in `local/diagnostics/node-usage-preview-v2-20261022` is qualified and its figure reviewed.
2. Peak observed occupancy is **8 of 16 chips**, not full saturation. The monitor emits only valid NRT periods and stopped emitting after native work ended near 10:44. Missing hardware counters stay unknown; owned-stage reservation clocks separately establish when this program holds chips. Do not promote conditional busy averages to whole-night utilization.
3. Monitor PID **2741371**, session **16559**, is the only remaining owned observer. After freezing the final snapshot, SIGINT its exact PID to close its `neuron-monitor` child cleanly. This stops observation, not the machine.
4. Review the final figure and JSON; add actual usage/coverage numbers and final links to the summary and status. Record current idle device inventory, storage and state readiness. Check current local links and clean/pushed main. Commit and push the completed handoff before 12:00.
5. Remain active through **12:00 UTC**, then give concise outcomes and results/model links. No new training campaign near the deadline.

## Execution context

Repo `/home/ubuntu/dario`; `source scripts/env.sh`; CPU `USE_TORCH_XLA=0`; `rg` is unavailable. Existing node only; preserve sources and completed outcomes. No new capacity, accelerator-stack changes, IAM/network changes, paid teachers or official TEST. Do not spawn agents.

Private controls remain under `local/diagnostics`: `uniform-lm-native-v1`, `scaffold-learning-overnight-v1`, `general-web-retention-20261022`, `uniform-mixture-20261022`, `overnight-common-panels-20261022`, `scaffold-standard-CPU-verification-20261022`, `uniform-mixture-CPU-check-20261022`, and `source-aux-gradient-20261022`. All scientific workers are finished. Immutable numerical workers and the model-delta codec remain unchanged.

[Detailed earlier handoff](archive/AGENT_HANDOFF_20261022_1118UTC.md).
