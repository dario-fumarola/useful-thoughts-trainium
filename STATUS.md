# Current status

Completed October 22, 2026, 11:56 UTC. [Overnight findings](docs/OVERNIGHT_RESULTS_20261022.md) · [Models](docs/MODELS_20261022.md) · [History](docs/archive/STATUS_through_20261021.md).

## Ordinary training works; the computation auxiliary does not improve it

The development-selected ordinary model improves fresh FineMath prediction by **0.014481 [0.013903, 0.015058] nats/token** on 4,096 documents. Its separate final 16-token gain is **0.032717 nats/span**. All four prescribed settings finish before development selection; confirmation follows freezing the choice. The selected model loses **0.008835 nats/token** on a separate educational-English panel. [Baseline](validation/uniform_lm_baseline_complete_20261022/REPORT.md) · [Retention](validation/general_web_retention_complete_20261022/REPORT.md).

All six matched raw/format/computation models finish. Computation loses to same-data ordinary training by **0.000944 / 0.000908 nats/token**, negative intervals in both source orders. It beats format-only supervision but also loses to the qualified ordinary reference. All arms improve over B0. The declared auxiliary-learning criterion **fails**. Independent standard CPU forwards preserve the negative trace-versus-raw result. [Matched study](validation/scaffold_learning_complete_20261022/REPORT.md) · [1,024-document CPU check](validation/scaffold_standard_CPU_verification_20261022/REPORT.md).

## Source mixing improves both declared fresh domains

Both fixed ordinary-training orders pass: Math gains **0.011346 / 0.011204**, English gains **0.008550 / 0.008756 nats/token**, all positive intervals. Their declared 50/50 target beats the qualified all-math reference by **0.006578 / 0.006610**. The reference remains stronger on Math alone. Each mixed-source system uses one model at inference. [Source-mixture confirmation](validation/uniform_mixture_complete_20261022/REPORT.md).

The common-panel comparison retains all twelve models. Every mathematical-only model loses on English; both mixed models improve on B0 in both domains. It is supplemental, without new fitting or model selection. [Every model on common panels](validation/overnight_common_panel_complete_20261022/REPORT.md).

## Evidence and artifacts are published

All twelve public model artifacts restore exact BF16 values. Anonymous download, checksums, standard HF loading and finite CPU forwards pass for every model; a clean CPU-only environment additionally restores and loads the selected ordinary model without Neuron or XLA. All six numerical archives and the mixture CPU supplement download and replay successfully. Original failures, sources, weights and optimizer snapshots remain retained.

Actual per-study assigned reservations total **56.899836 chip-hours**, including all original failed qualification attempts, within every separate ceiling. This is not the whole-node invoice. All **85 native attempts are closed**, and current hardware inventory shows **0 of 16 devices occupied**. All **2,074 CPU tests**, smoke and compilation pass. [Resource reconciliation](validation/overnight_resource_summary_20261022.json) · [Exact model artifacts](docs/MODELS_20261022.md).

The main overnight program and bounded TRAIN-only gradient diagnosis are complete. No parameter update was applied in the diagnosis; it does not identify a unique cause of the held-out result. [Gradient follow-up](validation/source_auxiliary_gradient_20261022/REPORT.md). Final synthesis and usage accounting are complete before the owner's 08:00 Eastern / 12:00 UTC deadline. Storage and available-host-memory floors remain 35 and 512 GiB.

The earlier selective-reader result remains: targeted calculation gains repeat, every declared natural endpoint loses. Neither the new ordinary-training gains nor the failed auxiliary recipe establishes autonomous thought discovery or internalized computation.

Observed peak occupancy is **8 of 16 chips**. The 02:53–11:55 UTC window schedules **28.90%** of available chip time to this program; counters cover **84.90%**, with **26.96%** conditional busy time. [Usage and gaps](validation/overnight_node_usage_20261022/REPORT.md) · [Final readiness](validation/overnight_deadline_readiness_20261022.json). The observer is closed; the node stays on for retained local optimizer states.
