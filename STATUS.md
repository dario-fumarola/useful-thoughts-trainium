# Current status

Updated October 21, 2026. [Full status history](docs/archive/STATUS_through_20261021.md).

## Running next: prediction-position population census

The owner requested a concrete fresh-distribution test. A read-only census will count the frozen arithmetic class at all bounded prediction positions, alongside document-level incidence and one hash-selected cut per source. It reads existing curator logs and identity-only cohort projections, with no model outcomes or new source acquisition. [Prospective plan](configs/reader_population_census.json). Bound: four CPU cores, 15 minutes, 256 MiB, zero model calls.

## Complete: selector optimization on cached TRAIN data

The original hard gate passes the practical criteria in **all six new orders**. The calibrated smooth gate passes in **four**, with two nearly inactive runs. Its prospective all-six criterion fails. Removing the cutoff did not stabilize learning here.

- Hard-gate gains: **0.42–0.46 nats per 16-token calculation span** versus B0; all six also beat constant, position and matched no-calculation controls.
- Ordinary-text costs remain positive but within tolerance. No natural-corpus gain or inference-efficiency claim follows.
- 4,096 FIT / 1,024 internal validation sources; all 24 terminal heads froze before scoring. This is internal development, not fresh confirmation.
- Independent NumPy reconstruction and source-free replay agree. All 1,979 pre-fit CPU checks passed.
- Four CPU cores; 215.82 seconds for fitting/readout and 13.19 for reconstruction; 9.01 MiB output. Zero new model calls or accelerator chip-hours. All workers closed successfully.
- Execution code: `10268da`; protocol: `435e81b1…`. No follow-on fit or accelerator job is running.

[Complete report](validation/selector_optimization_train_v1_complete_20261021/REPORT.md) · [All orders](validation/selector_optimization_train_v1_complete_20261021/all-orders.csv) · [Prospective design](configs/selector_optimization_train_v1.json) · [Verified release](validation/selector_optimization_train_v1_public_release_20261021.json)

## Earlier fresh-panel result remains unchanged

One frozen selector order gains **+0.457 nats per 16 calculation tokens** versus B0 [0.388, 0.526]; the other nearly closes its gate. The original joint criterion fails. The internal split and schedule changed in the new study, so it does not identify why that earlier order failed.

[Parent report](validation/causal_reader_owner_complete_20261021/REPORT.md) · [Verified public evidence](validation/causal_reader_owner_public_release_20261021.json)

V29 did not establish a better direct pretrained model; V30 closed its fixed predictive-reward branch. Improved pretraining, internalization and autonomous discovery remain open. Further CPT work needs a uniform next-token baseline and explicit compiler settings.

The installed compiler defaults to `--auto-cast=none`; flag omission does not establish BF16 execution. Existing numerical differences remain unresolved. [Read-only inspection](validation/executed_precision_review_20261021.json).

Production statistics now share an explicit-cluster helper in the active selector paths and generic paired-row API. All 120 study contrasts reproduce within 6.2e-16. Historical results and separate verification arithmetic remain unchanged. [Methods](docs/STATISTICS.md).
