# Current status

Updated October 22, 2026, 07:35 UTC. [History](docs/archive/STATUS_through_20261021.md).

## Ordinary pretraining reference passes

Fresh confirmation on 4,096 FineMath windows: B0 full-window NLL **1.443585**, selected uniform-LM model **1.429104**. Gain **0.014481 nats/token [0.013903, 0.015058]**, approximately **1.44% lower perplexity**. Final 16-token gain is separately **0.032717 nats [0.013546, 0.052275]**. The older calculation diagnostic is +0.021967 [−0.010545, +0.054757] nats/span, not fresh confirmation.

All four fixed settings finish 128 updates; all improve DEV. DEV selects warmup-cosine / peak LR 1e-5 after all terminals freeze, before confirmation. Each sees the same 32,768 source windows once, with equal coefficients on all 17,268,736 targets. Exact native recovery and original-score NumPy replay pass. Actual **16.607 assigned chip-hours**, including failed reservations, within 18. This is a FineMath adaptation result, not thought-training evidence or four training replications.

[Report and all settings](validation/uniform_lm_baseline_complete_20261022/REPORT.md) · [Design](configs/uniform_lm_baseline.json).

## Cross-domain diagnostic and bounded replay test

All four ordinary baselines regress on the fixed educational-English panel. The selected cosine model loses **0.008835 nats/token [0.008259, 0.009388]**; FineMath success is a domain result. [Diagnostic](validation/general_web_retention_baseline_20261022.json).

A fixed 50/50 FineMath/FineWeb-Edu training comparison is prepared in two source orders, with 2,048 fresh mathematical and 1,024 fresh English confirmation windows. The exact qualified numerical recipe stays fixed; source replay is the intervention. Both protocols freeze before scaffold outcomes. Execution follows the main study only if measured timing, all full-state storage and the additional **7.5 assigned-chip-hour** ceiling permit. [Preparation](validation/uniform_mixture_preparation_20261022/REPORT.md).

## Matched supervised-learning study is running

Three fresh training arms—raw, format-only and verified computation—across two source orders. All share 34,816 original windows, equal source coefficients, the selected cosine schedule and 128 updates. Optional work follows original text; labels are derived from the preceding prefix. This isolates a post-hoc learning scaffold, not an autonomous writer. All six terminals freeze before fresh natural confirmation; B0 and the qualified ordinary model remain absolute references.

Native loss, accumulation, LR, reducer, masters, replica and exact recovery pass; qualification cost **1.530 assigned chip-hours**. All three first-order fits and full-state exports are complete; all three second-order fits have launched on independent two-chip groups. Existing hardware only, up to three full trainers concurrently to preserve host-memory and checkpoint-export headroom. Measured complete projection 39.449, prospective total ceiling **40 assigned chip-hours**. Every phase and failed reservation counts. [Overnight plan](docs/OVERNIGHT_PLAN_20261022.md).

All **2,074 CPU tests**, smoke and compilation pass. CPU checks are separate from native qualification. All original sources, models and optimizer states remain retained; disk and host-memory floors stay 35 and 512 GiB.

## Earlier findings remain intact

Frozen selective inference improves all six fresh calculation panels; five beat every additional control. Every natural-text endpoint loses. A simple no-calculation constant ensemble improves natural spans by +0.0244 [0.0155, 0.0340]. These findings are unchanged by the new baseline.

[Reader report](validation/reader_fresh_confirmation_complete_20261021/REPORT.md) · [Verified release](validation/reader_fresh_confirmation_public_release_20261021.json).

The owner authorized continuous work on the existing node until 08:00 Eastern / 12:00 UTC. Current objectives are complete scientific readout, numerical replay and publication, including unfavorable results. No capacity purchase or stack change.
