# Current status

Updated October 22, 2026, 10:35 UTC. [History](docs/archive/STATUS_through_20261021.md).

## Ordinary pretraining reference passes

Fresh confirmation on 4,096 FineMath windows: B0 full-window NLL **1.443585**, selected uniform-LM model **1.429104**. Gain **0.014481 nats/token [0.013903, 0.015058]**, approximately **1.44% lower perplexity**. Final 16-token gain is separately **0.032717 nats [0.013546, 0.052275]**. The older calculation diagnostic is +0.021967 [−0.010545, +0.054757] nats/span, not fresh confirmation.

All four fixed settings finish 128 updates; all improve DEV. DEV selects warmup-cosine / peak LR 1e-5 after all terminals freeze, before confirmation. Each sees the same 32,768 source windows once, with equal coefficients on all 17,268,736 targets. Exact native recovery and original-score NumPy replay pass. Actual **16.607 assigned chip-hours**, including failed reservations, within 18. This is a FineMath adaptation result, not thought-training evidence or four training replications.

[Report and all settings](validation/uniform_lm_baseline_complete_20261022/REPORT.md) · [Design](configs/uniform_lm_baseline.json).

## Cross-domain diagnostic and bounded replay test

All four ordinary baselines and all six matched scaffold models regress on the fixed educational-English panel. The selected cosine model loses **0.008835 nats/token [0.008259, 0.009388]**; FineMath success is a domain result. [Complete diagnostic](validation/general_web_retention_complete_20261022/REPORT.md).

The fixed 50/50 source-mixture study **passes in both orders**: fresh Math gains **0.011346 / 0.011204**, English gains **0.008550 / 0.008756 nats/token**, all positive intervals. It beats the qualified all-math reference on the declared equal-domain target by **0.006578 / 0.006610**. The reference remains better on Math alone. Same qualified schedule, no thoughts; both protocols froze before scaffold outcomes. Actual **6.291 assigned chip-hours** within 7.5. [Complete result](validation/uniform_mixture_complete_20261022/REPORT.md).

## Matched supervised-learning study is complete: primary criterion fails

All six raw, format-only and verified-computation models finished 128 updates and froze before fresh confirmation. Computation supervision loses to matched ordinary training by **0.000944 [0.000845, 0.001048] / 0.000908 [0.000813, 0.001004] nats/token**. It beats format-only training but also loses to the qualified ordinary reference; all trained arms improve over B0. The two-order supervised-learning criterion **fails**. Final-continuation comparisons with raw training also lose. This is supervised post-document learning, not autonomous discovery.

All 26 original-score comparisons replay exactly; all six histories and models are retained. Actual **32.537 assigned chip-hours**, within 40, including qualification and scoring. [Full matched report](validation/scaffold_learning_complete_20261022/REPORT.md).

All twelve fixed models are now trained and retained. The source-mixture study passes as above. The complete English readout also loses in all six arms. The 1,024-document standard CPU comparison, common-panel readout, mixture CPU check and exact-model publication are in progress.

All **2,074 CPU tests**, smoke and compilation pass. CPU checks are separate from native qualification. All original sources, models and optimizer states remain retained; disk and host-memory floors stay 35 and 512 GiB.

## Earlier findings remain intact

Frozen selective inference improves all six fresh calculation panels; five beat every additional control. Every natural-text endpoint loses. A simple no-calculation constant ensemble improves natural spans by +0.0244 [0.0155, 0.0340]. These findings are unchanged by the new baseline.

[Reader report](validation/reader_fresh_confirmation_complete_20261021/REPORT.md) · [Verified release](validation/reader_fresh_confirmation_public_release_20261021.json).

The owner authorized continuous work on the existing node until 08:00 Eastern / 12:00 UTC. Current objectives are complete scientific readout, numerical replay and publication, including unfavorable results. No capacity purchase or stack change.
