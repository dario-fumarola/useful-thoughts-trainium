# Next research work

Uniform-LM source and CPU preparation are complete: all 38,912 windows independently check, the one-loss path passes dense-gradient/accumulation tests, and concrete qualification/four-setting protocols pass real-data cursor checks. [Report](validation/uniform_lm_preparation_20261021/REPORT.md) · [Design](configs/uniform_lm_baseline.json).

1. Qualify actual native training at batch 256 / accumulation 8: initial loss, one reducer, FP32 master update, actual optimizer LR after warmup/decay, replica agreement and full-state restore. Measure the complete workload against the proposed finite budget before fitting.
2. Compare the four declared ordinary-training settings on the same 32,768 sources, source order and 17.269 million tokens. Freeze all terminal checkpoints before DEV scoring; report every result.
3. Freeze the selected recipe/checkpoint before scoring the untouched 4,096-source CONFIRM cohort. Compare full-window NLL with B0 and report suffix/calculation diagnostics separately.

The completed selector study retains six targeted gains, one unresolved position-control comparison, every natural loss and the positive no-calculation ensemble. A competent uniform-NTP reference addresses the next pretraining question; it does not retroactively repair those results. [Selector report](validation/reader_fresh_confirmation_complete_20261021/REPORT.md).
