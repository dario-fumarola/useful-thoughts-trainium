# Matched post-document computation supervision

The prespecified supervised-learning criterion **fails**. Adding verified computation supervision makes direct prediction **worse than the same-data ordinary-training control in both source orders**, by 0.000944 and 0.000908 nats per token, with negative confidence intervals. It improves on format-only supervision, but also loses to the independently qualified ordinary-training reference. All six trained models improve over untouched B0. Ordinary continued pretraining works here; this auxiliary computation objective does not improve it.

These statements concern direct prediction of original text without generated work at evaluation. This is a small, consistent relative loss, rather than the broad base-model regression observed in earlier specialized recipes. Beating the weaker format-only control does not establish a computation-training benefit against the stronger matched raw control.

## Fresh natural confirmation

All six terminal update-128 checkpoints were fixed before any prediction on the 4,096 fresh confirmation documents. The reference cosine schedule and learning rate were selected in the separate completed ordinary baseline, before releasing these fits. No checkpoint or order is discarded.

| Model | Full-window NLL, lower is better | Final 16-token log probability, higher is better |
|---|---:|---:|
| B0 | 1.451454 | -20.606478 |
| scaffold_format-seed11 | 1.438331 | -20.609742 |
| scaffold_format-seed12 | 1.438177 | -20.614429 |
| scaffold_raw-seed11 | 1.436603 | -20.591240 |
| scaffold_raw-seed12 | 1.436599 | -20.593151 |
| scaffold_trace-seed11 | 1.437547 | -20.610876 |
| scaffold_trace-seed12 | 1.437507 | -20.613871 |
| uniform_reference | 1.436665 | -20.593448 |

The following gains use all **527 original target tokens**, in nats per token. Positive means better prediction. Each bracket is a paired 95% document-bootstrap interval.

| Source order | Trace versus matched raw | Trace versus matched format | Trace versus B0 | Trace versus qualified uniform reference |
|---|---:|---:|---:|---:|
| 11 | -0.000944 [-0.001048, -0.000845] | +0.000784 [+0.000649, +0.000920] | +0.013908 [+0.013438, +0.014442] | -0.000882 [-0.001058, -0.000712] |
| 12 | -0.000908 [-0.001004, -0.000813] | +0.000670 [+0.000539, +0.000809] | +0.013947 [+0.013462, +0.014491] | -0.000842 [-0.001059, -0.000634] |

The declared primary criterion requires positive lower bounds for trace versus raw and format in each source order, and versus B0 in both orders. The qualified uniform reference remains an additional practical comparison: a relative benefit against matched controls is not automatically superiority over a strong ordinary-training recipe.

An exploratory decomposition into eight fixed context-position bins finds negative trace-versus-raw gains in every bin in both orders. Their individual intervals are negative too; these are descriptive, non-simultaneous intervals on the completed panel, not new success criteria. The loss is not confined to the short-context positions that account for most of the original baseline's improvement. All six decompositions, including the favorable format comparisons, are retained: [order 11 versus raw](context-trace-11-versus-scaffold_raw-seed11.json), [order 12 versus raw](context-trace-12-versus-scaffold_raw-seed12.json), and the corresponding format/reference files in this directory. Weighted bins reconstruct each original aggregate exactly. This does not identify which training dynamics cause the loss.

## Final continuations remain a separate endpoint

These are gains over the entire final **16-token span**, in nats per span, on the same fresh natural documents. They do not substitute for the primary full-window endpoint.

| Source order | Trace versus matched raw | Trace versus matched format | Trace versus B0 | Trace versus qualified uniform reference |
|---|---:|---:|---:|---:|
| 11 | -0.019636 [-0.029297, -0.009996] | -0.001134 [-0.012754, +0.010312] | -0.004398 [-0.024316, +0.015382] | -0.017428 [-0.028762, -0.005902] |
| 12 | -0.020720 [-0.030358, -0.011644] | +0.000558 [-0.010934, +0.012020] | -0.007393 [-0.027239, +0.012784] | -0.020423 [-0.033191, -0.007149] |

The previously inspected 1,024-question calculation panel is reported completely in [all paired contrasts](all-paired-contrasts.csv) and [absolute scores](absolute-scores.csv). It is a diagnostic, not fresh confirmation. Every evaluation is teacher-forced original-text prediction; this is not free-running numerical accuracy.

## What is matched

Each model starts afresh from the same released Qwen3-1.7B-Base. All arms receive the same 34,816 original 528-token windows once: 32,768 ordinary windows plus 2,048 calculation windows. Each model has **18,348,032 source targets**, 128 updates, batch 272, eight ranks, microbatch 2 and 17 accumulated microsteps. Source orders 11 and 12 are crossed with raw, format and verified-trace arms. All arms use 13 warmup updates, peak LR 1e-5 and cosine decay to one-tenth peak, AdamW with clipping 1, betas (0.9, 0.95), epsilon 1e-8 and weight decay 0.1. Parameters and gradients use BF16; masters, moments and normalized CE use FP32; compiler auto-cast is explicitly disabled.

Every original source target has coefficient **1/(272×527)**. Complete original text precedes all optional work, so original-text prediction is never conditioned on that work. Calculation rows append a supplied header: the raw arm has no work labels, the format arm predicts only EOS, and the trace arm predicts verified steps followed by EOS. Work is independently normalized per example before the global batch mean; ordinary rows have zero work loss. Work-label totals are **0 /2,048 /73,210**. This matches explicit source coefficients and padded execution shapes, not every gradient magnitude, optimization difficulty or information budget. Global clipping can couple source and work gradients.

The work labels are derived from the first 512 source tokens only. Because this is a **post-document** scaffold, the trained writer itself can see all 528 original tokens when predicting work. Consequently this tests supervised auxiliary learning, not prospective writer behavior, autonomous discovery or proof of internalized computation. Those stronger claims require separate experiments.

The additional calculation sources are matched across all three arms. The qualified uniform reference uses only its original 32,768 ordinary windows and a different global batch 256; the matched raw arm is the source/exposure control. The independent ordinary reference remains useful for judging practical predictive value, while its different corpus means it cannot alone identify the auxiliary objective's effect.

## Source and numerical qualification

Fresh natural source inventory supplies 1,024 qualification and 4,096 confirmation windows. Every window matches the pinned original Parquet record. All 2,112 additional calculation rows (2,048 fitting,64 qualification) were independently checked against original records, including 5,722 exact rational arithmetic steps. Fitting, qualification and fresh confirmation are source-separated under the declared rules. Sources are not proven absent from B0's original pretraining, and the eligible-document-weighted FineMath frame is not all language.

Native qualification passes unchanged initial-loss, actual-LR, single-reducer, finite FP32-master, replica and full-state checks. Its CPU/native initial-loss error is 0.000425795nats/token. Restoring the complete state after update 2 reproduces updates 3/4, every terminal model tensor and all optimizer shards exactly. Qualification and replay consume 1.529985 assigned chip-hours. These establish the measured execution contract, not scientific success.

## Standard CPU forward diagnostic

A separately executed standard Hugging Face FP32 check scores all eight frozen models on 32 identity-fixed confirmation documents, using 256 CPU forward passes. The largest difference in mean paired gains from native execution is **0.000457 nats/token**; for the two trace-versus-raw contrasts it is **0.000051 / 0.000163**. Both CPU trace-versus-raw point estimates remain negative on this subset. Maximum single-document paired-gain disagreement is 0.003658 nats/token, while the largest individual token log-probability difference is **1.2671 nats**. Aggregate agreement therefore does not establish tokenwise equality. This is a numerical diagnostic on inspected documents, not a new independent confirmation or external training replication. [All paired checks and the frozen plan](../scaffold_learning_CPU_forward_check_20261022.json).

A larger fixed-subset CPU comparison was specified before this check completed. Its document count is selected only from measured runtime and the remaining deadline, before its prediction results; its outcome will be reported separately.

## Training behavior and resources

The following are descriptive training diagnostics. Clipping counts compare the recorded pre-clipping norm with 1; source/work losses are from the terminal training batch, not held-out quality.

| Model | Mean update seconds after first 3 | Updates with norm above1 | Terminal source loss | Terminal work loss |
|---|---:|---:|---:|---:|
| scaffold_format-seed11 | 68.39 | 33/128 | 1.348493 | 0.000000 |
| scaffold_format-seed12 | 69.11 | 36/128 | 1.419724 | 0.000001 |
| scaffold_raw-seed11 | 69.32 | 10/128 | 1.346649 | 0.000000 |
| scaffold_raw-seed12 | 68.38 | 11/128 | 1.418348 | 0.000000 |
| scaffold_trace-seed11 | 68.34 | 26/128 | 1.348018 | 0.000305 |
| scaffold_trace-seed12 | 68.40 | 25/128 | 1.419441 | 0.001968 |

Actual assigned reservations total **32.536880 chip-hours**: training 29.815051, scoring 1.191845, qualification and all other attempts 1.529985. Before long fitting, the initial 32-chip-hour planning value was amended to a hard 40 after actual same-shape timing projected 39.448608. No outcomes were used to revise the source, labels, models or endpoints. Every phase shares that global ceiling. At most three full trainers run concurrently on independent two-chip groups, leaving memory for checkpoint exports. Existing hardware only; free-disk 35GiB and host-available 512GiB floors remain enforced. Assigned costs are not the whole 16-chip-node invoice, utilization percentage or FLOP efficiency.

## Statistics and public replay

Each comparison resamples **unique source documents**, carrying all 527 scores together: 5,000 percentile-bootstrap draws, seed 17, central production helper. The two source orders use one pretrained starting checkpoint and the same source corpus; these are optimization repetitions, not independent pretrained systems. Intervals describe document sampling conditional on fixed checkpoints, not the full uncertainty of method development or training.

The source-free archive preserves every original native FP32 token score exactly, all six 128-update histories, source identities and fixed protocols without source text or source token IDs. A standalone NumPy reconstruction verifies all 26 paired contrasts and all declared decisions. Seven integrity fixtures reject altered verdicts, intervals, score/history hashes, units and missing controls. The reconstruction is within-repository numerical verification, not an external model replication or a recheck of omitted source text.

Training numerical source: `53adb6eb0942e527448118ac7361a766f0f242f0`. Evaluation source: `b55349f`. All original models and optimizer states remain retained outside Git. See [complete readout](readout.json), [prospective design](prospective-design.json), [resource amendment](resource-amendment.json) and [numerical reconstruction](independent-numerical-replay.json).
