# Initial source and auxiliary gradient diagnosis

**Matching explicit source-loss coefficients does not preserve the optimizer's source-learning direction.** On the exact first batch of the matched study, the format-only work objective has a gradient about 26 times larger than the source objective. The trace gradient is smaller and nearly orthogonal to the source gradient, with a slightly positive dot product. This does not support a simple initial anti-alignment explanation for the trace loss.

This is an exploratory **TRAIN-only derivative calculation at unchanged B0**, not a new training run, held-out result or proof of the cause of the completed 128-update comparison. It makes **330 CPU forward/backward calls, zero optimizer steps and zero accelerator calls**. Complete parameter fingerprints before/after are identical.

## One fixed batch and unchanged objective

The batch is the first 272 source documents from the actual order-11 raw/format/trace histories; their identities match exactly. It contains **29 calculation documents**, or 10.7%, compared with 5.9% in the whole fitting cohort. This initial batch is not a representative corpus estimate. Its 143,344 original target positions keep coefficient 1/(272×527). Each calculation-row work loss is averaged separately and divided by 272; ordinary rows contribute zero work loss. The format branch has 29 EOS targets; the trace branch has 1,058 work/EOS targets. Both counts match the original native first update.

The computation uses standard CPU FP32 gradients and streamed FP64 norms/dots. These differ from native BF16 gradient computation; this is not a native gradient-parity qualification.

| Initial quantity | Source only | Source + format work | Source + trace work |
|---|---:|---:|---:|
| Source gradient norm | 1.411065 | 1.411065 | 1.411065 |
| Auxiliary gradient norm | 0 | 36.762979 | 2.267196 |
| Source/auxiliary cosine | — | −0.013306 | +0.001008 |
| Combined gradient norm | 1.411065 | 36.771282 | 2.671654 |
| Global clip multiplier at norm 1 | 0.708685 | 0.027195 | 0.374300 |
| Relative source alignment of zero-moment adaptive gradient direction | 1.000000 | 0.156745 | 0.707216 |

For context, original native initial pre-clipping norms are **1.296053 / 36.788700 / 2.590437**. Their comparison is descriptive: FP32 CPU and native BF16 gradients are not equal.

## What the directional calculation means

For gradient g, the counterfactual gradient component of the first bias-corrected AdamW direction is proportional to c·g/(|c·g|+epsilon), where c is the specified global clipping multiplier and epsilon is 1e-8. The common learning rate is omitted; the common weight-decay term cancels in differences between directions from the same parameters. The table reports the source gradient's inner product with this adaptive gradient component, relative to source-only. **No parameter update is applied.** These ratios are not measured loss reductions, percentages of final learning or predictions of future validation performance.

A common gradient rescaling largely cancels in this zero-moment adaptive expression, so the observed difference must not be attributed to clipping alone. The auxiliary changes coordinate directions as well as the combined norm. A rule that only removes negative raw dot products would leave this slightly positive CPU trace vector unchanged at this point; it is not an established repair.

The result illustrates why a format-only control can be a weak optimization comparator despite matched scalar coefficients, and why the matched ordinary-training control remains essential. It does not establish that initial geometry uniquely causes the eventual negative trace-versus-raw result. Later moments, curvature, batches, precision and generalization are untested here. The original failed auxiliary-learning criterion and positive source-mixture criterion stay unchanged.

## Evidence and execution

The source/plan/worker hashes, exact batch identities, every scalar and both parameter fingerprints are in [the readout](readout.json). [Worker](CPU_gradient_worker.py) · [Fixed plan](prospective-plan.json) · [Measured runtime decision](measured-runtime.json). Actual complete runtime is **1,053.6 seconds** on 24 CPU cores. The first local prototype assumed optimizer constants were protocol fields; it was stopped after an incomplete source pass, then restarted with constants checked in the unchanged qualified native optimizer constructor. Original logs and the first plan are retained. No original model, source or native training code was modified.

This provides source and computation for review, rather than claiming an independent replay of unpublished full gradient arrays or an external replication. The original [fresh matched study](../scaffold_learning_complete_20261022/REPORT.md) remains the model-quality result.
