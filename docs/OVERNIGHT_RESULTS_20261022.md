# Overnight results — October 22, 2026

Updated 09:15 UTC; the owner deadline is 12:00 UTC / 08:00 Eastern. This interim synthesis names unfinished work explicitly.

**Ordinary continued pretraining works on the intended mathematical text. The added computation-supervision objective does not improve that learning, and mathematical specialization costs English prediction.** The source-mixture comparison and larger CPU verification are still running.

| Completed question | Result | Evidence |
|---|---|---|
| Can uniform next-token training beat B0? | Yes: +0.014481 [0.013903, 0.015058] nats/token on 4,096 fresh FineMath windows. DEV chooses among all four fixed completed settings before confirmation. | [Baseline report](../validation/uniform_lm_baseline_complete_20261022/REPORT.md) |
| Does verified post-document computation improve direct learning? | No under this recipe: −0.000944 / −0.000908 nats/token against same-data ordinary training, negative intervals in both orders. Trace beats format-only supervision but loses to the qualified ordinary reference. | [All six models and 26 comparisons](../validation/scaffold_learning_complete_20261022/REPORT.md) |
| Does the mathematical training preserve English prediction? | All ten models lose to B0 on the separate 1,024-document educational-English panel. The selected ordinary reference loses 0.008835 nats/token. | [Full retention diagnostic](../validation/general_web_retention_complete_20261022/REPORT.md) |
| Does searching existing calculations justify a writer update? | The inspected TRAIN-only fixed-reader check finds only +0.0266 nats/span hindsight best-of-eight gain over greedy; no actor is trained. | [Bounded saved-score diagnostic](../validation/writer_feedback_opportunity_20261022/REPORT.md) |

## What remains in progress

- Two fixed 50/50 FineMath/educational-English ordinary-training models, fresh confirmation in both domains, and comparison with the qualified all-mathematical reference. Their protocols froze before the computation-supervision outcomes.
- A 1,024-document standard Hugging Face FP32 numerical cross-check of all eight matched-study predictors, using 8,192 CPU forwards and no accelerator reservations. The sample size was chosen only from measured runtime and deadline, before those scores.
- A supplemental common-panel comparison of all twelve fixed trained models, preserving every parent outcome and using a declared equal-domain diagnostic target.
- Exact, losslessly compressed model artifacts for every trained arm, including controls and unsuccessful conditions. Ten models have passed actual tensor-bit round trips and uploaded to a draft release; public publication and anonymous restoration are pending.

## Interpretation

The baseline changes the interpretation of another failed auxiliary objective. This time the direct comparator learns successfully, and the computation treatment is a small relative loss against it. The source-matched raw arm is the causal recipe control; the independently qualified baseline is an additional practical reference with a different source mix and batch size.

All models use a mature 1.7B starting checkpoint and short continued training. These are not onset experiments, autonomous discovery or a broad evaluation of language ability. Teacher-forced token likelihood is distinct from free-running arithmetic accuracy. The post-document writer sees the original text before predicting work, so supplied prefix-derived labels do not make its training behavior prospective.

The earlier frozen-selector finding remains intact: generated calculation can help targeted real-text continuations, but the tested interface loses on its unenriched natural endpoint. A useful inference ensemble is separate from improved pretrained weights.

## Evidence and accounting

The [baseline](../validation/uniform_lm_baseline_public_release_20261022.json), [matched computation study](../validation/scaffold_learning_public_release_20261022.json) and [English diagnostic](../validation/general_web_retention_public_release_20261022.json) have public original-score archives, anonymous hash checks and standalone NumPy replay. Those verify numerical consistency rather than repeat training or establish an external replication. Original source text stays outside the source-free numerical archives.

The native work retains full original model and optimizer states. Standard CPU checks are separate: aggregate paired gains agree on the initial fixed subsets, while individual token log probabilities can differ by more than one nat. The larger check measures that uncertainty directly rather than assuming tokenwise equality.

Actual assigned costs so far: ordinary baseline **16.607**, matched computation study **32.537**, separate English diagnostic **0.502 chip-hours**. Remaining studies have their own fixed ceilings. These costs include their actual attempts and are not whole-node billing or a claim of full 16-chip occupancy. The node has 16 physical chips; training uses independent two-chip groups, with memory and retained full-state exports limiting safe concurrency. The 35 GiB free-storage and 512 GiB available-host-memory floors remain unchanged.
