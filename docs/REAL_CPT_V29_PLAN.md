# V29: computation supervision with an identical clean prediction objective

This is CPU preparation, **not a training release**. V28 stays closed. All
science models would start from unchanged Qwen3-1.7B-Base, not its specialized
permutation checkpoints. No RL or model-size change is proposed.

The question is: **does supervised prefix-computable work improve predictions
of real continuations, while preserving ordinary text prediction?** Training
benefit without written work and extra inference benefit remain separate claims.

## What changes from the earlier real-text studies

| Reference | Relevant result or limitation | V29 intervention |
|---|---|---|
| V10 | Raw/prior/post CPT failed its full-span likelihood gates; all active labels shared one mean. | Give clean text and auxiliary targets explicit, separate losses. |
| V15 | Strong arithmetic assistance, without source LM supervision; ordinary prediction regressed. | Preserve a complete clean-text forward on every document. |
| V17 | Source-balanced computation improved arithmetic answers. Direct continuation LL improved over its trained control, but remained below B0. Source loss excluded the clean continuation boundary; work and future shared a task mean. | Train the real continuation in the clean view as well, keep its auxiliary coefficient independent of work length, and retain B0 as an absolute reference. |
| V28 | Near-perfect generated execution within covered lengths, failed direct transfer and severe forgetting in all task-only arms. | Use real source text and a shared LM objective. Keep V28 as a positive control rather than extending it. |

The old V17 direct-LL differences are +0.333/+0.507 nats per16-token span
versus its trained direct controls, and −0.864/−0.802 versus untouched B0.
These are reconstructed historical development comparisons, not fresh evidence.
V29 is a new recipe, not a single-factor ablation of V17.

## Source and views

Use existing, provenance-recorded FineMath assets. TRAIN contains the same5,120
verified calculation documents used previously, plus15,360 ordinary TRAIN
documents. This deliberately enriches calculations to25%; it is not a natural
estimate of how often thinking should occur. Every document supplies its original
native prefix and16 actual continuation tokens. Nothing is converted into QA.

The common clean view is `prefix + continuation16`, with CE at every real
transition after the first token. It has no artificial EOS, inserted markup, or
teacher computation. Its IDs, labels, masks and coefficient are identical in all
arms. Ordinary documents receive only this view.

Calculation documents additionally receive one independent causal view:

| Arm | Auxiliary sequence after the original prefix |
|---|---|
| Raw | Original continuation16 |
| Format | Given `Calculation:` opening, learned empty `Answer:` boundary, continuation16 |
| Computation | Same given opening, verified full calculation, learned `Answer:` boundary, continuation16 |

The exact interpreter receives **only prefix IDs**. The complete trace includes
the root calculation; this is an ordinary supervised computation scaffold, not a
claim that partial information was sufficient. Source-RHS verification selected
the equation corpus, so selection remains privileged supervised curation.

The supplied opening has no loss. The first native `Answer:` token is a learned
stop action; the whole answer boundary is supervised. No synthetic EOS replaces
an ordinary source continuation. All native traces plus stopping fit128 actions
and the640-position training shape, or are rejected before any model evaluation.

## Objective and proposed exposure

For each source document:

\[
L=L_{\mathrm{clean}}+\mathbf1_{\mathrm{calculation}}
  (0.5L_{\mathrm{aux\ continuation}}+1.0L_{\mathrm{work/boundary}}).
\]

Each loss is a mean over its own tokens in that example, followed by the fixed
source-batch mean. The raw arm has no work component. Work weight1.0 is chosen
before learning or new held outcomes: do not recreate final-target dilution or
assume a weak auxiliary signal will teach the procedure. This is one proposed
weight, not a sweep. The original CPU packing snapshot used a placeholder0.1;
no learning used it and changing the coefficient changes no packed data.

The raw control is **weighted ordinary continued NTP**, with extra weight on
the selected real continuations. It retains the same continuation coefficient
as the other arms. It is not uniform whole-corpus NTP. The format control tests
augmentation beyond this raw recipe. Neither control matches semantic label
information or gradient difficulty; report those differences.

Proposed: two data-order seeds, two passes, global32 source documents and8
calculation auxiliary views per update,1,280 updates, terminal checkpoint primary.
Every source occurs exactly once/pass, with one calculation and three ordinary
documents per group. Separate source and auxiliary forwards accumulate before
**one** optimizer step/reduction. Auxiliary batch means need a0.25 factor to
preserve the displayed per-source-document objective.

Use the existing BF16/FP32-master runtime, constant1e−5 LR, AdamW and checkpoint
cleanup/recovery path. World8, source microbatch4 and auxiliary microbatch1 are
candidates requiring qualification. Shared padded shapes do not establish equal
FLOPs. Report native source tokens, targets by component, allocated positions,
actual reservations, generated tokens and latency separately.

## Evaluation to freeze before learning

The CPU cohort contains768 fresh arithmetic documents and4,096 ordinary DEV
documents, excluding documented old panels. Arithmetic additionally excludes
previous ordered exact-value expressions, canonical pages and source content.
These are source-TRAIN-derived development data, not official TEST or evidence
of novelty relative to Qwen pretraining. Existing near-duplicate filtering is
probabilistic; semantic duplicates remain possible.

1. **Training benefit:** native direct full16-token LL on the arithmetic panel,
   computation versus both trained controls, with B0 beside every contrast.
   An advantage over degraded controls alone is not a new best starting model.
2. **Inference benefit:** generated-work versus the same model's raw and empty
   paths, and the matched raw model. Report full-span LL and complete numerical
   answers separately; neither metric substitutes for the other.
3. **Content controls:** intact unrelated generated work and supplied correct or
   corrupted work. Oracle traces include the answer and are plumbing controls.
   They cannot establish writer attainability or autonomous improvement.
4. **Ordinary prediction:** per-document mean CE over all527 native transitions,
   and continuation16 separately, on the ordinary panel. Always include B0.

Resample documents, not token positions or sampled notes. Keep training-order
uncertainty distinct from conditional paired intervals. Preserve all errors,
caps and unsupported answers. Writer inputs cannot access held continuations.
No prompt/checkpoint/weight selection on the new panel. No fresh GSM8K/TEST claim.

Exact scientific success criteria and power assumptions remain to be frozen
after data readback and before a learning release. Historical V17 paired raw-LL
SDs1.99–2.48 nats suggest768 documents can resolve several tenths of a nat, not
arbitrarily small effects. A true effect exactly at a point-estimate threshold
cannot provide80% passage probability. Accuracy power requires its own analysis.

## Remaining release conditions

Independently read back the datasets and CPU packing; qualify the actual
two-forward full-token loss, native/master update, memory behavior and restart;
then set a measured reservation/storage budget and review the final protocol.
No accelerator job or long training run is released by this document.

A positive result would support a supervised calculation-augmented CPT recipe
on mathematical text. General-language gains, identical downstream post-training,
teacher withdrawal and autonomous improvement would still require later tests.
