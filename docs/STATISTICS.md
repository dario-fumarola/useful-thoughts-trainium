# Paired statistical comparisons

Production numerical resampling lives in `weekend_thinking/stats.py::cluster_bootstrap`. The generic paired-row wrapper and active causal-reader/internal-selector readouts call it. It takes paired item-by-contrast differences, explicit cluster IDs and a named independent unit. Shared draws preserve pairing across contrasts.

Resample independent clusters with replacement and carry all their items together. Estimate the item-weighted mean: sampled cluster sums divided by sampled item counts. This matters when cluster sizes differ; averaging cluster means is a different estimand. Percentile intervals use the frozen draw count and random stream for the study.

| Study structure | Independent resampling unit | Score contribution |
|---|---|---|
| Continued-pretraining full windows | Unique source document | Mean of all 527 original target log probabilities |
| Continued-pretraining final spans | Unique source document | Sum of the original final 16 token log probabilities |
| Declared two-domain target | Unique source document within each fixed stratum | Equal-weight sum of domain means and independent domain bootstrap draws |
| Current selector continuation panels | Unique source document | Complete original 16-token log probability |
| Related counterfactual program variants | Original map/program family | All paired variants in the sampled family |
| Multiple windows from one document | Original source document | All windows in that source cluster |

The selector wrappers require complete unique source IDs and retain their fixed 512/4,096-source contracts. Generic cluster resampling also supports repeated cluster IDs for related items. The paired-row API rejects missing, duplicated or differently clustered paired items. Reports record units, item/cluster counts, draw counts and RNG provenance. The generic compatibility wrapper calls its legacy `document_id` field a cluster; new callers should name its scientific meaning explicitly.

A confidence interval describes source-sampling uncertainty conditional on the fitted checkpoints. It is not uncertainty over independently trained models or adaptive experiment selection. Head-training orders are optimization repetitions over fixed predictors. Multiple reported intervals are unadjusted unless a study says otherwise.

Historical frozen producers and independent numerical checkers can retain separate code. They are not silently rerun under a new helper. In particular, reconstructing the production arithmetic independently is a useful verification boundary.

The October 21 consolidation preserves RNG streams and all 120 internal selector-study mean/interval contrasts within 6.2e-16, without refitting a head, opening completed DEV or changing the original readout. [Parity receipt](../validation/production_bootstrap_refactor_20261021.json). Tests compare unequal-size clusters with explicit resampling, reject item mismatches, preserve RNG continuation, and demonstrate how treating related tokens as independent falsely narrows intervals.

The full post-refactor CPU suite passes 1,992 checks, plus smoke and compilation. This does not qualify accelerator execution.

The October 22 continued-pretraining comparisons keep full-window nats/token and final-span nats/16 tokens separate. Mathematical and English strata use independent fixed streams (seeds 17 and 19) and 5,000 draws. Their declared 50/50 target is not an estimate of natural language prevalence. Backend comparisons pair the same documents, models and target positions before forming CPU-minus-native differences; individual token predictions are not independent resampling units. The final CPU suite has 2,074 passing tests, separate from native qualification.
