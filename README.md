# Useful Thoughts on Trainium

We study when intermediate computation improves prediction, whether it can improve learning, and how useful computation could eventually be discovered autonomously. Current empirical work focuses on selective inference over frozen models. Improved pretrained weights and autonomous discovery remain open.

## Latest result: repeated internal selection, unsuccessful smooth-gate repair

A CPU-only comparison fitted 24 selectors on existing TRAIN caches: hard/smooth gates, calculation/no-calculation ensembles, six fixed document orders. The original hard gate passes the practical criteria in **6/6 orders**, gaining **0.42–0.46 nats per 16-token calculation continuation** over the unchanged base. The calibrated smooth gate passes in **4/6**, with two effectively inactive runs. Removing the cutoff did not improve reliability in this comparison.

This source-hash split is internal development evidence: 4,096 fitting and 1,024 validation documents. It does not replace the earlier fresh-panel result or confirm improved pretraining. Ordinary prediction still has a small cost. Independent reconstruction and source-free numerical replay agree. Fitting/readout and reconstruction used about four minutes on four CPU cores, with no new model or accelerator calls.

[Complete comparison](validation/selector_optimization_train_v1_complete_20261021/REPORT.md) · [All orders](validation/selector_optimization_train_v1_complete_20261021/all-orders.csv) · [Prospective design](configs/selector_optimization_train_v1.json)

## What the earlier evidence establishes

| Question | Evidence and boundary |
|---|---|
| Can explicit computation help? | Strong arithmetic and in-range permutation-program results. Length extrapolation and direct transfer remain limited. |
| Can a causal selector use calculations on fresh real text? | One parent order gains **+0.457 nats per 16 tokens** [0.388, 0.526], beating the declared controls. The second order nearly closes its gate. The original joint criterion fails. |
| Does intermediate supervision improve learning? | One developmental document-QA signal survives a coefficient-matched control. Fresh confirmation remains absent. |
| Does it improve continued pretraining? | V29 fails its primary direct-prediction criterion. Its “raw” control reweights selected continuations, so it is not a uniform next-token baseline. |
| Can useful computation be discovered autonomously? | Unanswered here. V30's fixed predictive-reward branch closed without RL after failing its prerequisite. |

Both predictors in the selective-reader experiments remain frozen. A better inference interface is distinct from better model weights, internalized computation, natural-corpus improvement or computational efficiency.

[Parent selective-reader report](validation/causal_reader_owner_complete_20261021/REPORT.md) · [Verified public evidence](https://github.com/dario-fumarola/useful-thoughts-trainium/releases/tag/causal-reader-owner-approved-20261021) · [V29 report](validation/real_cpt_scaffold_v29_complete_20261020/REPORT.md) · [V30 report](validation/candidate_utility_v30_complete_20261021/REPORT.md)

## Working with the project

Read [STATUS.md](STATUS.md), [NEXT_STEPS.md](NEXT_STEPS.md) and [AGENTS.md](AGENTS.md). The [roadmap](docs/RESEARCH_ROADMAP.md) separates inference benefit, learning benefit and autonomous discovery. Completed reports and published outcomes stay intact while new recipes can be iterated transparently.

Use the existing environment (`source scripts/env.sh`). Run `bash scripts/run_checks.sh` after relevant source changes. CPU checks establish numerical and causal invariants; native execution needs its own qualification. Large models, source data and caches stay outside Git.

The installed compiler's default is no auto-casting; a BF16-by-default explanation of the prior qualification differences is unsupported. [Read-only precision inspection](validation/executed_precision_review_20261021.json). Existing deviations remain unresolved. Future protocols should specify precision explicitly.

[Full previous landing page and experiment history](docs/archive/README_through_20261021.md).
