# Useful Thoughts on Trainium

We study when intermediate computation improves prediction, whether it can improve learning, and how useful computation could eventually be discovered autonomously. Ordinary continued pretraining now supplies a qualified reference; the completed matched computation-supervision comparison does not beat ordinary training.

[Complete overnight release](https://github.com/dario-fumarola/useful-thoughts-trainium/releases/tag/overnight-results-20261022) · [Overnight findings](docs/OVERNIGHT_RESULTS_20261022.md) · [All twelve trained models and restore tools](docs/MODELS_20261022.md) · [All models on common panels](validation/overnight_common_panel_complete_20261022/REPORT.md)

## Ordinary source mixing improves both fresh domains

Both fixed 50/50 FineMath/educational-English training orders pass fresh confirmation. Direct-prediction gains over B0 are **0.011346 / 0.011204 nats/token on mathematics**, and **0.008550 / 0.008756 on English**, with positive paired document intervals in every case. The declared equal-domain target improves over the qualified all-math reference by **0.006578 / 0.006610 nats/token**, also with positive intervals. Each model trains once on 17.3 million original targets, with the same qualified optimizer schedule and no thoughts.

The all-math reference remains better on mathematics alone; mixing sources reverses its English regression. This establishes a stronger ordinary-training reference for the two declared frames, not a thought-training effect or a result on all language. Actual additional cost: **6.291 assigned chip-hours**, within 7.5. [Both source orders, absolute scores and all comparisons](validation/uniform_mixture_complete_20261022/REPORT.md).

## Uniform next-token baseline: fresh confirmation passes

The selected ordinary-training model improves full-window FineMath NLL from **1.443585 to 1.429104**, a gain of **0.014481 nats/token [0.013903, 0.015058]** on 4,096 untouched confirmation documents. This is about **1.44% lower perplexity** on the declared source population. Its final 16-token spans improve by **0.032717 nats [0.013546, 0.052275]**. This is a domain-adaptation reference, not an effect of thought supervision.

All four fixed LR/schedule settings improve DEV. The lowest full-window DEV loss selects **warmup-cosine, peak LR 1e-5**, after all four terminal checkpoints freeze and before confirmation. Each recipe trains once on the same 32,768 original windows, **17,268,736 equally weighted targets**, with 128 updates. Native accumulation and exact full-state recovery pass. Actual assigned cost, including all unsuccessful reservations, is **16.607 chip-hours** within 18; this is not the whole-node invoice. Four settings are not four training replications.

A separate fixed educational-English diagnostic finds regression in all four settings. The FineMath-selected model loses **0.008835 nats/token [0.008259, 0.009388]** there. The domain confirmation pass stands; it does not establish a general predictive improvement. The completed panel also finds negative intervals for every matched raw/format/computation model; all ten trained models lose to B0. [Complete cross-domain diagnostic](validation/general_web_retention_complete_20261022/REPORT.md). The completed source-mixture comparison above addresses that measured limitation with fresh confirmation and the same qualified objective and schedule. [Frozen preparation](validation/uniform_mixture_preparation_20261022/REPORT.md).

[Complete report and every setting](validation/uniform_lm_baseline_complete_20261022/REPORT.md) · [Fixed design](configs/uniform_lm_baseline.json) · [Public numerical evidence](https://github.com/dario-fumarola/useful-thoughts-trainium/releases/tag/uniform-lm-baseline-20261022) ([download/replay verified](validation/uniform_lm_baseline_public_release_20261022.json))

## Matched computation supervision: ordinary training wins

All six models completed 128 updates and froze before prediction on **4,096 fresh FineMath documents**. Verified-computation supervision loses to the same-data ordinary-training arm by **0.000944 [0.000845, 0.001048]** and **0.000908 [0.000813, 0.001004] nats/token** in the two source orders. It beats the format-only arms, but also loses to the independently qualified ordinary-training reference. Every arm improves over untouched B0. The declared supervised-learning criterion **fails**.

This isolates a post-document auxiliary training objective: original source targets precede optional work and retain equal coefficients. It does not test autonomous thought discovery. The full-window loss and the final-continuation loss against matched raw training are negative in both orders. All original scores, all six histories and the 26-contrast NumPy reconstruction are retained. Standard HF FP32 forwards on 1,024 fixed documents also find negative trace-minus-raw gains in both orders. [Independent CPU check](validation/scaffold_standard_CPU_verification_20261022/REPORT.md). Actual assigned cost is **32.537 chip-hours** within 40, including qualification and every attempt.

[Complete matched report](validation/scaffold_learning_complete_20261022/REPORT.md) · [Verified original-score evidence](https://github.com/dario-fumarola/useful-thoughts-trainium/releases/tag/scaffold-learning-20261022) · [Prospective overnight design](docs/OVERNIGHT_PLAN_20261022.md).

## Fresh confirmation: calculation gains, natural-text losses

All six frozen selectors improve **1,024 fresh calculation continuations by 0.421–0.506 nats per 16 tokens** over B0. Five also beat every additional control; order 45's comparison with the calculation position schedule spans zero, so the all-six calculation criterion fails.

On **4,096 unenriched natural windows**, every selector loses **0.020–0.052 nats per span**, with negative intervals. A fitted constant ensemble without calculations instead gains **0.0244 [0.0155, 0.0340]**. The natural prediction criterion fails in every order. Targeted computation benefit repeats; net natural predictive value does not follow.

Every head, control and source selection was fixed before fresh model outcomes. Both backbones remain unchanged. Independent reconstruction and public numerical replay agree. Actual cost: **1.782 reserved chip-hours including qualification**, below 4.25. All **2,004 CPU checks**, smoke and compilation pass.

[Full report and all orders](validation/reader_fresh_confirmation_complete_20261021/REPORT.md) · [Public evidence](https://github.com/dario-fumarola/useful-thoughts-trainium/releases/tag/reader-fresh-confirmation-20261021) ([download/replay verified](validation/reader_fresh_confirmation_public_release_20261021.json)) · [Prospective design](configs/reader_fresh_confirmation_design.json)

The strict verified arithmetic class occupies **0.001664% of prediction positions** in the existing source-clean frame. That is not every useful reasoning event or all FineMath; the natural result above is measured directly rather than obtained by reweighting the arithmetic panel. [Census](validation/reader_population_census_20261021/REPORT.md).

## Preceding development: repeated internal selection, unsuccessful smooth-gate repair

A CPU-only comparison fitted 24 selectors on existing TRAIN caches: hard/smooth gates, calculation/no-calculation ensembles, six fixed document orders. The original hard gate passes the practical criteria in **6/6 orders**, gaining **0.42–0.46 nats per 16-token calculation continuation** over the unchanged base. The calibrated smooth gate passes in **4/6**, with two effectively inactive runs. Removing the cutoff did not improve reliability in this comparison.

This source-hash split is internal development evidence: 4,096 fitting and 1,024 validation documents. It does not replace the earlier fresh-panel result or confirm improved pretraining. Ordinary prediction still has a small cost. Independent reconstruction and source-free numerical replay agree. Fitting/readout and reconstruction used about four minutes on four CPU cores, with no new model or accelerator calls.

[Complete comparison](validation/selector_optimization_train_v1_complete_20261021/REPORT.md) · [All orders](validation/selector_optimization_train_v1_complete_20261021/all-orders.csv) · [Prospective design](configs/selector_optimization_train_v1.json) · [Public numerical evidence](https://github.com/dario-fumarola/useful-thoughts-trainium/releases/tag/selector-optimization-train-20261021)

## What the earlier evidence establishes

| Question | Evidence and boundary |
|---|---|
| Can explicit computation help? | Strong arithmetic and in-range permutation-program results. Length extrapolation and direct transfer remain limited. |
| Can a causal selector use calculations on fresh real text? | All six fixed heads improve on B0 in the fresh calculation confirmation; five beat all controls. Every head loses on the fresh natural panel. The earlier parent joint failure remains unchanged. |
| Does intermediate supervision improve learning? | Earlier document-QA evidence remains developmental. The fresh, two-order post-document computation study loses to matched ordinary training in both orders. |
| Does it improve continued pretraining? | V29 fails its primary direct-prediction criterion. Its “raw” control reweights selected continuations, so it is not a uniform next-token baseline. |
| Can useful computation be discovered autonomously? | Unanswered here. V30's fixed predictive-reward branch closed without RL after failing its prerequisite. |

Both predictors in the selective-reader experiments remain frozen. A better inference interface is distinct from better model weights, internalized computation, natural-corpus improvement or computational efficiency.

[Parent selective-reader report](validation/causal_reader_owner_complete_20261021/REPORT.md) · [Verified public evidence](https://github.com/dario-fumarola/useful-thoughts-trainium/releases/tag/causal-reader-owner-approved-20261021) · [V29 report](validation/real_cpt_scaffold_v29_complete_20261020/REPORT.md) · [V30 report](validation/candidate_utility_v30_complete_20261021/REPORT.md)

## Working with the project

Read [STATUS.md](STATUS.md), [NEXT_STEPS.md](NEXT_STEPS.md) and [AGENTS.md](AGENTS.md). The [roadmap](docs/RESEARCH_ROADMAP.md) separates inference benefit, learning benefit and autonomous discovery. Completed reports and published outcomes stay intact while new recipes can be iterated transparently.

Use the existing environment (`source scripts/env.sh`). Run `bash scripts/run_checks.sh` after relevant source changes. CPU checks establish numerical and causal invariants; native execution needs its own qualification. Large models, source data and caches stay outside Git.

The active selector paths and generic paired-row API share [one cluster-bootstrap implementation](weekend_thinking/stats.py), with explicit source IDs and units. [Statistical methods](docs/STATISTICS.md). Historical independent replays retain separate arithmetic.

The installed compiler's default is no auto-casting; a BF16-by-default explanation of the prior qualification differences is unsupported. [Read-only precision inspection](validation/executed_precision_review_20261021.json). Existing deviations remain unresolved. Future protocols should specify precision explicitly.

[Full previous landing page and experiment history](docs/archive/README_through_20261021.md).

All twelve model downloads were anonymously hash-checked and restored exactly, and every restored model passes standard HF load/forward checks. The standalone tools also pass in a fresh CPU-only environment without Neuron or XLA. [Model catalog](docs/MODELS_20261022.md) · [Clean CPU receipt](validation/trained_model_clean_CPU_environment_20261022.json).
