# Prediction positions, source availability and the next confirmation

The verified arithmetic class is rare at natural prediction positions in the pinned source frame. **19,552 of 1,175,123,958 bounded positions qualify: 0.001664%, about one in 60,100.** This is a census of the existing source-clean 16-shard FineMath TRAIN frame, not a prevalence estimate for all FineMath or every kind of useful reasoning.

## The denominator matters

| Quantity | All source-clean documents | Sources unused by the previous reader cohorts |
|---|---:|---:|
| Documents | 1,379,348 | 1,366,004 |
| Documents with a complete 512-past/16-future window | 998,583 | 985,239 |
| Possible bounded prediction cuts | 1,175,123,958 | 1,149,991,897 |
| Verified full-prefix arithmetic boundaries | 19,552 | 7,850 |
| Position-weighted verified incidence | **0.001664%** | **0.000683%** |
| Document-balanced expected random-cut incidence | 0.001698% | 0.000709% |
| Documents containing such a boundary, among complete-window documents | 1.175% | 0.511% |
| Verified cuts from one fixed hash position per eligible document | 21 | 12 |

Each denominator includes every bounded native-token cut with 512 past and 16 future tokens, including positions without an equals sign or rejected by serialization checks. It is not conditioned on a correct calculation. Verified numerators use the original curator's source-supported, at-least-two-operation class before additional auxiliary-packing and expression-novelty restrictions.

Source identity and token length fix a prospective hash position independently of candidate labels. Both the exact position-weighted incidence and the document-balanced expected random-cut incidence are reported; they are different estimands. These are exact counts in a fixed finite frame, so no sampling interval is attached. The old source cleaning and deliberate removal of previously used calculation-heavy cohorts limit transport to a new corpus. In particular, depletion lowers incidence in the unused frame.

The pre-label-stage diagnostic passes 56,910 positions, but includes native whole-window serialization checks that can depend on future text. It is not a qualified causal invocation policy. The strict verified class also does not cover all mathematical computation or all places where generated notes could help.

## Consequence for interpretation

The internal selector study's break-even arithmetic prevalence was approximately 3.5–11.6% under its two-panel transport calculation. The census incidence is thousands of times lower. This comparison is a warning about transport, **not a measured whole-corpus prediction loss**: the enriched calculation and selected ordinary panels need not represent this frame, and useful effects outside the strict verified class are possible.

A fresh evaluation should therefore measure its unenriched natural panel directly. Reweighting a balanced arithmetic benchmark and calling that natural prediction would answer the wrong question. Calculation-specific benefit, ordinary-text tolerance and net prediction improvement remain distinct outcomes.

## A fresh test is feasible with existing data

The source inventory finds **1,048 residual examples** under the original one-equation-per-document selection, full-prefix/packing, source-separation and previous-expression rules. It does not require more downloaded shards. [Inventory](source-inventory.json) and [fixed rules](../../configs/reader_confirmation_source_inventory.json).

The next source plan selects **4,096 natural documents/cuts using identity and length only**, then **1,024 strict calculation documents** excluding every natural source. There are 1,046 strict candidates after that exclusion. All 5,120 selected windows have been reconstructed and independently checked against the pinned Parquet rows, including arithmetic support on every calculation example. No selected row was replaced. [Preparation](fresh-source-preparation.json), [independent source check](fresh-source-check.json), [prospective plan](../../configs/reader_fresh_confirmation_sources.json).

This prepares two separate endpoints: fresh calculation-mechanism confirmation and direct full-span prediction on an unenriched natural panel. Existing hard-gate heads and controls stay frozen. Freshness is relative to this project's source cohorts; novelty relative to base-model pretraining is unproven. A finite numerical/runtime qualification must precede any new predictor execution on these panels. No fresh predictive result exists at this source-preparation stage.

## Verification and resources

The census hashes both complete original curation logs and checks 1,379,348 distinct source identities. A separate source-free recount reproduces the counts, estimands and identity-only position selection. It does not re-tokenize or semantically re-curate omitted source text. [Census](census.json), [replay](projection-replay.json), [script](../../scripts/replay_reader_population.py), [prospective census plan](../../configs/reader_population_census.json).

The census took **41.36 seconds**, strict source inventory **11.09 seconds**, source preparation **35.59 seconds**, and independent pinned-Parquet checks **42.81 seconds**, each bounded to four CPU cores. These stages made **zero model calls and used zero accelerator chip-hours**. Source text, tokens, weights and hidden features are omitted from the public count projection.
