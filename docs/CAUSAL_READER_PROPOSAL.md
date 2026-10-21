# Selective use of calculation: proposal, execution held

The next question is whether a causal predictor can use a calculation where it
helps while retaining an actual unchanged B0 prediction where it does not.
This is a bounded inference mechanism proposal. It does not train better base
weights, demonstrate internalization, or reopen V30's closed reward branch.
No released-model feature extraction or experiment-head training has run.
CPU source feasibility is [closed without a model release](../validation/causal_reader_feasibility_complete_20261021.json):
the fixed12-shard pool supplies6,095eligible calculations versus6,672required.
The independent saved-pool recount agrees. No smaller cohort or relaxed exclusion
replaces the frozen design; this does not establish a global source shortage.

The [specified draft](../validation/causal_reader_gate_design_20261021.json)
contains the prediction contract, proposed head settings, controls and release
conditions. These are reviewable defaults, not a frozen execution protocol.

## What changes in prediction

Keep B0 and the first V29 computation checkpoint frozen. B0 sees only the original
prefix and previously observed continuation tokens. The computation model writes
one greedy note from that prefix, then predicts through the existing calculation
interface. At each target position, combine the two normalized distributions:

$$
p_t=(1-g_t)p_{B0,t}+g_t p_{\mathrm{calculation},t}.
$$

A small logistic linear head computes a score $s_t$, then uses the fixed rule
$g_t=\max(0,(s_t-0.01)/0.99)$. This supplies an exact zero-contribution region,
used identically in training and prediction. Inputs are detached hidden states
that predict the current token, distribution entropies, note length/stop status
and continuation position. The current or later target token, its probability,
arithmetic correctness and annotated numeral locations cannot be gate inputs.
Teacher forcing supplies preceding tokens only. Perturbing the current/future
tokens must leave the corresponding gate inputs unchanged.

When the selector chooses zero, return B0's prediction directly. A separate
disabled-system branch also skips the calculation. These identities still need
actual implementation checks. Active positive mixing has **no universal
no-regression guarantee**. A wrong gate can still damage prediction; fallback
availability is not proof that it will be selected correctly.

## How learning starts

Train only the head against full 16-token continuation likelihood, using actual
greedy notes, including wrong, empty, unsupported and capped outputs. The two
models remain fixed. The target probabilities enter the loss, never the gate's
input array. Start with zero head weights and a bias giving a score of 0.0199,
hence a 0.01 calculation contribution inside the learning region. That gives the
head a likelihood gradient without requiring writer policy optimization. A closed
gate has zero local gradient and may stall; this is a failure risk, not permission
to change thresholds after outcomes. Exact zero contribution also gives no writer
reward variation. This proposal does not solve autonomous writer bootstrapping.

Log the fraction of exactly closed gates, positive-gate quantiles, CE-gradient
norms and optimizer-step norms throughout training. A fixed TRAIN probe should
also show where gates close despite an available assisted-token advantage. Those
target-dependent diagnostics stay outside the head's inputs. A stall would be
evidence about this parameterization/optimization, not proof that the predictors
contain no complementary information. Do not change thresholds to rescue it.

Use the same generation protocol on calculation and ordinary prefixes. The gate
must not receive a curator's indication that the document contains arithmetic.
Forced closure after an incomplete note remains an explicit intervention and
all attempts remain in the evaluation. Source separation and sample-size
feasibility must be established before any new model calls.

## What it must beat

| Comparison | Uncertainty it resolves |
|---|---|
| Unchanged B0 | Does the actual system improve prediction? |
| One global mixture coefficient fitted on TRAIN | Is causal selection useful beyond a constant blend? |
| Matched causal B0 plus raw-V29 ensemble | Is the gain available without calculations? |
| Empty, unrelated and corrupted notes | How does note content affect the result? |
| Active protocol on ordinary prefixes | What predictive cost accompanies the gain? |

This is not an autoregressive-compute-matched comparison. Conventional CoT and
an exact interpreter remain competitors before any practical efficiency claim.
Also fit a secondary sixteen-weight schedule on TRAIN, using only continuation
position. It can exploit a generic number-first/prose-later pattern without
recognizing helpful context. This needs no extra backbone calls. Preserve the
eight original conjunctive requirements; any stronger context-adaptation claim
also needs an advantage over that simpler schedule.
The note interventions alone also cannot establish a faithful internal algorithm.
Teacher-forced likelihood does not establish complete-answer accuracy, and its
scoring latency is not an autoregressive serving benchmark. The calculation-enriched
cohort and tolerated ordinary cost also do not establish a natural-corpus gain.

The proposed fresh calculation endpoint is a full-span gain of at least 0.25 nats
over B0 with a positive paired 95% lower bound, plus positive lower bounds over
the constant blend and raw ensemble. The active ordinary suffix must have an NLL
regression upper bound at most 0.03 nats/token. Each of two head-optimization
orders must pass; neither is an independent backbone replication. The initial
1,024-per-panel placeholder is superseded by a [prospective feasibility plan](../validation/causal_reader_feasibility_plan_20261021.json)
with 4,096 per panel, fixed before new source outcomes. Its normal approximation
gives a joint passage lower bound of 0.917 only under the stated alternative:
base gain 0.5, each control gain 0.25, ordinary gain zero and per-contrast SD 4
nats/span. Smaller effects or larger/heavy-tailed variability are not covered
by that assurance. Do not expand the study after inspecting model outcomes.

These criteria do not establish a net corpus gain. If calculation spans gained
exactly 0.25 nats and ordinary spans lost 0.03 per token, the ordinary-span cost
would be 0.48 nats. At calculation-window prevalence $q$, the hypothetical gain is
$0.25q-0.48(1-q)$, breaking even only at $q\approx0.658$. This is a threshold
counterexample, not a measured result. Report the actual prevalence tradeoff
$q\Delta_{calc}+(1-q)\Delta_{ordinary}$ and its uncertainty. Natural prevalence
has not been measured; a calculation-enriched panel cannot supply it. Define
prevalence under the actual source-window selection policy, rather than counting
documents that happen to mention arithmetic.

## What the CPU review already rules out

The earlier pointwise-capped variant guaranteed at most 0.03 nats of regret per
token by bounding the calculation weight to $1-e^{-0.03}$. Its
[saved-score upper bound](../validation/reference_preserving_reader_review_20261021.json)
allows hindsight choice of both a candidate and each token's gate. Even then,
best-of-eight averages only **0.137 nats per 16-token span** on the inspected V30
documents. That variant cannot meet the earlier 0.25 point target for those fixed
candidates and should not be trained. This is proposal arithmetic, not a new V30
endpoint or a bound on every possible calculation/reader.

The requested [unrestricted greedy-note bound](../validation/greedy_token_opportunity_20261021.json)
uses only the original greedy calculation and B0 scores, with no best-of-eight.
It sums $\max(0,\ell_{greedy,t}-\ell_{B0,t})$ over each span. Across the same 256
inspected documents, this gives **2.063 nats per 16 tokens** on average, with a
median of 1.561. In contrast, using the calculation throughout loses **1.429 nats**
versus B0; the tokenwise oracle avoids 3.492 nats of harmful contribution. These
are saved-score statistics, not a learned policy result. Original native target
arrays and all public token differences agree on reconstruction. The
[standalone script](../scripts/greedy_token_opportunity.py) checks released file
hashes and paired target identities before computing the bound.

This establishes complementary token probabilities in the fixed pair. It does
not show that a causal gate can identify them, that the benefit comes from
calculation rather than ordinary ensembling, or that active ordinary prediction
is preserved. It therefore supports retaining the mechanism as a candidate,
without releasing feature extraction, learning or another model evaluation.

The revised unrestricted gate gives up that pointwise guarantee. It therefore
requires a fresh empirical retention test; it is not automatically safer merely
because B0 is frozen. If it only recovers B0, matches ordinary ensembling, or fails
the full-span/retention comparisons, close it without RL or scale escalation.

[Quiet-STaR, sections 4.3–4.4 and Appendix I](https://arxiv.org/html/2403.09629v2)
already motivates learned mixing and discusses bootstrap instability. Its method
describes logit interpolation; this proposal specifies an arithmetic probability
mixture and frozen models. Adding a gate is neither a novelty claim nor a
reproduction of its results.

CPU mechanics now pass independent tiny-HF, causality, normalization, exact raw
bypass, detached-gradient and optimization fixtures. The unpredictable-coin
control correctly recovers no causal gain from a positive hindsight bound. These
are implementation checks, not released-model results. The fixed source preparation
has closed below its required quota; actual-model qualification and scientific
execution were not reached. The explicit power/resource plan stays prospective. The [cost projection](../validation/causal_reader_cost_plan_20261021.json)
proposes an eight reserved-chip-hour hard total, including qualification. It uses
old timings and must survive profiling of actual ordinary notes and feature
export before any science release. Even a successful gate would
establish selective inference only. A return to pretraining needs a separate
learning intervention and a genuinely uniform-NTP control; internalization needs
an explicit transfer objective or curriculum rather than an assumed side effect.
