# Selective use of calculation: proposal, execution held

The next question is whether a causal predictor can use a calculation where it
helps while retaining an actual unchanged B0 prediction where it does not.
This is a bounded inference mechanism proposal. It does not train better base
weights, demonstrate internalization, or reopen V30's closed reward branch.
No new model, feature extraction, or head training has run.

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
The note interventions alone also cannot establish a faithful internal algorithm.
Teacher-forced likelihood does not establish complete-answer accuracy, and its
scoring latency is not an autoregressive serving benchmark. The calculation-enriched
cohort and tolerated ordinary cost also do not establish a natural-corpus gain.

The proposed fresh calculation endpoint is a full-span gain of at least 0.25 nats
over B0 with a positive paired 95% lower bound, plus positive lower bounds over
the constant blend and raw ensemble. The active ordinary suffix must have an NLL
regression upper bound at most 0.03 nats/token. Each of two head-optimization
orders must pass; neither is an independent backbone replication. The provisional
1,024 documents per evaluation panel is a minimum, **not a power certification**.
Freeze a finite sample size and joint-power assumptions before fresh outcomes;
do not expand a study after inspecting its confidence intervals.

## What the CPU review already rules out

The earlier pointwise-capped variant guaranteed at most 0.03 nats of regret per
token by bounding the calculation weight to $1-e^{-0.03}$. Its
[saved-score upper bound](../validation/reference_preserving_reader_review_20261021.json)
allows hindsight choice of both a candidate and each token's gate. Even then,
best-of-eight averages only **0.137 nats per 16-token span** on the inspected V30
documents. That variant cannot meet the earlier 0.25 point target for those fixed
candidates and should not be trained. This is proposal arithmetic, not a new V30
endpoint or a bound on every possible calculation/reader.

The revised unrestricted gate gives up that pointwise guarantee. It therefore
requires a fresh empirical retention test; it is not automatically safer merely
because B0 is frozen. If it only recovers B0, matches ordinary ensembling, or fails
the full-span/retention comparisons, close it without RL or scale escalation.

[Quiet-STaR, sections 4.3–4.4 and Appendix I](https://arxiv.org/html/2403.09629v2)
already motivates learned mixing and discusses bootstrap instability. Its method
describes logit interpolation; this proposal specifies an arithmetic probability
mixture and frozen models. Adding a gate is neither a novelty claim nor a
reproduction of its results.

The proposal is complete for review; execution remains held. Fresh-source
availability, prospective power, actual feature-causality/normalization checks
and a finite resource budget remain unqualified. Even a successful gate would
establish selective inference only. A return to pretraining needs a separate
learning intervention and a genuinely uniform-NTP control; internalization needs
an explicit transfer objective or curriculum rather than an assumed side effect.
