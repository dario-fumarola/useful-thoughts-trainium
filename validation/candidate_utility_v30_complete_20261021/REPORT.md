# V30: predictive reward diagnostic closes without RL

The frozen diagnostic **fails**. Even choosing the best of eight calculations
after seeing the continuation does not establish improvement over the same
reader's raw prediction, and remains significantly worse than the untouched
base. This reward branch is closed. No RL or rescue sweep follows.

## Completed design

256 fresh source/page/content/ordered-expression-disjoint FineMath TRAIN documents;
native 512-token prefixes and original 16-token continuations. The chronological
first V29 computation terminal is both frozen writer and reader. Each prefix gets
one greedy calculation and eight samples at temperature 0.7, cap 128 including stop.
All **2,304 attempts and 3,840 likelihood rows** complete without execution errors.
No capped or empty calculations; mean 34.17 actions. No parameters change.

Cohort, policy and gates were frozen before model calls. Independent source
checks cover 390 eligible documents and 1,030 teacher equations. Actual stochastic,
cache, behavior-probability and RNG qualification checks 72 rollouts/2,853 actions.
The final checker reconstructs all 3,840 contexts, primary paired statistics and
decisions, and agrees with four ordinary HF forwards. These are checks within
the repository, not external replication or scientific learning evidence.

## Full-continuation result

| Contrast | Mean nats per 16 tokens [paired 95% interval] |
|---|---:|
| Hindsight best8 − same-reader raw | **+0.082 [−0.241, +0.394]** |
| Hindsight best8 − untouched B0 raw | **−1.083 [−1.569, −0.616]** |
| Hindsight best8 − greedy calculation | +0.346 [+0.153, +0.568] |
| Mean sampled calculation − same-reader raw | −0.236 [−0.570, +0.096] |
| Correct supplied calculation − corrupted | +2.774 [+2.458, +3.109] |
| Correct supplied calculation − same-reader raw | +0.226 [−0.054, +0.497] |

Best8 is a future-aware oracle opportunity, **not deployed prefix-only selection**.
Its local gain over greedy does not satisfy either absolute raw comparison.
Resampling uses 256 documents, not 2,048 independent candidate draws. Intervals
condition on this checkpoint and fixed sampling policy.

## Correctness and ranking

1,807/2,048 samples meet the conservative verified-canonical execution definition;
182 contain an arithmetic error or end at the wrong root value; 59 are unsupported
or alternative procedures. Valid alternatives are not declared mathematically wrong.
242 documents have a canonical sample. On 98 documents a canonical sample beats
both raw baselines, passing the declared coverage requirement.

Only 25 documents contain both canonical and error samples, below the required 32.
Within those 25, canonical work scores **+5.112 nats [3.189, 7.215]** better on average.
The observed ranking is positive: the alignment gate fails minimum coverage,
not its estimated sign. Oracle/corrupt sensitivity also passes. The broader
full-continuation opportunity still fails.

For best8 versus same-reader raw, the numeral improves by **+0.596 nats**,
surrounding prose deteriorates by **−0.507**, and boundary/whitespace changes
by −0.006. Numerical improvement again fails to become a full-span win.

## Decision and limits

Preserve V29's numerical capability finding. A healthy, mostly correct writer
and an arithmetic-sensitive reader are insufficient for this predictive objective.
This is one TRAIN-only fixed-model diagnostic with supervised warm start—not
autonomous discovery, internalization, general pretraining improvement, or an
impossibility result. No temperature/prompt/checkpoint sweep or additional
training is released by this result.

The separate read-only V29 audit verifies both raw runs' 40,960 ordered exposures.
Its raw control is continuation-reweighted NTP: selected future tokens have
5.47–17.47 times the per-document coefficient of earlier tokens. Loss falls
strongly on the repeated pass. This identifies weighting/repetition hypotheses,
**not the cause** of fresh-document regression. No new CPT is released.

Reserved accelerator cost: 0.188266 science chip-hours plus 0.138581 qualification
chip-hours; CPU references/checks are separate. All assigned processes/devices
close cleanly. Original states, source records and every candidate/token score
remain preserved locally.
