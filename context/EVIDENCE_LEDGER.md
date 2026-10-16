# Evidence ledger and interpretation limits

This is the shortest route from claims to bundled evidence. It distinguishes
**source observations**, **prior review interpretations**, and **new proposals**.
No historical model forward pass was rerun while assembling this repo.

| Claim | Classification and source | Limit |
|---|---|---|
| A small NTP cost may be acceptable for downstream gain | Owner discussion; `transcript_project_excerpt.it.txt`, section beginning “Quello che non capisco…” | Not permission to ignore large damage or omit comparisons |
| Downstream tests should not wait for a positive likelihood probe | Owner discussion; section beginning “Quello che avevo notato…” | Report both outcomes |
| Old model initialization is from scratch | Direct code: `legacy/ema-trainium-pre-thinking/pre-training/train/model_init.py` | New loader must use real pretrained files |
| `--sync-grads` defaults to zero in the historical thought trainer | Direct code: `pre-training/train/think_train.py:113` under legacy root | Actual run history requires the manuscript/journal; do not infer all later runs from this default |
| Runs 13–74 lack intended global synchronization | Historical manuscript/runbook disclosure and prior review | Negative learning results refer to that actual exposure; late synced family is separate |
| OGeo final clean target was left zero | Source plus current `legacy_repair.py` geometry regression | Patch effect on historical science unknown; no weight rerun |
| Run 95 fixed-anchor oracle gain rises | Captured `agent/scratchpad/fixedanchor_0607.out` and matching script | 382 anchors, 48 sequences, K=1, oracle/silence, shared writer+reader changes |
| Ordinary thought policy remains negative in that comparison | Same captured output | “More helpful candidates exist” is narrower than deployed usefulness |
| Run 95 training used 128 rollouts | Current manuscript run entry; prior review of the launch history | Do not copy STATUS's “trained on six” statement |
| Claimed +0.30 ordinary-pretraining attribution is not a before/after test | Prior review plus preserved old initialization sweep and BANDS outputs | Treat the attribution as disputed/incorrect, not a positive causal result |
| Late own-versus-filler differences shrink at short horizons | `agent/scratchpad/fillerarm_0607.out` | Both can still lose to silence; do not retain “always worse than filler” |
| Local intervention totals are not full online-sequence likelihood | Prior review of the probe design | Teacher-forced independent branches and overlapping windows limit extrapolation |
| Late downstream behavior is not established in the supplied record | Prior review and historical status | Absence of recorded benchmarks is not evidence of no capability benefit |
| Primary C/R recipe is likely to work | **Not established**; a proposed favorable test | Hyperparameters and combined simplifications are not validated |
| New repo is a ready Neuron trainer | **False** at delivery | CPU helpers work; production adapter/qualification remain |

## Reading historical status responsibly

`legacy/ema-trainium-pre-thinking/STATUS.md` contains live conclusions from old
agent ticks. Some conflict with their own captured analyses and later review.
Keep the original bytes as evidence; do not make that file the new authority.
The journal and manuscripts also contain tentative interpretations. Use the
underlying script/output and proper units when making a factual claim.

For external work, use `docs/SOURCES.md` and `docs/REFERENCE_NOTES.md`. Research
paper summaries there are prior-review context, not freshly reproduced results.
