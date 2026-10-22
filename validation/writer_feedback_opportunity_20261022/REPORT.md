# Saved candidate opportunity under a fixed reader

This exploratory TRAIN-only diagnostic uses the **existing** V30 predictions on 256 source documents: eight sampled calculations and one greedy calculation each. There are **no new model calls or parameter updates**. It does not reopen V30 or establish autonomous thought learning.

The reader is the already training-fitted, frozen **position-only probability mixture** of B0 and the calculation model. Its 16 note weights and the separate no-note weights are preserved in [the readout](readout.json). For each token, mixture likelihood is `logaddexp(log1p(-g) + base_logp, log(g) + note_logp)`; sum the original 16 tokens before comparing candidates. An actual B0 null option selects a different predictor, rather than subtracting the same baseline from every candidate reward.

| Fixed-reader comparison | Gain, nats per 16-token continuation [95% document interval] |
|---|---:|
| Greedy calculation mixture versus B0 | +0.2282 [0.1591, 0.3031] |
| Mean sampled mixture versus B0 | +0.2292 [0.1629, 0.3019] |
| Hindsight best of eight versus greedy | +0.0266 [0.0086, 0.0548] |
| Hindsight best of eight or actual B0 versus greedy | +0.0697 [0.0475, 0.0998] |
| Greedy calculation mixture versus fixed no-note position control | +0.1924 [0.1297, 0.2586] |

The added opportunity from searching these eight calculations is small under this particular reader. A hindsight choice can see the future; it does not demonstrate a prefix-only policy that captures the gain. Null wins on 35.2% of these documents. This supports investigating selective use before investing in an actor update under this fixed interface; it neither rules out other writers/readers nor changes any historical scientific decision. No writer training follows from this diagnostic.

Every saved source score file is hash-checked against the [published V30 evidence](https://github.com/dario-fumarola/useful-thoughts-trainium/releases/tag/v30-candidate-utility-20261021-v2). Document/target pairing and all 16 finite token scores are required. The controls were fitted in the preceding selector-development study and remain fixed. All seven comparisons resample unique source documents, 5,000 draws, seed 17, with the central production helper. These are inspected training-pool results, not fresh confirmation, a new capability finding or an independent model replication.
