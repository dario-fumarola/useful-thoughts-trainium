# Cross-domain retention of fixed mathematical CPT models

On this fixed educational-English panel, **all ten trained models lose to B0, with negative full-window confidence intervals**: four ordinary mathematical-training settings and all six matched raw/format/computation models. Computation supervision does not establish better retention than matched ordinary training in either source order, and it loses to format-only supervision in both. Every result is shown below. This separate diagnostic neither changes the original FineMath confirmation criterion nor turns a tolerated regression into an improvement.

## Every fixed model

The panel has 1,024 unique original source windows, with 527 predicted tokens each. Positive gain means better prediction. Full-window nats/token and final 16-token nats/span are different measurements.

| Frozen model | Full-window NLL | Gain over B0, nats/token [95% CI] | Final 16-token gain, nats/span [95% CI] |
|---|---:|---:|---:|
| B0 | 2.495754 | Reference | Reference |
| scaffold_format-seed11 | 2.504187 | -0.008433 [-0.008968, -0.007910] | -0.062177 [-0.098426, -0.028592] |
| scaffold_format-seed12 | 2.503860 | -0.008106 [-0.008615, -0.007590] | -0.058825 [-0.094302, -0.024962] |
| scaffold_raw-seed11 | 2.504660 | -0.008906 [-0.009458, -0.008358] | -0.066742 [-0.100271, -0.034508] |
| scaffold_raw-seed12 | 2.504501 | -0.008747 [-0.009297, -0.008208] | -0.064269 [-0.097915, -0.030490] |
| scaffold_trace-seed11 | 2.504578 | -0.008824 [-0.009373, -0.008286] | -0.074119 [-0.110262, -0.039758] |
| scaffold_trace-seed12 | 2.504511 | -0.008757 [-0.009297, -0.008220] | -0.073151 [-0.108192, -0.039794] |
| warmup_constant-lr1e-05 | 2.509563 | -0.013809 [-0.014498, -0.013130] | -0.126392 [-0.169885, -0.083812] |
| warmup_constant-lr3e-06 | 2.499366 | -0.003612 [-0.004029, -0.003200] | -0.029456 [-0.056047, -0.003313] |
| warmup_cosine-lr1e-05 | 2.504589 | -0.008835 [-0.009388, -0.008259] | -0.073812 [-0.107397, -0.040786] |
| warmup_cosine-lr3e-06 | 2.497326 | -0.001572 [-0.001914, -0.001235] | -0.014251 [-0.037541, +0.008461] |

[All comparisons](all-paired-contrasts.csv) also retain trace versus matched raw, format and the qualified cosine reference in each source order. Relative improvement over a trained control and absolute improvement over B0 remain separate claims. Intervals are conditional paired document-sampling uncertainty, not uncertainty across base models or all possible training runs.

## Sources and chronological boundary

The frame is the fixed first 32,768 records of one [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) shard, revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9. Its source-file SHA is b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871. Identity and length, not arithmetic flags or model outcomes, select 1,024 documents and one intact 528-token window per document. Known mathematical canonical pages/content and duplicate pages are excluded. Independent reconstruction verifies all selected records, tokens, identity ordering and exclusions without row replacement.

These are educational-English web documents outside the known mathematical frame, not all English, a natural token-weighted corpus or sources proven absent from base pretraining. The new language evaluation is distinct from the earlier unenriched **FineMath** panels, which were still mathematical web text.

All four ordinary baseline terminals were fixed before this source plan. B0 and the four baseline models were scored first; the six scaffold models were scored only after all their terminals and primary scoring jobs closed. All eleven use direct original-text prediction with correctly supplied prior tokens. No generated note, selector or ensemble is used here. The ordinary baseline's English losses motivated the separately prepared source-mixture test; its new English confirmation cohort excludes this panel.

## Execution and numerical evidence

The outer source is **665dde897624134884eb902b17369eabccf7797f**. The mathematical likelihood worker is unchanged and hash-bound. The wrapper checks source identities, independent receipts, model hashes and all required terminals. Actual additional assigned reservations total **0.502299 chip-hours**, including every attempt, below the fixed 1.5 ceiling. One existing two-chip group runs at a time; there is no new training. The 35 GiB filesystem and 512 GiB available-host floors remain enforced. These assigned costs are not whole-node billing.

Each comparison resamples unique source documents, carrying all 527 target scores together: 5,000 percentile draws, seed 17. Every absolute score and all 16 comparisons are reproduced independently from the exact original native FP32 scores by the released NumPy script. Integrity fixtures check score hashes, required models, units and intervals. This is within-repository numerical verification, not an external model replication or source-text verification from the source-free archive.

The qualified ordinary model's positive FineMath confirmation remains valid within its declared domain. This diagnostic measures the additional cross-domain question directly, and the measured cost cannot be described as a general-language win merely because it is below an older 0.03 nats/token tolerance. The scaffold models similarly require both relative and absolute comparisons. See [readout](readout.json), [all comparisons](all-paired-contrasts.csv) and [numerical replay](independent-numerical-replay.json).

## Separate standard CPU forward diagnostic

A fixed inspected subset of 32 English documents was independently scored with standard Hugging Face FP32/eager CPU forwards. The selected ordinary model loses 0.011201 nats/token versus B0 on CPU, compared with 0.011281 natively on the same documents. The absolute mean paired-gain difference is 0.00007993 and the largest document difference 0.002554. The largest individual-token log-probability discrepancy is 2.017855: aggregate agreement does not establish tokenwise identity. All 64 forwards took 92 seconds with no accelerator use. This checks numerical consistency of the domain cost, not fresh scientific confirmation or external replication. [Complete receipt](../general_web_retention_cpu_forward_check_20261022.json).
