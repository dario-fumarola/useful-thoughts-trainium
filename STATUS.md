# Current status

Updated October 21, 2026. [History](docs/archive/STATUS_through_20261021.md).

## Fresh confirmation complete

Every fixed head improves the **1,024 fresh calculation spans** over B0: **+0.421 to +0.506 nats per original 16 tokens**. Five of six beat all additional controls. Order 45 versus the calculation position schedule is +0.075 [−0.042, +0.188]; the all-six calculation criterion fails.

Every head loses on **4,096 unenriched natural windows**: **−0.020 to −0.052 nats per span**, all intervals negative. The natural endpoint fails in every order. The simple no-calculation constant mixture instead improves by +0.0244 [0.0155, 0.0340]. These are separate outcomes, not a single ambiguous success/failure label.

[Complete report](validation/reader_fresh_confirmation_complete_20261021/REPORT.md) · [All orders](validation/reader_fresh_confirmation_complete_20261021/all-orders.csv) · [Verified public release](validation/reader_fresh_confirmation_public_release_20261021.json) · [Design](configs/reader_fresh_confirmation_design.json).

- All 12 heads and fitted controls stayed frozen; no new fitting or backbone update. All sources independently match pinned Parquet data, without replacement.
- All 64 scientific native workers closed successfully. Actual **1.781765 reserved chip-hours including 0.130007 qualification**, below 4.25, at most two physical chips.
- Independent NumPy reconstruction agrees on features, gates, mixtures, all 80 contrasts and both failed endpoint decisions. The source-free public replay agrees too. All **2,004 CPU tests**, smoke and compilation pass.
- Explicit auto-cast=none and the new prospective numerical contract passed; the historical stricter failure is unchanged.

## Interpretation and next question

Useful selective prediction repeats on targeted calculations; the current system does not improve its declared natural distribution. Frozen inference results do not establish better pretrained weights, internalization or autonomous discovery. The next pretraining baseline must be genuine uniform next-token training, without the V29 duplicated/upweighted continuation branch.

The [population census](validation/reader_population_census_20261021/REPORT.md) reports 0.001664% strict verified boundaries in its source-clean frame, not a prevalence claim for every kind of useful computation. Natural outcomes above are measured directly.

Earlier records remain: [internal hard 6/6 versus smooth 4/6](validation/selector_optimization_train_v1_complete_20261021/REPORT.md), [parent positive order and failed joint gate](validation/causal_reader_owner_complete_20261021/REPORT.md), V29's failed CPT criterion, and V30's closed reward branch.
