# Current status

Updated October 21, 2026. [History](docs/archive/STATUS_through_20261021.md).

## Fresh confirmation: numerical qualification running

The fixed test has **1,024 new calculation documents and 4,096 unenriched natural windows**. Every chosen window independently matches pinned Parquet data, with no replacements. All six existing hard-gate heads and six matching no-calculation heads stay frozen.

Calculation-mechanism confirmation and direct prediction on the natural panel are separate endpoints. Both face learned, constant and position-only no-calculation controls. Old retention tolerance is secondary. [Design](configs/reader_fresh_confirmation_design.json) · [Source check](validation/reader_population_census_20261021/fresh-source-check.json).

- Numerical qualification started at 18:18 UTC (supervisor 2415051, coordinator session 52227), running first with explicit `--auto-cast=none` and prospective empirical tolerances. Historical failed gates and exceptions stay unchanged.
- One existing-node test: two physical chips at most, **4.25 reserved chip-hours including qualification**, itself capped at 0.5. Parent actual plus the new maximum remains below 8.25. A successful numerical check and measured cost projection release scientific execution.
- No head or backbone fit. All **2,004 CPU checks**, smoke and compilation pass. [Freeze receipt](validation/reader_fresh_confirmation_freeze_20261021.json).

## New population result

The strict verified arithmetic class occupies **19,552 / 1,175,123,958 positions (0.001664%)** in the existing source-clean frame. Document presence is a different denominator. This is not all FineMath or every kind of useful reasoning. The natural panel measures prediction directly. [Census report](validation/reader_population_census_20261021/REPORT.md).

## Previous results stay intact

The internal comparison found hard-gate gains in six orders and calibrated smooth-gate gains in four; two smooth runs stalled. [Report](validation/selector_optimization_train_v1_complete_20261021/REPORT.md). The parent fresh-panel experiment improved in one order and failed jointly. [Parent report](validation/causal_reader_owner_complete_20261021/REPORT.md).

V29 did not establish improved direct pretraining; V30 closed its fixed reward branch. Uniform NTP remains a separate baseline requirement. Improved weights, internalization and autonomous discovery remain open.
