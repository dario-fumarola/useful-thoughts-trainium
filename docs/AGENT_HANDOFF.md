# Active handoff — October 21, 2026

The owner asked to keep progressing and supplied scientific/engineering critiques. The bounded TRAIN-only optimization study is complete. No goal or new subagents. Do not restart completed fits or inspect completed DEV for recipe selection.

- Repository: `/home/ubuntu/dario`; immutable execution worktree `local/worktrees/selector-optimization-train-v1`, code `10268da`.
- Control F: `local/diagnostics/selector-optimization-train-v1`; output O: `local/science-runs/selector-optimization-train-v1`. CPU PID 2407369 and independent PID 2408211 both closed 0; outer session 48168 is closed. Total wall seconds 215.82 + 13.19. No model/accelerator calls.
- Protocol SHA `435e81b1f88c31ea0da8844cf9962eb86ff944e8124cb989d89e9e1108d430ac`; split SHA `d388a6771e2a1ede8040198c1579b2d61b107d64447809aeac9e11250c242611`. Original readout SHA `3f30536642abbece5793ca1cc78fe2919d8abad9b824dafb6117b769e83c9ca4`.
- Result: hard 6/6 primary pass; calibrated smooth 4/6, with effective stalls in orders 41 and 43. All hard heads beat position control too. Smooth repair's prospective all-six criterion fails. Both gates use identical initial value/slope and paired FIT orders. No successful seed was selected.
- 4,096 FIT / 1,024 internal validation, 512 per kind; 24 terminal heads, ten passes/640 updates. All froze before internal scoring. Internal development only; original TRAIN data had been used previously.
- Independent NumPy reconstruction passes all head probabilities, routing, absolute scores, intervals and decisions. Source-free replay also passes. Full pre-fit 1,979 CPU checks passed.
- Public result/report: `validation/selector_optimization_train_v1_complete_20261021/`. Public numerical projection: `F/public-evidence/`. `F/publish.py` and `F/write_report.py` created reports. Do not rerun them over existing outputs or overwrite archived history.

**Next:** finish public packaging, commit/push, release and anonymous full-download/hash checks. Then consolidate active production bootstrap code with explicit cluster IDs/unit, leaving separate verification arithmetic independent. Keep all old outputs unchanged. Run checks after source edits. Do not launch a new backbone/CPT sweep under this study's authorization.

Current docs are concise, with full previous versions in `docs/archive/*_through_20261021.md`. The installed compiler's parser defaults auto-cast to none, so omission does not prove BF16 execution. Original numerical discrepancies remain unresolved; `validation/executed_precision_review_20261021.json` records read-only inspection. No native parity rerun occurred.

Parent result f6e0b65 / receipt 6c19864 remains closed: one fresh-panel order gains +0.457 nats/16; the other nearly shuts off; original joint criterion fails. Current changed FIT subset/schedule and new orders prevent identifying why. Both backbones remain frozen. Uniform NTP, natural prevalence, fresh confirmation, improved weights and autonomous discovery are still separate questions.

Preserve inputs/models/state/caches and 35 GiB free disk/512 GiB available memory. No cloud, stack or paid-service changes. Continue regular concise user updates. Latest owner steering governs; archived historical plans are not launch commands.
