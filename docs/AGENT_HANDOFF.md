# Current handoff — 2026-10-19 22:53 UTC

Continue autonomously; no final at a milestone, no goal creation. Agent quota
unavailable until October24; do not retry agents. Read STATUS/NEXT_STEPS first.

## V25 is complete

Report: validation/document_evidence_weight_v25_complete_20261019.json.
SHA256 ae7d3eacd701acb20eb969721dea531782c18f73d87bd79dd5bfacc8d6d86837.
C=local/diagnostics/document-evidence-weight-v25-ablation-v1.
O=local/nvme-cpt/runs/document-evidence-weight-v25-selection1-TRAIN2048-seed2026101923-v1.
All V25 training, protection, generation and analysis processes finished0.
Do not rerun any original producer or immutable analyzer.
Source W25=local/worktrees/document-evidence-weight-v25 @5414f8d remains frozen.
All C/release.json pipeline sources and postprocessing-sources.json remain frozen.
All256 updates/full128/256 states audited; terminal bound/protected. All original states remain.
Terminal model SHA1d918a183a8bd171d72f7afea84894ad87735c77e99ca89b838403d7bd11ef87.
Current full EBS local/backups/science/document-evidence-weight-v25-ablation-v1/TERMINALS/latest-full256.
Earlier EBS optimizer duplicates alone retired after newer full verification; all models/originals retained.

All1531 generation attempts/256 new ordinary likelihood rows completed, zero answer
errors/caps/unsupported. One held duplicate-ID selection used declared empty fallback.
All765 canonical answer replays stay in the answer field; no ID/header regeneration.
Native own and canonical own answers agree255/255; fallback action sequence equals empty.
Current-model native/cache and independent raw-token/count/bootstrap checks all PASS.
Originals/source:445 files66MB EBS readback/fsync protected under COMPLETE_READOUT.
Final public artifacts:88 files protected under PUBLICATION; see C/publication-protection.json.
Figures were inspected through PNG exports with identical numeric CSV bytes.

## Result and interpretation

Only selector coefficient changes0.1→1.0, one fresh1.7B B0, same TRAIN2048/order1923,
four passes/256 updates, source/final0.5/0.5, worker/optimizer/topology/precision unchanged.
Primary TRAIN exact sets124/128,64/96,15/32 versus117/128,59/96,12/32. Gate FAILS.
All256 TRAIN answers correct; annotation fitting is not semantic validity/generalization.
Held native direct138, own143; old matched direct127/old selector own129.
New own−old direct+6.3pp[1.6,11.4]; new direct−old direct+4.3[-0.4,9.4].
New own−same direct+2.0[-1.6,5.5]; canonical empty139/own143/oracle149.
Canonical own−empty+1.6[-1.6,4.7]; no established incremental inference benefit.
Held sets169/255 versus132, mainly two-support160/205; three9/46, four0/4 descriptive.
Reader still sees all20 passages; three-support own/oracle both31/46 despite imperfect sets.
OrdinaryNLL+0.0666vsB0,−0.0687vsold selector: smaller regression, not broad LM gain.
Own17.16 versus direct7.64 actions; science2.232chip-hours, qualification1.272 separate.
Single data-order seed and previously inspected held255: developmental, not confirmation.
Historical direct control has auxiliary0.1 versus new selector1.0; this is a recipe
comparison, not a coefficient-matched test of semantic support supervision.
No old V23/V24 report, score, parser or gate changed. Official DEV/TEST unopened.

## Next concrete question

Prepare ONE format-only auxiliary1.0 control under a new explicit experiment ID.
Fresh B0 evidence_direct, same TRAIN2048/data-order1923/four passes/256 updates,
source/final0.5/0.5 and numeric worker as V25. Compare with the completed selector1.0.
This tests whether the recipe advantage survives matching the auxiliary coefficient.
No coefficient sweep, new dataset, larger model, RL or official-final-data access.
Write a prospective bounded design and decision rule, inspect storage, then freeze.
Any driver change should be validation/identity only; preserve numerical-function ASTs
and all old-purpose guards. Run required CPU checks and actual bounded qualification
before a substantive learning release. No new learning job is released yet.

## Storage and engineering constraints

Keep35GiB on each filesystem and512GiB host available. About58GiB root,127GiB
NVMe-cpt,99GiB NVMe-study remain after V25. New full states are22.44GiB each.
Account for qualification, two science states and successive full EBS backups.
If inactive engineering BACKUP duplicates must relocate, retain original files,
verify original/old/new hashes and fsync/readback before unlinking only the old duplicate.
Never delete original states, active caches or science terminal weights as a shortcut.
Consider whether a bounded new qualification can use full4/8 plus fresh restore4,
rather than blindly copying the old four-full-state2/4/6/8 storage budget.
Inspect actual validator/driver support before choosing that protocol.

No package/stack upgrade, instance/IAM/network/bucket/paid-teacher changes.
Use source scripts/env.sh and the existing venv; CPU uses USE_TORCH_XLA=0.
For stdin Python, cwdroot overrides PYTHONPATH; run from the intended frozen worktree.
Local/analysis/CURRENT_V25_SESSION.json lists completed IDs and helper paths.
Future work must use new output/control/source identities; never mutate V25.
