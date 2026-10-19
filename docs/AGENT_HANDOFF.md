# Handoff — 2026-10-19 10:50 UTC

## Active task

Complete source-balanced V17, then review all eleven models once. Do not inspect
partial V17 accuracy or likelihood contrasts. Monitor execution, row counts,
numerical guards, memory, checkpoints and physical closure only. Keep working;
user authorizes existing-node training, public Git pushes, research and adaptation.
Agents are quota-unavailable until October24; do not retry. No cloud/IAM/network,
paid teacher, bucket, official TEST, broad RL, size jump or expanded split rescue.
Send meaningful commentary at least every60s. Avoid long speculative analysis.

## New training

Immutable source `37028a1`, branch/worktree `real-source-balanced-v17` under
`local/worktrees/`. All1,656 CPU checks plus smoke/compile pass. All10,240 native
TRAIN views independently checked. Existing causal engine, one ZeRO reducer,
BF16 active/gradients, FP32 masters/Adam. Same TRAIN5120/two passes/320 updates,
world8×micro4,width640,target640,constantLR1e-5. Arms source_direct/source_cot;
seeds2026101913/14. Objective perexample0.5meanCE(source prefix)+0.5meanCE(task).
Prefix labels4,489,846/seed botharms;task163,840/512,900;not equal target information.
Full160/320;terminal primary;fresh B0 each. Never edit immutable running source.

Actual qualification completed: control
`local/diagnostics/real-source-balanced-v17-world8-target640-qualification-v2`.
Status PASS_ACTUAL_REAL_SOURCE_BALANCED_FULL_RESTORE in independent audit.
11 actual updates,9 changed-master/cast/replica checks,24 native checks repeating
four contexts,6 full states, fresh optimizer restore6→8. Post-save4→6 PSS growth
1.877728GiB passes<2. All source/task masks/counts/weighted losses/state bytes
checked. No uninterrupted trajectory/device-RNG readback/stochasticRL claim.
V1 PATH failure before allocation preserved;V2 explicitly prepends actual venvbin.
Public `validation/real_source_balanced_v17_hardware_20261019.json`.

Science control: `local/diagnostics/real-source-balanced-v17-study-v1`.
Four tmux sessions `source-train-v17-{arm}-seed{seed}`,physical0/1,2/3,4/5,6/7.
All32 ranks captured;first finite update complete asof10:48;no failures.
Outputs `local/nvme-cpt/runs/real-source-balanced-v17-{arm}-TRAIN5120-seed{seed}-v1`.
Finite deadline18:00UTC;no retry/extensions/model selection.
Health: `source scripts/env.sh` then
`python local/analysis/live_real_source_v17_health_20261019.py`.

Terminal pipeline scripts in `local/analysis/`:
`real_source_state_audit_v17_20261019.py`,
`audit_real_source_four_terminals_v17_20261019.py`,
`protect_real_source_four_terminals_v17_20261019.py`,
`bind_real_source_four_terminals_v17_20261019.py`.
All parsed; independent two-pass order/label counts tested against actual protocols.
Tmux `source-v17-terminal-pipeline` waits all four closures, then audit/protect/bind.
Its shell launches from training worktree with correct PYTHONPATH;do not duplicate.

## Frozen evaluation

Source c9b399b, branch/worktree `real-source-eval-v17`;1,668 CPU checks pass.
New confirmation512 arithmetic plus ordinary256 from pinned FineMath TRAIN
shards63/62; excludes prior retained pages/content and ordered arithmeticASTs.
No model-based selection. Earlier TEST/unlisted raw text stayed opaque.
Manifest `local/nvme-study/data/real-source-balanced-v17-confirmation-v1/manifest.json`
SHA15d4e0d4c85f07d9cdb09d545203e8c770f05c2ca80a3acd2118c5da0f55c0e6.
Data/design/source published before held forwards/new training.
`validation/real_source_balanced_v17_design_20261019.json`
SHA3f902839f1b0c9cd3db4a14e2123f9d4a424153b56db9a746e8bcbcf014c812a.
Inputs/gold separately frozen and EBS-protected under local/nvme-cpt/data/.
Prefix-only writer inputs item_id/document_id/prefix_ids. All512 scalar partitions exact.

Master `local/diagnostics/real-source-eval-v17-study-v1/plan.json`,deadline20:00UTC.
Eleven fixed models: B0;two each old target_direct,target_cot,real_prior;four new.
Slots0..6 released tmux `source-eval-v17-slot{slot}` onphysical8,9,10,11,14,15,13.
No numerical/operational failures seen asof10:48. Own notes512 complete for three
old models;other baseline notes progressing. No capability values inspected.
New terminal slots7..10 onphysical0,1,2,3 wait audit/protection/bindings.
Watcher tmux `source-v17-new-evaluations`;source
`watch_real_source_new_evaluations_v17_20261019.py`. Do not duplicate preparation.
Per-model registries `local/diagnostics/real-source-v17-{id}-evaluation-v1/evaluation-binding.json`.
Four-slot native LL queue tmux `source-v17-score-queue`,cores48..51 physical12.
Three models' two panels finished;queue nofailures. All models get same-model
first4 CPU references/mode at unchanged0.001 parity guards. Master plan immutable;
scoring topology in separate score-queue-plan.json.
Final totals28,160 scalar attempts/5,632 notes/30,976 likelihood rows.

Primary eachseed: source_cot.own−source_direct.raw>=3pp andpositivepaired95%lower;
own−sameCoTempty andown−B0raw lowerpositive;zero executionerrors.
AND ordinary raw NLL upper95%CI versusB0<=+0.03nat/token. Bothseedsrequired.
20k document bootstraps seed2026101917;no pooledseeds. Same initial-scalar-v2,
leading complete scalar endpoint,notcompleteQA;note128actions/answer32,allfailures.
Secondary training-only effects remain separately reportable,notredefinedmain gate.

Still implement complete-only V17 analyzer/checker/figures/CSV/examples and watcher.
Reuse V15 sources explicitly adjusted to11×512 andnew quality gate. All original
attempts/numerical fallback lineage retained. V15 example exporter required ID
reindexing to frozen order before unchanged donor permutation;retain that fix.
Publish actual training report after allstates audited,with costs/label components.

## Storage and backups — important order

Keep35GiB free/filesystem and512GiB hostavailable. Root~59.3GiB,NVMe-cpt~1004GiB,
NVMe-study~205GiB at10:48. Never mkfs;mounts already prepared. Compilation cache
symlink points NVMe-cpt;do not move while actorslive. EBS same-instance only.
All original160/320 checkpoints stay NVMe. EBS retains all terminal models and
latest global fulloptimizer. V10 oldEBS optimizerduplicates were retired after
fresh allpayload hashes on both sides and verified newerV15full protection:
`local/diagnostics/v17-scientific-backup-budget-v1/completion.json`.
No originals/modelweights deleted;oldEBS sidecar discloses retired duplicate state.
Current latestfull EBS:
`local/backups/science/real-target-sft-v15-study-v1/TERMINALS/latest-full320`.

New V17 protection MUST copy new full320 FIRST (~22.4GiB),readback/fsyncverify;
then freshcheck oldV15original+EBS andnewfull;retireONLYoldV15EBS8optimizerduplicates;
then copy remaining3terminalmodels,hardlink latestmodel withinEBS. Keeps35GiBfloor.
`protect_real_source_four_terminals_v17_20261019.py` implements this order.
Never blindly reuse older models-first protection script. All originalstates remain.

## Latest completed science — V15 positive inference, substantial ordinary cost

Public report `validation/real_target_sft_v15_complete_20261019.json`
SHAf8bc04e2c759cbad2a9438754aee828d5f99c6b4747dc41198fa1e4d3a43e033.
All9models close:17,280 scalar/3,456notes/19,584LL,37 gzrawpayloads.
Fresh384:CoT own342/335,directraw275/285,CoTempty261/263,B0raw257.
Own−direct+17.45pp[13.28,21.61]/+13.02[8.85,17.19];allpredeclaredgatesPASSboth.
Zeroexecutionerrors;ownunsupported24/26;noempty/cappednotes,naturalAnswerboundary.
Most answerscap32;leading-scalar endpointonly. Full16versusdirect+.603/+.446nat,
but versusB0−1.979/−2.119. OrdinaryNLLregresses+.128/+.147nat/token vsB0.
IndependentnativeID/text/value/gold/count/all20kCI/literalgates PASS. Figures checked;
15 disclosedexamples andparsertransitionaudit public. Mechanism is supervised
arithmetic inference,not spontaneous emergence,broadQA ortraining-only benefit.
Alloriginaloutputs/readout/modelsprotected;latestfull320=target_direct seed1913.

Other closed:V10 full-spanprimarytraining/inferencegatesFAILbothseeds;secondary
arithmeticpositive. VariabledepthV11composition124→306/512,one bounded diagnostic;
no expanded split rescue. MuSiQue TRAIN CPU-only:directed evidence-exclusion
leaves502/2983train at2048/3072 for512held;no model/cohort/longcontextrelease.
OldR1234staysstopped63. Sources/papers in docs/REFERENCE_NOTES.md.

## Environment

Venv /home/ubuntu/ema-trainium/.venv;always source scripts/env.sh then override
PYTHONPATH with desired immutable worktree first plus local/analysis/test-deps.
CPU USE_TORCH_XLA=0. tmux PATH must explicitly include actual interpreter's bin;
verify libneuronpjrt-path. No dependency upgrades. pytest via python -m pytest.
No `rg`;use targeted Python/grep. Avoid broad process argument/environment dumps.
Git public owner-authorized repository dario-fumarola/useful-thoughts-trainium.
