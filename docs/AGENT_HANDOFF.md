# Current research handoff — 2026-10-19 08:16 UTC

Continue autonomous work until the owner stops it. Do not end at a milestone.
Use tools promptly; avoid minutes of speculative internal planning. Give useful
commentary within60 seconds. No active goal; do not infer/create one. Agents are
quota-unavailable until October24; do not retry. Read STATUS and NEXT_STEPS.

## Active study

`real-target-sft-v15` closes a missing strong supervised reference. Four models:
`target_direct`, `target_cot`, seeds2026101913/14. Every model starts fresh B0,
320updates/global32/two passes on the SAME verified real TRAIN5120. Prefix and
supplied opening unscored. Direct future16; CoT full trace+Answer+future16.
Source829d9a2 at `local/worktrees/real-target-sft-v15`, immutable while live.
All1,647 CPUchecks+smoke/compile pass. All10,240 actual TRAIN packs validated.
Target counts per seed163,840 direct /512,900 CoT; paddedwidth640,target128.
Actual world8/micro4 BF16-active/F32-master qualification PASS:11finite updates,
9 explicit changed-master/cast/replica witnesses,6full states,actual full6 restore
and ordered7–8. All48 saved CPU RNG payloads are settable; saved6/resumed8 CPU
states match onall8ranks. Immediate device-RNG readback and uninterrupted parity
remain unverified. This is not a stochastic-RL qualification.
Public:validation/real_target_sft_v15_hardware_20261019.json.

Four science jobs released08:00UTC on physical0/1,2/3,4/5,6/7. At08:16 they reach
154–156/320,~5.5sec/update,no failures. Full160/320,terminal primary,no extensions.
Control:`local/diagnostics/real-target-sft-v15-study-v1`. Tmux names:
`sft-v15-target_direct-seed2026101913/14` and `sft-v15-target_cot-seed2026101913/14`.
Finite owned supervisors monitor host/disk and preserve complete-boundary STOP.

Exact next command (mechanical counts only):
`python3 /home/ubuntu/dario/local/analysis/live_real_target_v15_health_20261019.py`
Do not inspect partial held answers or numeric likelihood outcomes.

Tmux `target-sft-v15-terminal-pipeline` waits allfour physical closures,then runs
independent audit/protection/binding once. Helpers in local/analysis:
`real_target_state_audit_v15_20261019.py`,
`audit_real_target_four_terminals_v15_20261019.py`,
`protect_real_target_four_terminals_v15_20261019.py`,
`bind_real_target_four_terminals_v15_20261019.py`,
`watch_real_target_terminal_pipeline_v15_20261019.py`.
It verifies all1,280 ordered metrics,32native vectors/eight full states,then
protects allfour terminal models and latest full320. The independent label formula
is16 per direct example,trace_tokens+2+16 per CoT example. All original states stay
NVMe. Latest-full model is hardlinked inside the EBS archive. Do not duplicate it.
Expected completion:control/terminal-pipeline-completion.json with
`ALL_FOUR_MODELS_AUDITED_PROTECTED_AND_FROZEN`;bindings incontrol/frozen-models.

## Fixed evaluation

Prospective design:validation/real_target_sft_v15_design_20261019.json;
source protocol:docs/REAL_TARGET_SFT_V15.md. Fresh384 is drawn from unused source
TRAIN documents,disjoint from actualTRAIN5120 and both earlier arithmetic panels
by document and ordered exact-value AST. Operator patterns/equal answers can recur.
Source data:`local/nvme-study/data/real-target-sft-v15-held-v1`.
Held SHA8b9bc93656eb536981c8331d04dfd120117e373364d00b0806691907e005581c.
No officialTEST. Cohort/design/native inputs and separate gold are EBS readback protected.

Primary:CoT-own minus direct-SFT-raw complete-scalar exact rational accuracy.
Require>=3pp and paired95% lower>0 in EACH seed,plus own-empty and own-B0raw positive
lowers in each seed,zeroexecutionerrors.20k document bootstrap,seed2026101917.
All unsupported/error outputs remain in384. Likelihood/full16 components and ordinary
FineMath256 raw NLL are secondary. No pooled seeds or post-hoc primary substitution.
Passing establishes a supervised real-arithmetic reference,not CPT benefit/emergence.

Eval source41a5bfd at `local/worktrees/real-target-eval-v15`,immutable once jobs live.
All1,664 CPUchecks+smoke/compile pass. Legacy256 scopes unchanged;fresh384 has new IDs.
Notes cap128/prefill544;scalar cap32/prefill672,EOS only. Fixed Answer closure2tokens.
Unchanged initial-scalar-v2 scoring after physical closure;whole donor controls intact.
Optional one-core FP32 span scorer compares first4 frozen docs in EVERY mode to CPU
(max tokenerror<=.001,absmean span drift<=.001) before scoring. Preserve failures;
if needed run a separately identified canonicalCPU fallback,never change the models.
`--device cpu` remains the canonical score path. B0all1280 parity profile is separate.

Global evaluation registry:`local/diagnostics/real-target-eval-v15-study-v1/plan.json`.
Exactly9 models:B0,oldV10raw/prior eachseed,and the four new terminal models.
Five baseline controllers are active on physical8,9,10,11,14. Their native/cache
checks pass and notes are progressing. Each copied controller runs notes→scalar,
reuses original notes and keeps all attempts. Tmux `target-eval-v15-slot0` through4.
Per-model registry path is in the master plan. Controller completion status:
`COMPLETE_FIXED_V15_MODEL_NOTES_AND_SCALAR`. No held capability review until all9finish.

Preparation factory (run from eval worktree with correct PYTHONPATH):
`local/analysis/prepare_real_target_model_evaluation_v15_20261019.py`
requires --model-id,--physical,--slot,[--binding]. Baseline5 already prepared/released;
do not repeat. Future four use slots5–8,physical0–3,only AFTER allfour terminal
bindings/protection exist. Factory writes protected control/CPU references and
launch-evaluation.sh but does NOT launch. Root then launches each once.

The four-slot one-core score queue is now live onlogical48–51 after note bundles
complete:tmux target-v15-score-queue. Helpers run_real_target_scores_v15_20261019.py
and watch_real_target_score_queue_v15_20261019.py;statecontrol/score-queue-state.json.
STILL TO BUILD:finite watcher to prepare/release fournewmodels afterterminalbinding;
final all-nine archive/readout/independent check/plots/examples. Do not let this
engineering become a reason to inspect/select partial outcomes. Full source/gold
separation and all original artifacts must remain. The5baseline CPU prep commands
finished successfully;their source/controls were EBS verified before release.

## Completed findings that stay unchanged

MainV10:both predeclared full16 likelihood gates FAIL. Post raw gains~0.01nat;
prior-own versusrawCPT+0.097/−0.139nat,intervals crosszero. The prospectively declared
scalar secondary is positive:216/213 of256 versusrawCPT193/196,+9.0/+6.6pp with
positive intervals;unsupported25/24 retained. Same-prior empty175/172. This is a
narrow extra-computation arithmetic-answer gain,not ordinary training benefit.
All19,712 LP rows/3,584notes/17,920scalar attempts,figures,28disclosed examples and
independent bootstrap checks are public in main c6bf246/c60fb3d and follow-ups.
Posthoc partition331ce2e:tail damage~−.21nat inbothseeds;numeral gainclear onlyseed13.
V10models/latest full320 protected;alloriginal160/320 states remain. Never alter raw
reports SHAeb5f6b07aa4b39da6f708a7941fc2bc3bbaf1580b5bc3942979730acab2e1bf4
and c25b6a06f855684622084a9e4744b2c55ef8f7290f31ee108e0aca8537e1855f.
Bounded variable-depth repair is closed and positive only inits narrow scope;
no expanded split rescue. OldR1234 stays stopped63. No broadRL orunqualified8B.
MuSiQue is CPU-only preparation;19,938TRAIN questions,the visible-evidence split
risk is documented. It is not a third released learning program.

## Runtime/storage discipline

Source scripts/env.sh,then explicitly set PYTHONPATH to the DESIRED WORKTREE plus
local/analysis and local/test-deps. Python heredocs run from main otherwise import
main before PYTHONPATH;every preparer must assert ROOT. Use correctcwd. No stack
upgrades,cloud/IAM/network changes,paid teacher services or unapproved bucket.
Do not print broad process args or environment;unrelated services may hold secrets.

Compile cache nowbyte-verified on existingNVMe:
`local/neuron-cache-driver` is a symlink to
`local/nvme-cpt/caches/neuron-cache-driver-20261019`. Original runtimepath unchanged,
all10,908entries/18,495,161,350bytes verified. No model/checkpoint/data removed.
Report:validation/neuron_compile_cache_NVMe_move_20261019.json.
Rootfree~74.6GiB after qualification backups;projected newmodels+latestfull~35.3GiB,
35GiB reserve retained. NVMe has ample room for original states. Keep512GiB host
availability and35GiB perfilesystem. EBS is same-instance,not off-instance backup.
NVMe-cpt UUID5119e7ab-0ca6-4295-b8f0-e856d87b4b67 is ALREADY formatted/mounted;
NEVER mkfs. No new reviewZIP today;one was already shared. Main publication of
V15hardware/active status/cache move is included in the next reviewed commit.
