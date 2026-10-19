# Current research handoff — 2026-10-19 05:58 UTC

Continue autonomous work until the owner stops it. Do not finish at a milestone.
Read STATUS.md and NEXT_STEPS.md. Agents are quota-unavailable; do not retry.
Keep commentary meaningful and no more than60 seconds apart. Avoid long internal
planning while runs progress. No active goal; do not infer/create a goal.

## Exact next actions

1. Monitor the existing evaluation jobs; do not duplicate launchers.
   Main control:local/diagnostics/real-trace-cpt-v10-study-v1.
   All six raw/prior/post models, seeds2026101913/14, completed320 updates and
   physically closed. All12 full160/320 states and48 native checks independently
   pass. All six terminal weights and latest complete optimizer are EBS-protected.
   Latest full320:real_raw seed1914. Terminal bindings in control/frozen-models.
   Immutable training3d8666b,worktree local/worktrees/real-trace-cpt-v10.
   Public:validation/real_trace_cpt_v10_terminal_training_20261019.json.

2. Main likelihood evaluation: source29c8b24 at
   local/worktrees/real-trace-eval-v10,1,646 CPU checks. NEVER edit this source
   while its generation/scoring jobs live: they verify every dependency hash.
   The release watcher launched all six fixed models. Each has cpt-eval-{arm}-seed{seed}
   and cpt-score-{arm}-seed{seed}. Full registry:control/evaluation-registry.json
   confirms ALL_SIX_FROZEN_TERMINAL_EVALUATIONS_LAUNCHED. B0 notes are complete; cpt-v10-B0-CPU-scoring
   continues its two arithmetic panels and explicitly reuses ordinary-reference
   observations. Do not inspect numeric trained outcomes before all models/panels finish.

   Tmux cpt-v10-final-readout waits for all seven CPU completions, then runs archive
   and analysis once. Source:local/analysis/watch_real_trace_final_readout_v10_20261019.py.
   Result:validation/real_trace_cpt_v10_complete_20261019.json and matching directory.
   It saves all35 payloads:19,712 native16 likelihood rows and3,584 original notes.
   Afterward run check_real_trace_readout_v10_20261019.py and
   render_real_trace_readout_v10_20261019.py from the evaluation worktree with
   explicit PYTHONPATH; inspect the figures and independently assess all claims.
   Complete examples/interpretation/publication remain to do. No automatic Git push.

3. The already-declared scalar secondary is now implemented, not a new training
   study. Public prospective design:a9c4334 on main. Source4297c1d at
   local/worktrees/real-trace-answers-v10; full1,657 CPU checks plus latest11 affected
   checks after tighter binding validation. NEVER change its source while jobs live.
   Gold/contexts are separate. Existing initial-scalar-v2 unchanged. Greedy32,
   EOS-only; all256 documents per mode, errors/unsupported/caps retained.
   Seven fixed models, AST256 and legacy256 separately, five modes:17,920 attempts.

   B0 tmux cpt-scalar-B0, physical12, is generating first-panel answers.
   All four actual native and cache672/cap32 checks pass. Qualification:
   local/diagnostics/real-trace-v10-scalar-secondary-study-v1/B0-shape-qualification.json.
   Tmux cpt-v10-scalar-release waits for each model's complete original note bundles,
   prepares/protects exact contexts, then launches it on physical6–11. Every trained
   model checks itself again. No retry or source/notes/model selection.
   Registry appears after all seven releases. Existing finite watcher is
   local/analysis/watch_real_trace_scalar_secondary_v10_20261019.py.
   Tmux cpt-v10-scalar-readout waits for all completions, then archives/rechecks
   and publishes local JSON/payloads using analyze_real_trace_scalar_secondary_v10_20261019.py.
   Secondary result path:validation/real_trace_cpt_v10_scalar_secondary_20261019.json.
   Figures, direct independent checks and interpretation still need completion.
   This measures the initial computed scalar at a curated document boundary;
   do not call it general problem QA or change the primary likelihood gates.

## Fixed data and decisions

TRAIN5120 unchanged, SHA437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915.
Primary held V2:local/nvme-study/data/real-trace-cpt-v10-held-panels-v2,
manifest22bd05cb3ad07e08a17e30578bebe07629216c9ad3910e9270bf0794b34dff24.
All256 ordered exact-value ASTs are distinct and absent from TRAIN. Ten aliases
were corrected before learning; preserve V1, old panels and all observations.
Evaluation questions/targets separate under
local/nvme-cpt/data/real-trace-cpt-v10-evaluation-inputs-v2.
Scalar scoring-only values under local/nvme-cpt/data/real-trace-v10-scalar-secondary-v1.

Training gate:post.raw−raw.raw full16 LL>=0.25nat and paired95% lower>0 in both
seeds; ordinary-text token NLL regression mean<=0.02,95% upper<=0.05.
Inference gate:prior.own−raw.raw>=0.25nat and positive lower in both seeds;
own−same-model raw and empty must also have positive lower; no execution errors.
Independent gates. Also report prior.own−raw.own, own−other, B0 absolute paths,
ordinary regressions, all caps/errors and actual token costs. Never turn beating
an impaired control into a new-best-base claim. Bootstrap20k documents,seed1915;
intervals condition on checkpoints, not training-seed uncertainty. All panels DEV.

## Completed diagnostic and broader scope

Variable-depth repair is complete/public:a7835c7,
validation/variable_depth_writer_v11_complete_20261019.json.
Composition512:fixed124→variable306, paired+35.5pp[31.3,40.0]; canonical/natural91→409.
FrozenfullCoT273, but curricula/target tokens differ and only one diagnostic seed.
Freshdepth2 variable182/256 versusCoT193; unseen depth6 variable10/256 versusoracle78.
All9,216 answers/2,048 notes are audited/public. Do not launch a larger split rescue.
Old two-seed v7 is public negative;a0ae5ec. R1234 stays stopped at63.
MuSiQue broader-document work is CPU preparation only:7c5c264,correct composed
question field,all visible paragraphs,no gold support flags in input. No cohort,
model, official DEV/TEST or broader-learning release. Decide after mainline readout.
Read-only TRAIN split audit is public:document_evidence_v12_split_risks_20261019.json.
A hypothetical question hash split reuses training-visible support for3846/3878held
questions. Exact visible-paragraph graph is connected; do not claim evidence novelty
from a naive question split. This released no cohort or model.
Actual CE coefficient allocation:real_trace_cpt_v10_loss_allocation_20261019.json;
added trace targets receive about3.5% of per-example token-loss weight, not a
measured gradient fraction. Fixed objective and success gates remain unchanged.

## Storage and runtime invariants

Keep35 GiB free per filesystem and512 GiB host availability. All original full
checkpoints stay on NVMe. EBS retains every terminal model and latest full320.
Superseded EBS first160 optimizer duplicates were retired only after fresh full
hashes on BOTH old original/copy and newer original/copy. Eight duplicates only;
all originals/model weights remain. Receipt:
validation/checkpoint_retention_after_real_CPT320_20261019.json. Root now about64 GiB free.
Current EBS latest:local/backups/science/real-trace-cpt-v10-study-v1/TERMINALS/latest-full320.
Historical FIRST_FULL160 has an explicit retirement annex; do not load its removed
rank copies. Original full160 remains intact on local/nvme-cpt/runs.
Backups are same-instance, not off-instance.

local/nvme-cpt already formatted/mounted UUID5119e7ab-0ca6-4295-b8f0-e856d87b4b67;
NEVER mkfs again. local/nvme-study UUID63bee3f4-6b43-4290-a0f8-1c8946b7a509 intact.
Use existing venv and source scripts/env.sh, then ALWAYS override PYTHONPATH:
export PYTHONPATH="$PWD:/home/ubuntu/dario/local/analysis:/home/ubuntu/dario/local/test-deps".
Absolute analysis scripts otherwise import main instead of the selected worktree.
No rg installed; use targeted Python/grep/find. No cloud/IAM/network/paid teacher/
new bucket changes. Do not inspect broad process environments: unrelated daemon
arguments can contain secrets. Read only owned PIDs/selected fields.

## Latest prepared readout additions

All four augmentation models have finished both original-note panels; their
scalar jobs are now released on physical8–11. Raw control notes are still
finishing, so their scalar jobs remain pending. No numerical capability outcomes
reviewed. Main CPU likelihood scoring continues; some primary panels have
several hundred rows but none of the six full model/panel sets is complete.

All306 top-level training evidence files (native vectors, initial master witnesses,
metrics and completion/qualification JSON) are EBS-readback/fsync protected under
TERMINALS/TRAINING_EVIDENCE. Public training report now links all six original
320-row compressed training traces; no new observations.

Static old-R padding finding:validation/retired_R_empty_padding_CPU_20261019.json.
Tiny random CPU Qwen confirms a single empty note still projects1024 replay head
rows (2048 with checkpoint recompute), while a cropped8-slot view preserves loss,
reader likelihood and parameter gradients (maxdiff1.49e-8) in this one fixture.
No production optimization, real-model hardware profile, speedup or RL restart.
Do not treat this as full mixed/capped/bucket-boundary qualification.

Secondary renderer prepared:local/analysis/render_real_trace_scalar_secondary_v10_20261019.py.
Run only after complete secondary report. Its all-mode CSV preserves every parser
category; bars display raw/own plus oracle/empty/other reference points. Review
figures visually before publication. Do not alter either live worktree.
