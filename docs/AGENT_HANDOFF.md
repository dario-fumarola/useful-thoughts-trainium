# Current research handoff — 2026-10-19 07:18 UTC

Continue authorized autonomous work; do not stop at a milestone. Read STATUS.md
and NEXT_STEPS.md. Agents remain quota-unavailable;do not retry. Updates<=60sec.
No active goal;do not create one. No8B/broadRL/officialTEST/cloud changes.

## Latest completed evidence and exact next actions

Both fixed real-data readouts are complete. All producers/watchers closed normally.
Main report:validation/real_trace_cpt_v10_complete_20261019.json.
Primary report SHAeb5f6b07aa4b39da6f708a7941fc2bc3bbaf1580b5bc3942979730acab2e1bf4.
Independent checker PASS19,712likelihood rows/3,584notes/all20kbootstrap intervals.
Both primary gates FAIL:post.raw−raw.raw+0.012/+0.006nat;prior.own−raw.raw+0.097/−0.139.
Ordinary FineMath regression bounds pass. Own−empty/other positive in both seeds,
but not sufficient to rescue either primary gate. B0raw remains a strong LL control.
Scalar report:validation/real_trace_cpt_v10_scalar_secondary_20261019.json.
SHA c25b6a06f855684622084a9e4744b2c55ef8f7290f31ee108e0aca8537e1855f.
Independent check PASS17,920answers,native stops,exact numbers,all paired intervals.
AST256priorown216/213 versusrawCPT193/196 andsamepriorempty175/172.
Paired+9.0/+6.6pp versusrawCPT,positive intervals;unsupported25/24 retained.
No execution errors. This is a prospective secondary,not primary success or generalQA.
All35+14raw payloads,CSV,plots and28 disclosed examples are ready for publication.
Sources/checkers/renderers copied into each result directory/analysis-source.
Original raw reports MUST stay immutable;their SHA bindings are checked.
Posthoc error-transition-source.py independently reproduces every descriptive
partition;gain includes numerical corrections and improved format coverage.

Next:finish presentation-annex EBS readback protection;review and commit/push
complete mixed result/README/STATUS/NEXT. Then close the forward-only scoring
engineering result below and select a bounded next scientific intervention.
No new model is running. All six terminal models/latest full320 are protected.
Immutable training3d8666b,likelihood29c8b24,scalar4297c1d;do not edit old identities.
No training extension or checkpoint selection follows from either readout.

Forward-only engineering:local/diagnostics/real-span-XLA-B0-v14-v2/span_probe.py,
outputs local/nvme-cpt/diagnostics/real-span-XLA-B0-v14-v2. Exit0,all1280rows,
configuredone logicalcore,FP32,width672,batch4. Max tokenerror0.00024569,
absmean span drift0.00021813;declared<=.001 parity passes. Estimated warm
speedup15.72x versus16-threadCPU,warmbatch~.33sec;firstbatch40.4sec,total161.8sec.
No gradients/sampling/weight-sync/trained-model qualification orRLstepclaim.
V1failed before scoring because profile.py shadowed stdlibprofile;preserved.
Protect allV2originals andV1failure,then publish separate engineering report.

Missing practical reference:answer-focused direct SFT versusfulltraceCoTSFT
on the SAME prefix-computable real data,unscoredprefix. Existing CPT givesadded
trace~3.5% ofnormalizedCEcoefficientmass,not gradientmass. Newobjective needsnew
identity/fixedprospectivecomparison/workloadqualification. Finish publication first.
The bounded split repair stays closed;broader MuSiQue remainsCPUonly preparation.

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

Mechanical progress helper (no score/answer parsing):
python3 local/analysis/live_real_trace_v10_health_20261019.py
It prints all row counts, completion/failure flags, storage and host availability.
B0 likelihood is fully complete; both raw-control secondary jobs are still waiting
for their original legacy notes. Four augmentation-model secondary jobs run.
Posthoc completed-v11 error decomposition is prepared/public-source reproducible:
variable_depth_writer_v11_error_decomposition_20261019.json. On composition,104/206
variable-system errors have exact teacher inputs; all409 matching-input native
answer sequences agree across independent jobs.71 errors have changed inputs where
the teacher succeeds;31 are changed inputs where both fail. No new observations.
