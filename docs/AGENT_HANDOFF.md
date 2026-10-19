# Current research handoff — 2026-10-19 04:04 UTC

Read NEXT_STEPS.md and STATUS.md first. The owner authorizes autonomous research,
maintaining the public repository, and existing-node training. Continue until
stopped. Agents are quota-unavailable; do not retry spawning them.

The main question is whether a single model trained with verified computations
improves fresh real-document predictions over source-matched raw CPT. Training
benefit and extra-inference benefit are independent endpoints. Keep the separate
writer–reader system as one bounded diagnostic, not the main architecture.

## Exact next action

The real-context eight-rank qualification has completed all eight updates and both
processes are physically closed. The independent audit is RUNNING:

- Control: local/diagnostics/real-trace-cpt-v10-world8-width640-qualification-v2
- Audit log: local/analysis/real-trace-v10-qualification-independent-audit-20261019.log
- Audit program: local/analysis/audit_real_trace_qualification_v10r2_20261019.py
- It has already independently passed all 16 native vectors and master witnesses.
- Require actual PASS_ACTUAL_REAL_TRACE_SFT_FULL_RESTORE, including protected full8.

After PASS, use immutable training worktree local/worktrees/real-trace-cpt-v10,
commit 3d8666b. Prepare with local/analysis/prepare_real_trace_study_v10_20261019.py,
review all six protocols, then local/analysis/release_real_trace_study_v10_20261019.py.
The release program protects source/protocols/evidence on EBS before launch.
Launch three tmux controllers, each C/supervise-partition.py with argument
real_raw, real_prior, or real_post. Physical partitions are 0/1, 2/3, and 4/5.
Two seeds run serially per arm. All six start fresh B0; 320 updates, terminal primary.
No automatic retry, outcome-based extension, or qualification-model warm start.

Training shape: world8 × micro4, global32, width640, target640, head chunk32;
BF16 active parameters/gradients, FP32 masters/Adam, one ZeRO reduction owner.
Warm fresh updates measured about six seconds; cold compilation took much longer.
Do not interpret compilation time as sustained throughput or finite loss as capability.

## Frozen data and endpoints

TRAIN5120 is unchanged, SHA437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915.
Primary held data is local/nvme-study/data/real-trace-cpt-v10-held-panels-v2.
Manifest SHA22bd05cb3ad07e08a17e30578bebe07629216c9ad3910e9270bf0794b34dff24.
V1 string novelty missed 10 TRAIN-equivalent ASTs. V2 was frozen before learning;
256 documents contain 256 distinct ordered exact-value ASTs, all absent from TRAIN.
TRAIN has 5,120 documents and 4,299 distinct ASTs. Preserve V1 and all old observations.
The ordinary256 panel is unchanged. Legacy256 remains separate, with expression overlap.

Primary training: post.raw − raw.raw full16 LL >=0.25 nat and positive paired CI in
both seeds, with ordinary mean NLL regression <=0.02 and CI upper <=0.05 nat/token.
Primary inference: prior.own − raw.raw >=0.25 nat and positive CI in both seeds;
own minus same-model raw/empty must also have positive CIs. Report all ordinary
regressions and unchanged-B0 absolute paths. Do not add gates after outcomes.
20,000 paired document bootstrap draws, seed2026101915. Checkpoint-conditional
uncertainty is not training-seed uncertainty. Preserve all failed/capped attempts.

## Active evaluations

Math diagnostic: both fresh fixed/variable writers finished160 and passed the
full independent six-checkpoint audit. Both models and latest variable160 full
optimizer are EBS protected. Training source5af3431, evaluation source11512db,
immutable worktrees variable-depth-writer-v11 and variable-depth-eval-v11.
Registry: local/diagnostics/variable-depth-writer-v11-v1/evaluation-jobs.json.
Five V3 four-shard jobs use physical8/9/10/11/14. Reader own/other controllers wait
on6/7 for original notes. All20 native checks pass; new cache shapes still compiling.
V1/V2 preparation runtime mismatches stopped before any accelerator worker;
retain holds and identical reused CPU references. No held-out accuracy result yet.

Real baseline: local/diagnostics/real-trace-v10-B0-DEV_AST256-notes-v1,
tmux real-trace-B0-notes-v10, physical15/logical60–63. All4 native checks pass;
new544 prefill compiles. No raw continuation targets enter generation.
Real evaluation source is local/worktrees/real-trace-eval-v10, latest29c8b24,
1,646 CPU checks. New CPU scorer reproduces all256 ordinary reference vectors
exactly. Frozen projections are local/nvme-cpt/data/real-trace-cpt-v10-evaluation-inputs-v2,
EBS protected. V1 preparation omitted leading whitespace in a secondary component
check and was held before model execution. V2 retains all primary16-token targets;
secondary components are unsupported on8 AST/5 legacy items, explicitly reported.
Raw-training eligibility remains independent of note-generation errors.

## Storage and operational invariants

Keep35 GiB free per filesystem and512 GiB host availability. Root free about89 GiB
before the real qualification archive. All original full checkpoints stay on NVMe.
EBS retains all terminal models and latest complete resumable scientific state,
currently variable160. Older inactive duplicate optimizer payloads retire only
after fresh original/EBS full hashes and newer full recovery protection. Model
files, metadata, receipts, and original states remain. This is same-instance backup.

NEW local/nvme-cpt is already formatted, fsync/readback tested and mounted:
UUID5119e7ab-0ca6-4295-b8f0-e856d87b4b67. NEVER rerun mkfs; remount UUID if needed.
Original local/nvme-study UUID63bee3f4-6b43-4290-a0f8-1c8946b7a509 is intact.
Use local/nvme-cpt/runs for the six real runs. No fstab/cloud/IAM/network changes.
No paid teacher service or unapproved bucket upload. Never print broad process
arguments or environment values; unrelated processes can contain credentials.

MuSiQue TRAIN preparation is complete and CPU-only: 19,938 authored questions,
correct composed_question_text, all paragraphs retained. V2 inventory and source
are protected; no cohort/model/evaluation release. Do not start a third large study.

The completed two-seed v7 study remains a negative architecture comparison:
full CoT wins all eight cells; generated composition answers20/19 of512 versus
fullCoT273/234. Original source57d3220 is immutable. Main result a0ae5ec and linked
reports preserve all51,200 answers/4,096 notes. Never resume R1234, stopped63.
No broad RL, unqualified8B, or official TEST. Existing review ZIP was linked today;
do not send another unless needed. Papers are in docs/REFERENCE_NOTES.md.

Keep analysis short and act on the next concrete task. Send meaningful updates
within60 seconds, wait at most45 seconds at a time, and let legitimate compiles finish.
