# Current research handoff — 2026-10-19 05:00 UTC

Read NEXT_STEPS.md and STATUS.md first. The owner authorizes autonomous research,
maintaining the public repository, and existing-node training. Continue until
stopped. Agents are quota-unavailable; do not retry spawning them.

The main question is whether a single model trained with verified computations
improves fresh real-document predictions over source-matched raw CPT. Training
benefit and extra-inference benefit are independent endpoints. Keep the separate
writer–reader system as one bounded diagnostic, not the main architecture.

## Exact next action

Mainline training is LIVE. Control: local/diagnostics/real-trace-cpt-v10-study-v1.
All three seed2026101913 models reached320,savedterminalandphysicallyclosed.
Fresh8rank seed2026101914 runs use physical0/1 raw,2/3 prior,4/5 post;alllive. Tmux: cpt-v10-real_raw,cpt-v10-real_prior,
cpt-v10-real_post. Secondseedrunsare past115updates;firstseedcheckpointsremainunchanged. Immutable training source3d8666b,local/worktrees/real-trace-cpt-v10.
All six fresh B0;320 updates,full160/320,terminal primary. Monitor active logs,
metrics,memory and device ownership. No automatic retry or outcome-based extension.

The real-context eight-rank qualification independently PASSES:all4fullstates,
native16,actualsaved6restored8optimizerdigests,finiteFP32masters/moments,BF16casts,
counters and memory bound. Late groupedPSSgrowth1.467GiB/save<2. Latestfull8 EBS
readback verified. Public qualifier and launch reports in validation. The complete
sciencePRE_LAUNCHsource/protocol/release archive was EBS-readback verified before
launch. No terminal scientific result yet. Protect first newsciencefull160 when
available;onlythenrotateolderEBSoptimizerduplicates with fresh readback.

Training shape: world8 × micro4, global32, width640, target640, head chunk32;
BF16 active parameters/gradients, FP32 masters/Adam, one ZeRO reduction owner.
Warm fresh updates measured about six seconds; cold compilation took much longer.
Do not interpret compilation time as sustained throughput or finite loss as capability.

The first science160 is fully independently audited,readback verified and fsynced:
real_prior seed1913. Full optimizer/model on EBS at
local/backups/science/real-trace-cpt-v10-study-v1/FIRST_FULL160.
Older variable160 and qualification8 EBS optimizer duplicates were retired only
after fresh full original/backup hashes and newer protection. All originals and
all EBS model weights remain. Receipt:checkpoint_retention_after_real_CPT160_20261019.json.

Automatic terminal watcher is LIVE:tmux cpt-v10-terminal-pipeline. After all6
closures it runs independent all-state/native/counter audit,protectsall6models
and one latestfull320,then freezes evaluation bindings. No retries.
Automatic evaluation watcher is LIVE:tmux cpt-v10-six-evaluations. Afterbindings
it prepares6canonicalCPUreferences,reuses each forlegacy,andlaunches6four-core
notejobs(physical0–5)and6CPUscorers(16threadseach). Code in local/analysis:
watch_real_trace_terminal_pipeline_v10_20261019.py,
watch_real_trace_six_evaluations_v10_20261019.py.
Inspect these finite watcher logs and receipts;do not duplicate their work.

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

Math diagnostic COMPLETE and published:main a7835c7.
All7controllers/3panels,9,216answers/2,048notes independently audited.
Composition512:fixed124,variable306(+35.5pp[31.3,40.0]),teacher377,fullCoT273,B0262.
Canonical/natural91→409;predeclaredrepairgatePASS. Freshdepth2variable182vsCoT193;
unseendepth6variable10/256vsoracle78. One seed,differentcurricula/targettokens,
frozencompetitors:positiveboundedcurriculumrepair,notmatchedarchitectureadvantage.
Publicreport:validation/variable_depth_writer_v11_complete_20261019.json.
Allpayloads,figures,CSV,deterministicexamples,auditsourceandEBSarchive preserved.
Do notlaunchanotherlargersplitrescue. Training5af3431/evaluation11512db immutable.

B0realnotegeneration COMPLETE for bothAST256andlegacy256. Primarycontrol:
local/diagnostics/real-trace-v10-B0-DEV_AST256-notes-v1;legacyanalogous.
CPUlikelihoodscoringLIVE:tmux cpt-v10-B0-CPU-scoring,control
local/diagnostics/real-trace-v10-B0-CPU-scores-v1. Ordinaryreference reused explicitly,
not newindependentobservations. No trainedreal-model capability result yet.

Real evaluation source local/worktrees/real-trace-eval-v10,29c8b24,1,646CPUchecks.
CPUscorer reproduces all256ordinary reference vectors exactly. Fixedprojections:
local/nvme-cpt/data/real-trace-cpt-v10-evaluation-inputs-v2,EBSprotected.
V1preparationomittedleadingwhitespaceinsecondarycomponentcheck;heldbeforemodelrun.
V2retainsallprimary16targets;componentsunsupported8AST/5legacy,neverdropprimaryrows.
Rawtrainingeligibilityisindependentofnotegenerationerrors. Othernotesarenative,
intactandgloballyderanged;no trim,repair,utilityfilterorbest-ofselection.

## Storage and operational invariants

Keep35 GiB free per filesystem and512 GiB host availability. Root free about83 GiB
after verified duplicate rotation. All original full checkpoints stay on NVMe.
EBS retains all terminal models and latest complete resumable scientific state,
currently real_prior seed1913 step160. Older inactive duplicate optimizer payloads retire only
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
