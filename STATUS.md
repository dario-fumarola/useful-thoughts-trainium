# Current state — 2026-10-17 21:43 UTC

**Both C runs finished256/exit0/cleanup and verified terminal backups. R1234 is
at59/256. Held C1234@256 is complete/analyzed/backed up; held C5678@64 and
local C1234@64 are running. The matched R/C capability result is pending.**
The owner authorizes sustained autonomous work, agents and public Git updates.
Follow `CODEX_NEXT_DAY.md`; its companion review file was not found. Weeks are
available. The frozen14-day window is a planning bound, not an instance shutdown.
Public repo: https://github.com/dario-fumarola/useful-thoughts-trainium .
Only `/home/ubuntu/dario` and its authorized local assets/outputs are used.

## Active work and ownership

| Job | Current state | Control / output | tmux / actual processes |
|---|---|---|---|
| R1234 | Complete59 at21:41:19UTC,676.415s;7,552 records,3,874,176 source tokens,1,777,519 sampled actions; finite sampled updates, replicas agree | `local/science/control/qwen-r-seed1234-science-256-001` / `local/nvme/runs/qwen-r-seed1234-science-256-001` | `dario-science-R1234-256`; launcher1668169, timeout/PGID1668180, driver1668181,32 workers on logical0–31 |
| Local C1234@64 |130/512 notes at21:43; started20:14:13.534663UTC, all512 original notes before scoring | `local/evaluation/local-dev512-v1/trained-plans-v1/C1234-step0064` / `local/nvme/evaluation/local-dev512-v1-C1234-step0064-001` | `dario-dev-local-C1234-64`; launcher2030185, generation timeout2030200/worker2030201; core32/device8, ports64501/64502 |
| Held C5678@64 |10/128 direct at21:43; started21:41:24.431770UTC | `local/evaluation/held-dev128-v1/c-plans-v1/C5678-step0064` / `local/nvme/evaluation/held-dev128-v1-C5678-step0064-001` | `dario-dev-held-C5678-64`; launcher2082723, direct timeout2082732/worker2082733; core36/device9, ports64471/64472 |

PIDs are observations, not launch authority; recheck start times, ancestry and
actual topology at each action. Each evaluator changes its child at stage changes.
All held C1234@64/@256 and B0 local processes/groups are gone; their earlier PIDs are
historical. C1 local root first-live proof SHAa794aa84…3fb885 verifies its actual
source/environment/private RAM/device8, real C64 checkpoint, all32 original R
identities and separate held C256 device9 at capture. Release07d61d34…0f940/host5f8a23db…662a14.
Current C5678@64 release6d754ca9…f04fb8/host4d0732b5…caade; first-live96a68e1f…3375ac
verifies actual source/environment/private RAM/device9 and the unchanged R/local groups.
B0 local and the now-complete C64 suite started at15:37–15:38 on separate devices;
`validation/science_development_start_20261017.json` pins their actual allocations.
They are nonfinal development, with qualification=true and final identity null.
No prompt/cap/seed/terminal rule was changed after observing outputs.

Existing agents: `/root/distributed` supervises R through32/48/64 and verified
EBS64; `/root/data` supervises the held suite through complete coverage/exit/cleanup,
then computes the full paired comparison and the same fixed-first4 example set.
It now watches C5678@64, without partial accuracy inspection. Both closed
C1234 held reports and B0 local are saved and backed up. `/root/review` is idle after final-controller and DEV-summary preparation/review. Root owns all
releases, launches, backups, signals and public updates. No automatic retry/run-all.

Root-owned shared supervisors remain RUNNING:

- Backup1606740, parent timeout/PGID1606728, tmux `dario-science-backups-256`;
  roster `local/science/backups.roster.json`, state `local/science/shared-backups.state.json`.
  Both C runs are terminal-verified. One serialized queue, latest-two EBS/run,
  allowance for two incoming payloads before prune; all NVMe originals retained.
- Node helper1606741, producer1606742, gzip1606743, timeout/PGID1606734, tmux
  `dario-science-node-256`. Config `local/science/monitor.config.json`; execution
  `local/nvme/science/shared-node-monitor/{state,launch}.json`, ongoing
  `node.jsonl.gz.partial`. Exactly one global producer. The initial static gzip
  argv capture was pre-exec; actual live argv is correct. Do not mistake static
  startup metadata for current process state or an active prefix for a closed log.

## Immediate next actions

1. Let R1234 finish the fixed256 updates. Preserve64/128/256 plus12-hour saves;
   no R checkpoint has published yet. Do not stop for early short notes or poor
   TRAIN utility. R5678 remains unreleased.
2. Let held C5678@64 finish all3 full128 modes/exit0/cleanup and its data-agent
   complete report/fixed-first4 rendering/all-control-writers-stopped callback.
   Root verifies and archives before another held release. If actual R1234@64
   and its verified EBS backup are ready then, evaluate that fixed milestone next
   with the reviewed R helper; otherwise release C5678@256 with the original
   helper. Every fixed64/terminal cell stays required; availability determines
   scheduling, not scores. One held suite, no retry/checkpoint selection.
3. Closed held/same-model development output backup:

   `python3 local/evaluation/archive_completed.py --control <exact-slot-directory> --output <exact-NVMe-output>`

   Reviewed SHA3d10ad2a…3bdce. Requires complete successful exit, departed owned
   PID/groups and idle owned device, then hashes every ordinary small source/copy
   file on distinct NVMe/EBS. Original predictions/scores/failures/logs remain.
   This helper intentionally excludes the separately labeled crossed-reader pack.
4. Let local C1234@64 finish all512 original notes and9216 scores, exit/cleanup.
   Reuse the single actual B0 map `local/analysis/local-dev512-v1.strata.json`,
   SHAfec318f7…7d5598; never rerun freeze_dev_strata.py or rebin trained outcomes.
   Root verifies artifacts/owned cleanup, then uses reviewed summarize_dev_strata.py
   SHA194281ae…5fc90 (small root-verification schema in its docstring). All18
   absolute arms, all statuses and fixed strata remain; invalid pairs withhold CIs.
   Archive the closed output/control before reusing device8 for another fixed
   development slot. All prospective cells remain required; no checkpoint selection.
5. The reviewed additional helper `local/evaluation/release_idle_development.r-ready.py`
   SHA279c984b…e31f4 supports `held-R R1234-step0064` once that real checkpoint
   exists. It reuses the pinned R pack checker and binds actual counters; no
   forecasted actions or changed numerical gate. Root review
   `release_idle_development.r-ready.root-review.json`, SHAc315e460…2e8a.
   Current original helper remains bf5460b4…8b213. Both have pending-release
   reservations, exact process/leader/phase, host/ports/private-temp checks and
   dynamic EBS retention accounting. Only the R1234-lower/idle-upper phase is
   supported now. Later R5678-upper/post-training releases need actual-phase
   support. R1 terminal development waits for that phase; do not delay R2 with it.
6. After actual R1234 terminal256, exit/whole-group cleanup and verified EBS256,
   release independently fresh-B0 R5678 using
   `local/science/preparation-256/release_next_arm.py` (SHAc25a8347…6128d) and its
   reviewed finite launcher. Require fresh runtime topology, free ports, actual
   headroom, matching leaders and private RAM tempfile proof. Remove the probe's
   empty outer TMPDIR before the launcher creates it. Never execute old coordinators.

## Complete unchanged-B0 local DEV512 audit

Completed20:11:02.407720UTC/exit0/cleanup, all512 notes and9216 scores statusok.
Root verified exact IDs/documents/18 keys, original notes/controls/protocol/model/
source hashes and all18 absolute means. Root proof2e65b744…de7a07.
Public exact `validation/local_dev_B0_20261017.json`, SHA706f7081…c8c034:

| K1 condition | Mean true-next-token log likelihood, nats |
|---|---:|
| Raw prefix |−1.118239 |
| Empty format |−7.360996 |
| Own sampled note |−11.203003 |
| Length-matched filler |−9.301484 |
| Shuffled own content |−7.326900 |
| Other note |−10.918726 |

Own−raw=−10.084765,95% conditional CI[−10.517006,−9.681984]; own−empty=−3.842007,
CI[−4.216340,−3.494004]. This is a baseline interface/content cost on fixed random
DEV documents, not an R result or proof training cannot help. All K4/K16 and
content contrasts/strata remain in the report. No protocol was changed.

Single B0 map fec318f7…7d5598 is frozen:303 letter/117 punctuation/59 digit/33
whitespace; four surprisal bins128 each, no failures/unassigned. Reuse memberships
unchanged for every trained reader. Actual summary helper review7afeb930…ac067c;
four synthetic tests and actual full B0 execution passed, without a new math gate.
Writer293480 actions/mean573.203,329 EOS/183 caps; original wall15373.6455s;
reader975.6816s; launcher16392.8894s. Shared-host/cold costs retained.
Execution report cc0b0007…a0b97; backup/start report740fca0d…d9116.
EBS `local/backups/evaluation/local-dev512-v1-B0-001/VERIFIED.json`, SHA325f94bd…ba90b,
31 files/28,195,977B, source/copy hashes verified. Includes complete originals,
strata map, reports and the first4 frozen-order readable prefix/note/DEV-future/
all-control examples; no score selection. All B0 control writers are stopped.

## Complete C1234 held development64 and terminal256

C256 completed21:37:20.790107UTC/exit0/all identities and7 groups gone. Root
verified actual terminal checkpoint8c20a705…6706/modelb2b8e2eb…fc6a1, exact B0
protocols/ordered128 TRAIN IDs/gold/output hashes and all9 cells' score/status/
cap/action/token/latency totals. Raw proofdc03c054…177e1a; report proof7bca83a9…b91f45.
Final public report `validation/held_dev_C_seed1234_step256_20261017.json`,
SHAa10ee312…9379de (all superseded analysis drafts preserved outside closed control):

| Mode | B0 | C64 | C256 | C256−B0 pp,95% conditional interval |
|---|---:|---:|---:|---:|
| Direct |113 |111 |109 |−3.125 [−8.59375,+1.58203] |
| Empty |107 |108 |109 |+1.5625 [−3.125,+6.25] |
| Thought |63 |60 |91 |+21.875 [+11.71875,+32.03125] |

C256 thought still trails its own direct/empty by18/128, CI[−21.875,−7.03125]pp.
Its128 thoughts all cap1024, with131072 thought/16130 answer actions, mean91.414s
per item. B0 thought has4 caps/33 format errors/14 answer truncations; C256 has128
caps/0 format errors/14 truncations. Of the33 B0 format-error cases,26 become
correct; other B0-status cohorts contribute−8(ok)/+10(truncated), net+28. This is
an observed status decomposition, not isolated reasoning-content causation.
Both64 and256 were fixed before outcomes; terminal256 remains primary. No R or
final-test result is established by this C development comparison.

EBS `local/backups/evaluation/held-dev128-v1-C1234-step0256-001/VERIFIED.json`,
SHA00331dc2…94538:54 files/10,268,574B, every ordinary source/copy verified.
Includes all raw outputs, reports and the same first4 frozen-order TRAIN cases,
B0+C256 ×all3 modes, with verbatim outputs/gold/score/status/cost/stop metadata.
Public backup/next-start report8bc891f3…001f9. C256 control writers remain stopped.
Earlier C64 report2c4ef1cf…aff178 and its53-file EBS backupc16c61e4…c1215d remain.

## Scientific execution and saved evidence

Both C seeds consumed exactly32,768 unique TRAIN records /16,809,984 source
tokens, with exact within-seed planned order. All original35 processes/groups
are gone. Latest-two verified EBS copies are128/256; NVMe64/128/256 remain.

| Arm | Terminal / exit UTC | All254 steps3–256: mean / median / linearP90 | Save-publication total / release-to-exit | Reports in `validation/` |
|---|---|---|---|---|
| C1234 |09:29:22 /09:29:37 |10.0564 /10.0566 /10.0963s |113.920 /3100.071s |`science_C_seed1234_{execution,backup}_20261017.json` |
| C5678 |10:27:17 /10:27:32 |10.0298 /10.0312 /10.0674s |110.654 /3093.213s |`science_C_seed5678_{execution,backup}_20261017.json` |

C1234 terminal COMPLETE8c20a705…6706/modelb2b8e2eb…fc6a1;
C5678 terminal16f22b7c…97bb/model740b4023…4cc. Root checked exact source/copy
manifests, helper receipts and all33 payload sizes; the pinned queue performed
full payload hashes. Raw C run metadata, metrics, all checkpoint manifests and
launch logs also have verified EBS copies in `local/backups/run-artifacts`.
C5678's earlier pre-loop label was a bounded-tail monitoring error, corrected
from its whole log and metrics; original captures remain preserved.

R1234 released10:32:53, started10:33:30; freshB0/world32/global128/accum4/G4/
cap1024/K1/T.7/FP32/LRU2/private RAM temp. Raw log:
`local/nvme/science/launch-logs/qwen-r-seed1234-science-256-001.log`.
The public first-live receipt observes26 at15:37 (not a fabricated early capture):
`validation/science_R_seed1234_progress_20261017.json`. All first32 metric rows
are preserved in `validation/science_R_seed1234_first32_20261017.json`, SHAe2e50f5f…7e7c5d,
with standalone plot `validation/figures/science_R_seed1234_first32_20261017.svg`.
All30 steps3–32: mean662.489s, median623.155s, P90811.018s; all observations and
shared-host development costs retained. These are changing-workload TRAIN
measurements, not a fixed-anchor learning curve or downstream result. Short-note
updates were followed by increased generation; no permanent stopping pattern.
R32 live identity receipt SHA43a4e38c…3d239 is in its control directory.
R48 completed19:40:28,650.880s. Its immutable health receipt SHA2f06c80e…6cf5b6
retains all46 steps3–48: mean654.096s, median641.035s, linearP90772.742s, no drops
or overlap subtraction. Root independently checked its original48-row byte-prefix
SHA, step coverage, counters and timing statistics. All32 identities/finite values/
sampled replicas pass. Actual R checkpoint/order metadata and EBS64 remain pending.

All32 completed rank0/micro0 example bundles, run snapshot, first32 metrics and
explicit ongoing-log byte prefix have verified EBS copies:
`local/backups/run-snapshots/qwen-r-seed1234-science-256-001/step-0000032/VERIFIED.json`,
SHAabcdbdb7…cff03,39 files/22,205,371B. This is **not a model/optimizer checkpoint**.

## Frozen source, qualification and interpretation

Fixed spec653b4efa43e933839733075eb5c2184a80de6dd24404263a8c9297adb9e241f7,
`local/science/experiment-256.json` file SHA22157721…c0dcda; exact deadline
**2026-10-31T08:27:49+00:00**. Immutable source `local/science/source-256-001`,
manifest SHA8a0641c6…77faa4, qualified61-file fingerprint
**244e2f13f91dfc746f0829b236a35b8f2c5452f5d62da7a1bc3b3bbe31287ff7**.
Numerical source remains unchanged. Use archive cwd and override PYTHONPATH to
archive+`local/test-deps` **after** sourcing env.sh, which hardcodes live-root paths.
Python `/home/ubuntu/ema-trainium/.venv/bin/python`. Installed CLI is
`/opt/aws/neuron/bin/neuron-ls --json-output`; default view identifies allocations.
Normalized format changes only tokenizer_dir to the same absolute B0 tokenizer;
original format and failed first freeze are preserved with the public path bridge.

Qualification is CLOSED: actual nine-check receipt
`local/science/qualification-256.json`, SHA3785fed6…45fd9. Public exact copies:
`validation/science_256_{frozen,qualification,resources}_20261017.json`.
R14 full execution/recovery pass; actual two-R execution observed, not ten steady
simultaneous updates or a2× credit. Reports `r_fp32_ram_{execution,resume}_20261017.json`.
C12 recovery passes its declared distinctions. Original coordinate v1 FAIL and
prospective held-seed v2 derivative PASS remain; no v3/FP64 investigation without
an actual new fault. Latest required233 CPU checks +toy PASS in
`local/checks/tests.XAA5WQ` (existing `local/test-deps` on PYTHONPATH). Qualification
EOS/negative-utility diagnostics are preserved, not capability evidence.

The first study is one serial C1234,C5678,R1234,R5678 reference, independently
freshB0 in every arm. Within each seed, exact source order/exposure match; cross-seed
overlap3,182 is disclosed. Terminal256 is primary, fixed64 development never
selects a checkpoint. Primary R-thought−C-thought; also R-direct−C-direct,
R-thought−R-empty and each arm versus B0. Direct permits ordinary calculations.
Format recovery or C degradation alone is not capability above the starting model.

Actual Qwen3-1.7B-Base/FineMath revisions, asset hashes and audited corpus are in
the frozen spec:329535 TRAIN/18360 DEV/18239 TEST, random512-prefix/16-future
anchors; no whitespace/entropy filtering/cycling. B0 held128 GSM8K **TRAIN**
baseline is113 direct/107 empty/63 thought, all failures counted. Its full51-file
payload archive is verified on EBS; public `validation/held_dev_B0_20261017.json`.
Local fixed DEV512 uses one original sampled note/doc (seed1729/T.7), all global
notes before controls,6 absolute conditions/K1 primary/K4,16 secondary. Reuse one
B0-derived stratum map for all readers; `validation/local_dev_strata_protocol_20261017.json`
was declared before local likelihood outcomes. Paired5000/seed17 document
bootstrap is conditional on checkpoints, not training-seed uncertainty.

Reviewed packs are under `local/evaluation/held-dev128-v1/{c,r}-plans-v1` and
`local/evaluation/local-dev512-v1/{b0-plan,trained-plans,crossed-plans}-v1`.
The12 prospective terminal DEV crosses reuse original complete global writer
bundles; no resampling or deployed-policy claim. Root releases one local job and
one held suite at a time, only with actual host headroom. Current two packs use
core32/36 during R1; opposite-phase plans use0/4 during R2 or honest post-training.

## Final evaluation — prepared, unlaunched

Actual metadata-only plan-v1.json was declared20:14:31.316051UTC,
SHA97f6bb40…51a6e, PREPARED_NOT_LOCKED; public exact
`validation/final_evaluation_plan_20261017.json`. No actual model lock, official
projection, evaluation freeze or execution roster exists. Earlier exclusion
prepared question/context screening data, with the LAMBADA final target removed;
do not claim test questions were never accessed. No official performance scores
have been computed.

Reviewed public controllers in `validation/orchestration/final_protocol` have
exact staged copies in `local/evaluation/final-protocol-v1`;15 synthetic tests
pass. The declaration reused exact completed B0 protocol dictionaries and ID-only
metadata; it never relabels DEV output. Only after all four actual terminal256/exit/cleanup/
verified EBS states: full existing final-model payload verification, immutable
pre-gold recipe/model lock, then canonical official projections, then separately
timed evaluation-freeze-v2. Never relabel DEV bundles as final TEST2048.
Public report: `validation/final_evaluation_preparation_20261017.json`.

Additional reviewed **ignored preparation only**, `local/evaluation/final-execution-v1`:
plan.py SHA3a15e3bd…5bfca, aggregate.py SHAd47178f9…fddde, root review
SHAabffe6fb…981b. Nine tiny synthetic tests pass; no real rows/models/gold read.
Planner binds all5 models/13 protocols and supports1/2/4/8/16 shards/concurrency
(default4). All global writer notes precede scoring; every reader receives all
its writer shards. Controls are globally constructed but output partitioned.
Same-model cells alone are scheduled; writer-specific protocols are not a reader/
writer pair allowlist. Aggregator uses exact sealed summary/bootstrap ASTs,
full global coverage and failures, one complete cell only.

The finite foreground `execute_group.py` is also reviewed preparation only,
SHAbcf73aa5…5759ac; `executor.root-review.json` SHA9ededf0b…1e4831.
It executes exactly one explicit root-released roster group, checks real final
metadata/source/dependencies and all-writer barriers, then freshly admits actual
idle devices/headroom/private RAM/ports. Pending/preallocation development jobs
and prior final owned survivors block admission. Remaining-deadline GNU timeouts,
actual PID/start/environment/mapping and process-local subreaper cleanup produce
immutable exits and aggregator-compatible snapshots. Nine tiny synthetic tests
pass; two pre-use draft defects (missing note-helper SCHEMA and incomplete live-job
exclusion) were corrected and specifically tested, with drafts preserved. No
actual final inputs, probes, roster/release or jobs were used. Actual root resource
floors/releases and successful execution/cleanup receipts are still required;
SIGKILL/host loss needs manual recovery. Whole-suite contrasts/report follow real
complete coverage. No numerical driver/evaluator or protocol change.

## Storage and review snapshot

Actual EBS free190.7GiB /NVMe~871GiB at this observation. Reserve eight science
EBS payloads total plus two rollover copies and20GiB; compute remaining capacity
from actual retained count, not a stale flat floor. All NVMe originals remain.
Obsolete qualification EBS duplicates were retired only after verifying all33
preserved NVMe payloads; receipt `local/qualification/ebs-retirement-before-science-20261017.json`.
Latest qualification EBS C12/R00414/RAM00514 remain, with unique upper continued4
at `local/backups/qwen-r-fp32-ram-resume-004/step-0000004`. No new disk formatted.
Cumulative per-process HWM overcounts shared pages/non-simultaneous peaks; use
actual MemAvailable plus frozen future-loader/workspace/safety reservations.
Current available~1TiB. Serial science was selected before its outcomes.
**Off-instance backup has no approved destination**; EBS and NVMe share this instance.
No new instances/IAM/networking/stack upgrade/paid teachers/unapproved uploads.

One review ZIP was built/shared today, from public HEAD04e8971 at08:59:
`local/review/2026-10-17-first-science/dario-first-science-20261017-checkpoint64.zip`,
3,886,635B, SHA17f3b507…6e5a9,388 entries. It preserves source/notices/frozen
reports, fixed TRAIN examples and full B0 held128 outputs; no weights/corpus/
cache/secrets. It is an earlier snapshot, not current HEAD. Next daily review
around Oct18 or if actually stuck; do not repeatedly rebuild or upload it.
