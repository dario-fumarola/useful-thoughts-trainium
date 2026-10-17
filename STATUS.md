# Current state — 2026-10-17 18:32 UTC

**Both C runs finished256/exit0/cleanup and verified terminal backups. R1234 is
at41/256. C1234@64 development is complete; its terminal thought evaluation is running.**
The owner authorizes sustained autonomous work, agents and public Git updates.
Follow `CODEX_NEXT_DAY.md`; its companion review file was not found. Weeks are
available. The frozen14-day window is a planning bound, not an instance shutdown.
Public repo: https://github.com/dario-fumarola/useful-thoughts-trainium .
Only `/home/ubuntu/dario` and its authorized local assets/outputs are used.

## Active work and ownership

| Job | Current state | Control / output | tmux / actual processes |
|---|---|---|---|
| R1234 | Complete41 at18:30:16UTC,640.191s;5,248 records,2,692,224 source tokens,1,690,126 sampled actions; finite sampled updates, replicas agree | `local/science/control/qwen-r-seed1234-science-256-001` / `local/nvme/runs/qwen-r-seed1234-science-256-001` | `dario-science-R1234-256`; launcher1668169, timeout/PGID1668180, driver1668181,32 workers on logical0–31 |
| B0 local DEV512 |347/512 original notes at18:32; likelihood scoring not started | `local/evaluation/local-dev512-v1/b0-plan-v1` / `local/nvme/evaluation/local-dev512-v1-B0-001` | `dario-dev-local-B0-512`; launcher1857552, generation timeout1857562, worker1857563; core32/device8 |
| Held C1234@256 |Direct and empty-format128/128 generated/scored; thought6/128 at18:32; started17:44:55.869722UTC | `local/evaluation/held-dev128-v1/c-plans-v1/C1234-step0256` / `local/nvme/evaluation/held-dev128-v1-C1234-step0256-001` | `dario-dev-held-C1234-256`; launcher1934659, current thought timeout1958512/worker1958513; core36/device9 |

PIDs are observations, not launch authority; recheck start times, ancestry and
actual topology at each action. Each evaluator changes its child at stage changes.
All C64 processes/groups are gone. Its earlier PIDs are historical.
B0 local and the now-complete C64 suite started at15:37–15:38 on separate devices;
`validation/science_development_start_20261017.json` pins their actual allocations.
They are nonfinal development, with qualification=true and final identity null.
No prompt/cap/seed/terminal rule was changed after observing outputs.

Existing agents: `/root/distributed` supervises R through32/48/64 and verified
EBS64; `/root/data` supervises the held suite through complete coverage/exit/cleanup,
then computes the full paired comparison and the same fixed-first4 example set.
It now watches C256, without partial accuracy inspection. The completed C64/B0
report is saved and backed up. `/root/review` is idle after final-controller preparation/review. Root owns all
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
2. Let the current C1234@256 suite finish all three128 modes/exit0/cleanup and
   its data-agent report/progress-writer shutdown. Verify/archive, then explicitly
   release `python3 local/evaluation/release_idle_development.py held-C C5678-step0064`
   and launch that slot's reviewed `launch.sh` in fresh tmux/its own launcher.log.
   Queue C5678@256 and actual R64 without checkpoint selection, one held suite.
   The helper only releases; it never launches/retries.
3. Closed held/same-model development output backup:

   `python3 local/evaluation/archive_completed.py --control <exact-slot-directory> --output <exact-NVMe-output>`

   Reviewed SHA3d10ad2a…3bdce. Requires complete successful exit, departed owned
   PID/groups and idle owned device, then hashes every ordinary small source/copy
   file on distinct NVMe/EBS. Original predictions/scores/failures/logs remain.
   This helper intentionally excludes the separately labeled crossed-reader pack.
4. After complete B0 local512 notes **and9216 score rows**, exit/cleanup and backup,
   run `python3 local/analysis/freeze_dev_strata.py` once. It implements the already
   public token-class and tie-broken B0 raw-K1 surprisal quartiles using original
   score-row target texts, without opening corpus/final targets. Synthetic
   class/tie/failure checks pass; no actual likelihood/map has been read/created.
   Then release `local-trained C1234-step0064` with the same root helper and use
   `local/evaluation/local-dev512-v1/trained-plans-v1/launch.sh C1234-step0064`.
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

## Complete first trained development check

C1234@64 finished17:37:02.403816UTC/exit0, all original identities and six timeout
process groups gone. Same fixed128 GSM8K TRAIN items, three exact B0 protocols,
FP32/cap1024/answer256/parser/source/checkpoint/output hashes verified. Full report
`validation/held_dev_C_seed1234_step64_20261017.json`, SHA2c4ef1cf…aff178:

| Mode | B0 /128 | C64 /128 | C−B0,95% conditional interval (percentage points) |
|---|---:|---:|---:|
| Direct |113 |111 |−1.5625 [−6.25,+3.125] |
| Empty format |107 |108 |+0.78125 [−4.6875,+6.25] |
| One thought |63 |60 |−2.34375 [−11.71875,+7.03125] |

No clear C−B0 change on this small fixed development subset. Thought caps C32/B04,
answer truncations36/14, format errors11/33; all stay counted. Mean thought/answer
actions326.242/151.547 for C versus124.484/126.297 for B0. Mean item wall37.994/22.858s;
actual cold/cache/shared-host observations, no subtraction/causal throughput claim.
Terminal256 remains primary; R comparison pending. Two analysis-only field-projection
errors (device xla:0 and public asset-field naming) are preserved as helper errors,
not evaluation failures; source/output/criteria never changed.

Root verified every score/ID/protocol/hash, then all6 cells' caps/tokens/latency.
EBS `local/backups/evaluation/held-dev128-v1-C1234-step0064-001/VERIFIED.json`,
SHAc16c61e4…c1215d,53 files/8,966,243B, source/copy hashes verified. Includes full
predictions/scores/logs/control/report and a human-readable fixed-first4 example
render for B0/C64×all3 modes (verbatim outputs, failures, counts/stops/timing).
Public backup/start receipt: `validation/held_dev_C_seed1234_step64_backup_20261017.json`.
C256 actual release053a645c…2522b8; hostc1a6ae6a…7e1825a; root first-live
f27f662b…65b2f verifies source cwd/environment/private RAM temp/actual device9,
B0 device8 and the unchanged32-worker R group. Its first capture is honestly
later than startup; exact UTC is in that control's `processes.first-live.json`.

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

No actual final plan/model lock/projection/freeze/roster exists. Earlier exclusion
prepared question/context screening data, with the LAMBADA final target removed;
do not claim test questions were never accessed. No official performance scores
have been computed.

Reviewed public controllers in `validation/orchestration/final_protocol` have
exact staged copies in `local/evaluation/final-protocol-v1`;15 synthetic tests
pass. After complete B0 local scores, `prepare_final.py declare` can reuse exact
DEV protocol dictionaries. Only after all four actual terminal256/exit/cleanup/
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
