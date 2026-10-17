# Current state — 2026-10-17 09:37 UTC

**Scientific C1234 completed 256 updates; C5678 is running. No trained C/R comparison is available.**
The first study remains **256 terminal updates**, with fixed step64 development.
Public repository: https://github.com/dario-fumarola/useful-thoughts-trainium .
Work is isolated in `/home/ubuntu/dario`; original repositories are unchanged.
The owner authorizes sustained autonomous work, review agents and public updates.
`CODEX_NEXT_DAY.md` governs the queue. Its companion review file was not found.
Weeks are available; the14-day planning window is not an instance shutdown date.

## Current work and ownership

**C1234 completed all256 scientific updates**, fresh released B0, FP32/world32/
global128/accum4, on the audited production TRAIN pool. Terminal COMPLETE
published09:29:22; exit0 at09:29:37; all35 original identities and owned process
groups are gone. Exact planned order and32,768 records /16,809,984 source tokens.
Steps1/2 took237.866/94.141s; all254 steps3–256 retained: mean10.0564s, median
10.0566s, P90 10.0963s. Three save-publication walls total113.920s, separate from
step elapsed; release-to-exit3100.071s. Sampled replica checks pass throughout.
`validation/science_C_seed1234_execution_20261017.json`, SHAbc04dec0…21209.

Root's shared queue verified terminal EBS256 at09:33:48 and completed retention
at09:34:28: latest two128/256, all original NVMe64/128/256 preserved. Root checked
exact source/copy manifests, receipts and all33 payload sizes; the pinned helper
performed full payload hashing. `validation/science_C_seed1234_backup_20261017.json`,
SHA45847387…7df9d. No model-quality claim follows from execution.

**C5678 is active**, independently fresh released B0, sealed source-256-001,
same spec/FP32/world32/global128/accum4/default C cache/EBS temp. Upper logical
32–63; ports64411/64412. Actual32 workers occupy devices8–15, four/device;
lower0–7 are free. Launcher1638841, timeout/PGID1638853, actual driver1638854.
Root release09:35:59.192303UTC SHA5907f8da…264ae, hostc9f3b74b…e3442a.
All previous terminal/exit/backup/cleanup, live shared leader, source, host memory,
storage and port checks passed. Run started09:36:37; no update was complete at
first capture. Distributed owns read-only startup/mapping/health. Receipt
`validation/science_C_seed5678_start_20261017.json`, SHAf1996cd0…8c3bf; full
process/start/ancestry/env bindings are in control/processes.json, SHA4da9630b…20ce9.
Control `local/science/control/qwen-c-seed5678-science-256-001`;
output `local/nvme/runs/qwen-c-seed5678-science-256-001`; rawlog
`local/nvme/science/launch-logs/qwen-c-seed5678-science-256-001.log`;
tmux `dario-science-C5678-256`. Let healthy startup and all256 updates finish.
Runtime topology uses `neuron-ls --json-output`; `--show-all-procs` also lists
monitor/query processes. Earlier unsuccessful preflight views are preserved locally;
no training attempt or guard/source change preceded the successful release.

Root owns both matching shared supervisors: backup1606740; node helper1606741,
producer1606742, gzip1606743, timeout/PGID1606734. Roster/config
`local/science/{backups.roster,monitor.config}.json`; states
`local/science/shared-backups.state.json` and
`local/nvme/science/shared-node-monitor/state.json`. Both RUNNING. Only C1234 is
terminal-verified. No R science or evaluation is released. Qualification jobs,
watchers and original v1 coordinators are inactive. Unique upper qualification
continued4 has a verified EBS copy at
`local/backups/qwen-r-fp32-ram-resume-004/step-0000004`.

Frozen spec **653b4efa43e933839733075eb5c2184a80de6dd24404263a8c9297adb9e241f7**,
`local/science/experiment-256.json`, file SHA22157721…c0dcda; deadline
**2026-10-31T08:27:49+00:00**. Actual nine-check receipt
`local/science/qualification-256.json`, SHA3785fed6…45fd9; all ten timings included.
Original archive `check_ready.py` succeeds, stdout/stderr in preparation-256.
Public byte-identical copies: `validation/science_256_{frozen,qualification,resources}_20261017.json`.

## Qualification evidence

- **R full execution PASS:** RAM005 freshB0/seed5678, lower0–31, world32/global128/
  accum4, FP32/LRU2/private RAM temp, G4/cap1024/K1/T.7;14 complete updates.
  Fixed **all steps5–14**: mean735.977s, median704.789s, linear P90 869.283s.
  Preserve all shared-host recovery/CPU comparison/backup overlaps without
  subtraction. Checkpoint saves43.617–46.908s are separate; issued-to-exit wall
  11,279.171s. Actions shrink substantially: timing is this measured workload,
  not an isolated TMPDIR speedup. Report:
  `validation/r_fp32_ram_execution_20261017.json`, SHA57116908…25a3aa4e.
- **Strict R recovery PASS:** actual restored3 model/all32 optimizer shards/RNG/
  order/counters byte-exact; all128 first-resumed4 tapes identical. Full continued4
  meets original1e-7abs+1e-6rel, with retained non-bitwise differences (model max
  9.095e-13, optimizer2.910e-11). Sole update4/unchanged STOP/exit0/cleanup.
  `validation/r_fp32_ram_resume_20261017.json`, SHA099634ce…f0f321.
- **Actual coexistence observed:** both mapped32-worker groups executed in two
  same~5s raw windows, one after upper gradient work. This is one strict recovery
  episode with cold optimizer compilation, not ten simultaneous steady timings.
  No concurrency multiplier is credited. Evidence is in the execution report.
- **C qualification PASS:**12 updates, ten steady3–12 mean10.007s, strict actual
  restore/continued-state recovery with declared bitwise/tolerance distinctions.
  `validation/c_fp32_final_resume_20261016.json`.
- Qualified active source **3aaaf8a**, fingerprint
  **244e2f13f91dfc746f0829b236a35b8f2c5452f5d62da7a1bc3b3bbe31287ff7**,
  unchanged. Latest required checks233CPU+toy smoke PASS. Signed P2 derivative,
  original device future-perturbation counterfactual and FP32 evaluator parity
  pass with their explicit older-component provenance limits. **C coordinate v1
  FAIL remains preserved; separately prospective held-seed v2 derivative PASS.**
  Do not add v3/FP64 work without a concrete fault.

Final qualification TRAIN tapes show **511/512 empty notes**, all512 stop at base
EOS; mean raw gain is negative. R004's terminal also had509/512 empty notes. These
are stopping diagnostics, not useful-thought or downstream efficacy evidence.
All1,792 RAM005 tapes and14 rank0 example traces remain local. The prospective
256 terminal, objective, prompts and evaluation comparisons remain unchanged.

## Production plan and next actions

1. Observe active C5678 through real updates, checkpoint64/128/256 and verified
   EBS copies. C1234 is terminal and backed up. Require actual terminal256/exit/
   backup/cleanup before independently fresh-B0 R1234, then R5678.
2. Launch **one32-core training group at a time: C1234, C5678, R1234, R5678**.
   Every arm independently starts from released B0, never C/qualification weights.
   Seed1234 uses lower0–31, seed5678 upper32–63; C/R share each seed's exact order
   and32,768 fresh TRAIN records. Cross-seed overlap3,182 is disclosed.
   Serial fallback was selected before science from full-pool memory uncertainty.
   Use one reviewed shared EBS queue and one compressed node monitor; bind their
   live matching identities before each finite root release. C1234 and C5678 have separate actual releases;
   matching shared supervisors are active. Idle cores may serve genuine development work
   subject to actual host headroom.
3. Complete fixed held-development C/R64/256 in direct/empty_format/one_thought,
   plus the local fixed DEV512 controls. Never select the terminal by dev score.
   Keep official final outcomes sealed until protocol/model inventory is fixed.

Immutable `local/science/source-256-001` matches all61 qualified active files,
archived from commit85c64a9. Its manifest SHA8a0641c6…77faa4 binds the tar and
qualified source. Four finite launchers passed independent review; C1234/C5678 are released, R arms are not. Use archive cwd/PYTHONPATH and absolute evidence/control paths;
current root docs govern operations. Frozen training spec and actual qualification
receipt now pass original readiness. The normalized production format changes
only tokenizer_dir to the same absolute B0 path; original format remains intact.
The failed first freeze is preserved; reviewed bridge is public. No new numerical
gate or changed token/stop/sampling policy.

## Data, development and interpretation

Actual released Qwen3-1.7B-Base revisionea980cb0…604c1,1,720,574,976 unique values,
original embedding/head tie, FP32 both arms. FineMath production revisione92b25a6…,
audited329535train/18360dev/18239test, fixed512-prefix/16-future random anchors;
K1 training, no whitespace/entropy filtering or cycling. Source/load/format and
production order hashes are in `local/science/preparation-256/manifest.json`.

Fixed128 held GSM8K **TRAIN** development B0 is complete: direct113/128,
empty107/128, thought63/128. All failures/truncations and51 payload files are
preserved with independently verified EBS copies. Primary R-thought−C-thought;
also R-direct−C-direct, R-thought−R-empty and every arm versusB0. Format recovery
or C degradation alone is not capability above B0. Direct allows calculations.
Local K1 primary/K4,16 secondary; retain absolute clean/empty/own/filler plus
content controls and fixed random token-class/surprisal strata. No official final
outcomes read; earlier exclusion preparation used question-only benchmark data.

Prepared/reviewed finite development packs, none released: held128 C suites in
`local/evaluation/held-dev128-v1/c-plans-v1` (manifest8506c2d7…), local DEV512 B0
`local/evaluation/local-dev512-v1/b0-plan-v1` (prepared0b54fd2d…), and eight trained
local slots `local/evaluation/local-dev512-v1/trained-plans-v1` (manifest7cbdbc51…).
Use full global512 note bundles before controls; preserve writer bundles for later
terminal reader crosses. Planned idle cores32/36 during lower R1234,0/4 during
upper R5678 or honest post-training phase; one local job and one held suite at
a time, only with actual host headroom. All children use the same frozen deadline.
Source, sampling, prompts, controls and B0 pack are unchanged; no new numerical gate.

## Review snapshot

Local review ZIP was built from public HEAD04e8971 after fixed checkpoint64,
then independently checked:388 entries,3,886,635bytes, SHA
`17f3b5070bd5980d093953ca151bd0e5b5a29210ea23d101992949484006e5a9`.
Path `local/review/2026-10-17-first-science/dario-first-science-20261017-checkpoint64.zip`.
Includes source/notices/frozen reports, fixed rank0 TRAIN examples1/7/14 for
R004/RAM005, and unchanged held128 B0 predictions/scores. Eight raw metadata
omissions are explicit in the manifest; no tensors/corpus/cache/full environment.
Link shared with owner; training continued. This snapshot has no trained C/R
comparison or final test score. It remains local, with no external upload.

## Storage and limits

Root independently rehashed **all132 source/EBS payload reads** for RAM0057/14,
PASS: `local/qualification/ram-sequence-v2/lower/terminal-backup-verification.json`,
SHAd92787e4…8d084b7. The watcher retained7/14 then exited normally. Afterward,
root verified all33 preserved NVMe payloads for each obsolete EBS duplicate before
retiring C6, R0031, R0047 and RAM0057. All original NVMe states remain. Aggregate
retirement receipt `local/qualification/ebs-retirement-before-science-20261017.json`,
SHA011afcd5…6366b. Current latest qualification EBS copies: C12/R00414/RAM00514;
unique resumed4 is verified. Actual resource projection leaves12.38GiB EBS beyond
all retention/rollover/allowances and156.51GiB NVMe beyond saves/logs/contingency.
Science reserves latest-two EBS copies/run plus **two new payloads+20GiB** for
backlogged rollover, about71.3GiB. Only one NVMe is formatted; other3 are untouched.

Pilot lowest ordinary MemAvailable point285.93GiB is not a continuous minimum;
node-monitor total-minus-used is not MemAvailable. One saved exact32-worker
snapshot sums cumulative VmHWM1,237,113,118,720bytes, overcounting shared pages and
non-simultaneous peaks. Successful full-loader freeze peak6882680KiB ×32×1.25, plus worker/owner peaks,
idle host use,64GiB temp and128GiB reserve leaves372.75GiB extra. This is a
conservative planning projection, not a mathematical peak guarantee. Full resource
receipt `local/science/preparation-256/resource-feasibility.json`, SHAd7880705…eed8.
EBS/NVMe share this instance. **Off-instance backup remains unresolved**; no
approved external destination. No stack upgrade, new instances/IAM/paid teachers,
unapproved uploads, or additional disk formatting.
