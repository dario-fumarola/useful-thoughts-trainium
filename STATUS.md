# Current state — 2026-10-17 08:45 UTC

**First scientific C1234 is running; no trained C/R comparison is available.**
The first study remains **256 terminal updates**, with fixed step64 development.
Public repository: https://github.com/dario-fumarola/useful-thoughts-trainium .
Work is isolated in `/home/ubuntu/dario`; original repositories are unchanged.
The owner authorizes sustained autonomous work, review agents and public updates.
`CODEX_NEXT_DAY.md` governs the queue. Its companion review file was not found.
Weeks are available; the14-day planning window is not an instance shutdown date.

## Current work and ownership

No accelerator training/evaluation job is active. RAM005 completed14 updates,
exit0 at08:13:18; all original workers/driver/launcher/monitor/observer are gone.
All16 devices were empty at cleanup. Root's lower backup watcher also exited0,
all owned processes gone. Original v1 coordinators remain unused.

**C1234 is active**, fresh released B0, source-256-001, FP32/world32/global128/
accum4/default qualified C cache/unset LRU/EBS temp; production329535 TRAIN pool,
terminal256/fixeddev64. Lower logical0–31; actual32 workers occupy devices0–7
(four/device), upper8–15 free. Launcher1607173, timeout/PGID1607186, driver1607187;
all32 actual worker/start/ancestry/env/device bindings are in control/processes.json.
Root release08:37:57.545140UTC SHA6647f1ec…7154e; run started08:38:35; firstupdate
loop marker08:39:19. **First real optimizer update completed08:43:17**,237.866s, norm1.677426,
sampled replicas exact,98.9617% sampled entries changed;128records/65664source
tokens. Startup step is separate from qualified warm timing; no efficacy claim.
Receipt `validation/science_256_first_update_20261017.json`, SHA705dd2e1…caaa7.
Distributed owns read-only startup/progress observation; no extra monitor.
Control `local/science/control/qwen-c-seed1234-science-256-001`;
output `local/nvme/runs/qwen-c-seed1234-science-256-001`; rawlog
`local/nvme/science/launch-logs/qwen-c-seed1234-science-256-001.log`.
Tmux `dario-science-C1234-256`. Let healthy startup and all256 updates finish.

Root owns both active shared host supervisors, matching the frozen spec/deadline:
backup helper PID1606740; node helper1606741, producer1606742, gzip1606743,
node timeout/PGID1606734. Roster/config `local/science/{backups.roster,monitor.config}.json`;
state `local/science/shared-backups.state.json` and
`local/nvme/science/shared-node-monitor/state.json`. Both RUNNING. No second
training group or evaluation is released. Other qualification processes are gone.
Unique upper continued4 has a verified EBS copy at
`local/backups/qwen-r-fp32-ram-resume-004/step-0000004`; its receipt is
`local/qualification/ram-sequence-v2/upper/backup.receipt.json`.

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

1. Observe firstC through real updates, checkpoint64/128/256 and verified EBS
   copies. Preserve actual order/counters and all old qualifications. Require
   terminal256/exit/backup/cleanup before the next training group.
2. Launch **one32-core training group at a time: C1234, C5678, R1234, R5678**.
   Every arm independently starts from released B0, never C/qualification weights.
   Seed1234 uses lower0–31, seed5678 upper32–63; C/R share each seed's exact order
   and32,768 fresh TRAIN records. Cross-seed overlap3,182 is disclosed.
   Serial fallback was selected before science from full-pool memory uncertainty.
   Use one reviewed shared EBS queue and one compressed node monitor; bind their
   live matching identities before each finite root release. Only firstC is released;
   matching shared supervisors are active. Idle cores may serve genuine development work
   subject to actual host headroom.
3. Complete fixed held-development C/R64/256 in direct/empty_format/one_thought,
   plus the local fixed DEV512 controls. Never select the terminal by dev score.
   Keep official final outcomes sealed until protocol/model inventory is fixed.

Immutable `local/science/source-256-001` matches all61 qualified active files,
archived from commit85c64a9. Its manifest SHA8a0641c6…77faa4 binds the tar and
qualified source. Four finite launchers passed independent review; only C1234 is released. Use archive cwd/PYTHONPATH and absolute evidence/control paths;
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
