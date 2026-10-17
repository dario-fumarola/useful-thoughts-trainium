# Current state — 2026-10-17 08:23 UTC

**Finite R qualification has passed; no scientific C/R arm has launched.**
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

Root is finishing storage and production resource preparation. The only active
copy is the unique strict upper continued4 checkpoint, using the sealed source's
`scripts/backup_checkpoint.py`. Source:
`local/nvme/qualification/qwen-r-fp32-ram-resume-004/qwen-r-fp32-nvme-tmp-004/step-0000004`.
Destination: `local/backups/qwen-r-fp32-ram-resume-004/step-0000004`.
Result/stderr: `local/qualification/ram-sequence-v2/upper/backup.{receipt.json,stderr.log}`.
Wait for successful copy/readback before the timed freeze. No automatic relaunch.

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

1. Finish the current upper4 EBS copy. Then run the sole necessary timed full-asset
   freeze, capture peak RSS, and complete the conservative32-loader memory and
   storage/timing projection. Assemble the actual nine-check qualification receipt
   against the exact frozen spec and run `scripts/check_ready.py` from the archive.
2. Launch **one32-core training group at a time: C1234, C5678, R1234, R5678**.
   Every arm independently starts from released B0, never C/qualification weights.
   Seed1234 uses lower0–31, seed5678 upper32–63; C/R share each seed's exact order
   and32,768 fresh TRAIN records. Cross-seed overlap3,182 is disclosed.
   Serial fallback was selected before science from full-pool memory uncertainty.
   Use one reviewed shared EBS queue and one compressed node monitor; bind their
   live matching identities before each finite root release. No supervisor or
   science release exists yet. Idle cores may serve genuine development work
   subject to actual host headroom.
3. Complete fixed held-development C/R64/256 in direct/empty_format/one_thought,
   plus the local fixed DEV512 controls. Never select the terminal by dev score.
   Keep official final outcomes sealed until protocol/model inventory is fixed.

Immutable `local/science/source-256-001` matches all61 qualified active files,
archived from commit85c64a9. Its manifest SHA8a0641c6…77faa4 binds the tar and
qualified source. Four finite launchers passed independent review and remain
unreleased. Use archive cwd/PYTHONPATH and absolute evidence/control paths;
current root docs govern operations. Existing local qualification pre-freeze
assembly contains all9 actual checks and all10 timings, but spec_hash=null:
**not ready until the actual freeze and resource review**. No new numerical gate.

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
unique resumed4 is copying. Before that copy, EBS free319.6GiB, NVMe1.001TiB.
Science reserves latest-two EBS copies/run plus **two new payloads+20GiB** for
backlogged rollover, about71.3GiB. Only one NVMe is formatted; other3 are untouched.

Pilot lowest ordinary MemAvailable point285.93GiB is not a continuous minimum;
node-monitor total-minus-used is not MemAvailable. One saved exact32-worker
snapshot sums cumulative VmHWM1,237,113,118,720bytes, overcounting shared pages and
non-simultaneous peaks. Add32 conservative full-loader copies, RAM temp and host
reserve using the necessary freeze RSS; this is an estimate, not a new model run.
EBS/NVMe share this instance. **Off-instance backup remains unresolved**; no
approved external destination. No stack upgrade, new instances/IAM/paid teachers,
unapproved uploads, or additional disk formatting.
