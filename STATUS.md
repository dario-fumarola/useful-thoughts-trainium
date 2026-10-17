# Current state — 2026-10-17 05:54 UTC

**R qualification is running; no scientific C/R pair has launched.**
The first-study terminal target is prospectively **256 updates**, with fixed
step64 development evaluation. Public repository:
https://github.com/dario-fumarola/useful-thoughts-trainium . Work is isolated in
`/home/ubuntu/dario`; original repositories remain unchanged.

The owner authorizes sustained autonomous work and review agents, with weeks
available. `CODEX_NEXT_DAY.md` governs the next-day queue; its snapshot5cd4c79
predates the completed R updates below. Referenced `REVIEW_AND_NEXT_24H.md` was
not found locally. The 14-day schedule is a planning limit, not a shutdown date.

## Live jobs and ownership

**Distributed owns active RAM005**, lower logical0–31, freshB0/seed5678,
world32/global128/accum4, G4/cap1024/K1/T.7, FP32/LRU2, max14. All32 workers
are mapped; **update3 completed at05:51:28**,869.016s, finite norm73.19961,
sampled replicas equal and replay error.00003791. Update4 is running. Full-step
times1–3 are939.926/867.743/869.016s; no isolated host-policy speedup claim. The prospective
steady timing window is updates5–14. No scientific arm launched.

- Control `local/qualification/ram-sequence-v2/lower`; output
  `local/nvme/qualification/qwen-r-fp32-ram-tmp-005`, raw logs/monitor in its
  adjacent `-observation` directory. tmux `dario-ram-v2-lower005`.
- Launcher1246580, timeout/PGID1246667, driver1246669, monitor timeout1246666.
  Actual32 worker identities are in control `{processes,rank-processes}.json`;
  actual launch receipt `launch.execution.json`. Ports64331/64332.
- Root release at05:05:19 UTC, SHA2a3a46e9…b74f83. Preflight passed; run metadata
  started05:06:33. Specb61afe63…bac80c. Hard deadline **13:05:19.591488 UTC**,
  30s kill grace. Original v1 prepared96068a0d…cc9241 stays immutable/unreleased.
- Private0700 TMPDIR `/dev/shm/dario/qwen-r-fp32-ram-tmp-005` on tmpfs26;
  host policyc6cc7268…4ec21. Persistent cache remains EBS66305; logs/checkpoints
  remain NVMe66310. Qualified source commit3aaaf8a and code SHA
  **244e2f13f91dfc746f0829b236a35b8f2c5452f5d62da7a1bc3b3bbe31287ff7**
  are unchanged. Latest required source checks:233 CPU tests + toy smoke PASS.
- **One-shot COMPLETE3 readiness capture finished**, exit0 at05:52:19, all32
  actual workers/ancestry/env/mapping validated. Ready SHAb674fd49…7d9b0 was
  immutable and fresh when root independently reviewed/released upper. Capture
  watcher1271879 is gone; no retry or later-milestone substitution occurred.
- **Root owns the only active backup watcher**, tmux
  `dario-ram-v2-lower005-backup`: launcher1272281, timeout/PGID1272286,
  Python1272287, startticks4830210/211/212. Same deadline; state/control in v2
  lower, raw events/logs in NVMe observation. Destination
  `local/backups/qwen-r-fp32-ram-tmp-005`, retain2/all NVMe originals preserved.
  COMPLETE3ed24af26…7f1336 published05:52:15.789; first EBS copy/verification is
  active. Stop this watcher if the driver fails/stops early.
- Distributed reuses the existing read-only host observer in tmux
  `dario-ram005-observe` for actual overlap/HBM evidence. No extra device job.

**R004 completed14 updates and exit0 at04:58:17 UTC**, all original workers,
launcher, monitor and observer gone. Full-step14:655.480s; CP14 save44.802s,
COMPLETE39a26e7f…276673e. All post-cold updates3–14: median734.750s,
P90 873.985s, mean761.221s. Workloads shrink greatly; no TMPDIR-only causal
speedup is claimed. Cold work, checkpoint wall and shared-host intervals are
separate. Single report `validation/r_fp32_nvme_execution_20261017.json`.

R004 final TRAIN tapes have **509/512 empty notes**, all512 stop at base EOS;
action lengths1/2/3 occur509/2/1 times. Mean raw gain remains negative. This is
qualification behavior, not useful-thought evidence or a trained C/R result.
The fixed research protocol and prospective256 terminal remain unchanged.

R004 EBS **7/14** are verified, terminal copy published05:02:16; watcher exit0 at
05:02:57 and all its owned PIDs gone. Root independently rehashed all33 original
and EBS payloads for BOTH checkpoints, PASS:
`local/qualification/qwen-r-fp32-nvme-tmp-004/terminal-backup-verification.json`,
SHA4e614878…5ee89. All NVMe originals remain.

The original strict upper **cp3-only verified hardlink fork and precreated STOP**
were prepared at05:01:07; no future metrics/tapes copied. Original fork receipt
`local/qualification/qwen-r-fp32-ram-resume-004/fork.json`; STOP SHA9a9df879…eecf8.
**Upper recovery is active**, control `local/qualification/ram-sequence-v2/upper`,
tmux `dario-ram-v2-upper004`; original v1 stays unreleased. Root issued release
at05:52:33, SHAf882c0fd…ca035, binding actual lower readyb674fd49…7d9b0 and the
unchanged original fork/STOP. Full original fork preflight passed before readiness
expired05:57:19. Launcher1327148, timeout/PGID1328412, driver1328414,
monitor timeout1328411. Review independently verified all32 actual workers/env/
ancestry on physical devices8–15 at05:54:32; all lower identities remain exact.
Receipt `upper/review.processes.json`, SHA90af9654…1c64d. Review owns STOP supervision and original tape/state
comparators; distributed owns actual simultaneous execution evidence. Output
`local/nvme/qualification/qwen-r-fp32-ram-resume-004/qwen-r-fp32-nvme-tmp-004`.
Strict restored3→one update4/save/exit is required; same deadline13:05:19 UTC.

**B0 held-development suite complete, exit0**, all128 fixed GSM8K TRAIN items in
three modes: **direct113/128 (88.28125%)**, **empty107/128 (83.59375%)**,
**thought63/128 (49.21875%)**. Empty has2 format failures/1 answer truncation;
thought33/14 plus4 capped thoughts. Runtime errors0. Every failure stays counted.
This is a substantial B0 protocol cost, not a trained C/R or official-test result.
No subset, prompt, cap, parser or prospective256 target changes follow it.

Started02:45:18.503479, completed04:18:35.923680 UTC, total93.29min including
loading/cold work. Generated tokens13,590/15,119/32,100 respectively. Mean item
wall9.80/10.53/22.86s includes first-item cold work; generation sums and separate
outer wall are retained. Public report with paired descriptive intervals and all
input/output hashes: `validation/held_dev_B0_20261017.json`.

At B0 cleanup all owned processes/PGID1079002 were gone and lower devices0–7
were idle; all32 original R004 workers remained unchanged. The PID-recording correction
stays preserved in control `local/evaluation/held-dev128-v1/b0-plan-v1`.
Raw outputs `local/nvme/evaluation/held-dev128-v1-B0-001` and approved EBS copy
`local/backups/evaluation/held-dev128-v1-B0-001` retain every prediction, score,
log and process receipt. Root independently rehashed **all51 source/copy payloads
plus the copied verification receipt**, PASS. Reviewer result52d82625…c4af7a,
backup receiptf472897d…42ffbc. No B0 process or deadline remains active.

## Completed evidence

Actual Qwen3-1.7B-Base revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`,
released weights/original embedding tie, FP32 both arms. Audited FineMath
production: **329,535 train /18,360 dev /18,239 test**. No official final benchmark
outcomes read. Earlier exclusion preparation used question-only benchmark data.

- **C32 recovery PASS:**12 updates, ten steady mean10.007s. Actual restored full
  model/all optimizer shards/metadata/counters/CPU-device RNG raw-byte exact;
  continuation through12 meets original tolerance, RNG/Adam steps exact, weights
  not bitwise equal. Verified EBS6/12; all owned processes gone.
  `validation/c_fp32_final_resume_20261016.json`.
- **Preserve C CPU-coordinate v1 FAIL:**754/1,720,574,976 coordinates. Independent
  identical-gradient Adam passes. Separate FP64 diagnosis finds device derivative
  closer than CPU FP32 overall, with residual sign disagreements retained.
  Prospectively declared held-seed **v2 derivative PASS**, relativeL2.00010316,
  cosine.9999999947, worst tensor.004933<.01, all8 gates.15,186 diagnostic sign
  differences retained. No v3/FP64 expansion absent a concrete new fault.
  `validation/c_fp32_accuracy_v2_{20261016,result_20261016}.json`.
- **R full-cap math PASS:** FP32 cached replay fixes the trajectory that failed
  BF16 at.001. Signed full-model backward passes CPU loss/norm and declared
  cosine/L2 bounds. Float counting repairs the device int64 clamp/sign defect.
  `validation/p2_repaired_signed_gradient_20261016.json` and
  `validation/int64_clamp_mechanism_20261016.json`.
- **R003 first full update PASS**, cooperative STOP/save/exit0 by01:59:02. Full
  serialized model/initialized masters/moments inspection passes; all33 original
  and EBS payload hashes independently verified, COMPLETEe563c4c2…2d966. Every
  owned worker/monitor and obsolete watcher is gone. Original and EBS retainstep1.
  `validation/r_fp32_first_update_20261017.json`.
- **Host I/O/parity:** EBS temporary NEFF copies observed125.07MiB/s; NVMe placement
  verified. All128 R003/R004 first-step tape pairs pass, tensor payloads byte
  identical. This is not a TMPDIR-only causal estimate or strict recovery proof.
  `validation/r_warm_cache_io_20261017.json` and
  `validation/r_host_io_tape_parity_20261017.json`.
- **Prefix-only accelerator check PASS:** actualB0 G4/cap1024, fixed writer RNG,
  same prefix with K1 target10→220; actions/masks/lengths/metadata/log-probs exact.
  One TRAIN prefix, two EOS/two caps; all owned processes gone. Original CPU
  fixture separately retained. `validation/prefix_only_fp32_20261017.json`.
- **Full-cap FP32 evaluation parity PASS:** local192 token comparisons max.00023079;
  secondary90 max.000070572, no ranking/token mismatch. All jobs exited.
  `validation/evaluator_fp32_fullcap_20261016.json`.
- **HBM accounting:** adjacent physical NC pairs share banks inLNC2; sampled R004
  max pair21.7915GiB, exposed allocator23.984375GiB. Residual is not guaranteed
  contiguous free capacity. `validation/r_hbm_capacity_20261017.json`.

All failures and fixed first-batch examples are preserved. R003's four saved
notes and first full batch have negative absolute raw gains; centered advantages
and qualification success do not establish useful deployed thoughts.

## Exact next sequence

Original baseline cleanup/fork preparation and one-shot lower readiness are
complete; v2 lower training and upper strict recovery are active.

1. Follow **v2 upper** strict restored3→one update4/save/exit. Review independently
   compares actual restored state/RNG raw bytes, all128 new stochastic tapes and
   complete post-update state using original criteria **5558cf95…8a00a4** and
   tolerances1e-7+1e-6. Never force RNG or loosen bounds. Original v1 stays unused.
2. Lower continues14. Report ten full timings5–14 and genuine overlapping device
   execution separately. No extra upper profiling job or assumed simultaneous
   throughput multiplier. Finish verified backups and owned-process cleanup.
3. Choose qualified host placement, account for production-loader RAM/storage and
   actual measured costs, freeze256, then launch the matched fresh-B0 seed chains.

Reviewed controls `local/qualification/ram-sequence-v2`, manifest
**daf199d05b620a05105520f734a3d53622ed1771cbb6662557125c11243ac1c6**;
public `validation/r_ram_sequence_20261017.json`. Six host fixtures and both static
probes pass; **lower running; upper strict recovery launched after actual ready review**. All original v1 files stay
immutable; **never release original v1 launchers**, old R003 resume, or old
upper-overlap-001. Lower0–31 ports64331/2, upper32–63 ports64321/2. Private RAM TMPDIRs
are separately bound; raw logs/checkpoints stay NVMe, compiler cache stays EBS.

Then select qualified host placement, budget using actual timing/storage, freeze
256 and launch **C1234→freshB0 R1234** and **C5678→freshB0 R5678** on32-core
partitions. Explicit serial fallback remains available. Prospective production256
exposure is prepared/reviewed at `local/science/preparation-256/manifest.json`:
32,768 unique TRAIN IDs/seed, exact C/R within-seed order,3,182 cross-seed overlaps
disclosed. No science freeze/release exists.

Held-development protocol `validation/held_dev_protocol_20261017.json` fixes128
GSM8K TRAIN items excluding all11 past demo/interface cases, B0 plus both C/R seeds
at64/256, all3 modes. Report primary R-thought−C-thought, R-direct−C-direct,
R-thought−R-empty and each arm versusB0, retaining errors/truncations/token costs.
Direct permits ordinary calculations. Official final outcomes stay sealed until
protocol/model inventory is frozen. Local K1 primary, K4/K16 secondary; preserve
absolute clean/empty/own/filler and fixed random dev token-class/surprisal strata.

## Storage and operational limits

At05:09 EBS free **269.8GiB**, mounted NVMe **1.15TiB**, tmpfs about1,000GiB.
Only one NVMe is formatted; other3 SSDs remain untouched. Science uses one reviewed
serialized backup queue and one compressed shared node monitor. Backup helper at
`validation/orchestration` passes its6 host fixtures/TERM review, unlaunched.
The node monitor wrapper also passes6 host pipe/cleanup checks, independently
rerun by root; no live capture exists. Both helper hashes are in the draft site.
Reserve **two new checkpoint payloads +20GiB≈71.3GiB** for a backlogged rollover.
Before science, reclaim only obsolete EBS qualification duplicates after full
verification of every preserved NVMe payload; retain recoverable latest states.
Existing retirement/relocation receipts stay under `local/qualification`.

EBS and NVMe share this instance. **Off-instance backup remains unresolved**;
no external destination is approved. No stack upgrades, extra disk formatting,
new cloud resources, IAM changes, paid teachers or unapproved uploads.
