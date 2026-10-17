# Current state — 2026-10-17 03:49 UTC

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

**Distributed owns R004**, upper logical32–63, freshB0/seed1234, world32/global128/
accum4, G4/cap1024/K1/T.7, FP32/LRU2, max14. Latest complete **update8 at03:47:55**,
761.552s; update9 runs; next checkpoint14. Steps3–8 take746.696–897.506s;
ten full steady timings and median/P90 are still pending. All recorded norms are
finite, sampled replicas equal, replay errors below.001. Updates1/2 include cold
optimizer work and are excluded from steady samples. Checkpoint3/4/7 save wall
49.368/47.404/47.230s is separate from per-step elapsed.

- Common source commit `3aaaf8a18baeccb8d1a543d348a53f6863599114`, active code
  SHA `244e2f13f91dfc746f0829b236a35b8f2c5452f5d62da7a1bc3b3bbe31287ff7`.
  No active-code edits during these jobs. Latest required source checks:
  **233 CPU tests plus toy smoke PASS**, `local/checks/tests.Yr1uiR`.
- Spec `92f866c86cb654b0daed9079e634837ae3aa5d950ffad21415022e24195ed7ac`.
  Control `local/qualification/qwen-r-fp32-nvme-tmp-004`, output
  `local/nvme/qualification/qwen-r-fp32-nvme-tmp-004`, raw logs/monitor in its
  adjacent `-observation` directory. Prepared49c770b7…, root releasecb29f2b8….
- Launcher871318, timeout/PGID871404, driver871406; recorded32 worker identities
  in `{processes,rank-processes}.json`. Ports64301/64302. Hard deadline
  **2026-10-17 09:38:07.402054 UTC**,30s kill grace.
- TMPDIR `local/nvme/tmp/qwen-r-fp32-nvme-tmp-004`, device66310; persistent cache
  `local/neuron-cache-driver` on EBS66305. Separately bound host policy9d2e1f20….
- Root owns backup watcher: tmux `dario-r-nvme-tmp004-backup`, launcher897722,
  timeout/PGID897725, Python897726, startticks3586343/3586344, same deadline.
  `backup-supervisor.state.json` records **verified EBS steps4/7**, latest7 at
  03:39:32, COMPLETE5a98b70b…708caab. Retain2 copies and every NVMe original.
  Root also independently rehashed all33 original/EBS checkpoint3 payloads:
  PASS, COMPLETE88190797…e8bbe0,27,542,553,263bytes. Its serialized model/all32
  optimizer shards are finite, valid master slices match, all Adam steps3.
  Stop this watcher if the driver fails or stops before terminal14.

**Review owns B0 held-development evaluation**, one lower logical core0, three
sequential freshB0 modes direct→empty_format→one_thought. Fixed128 GSM8K TRAIN
questions, FP32 compute/head, unchanged4shot prompt, caps1024thought/256answer.
Direct complete **113/128 correct (88.28125%)**, no errors/truncations;13,590
answer tokens, generation wall1,255.0s, first item222.53s, median7.63s.
Empty-format complete **107/128 (83.59375%)**,2 format failures/1 answer-cap
truncation/0 runtime errors, generation wall1,347.9s. Thought mode is active; all
failures/truncations stay in the128 denominator. This is a development baseline, not an official test or training gain.

- Started02:45:18.503479; hard deadline **06:45:00.434277 UTC**,30s grace.
  tmux `dario-held-dev128-B0-001`. Control
  `local/evaluation/held-dev128-v1/b0-plan-v1`; output
  `local/nvme/evaluation/held-dev128-v1-B0-001`.
- Root launcher1078999; **actual timeout/PGID1079002**, startticks3981062;
  mode shell1079049; current thought evaluator1135219/startticks4246270.
  Original execution.json mistakenly recorded an exited command-substitution
  subshell1079003. Preserve it; correction is in `supervision.json`.
- Core0, ports64351/64352, private0700 TMPDIR
  `/dev/shm/dario/held-dev128-v1-B0-001`, same EBS cache. Prepared95e25064…,
  launch9bfb2a10…, host policyaafe605d…. No additional node monitor.
- On full suite completion and owned-process cleanup, review verifies/copies the
  stable small output tree to `local/backups/evaluation/held-dev128-v1-B0-001`
  on EBS with per-file hashes. No weights/compiler payload or external upload.
  R004 timing discloses the overlapping one-core host workload.

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

After **R00414/exit/all original workers gone and B0 cleanup/all64 cores free**:

1. Use ORIGINAL upper `local/qualification/qwen-r-fp32-ram-resume-004/launch.sh
   --prepare-fork` to create fully verified cp3-only hardlinks and strict STOP.
2. Root releases **v2 lower**, then launches freshB0/seed5678 RAM005 through14.
   At fixed lower COMPLETE3, promptly capture actual32 worker/env/ancestry/topology
   with v2 `lower.launch.sh --ready`. Readiness is published once, expires300s;
   no later milestone substitution or automatic retry.
3. Root reviews actual readiness, releases **v2 upper**, launches original strict
   restored3→one update4/save/exit. Compare actual restored state/RNG raw bytes,
   first stochastic tapes and complete post-update state using original criteria
   **5558cf95…8a00a4** and tight float tolerances; never force RNG or loosen bounds.
4. Lower continues14. Report its ten full timings5–14 and genuine upper/lower
   overlapping execution separately. No extra upper profiling job or assumed
   simultaneous-throughput multiplier. Back up terminal states and verify cleanup.

Reviewed controls `local/qualification/ram-sequence-v2`, manifest
**daf199d05b620a05105520f734a3d53622ed1771cbb6662557125c11243ac1c6**;
public `validation/r_ram_sequence_20261017.json`. Six host fixtures and both static
probes pass; **no fork/STOP/release/RAM training yet**. All original v1 files stay
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

At03:49 EBS free **269.9GiB**, mounted NVMe **1.20TiB**, tmpfs about1,000GiB.
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
