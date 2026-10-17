# Current state — 2026-10-17 02:06 UTC

Real-asset Trainium qualification; **no scientific C/R pair has launched**.
Public repo: https://github.com/dario-fumarola/useful-thoughts-trainium . Work is
isolated in `/home/ubuntu/dario`; original repositories remain unchanged. The
owner authorizes sustained autonomous work and review agents, with weeks available.
The prospective 14-day science schedule is a planning limit, not a shutdown date.
The owner's `CODEX_NEXT_DAY.md` is the current next-day guidance. It references
snapshot5cd4c79; completed updates/save/backup below supersede that snapshot.
Adopt **256 updates** as the prospective first-study terminal target, subject to
qualified measured feasibility. Do not automatically maximize to2048. Its companion
`REVIEW_AND_NEXT_24H.md` was not found in the workspace; work continues from the
available handoff. No scientific outcomes informed this budget choice.

## Completed evidence

- Released Qwen3-1.7B-Base revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`,
  actual weights and original embedding tie; FP32 in both proposed arms.
- Audited FineMath production slice: **329,535 train / 18,360 dev / 18,239 test**.
  Benchmark overlap exclusions and fixed evaluation IDs are saved. No official
  final benchmark outcomes have been read.
- Full-cap cached FP32 generation/replay passes the original .001 bound on the
  exact trajectory that failed BF16. The precision change preceded science.
- Full-cap signed full-model backward passes CPU loss/norm and prospectively
  declared signed-gradient cosine/L2 bounds. The original integer normalization
  returned -1 on device; float counting repairs it without changing the objective.
  See `validation/p2_repaired_signed_gradient_20261016.json` and
  `validation/int64_clamp_mechanism_20261016.json`.
- **C32 recovery PASS, common source `3aaaf8a`:** 12 updates; ten steady steps
  average **10.007 seconds**. Fresh-process actual restored state is raw-byte
  exact across model, all 32 optimizer shards, metadata, counters and CPU/device
  RNG. Continued updates through step 12 meet the original numerical tolerance;
  all RNG states and Adam steps remain exact. Continued weights are not bitwise
  identical. Verified EBS backups retain steps 6 and 12. All C workers and supervisor
  exited. See `validation/c_fp32_final_resume_20261016.json`.
- Both full-cap FP32 evaluator suites pass .001 CPU parity: local controls and
  crossed readers, 192 token comparisons, max error .00023079; secondary paths,
  90 token comparisons, max error .000070572, no ranking/token mismatches.
  All workers exited; only development/interface items used. See
  `validation/evaluator_fp32_fullcap_20261016.json`.
- Latest required source checks: **233 CPU tests plus toy smoke PASS**,
  `local/checks/tests.Yr1uiR`. Active source remains unchanged; later commits add
  documentation, receipts and identical reviewed diagnostic copies. Their four
  public-path CPU fixtures also pass.

## C numerical diagnosis and held-seed validation

The original three-step CPU FP32 per-weight comparison remains **FAIL**:
754 of 1,720,574,976 coordinates exceed its fixed bound. Correcting the CPU norm
reduction does not remove this failure. A separate identical-gradient hardware
AdamW test passes; first-step checkpoint moments locate the discrepancy before
Adam, where tiny cancelled gradients can differ in sign.

An independent FP64 differentiable-arithmetic diagnostic is complete (all jobs
exited). It uses exact released FP32 weights cast to double, promoted RMSNorm and
softmax, and frozen native-CPU FP32 RoPE constants. All 310 parameter-gradient tensors are
saved. Against it, device relative gradient L2 error is **.0001083**, versus
**.0003425** for the original CPU FP32 reference. Device is closer on 672 of the
754 flagged coordinates, but 90 selected sign disagreements remain. This is
numerical diagnosis, not a retroactive v1 pass or a scientific result. Receipts:
`validation/c_firstbatch_fp64_derivative_20261016.json` and
`validation/c_fp32_gradient_discrepancy_20261016.json`.

A distinct prospective **v2 PASS** on held seed 5678. Exact first 128 TRAIN IDs
and criteria were committed/pushed before outcomes; 6/128 records overlap the
old batch. All 1,720,574,976 derivatives / 310 tensors are compared once against
FP64. Relative L2 **.00010316**, cosine **.9999999947**; worst tensor relative
L2 **.00493315** is below its fixed .01 bound. Scalar loss/norm checks and all
eight mandatory gates pass; comparator exited 0. Diagnostic sign differences
remain 15,186. Preserve v1 FAIL. Frozen criteria:
`validation/c_fp32_accuracy_v2_20261016.json`; full local result:
`local/qualification/qwen-c-heldseed-001/comparison.result.json`. All 11 targeted
fixtures/source review pass. Independent outcome audit passes, including all
9,920 first-step Adam counters. Public result with all tensor diagnostics:
`validation/c_fp32_accuracy_v2_result_20261016.json`.

## Active work and process ownership

**Distributed owns active R004**, upper logical cores 32–63.
Launcher 871318, timeout/PGID 871404, driver 871406; all 32 worker identities in
`local/qualification/qwen-r-fp32-nvme-tmp-004/{processes,rank-processes}.json`.
Source `3aaaf8a18baeccb8d1a543d348a53f6863599114`, active code hash
`244e2f13f91dfc746f0829b236a35b8f2c5452f5d62da7a1bc3b3bbe31287ff7`.
World32 / global128 / accum4, G4/cap1024/FP32/seed1234/LRU2, max14.
Spec `92f866c86cb654b0daed9079e634837ae3aa5d950ffad21415022e24195ed7ac`.
Ports 64301/64302; hard deadline **2026-10-17 09:38:07 UTC** plus 30s grace.
Reviewed prepared SHA `49c770b7…b9fec9a`, root release `cb29f2b8…048315a3`;
actual preflight passed 01:38:30. Output:
`local/nvme/qualification/qwen-r-fp32-nvme-tmp-004`; raw logs/monitor in adjacent
`qwen-r-fp32-nvme-tmp-004-observation`. TMPDIR:
`local/nvme/tmp/qwen-r-fp32-nvme-tmp-004`, NVMe device66310; persistent compiler
cache stays EBS device66305. Separately bound host I/O policy SHA
`9d2e1f20f3f38cb30ce41a80f4c3b453f44c4471aed15b382a84a2d8dd8cfb21`.

**R004 first full update passed at 01:57:33 UTC**, 1,093.263 seconds including
cold shared-cache work. Norm37.18666, sampled FP32 changed fraction .9895161,
sampled replicas bitwise equal; replay error .00005579. All 32 rank-specific
optimizer cache hits match actual R003 compilation completions. Update2 began
01:57:37. Warm-step times and ordinary checkpoint3 state remain pending;
distributed prepared a reviewed CPU-only state inspection for that checkpoint.
Transient CCOM RX retries during rank convergence cleared into the successful
update. No recorded device execution failure. No steady-speed claim yet.

**Root owns active R004 backup supervisor:** launcher897722, timeout/PGID897725,
Python897726; startticks3586343/3586344; tmux `dario-r-nvme-tmp004-backup`.
Same deadline. State/control in R004 qualification directory, events/logs in its
NVMe observation directory. Destination `local/backups/qwen-r-fp32-nvme-tmp-004`;
retains two verified EBS copies, preserving every NVMe original. Stop the watcher
if R004 fails or stops before terminal backup.

**R003 completed one full update, cooperatively saved and exited0 by 01:59:02.**
All workers, launcher and monitor are gone. STOP was requested at01:40:24 for
measured EBS host-I/O cost; no driver signal was sent. Its update metrics match
R004 exactly except elapsed/time. All 310 model tensors and 32×310 FP32 masters
and moments pass serialized-state checks; all 9,920 Adam steps equal1, and every
valid master slice equals the saved model. Original:
`local/nvme/qualification/qwen-r-fp32-final-003/step-0000001`.
Root rehashed all33 payload files at both original and EBS backup, identical
COMPLETE `e563c4c2…2d966`, total27,542,553,199bytes. Verified copy:
`local/backups/qwen-r-fp32-final-003/step-0000001`.
After verification, root retired its obsolete terminal14 watcher by identity-
checked SIGTERM; launcher849074/timeout849075/Python849077 are all gone. Expected
watcher exit1 reflects explicit retirement, not driver/checkpoint failure.
State and verification: `local/qualification/qwen-r-fp32-final-003.backup-*`.
Public receipt: `validation/r_fp32_first_update_20261017.json`.

**Host-I/O diagnosis and stochastic parity:** EBS temporary NEFF copies sustained
125.07MiB/s (3.948GB/30.10s); open worker destinations were `/tmp/tmp*/model.neff`.
NVMe placement is now actually observed. All128 first-step R003/R004 tape pairs
pass original exact metadata/actions/masks and1e-7+1e-6 float bounds; every tensor
payload is also byte-identical. Root reverified all256 hashes. These are
engineering observations, not a perfectly isolated TMPDIR performance estimate
or full-state recovery. Receipts: `validation/r_warm_cache_io_20261017.json` and
`validation/r_host_io_tape_parity_20261017.json`.

**Conditional RAM preparations, independently reviewed, unlaunched:**
upper `local/qualification/qwen-r-fp32-ram-resume-004/prepared.json`, SHA
`304c76caf53ea56c98774db686a9c90cc9bdeac97dabd7f3508c4a3bd6a32e9a`,
retains original criteria `5558cf95…8a00a4`, exact restored state and strict
checkpoint3→4 with STOP precreated. No fork/STOP/release exists yet. Source is
R004's immutable archive; ports64321/64322, upper partition, same spec/run basename.
Lower `local/qualification/qwen-r-fp32-ram-tmp-005/prepared.json`, SHA
`96068a0dd96489aa53c6bb05e86cb6f32dcd9fb5ab86e3a90217159a76cc9241`,
is fresh B0 seed5678/max14, lower partition, ports64331/64332. It can provide
RAM warm timings alongside two-R coexistence. Seven lower/three upper guard
fixtures and actual host static probes pass. Shared host helper SHA
`4117b0a04ea7149eac4c6df47bd84dc19bd655e7f38709a28f8614a8106f6b0e`.
Both keep checkpoints/raw logs on NVMe and the existing compiler cache on EBS;
private0700 TMPDIRs under `/dev/shm/dario`, filesystem26. Observed available RAM
exceeds1.2TiB; largest64 temporary NEFF copies total24.23GiB. Launch guards require
128GiB actual MemAvailable and64GiB tmpfs free. Host policies are separately bound.
Upper requires completed R00414/exit/all64free and a fresh root release binding
fork/exit/COMPLETE14. Lower additionally requires actual mapped upper32-worker
restored3 readiness plus immutable STOP and lower-free topology. No RAM speed
claim; final science policy follows measurements. Do not launch original R003
resume or old `qwen-r-fp32-upper-overlap-001`: their terminal14 guards cannot pass.
Preserve their original bytes.

All four fixed step1/rank0/micro0 TRAIN notes are saved in full at
`local/qualification/qwen-r-fp32-final-003.step1-rank0-micro0.example.{json,txt}`
and checked against the tape. Lengths342/1024/146/298, three EOS/one cap;
all raw gains negative on this prefix. The initial full128-prefix batch's mean
gain is also negative. Preserve these observations; neither is a downstream
result or an outcome-based stop condition.

**Held-seed jobs complete:** C32 update/COMPLETE and FP64 full derivative both
exited 0; comparator exited 0/PASS. All upper-core workers and supervisors are
gone. Checkpoint: `local/nvme/qualification/qwen-c-fp32-heldseed-001/step-0000001`.
C peak sampled device allocation 18.455 GiB, sampled errors zero. Exact preparation
SHA `26a6081f…2313d4b6`; state/receipts under
`local/qualification/qwen-c-heldseed-001/`. Public receipt and independent audit are complete; all original failure receipts
remain.
Root owns integration, decisions and public pushes. Do not edit common active
code during these jobs; later documentation commits do not alter its fingerprint.
The ignored local site draft now binds recorded per-arm runtime policies,
12-hour checkpoint interval, per-seed partitions, NVMe outputs and EBS backups.
Terminal exposure and finite launch deadline remain unset until R qualification.

## Immediate commands and remaining work

```bash
tail -1 local/nvme/qualification/qwen-r-fp32-nvme-tmp-004/metrics.jsonl
cat local/qualification/qwen-r-fp32-nvme-tmp-004/backup-supervisor.state.json
```

Complete distributed R004 timing/checkpoints/exit, then the reviewed strict upper
RAM recovery and lower fresh RAM timing/coexistence qualification. Preserve the
strict STOP-at-4 proof. Final host-I/O policy requires actual measurements. Prefer parallel seed chains C1→R1 and C2→R2 on the existing
32-core partitions; ideal world 64 saves no study time against that schedule.
Freeze the prospective256 target using measured simultaneous costs and fresh-data
supply before final-test access. Complete fixed-development B0/C/R direct/empty/
thought comparisons; step64 never selects the primary terminal. Save
all failures, examples and outputs; no outcome-based terminal selection.

Large artifacts remain ignored under `local/`. One verified unused NVMe is
mounted at `local/nvme`; the other three SSDs are untouched. Latest space:
EBS about323 GiB free after R003 backup, NVMe 1.3 TiB free. An obsolete EBS C duplicate was retired
after full verification of its preserved NVMe original; receipt:
`local/qualification/ebs-retirement-c-fe-20261017.json`. The science backup queue
will serialize copies across runs; its six host fixtures and independent TERM
review pass, and it remains unlaunched. A backlog can need two new payloads
before pruning. The old C4 checkpoint and its hardlink alias were relocated to
verified NVMe archives, preserving all bytes and receipts:
`local/qualification/c4-checkpoint-relocation-20261017.json`.
EBS backups and scratch share this instance;
no external backup location is approved. The reviewed backup supervisor retains
two verified complete states per run; see `docs/CHECKPOINT_BACKUP.md`.
