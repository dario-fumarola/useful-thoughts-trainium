# Current state — 2026-10-16 23:01 UTC

Real-asset Trainium qualification; **no scientific C/R pair has launched**.
Public repo: https://github.com/dario-fumarola/useful-thoughts-trainium . Work is
isolated in `/home/ubuntu/dario`; original repositories remain unchanged. The
owner authorizes sustained autonomous work and review agents, with weeks available.
The prospective 14-day science schedule is a planning limit, not a shutdown date.

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
  RNG. Continued updates through step12 meet the original numerical tolerance;
  all RNG states and Adam steps remain exact. Continued weights are not bitwise
  identical. Verified EBS backups retain steps6/12. All C workers and supervisor
  exited. See `validation/c_fp32_final_resume_20261016.json`.
- Both full-cap FP32 evaluator suites pass .001 CPU parity: local controls and
  crossed readers, 192 token comparisons, max error .00023079; secondary paths,
  90 token comparisons, max error .000070572, no ranking/token mismatches.
  All workers exited; only development/interface items used. See
  `validation/evaluator_fp32_fullcap_20261016.json`.
- Latest required source checks: **233 CPU tests plus toy smoke PASS**,
  `local/checks/tests.Yr1uiR`. Later commits change documentation/receipts only.

## C numerical diagnosis and next validation

The original three-step CPU FP32 per-weight comparison remains **FAIL**:
754 of 1,720,574,976 coordinates exceed its fixed bound. Correcting the CPU norm
reduction does not remove this failure. A separate identical-gradient hardware
AdamW test passes; first-step checkpoint moments locate the discrepancy before
Adam, where tiny cancelled gradients can differ in sign.

An independent FP64 differentiable-arithmetic diagnostic is complete (all jobs
exited). It uses exact released FP32 weights cast to double, promoted RMSNorm and
softmax, and frozen native-CPU FP32 RoPE constants. All 310 full derivatives are
saved. Against it, device relative gradient L2 error is **.0001083**, versus
**.0003425** for the original CPU FP32 reference. Device is closer on 672 of the
754 flagged coordinates, but 90 selected sign disagreements remain. This is
numerical diagnosis, not a retroactive v1 pass or a scientific result. Receipts:
`validation/c_firstbatch_fp64_derivative_20261016.json` and
`validation/c_fp32_gradient_discrepancy_20261016.json`.

A distinct prospective **v2** is being prepared on held seed5678, with exact
first128 training IDs fixed before outcomes. It compares a fresh ordinary C32
first update against the full FP64 derivative. Preserve v1 FAIL. Freeze the
loss/norm and global/per-parameter gradient criteria before launching. Seed5678
is a new order, not necessarily disjoint data; report overlap explicitly.

## Active work and process ownership

**Data agent: R32 qualification attempt003**, lower logical cores0–31.
Launcher779314, timeout/PGID779372, driver779374, 32 workers779498–779532.
Started22:51:17 UTC; hard deadline **2026-10-17 06:51:17 UTC** plus30s grace.
Source `3aaaf8a18baeccb8d1a543d348a53f6863599114`, active code hash
`244e2f13f91dfc746f0829b236a35b8f2c5452f5d62da7a1bc3b3bbe31287ff7`.
World32/global128/accum4, G4, cap1024, FP32, seed1234, XLA executable LRU2.
Host store127.0.0.1:64231; Neuron root127.0.0.1:64194. Prior attempt002 failed
before loading because the default host-store port collided with C; its failure
and bounded cleanup are preserved. Attempt003 has loaded actual weights and
started update1, with cold sampling compilation in progress. **No complete R
update yet.** Watch first versus accumulated backward memory; do not infer a
pass from the single-backward probe. Separate sysfs memory sampling is read-only.

Output: `local/nvme/qualification/qwen-r-fp32-final-003`.
Receipts/logs: `local/qualification/qwen-r-fp32-final-003*`.
After checkpoint3 exists, prepare a fresh process with STOP precreated: export
actual restored state before sampling, run exactly update4, compare all32×4
stochastic tapes and full post-update checkpoint. The prepared resume is not
launched. Require ten complete steady updates before exposure selection.

**Distributed agent:** prepare held-seed C32/FP64 validation, upper cores32–63;
no seed5678 outcomes authorized until prospective criteria and IDs are committed.
**Review agent:** independently freeze/review v2 criteria and exact batch.
Root owns integration, decisions and public pushes. Do not edit common active
code during these jobs; later documentation commits do not alter its fingerprint.

## Immediate commands and remaining work

```bash
tail -20 local/qualification/qwen-r-fp32-final-003.log
cat local/qualification/qwen-r-fp32-final-003.state.json
cat validation/c_fp32_final_resume_20261016.json
```

Complete the held-seed C numerical check and distributed R update/resume/timing.
Then consider sequential full-node64 qualification for better use of hardware,
keeping global128 and the objective fixed. Freeze terminal exposure from measured
complete-step costs and fresh-data supply before any final test access. Save
all failures, examples and outputs; no outcome-based terminal selection.

Large artifacts remain ignored under `local/`. One verified unused NVMe is
mounted at `local/nvme`; the other three SSDs are untouched. Latest space:
EBS303GiB free, NVMe1.4TiB free. EBS backups and scratch share this instance;
no external backup location is approved. The reviewed backup supervisor retains
two verified complete states per run; see `docs/CHECKPOINT_BACKUP.md`.
