# Current state — 2026-10-17 00:18 UTC

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

**Data agent: R32 qualification attempt 003**, lower logical cores 0–31.
Launcher 779314, timeout/PGID 779372, driver 779374, 32 workers 779498–779532.
Started 22:51:17 UTC; hard deadline **2026-10-17 06:51:17 UTC** plus 30s grace.
Source `3aaaf8a18baeccb8d1a543d348a53f6863599114`, active code hash
`244e2f13f91dfc746f0829b236a35b8f2c5452f5d62da7a1bc3b3bbe31287ff7`.
World 32 / global batch 128 / accumulation 4, G4, cap 1024, FP32, seed 1234, XLA executable LRU 2.
Host store 127.0.0.1:64231; Neuron root 127.0.0.1:64194. Prior attempt 002 failed
before loading because the default host-store port collided with C; its failure
and bounded cleanup are preserved. Attempt 003 has loaded actual weights and
started update 1. Its first backward graph compiled at 23:48:39 UTC after
43 minutes. All 32 ranks have since completed the first backward with zero
recorded execution failures and hold gradient-bearing tensors. First backward
bank allocation was about 19.78 GiB. The next scalar loss passed at 23:54:13.
First accumulated backward `MODULE_7801530274365426598` compiled at 00:10:50
and executed successfully on all 32 workers by 00:14:12: every success counter
increased by one, with zero failure increments. All main banks fell to about
16.52 GiB, consistent with the compiled gradient-buffer aliasing. The next scalar
loss `MODULE_2206265331420854255` is cold-compiling. Later-micro accumulation,
initialized optimizer state and a complete update remain pending.
**No complete R update yet.** Separate sysfs memory sampling is read-only.

Output: `local/nvme/qualification/qwen-r-fp32-final-003`.
Receipts/logs: `local/qualification/qwen-r-fp32-final-003*`.
Root owns the active EBS backup supervisor: launcher 849074, timeout/PGID 849075,
Python 849077, tmux `dario-r-final003-backup`, same 06:51:17 UTC deadline.
It awaits the first COMPLETE checkpoint and retains two verified EBS copies in
`local/backups/qwen-r-fp32-final-003`; source NVMe checkpoints remain untouched.
State: `local/qualification/qwen-r-fp32-final-003.backup-supervisor.state.json`.
Stop this waiting supervisor if the original driver fails or stops early.
After checkpoint 3 exists, prepare a fresh process with STOP precreated: export
actual restored state before sampling, run exactly update 4, compare all 32×4
stochastic tapes and full post-update checkpoint. The prepared resume is not
launched. Require ten complete steady updates before exposure selection.
The upper one-update R overlap smoke is prepared/reviewed, not launched:
`local/qualification/qwen-r-fp32-upper-overlap-001/prepared.json`, SHA
`dee151f1af85b76f86eb64ebb49c293d796463466e0c801a94e682a8db1d18f4`.
It binds actual lower fork/checkpoint 3/STOP readiness, source/settings and fresh
root release; separate host/Neuron ports 64251/64252, upper cores 32–63.
Data owns lower recovery; distributed prepared upper; root coordinates overlap.

**Held-seed jobs complete:** C32 update/COMPLETE and FP64 full derivative both
exited 0; comparator exited 0/PASS. All upper-core workers and supervisors are
gone. Checkpoint: `local/nvme/qualification/qwen-c-fp32-heldseed-001/step-0000001`.
C peak sampled device allocation 18.455 GiB, sampled errors zero. Exact preparation
SHA `26a6081f…2313d4b6`; state/receipts under
`local/qualification/qwen-c-heldseed-001/`. Public receipt and independent audit are complete; all original failure receipts
remain.
Root owns integration, decisions and public pushes. Do not edit common active
code during these jobs; later documentation commits do not alter its fingerprint.

## Immediate commands and remaining work

```bash
python3 local/qualification/qwen-r-fp32-final-003.status.py
cat local/qualification/qwen-c-heldseed-001/comparison.result.json
```

Complete distributed R update/resume/timing.
Overlap the required lower R resume with one bounded upper R update to qualify
two-R coexistence. Prefer parallel seed chains C1→R1 and C2→R2 on the existing
32-core partitions; ideal world 64 saves no study time against that schedule.
Freeze terminal exposure from measured complete-step costs and fresh-data supply before any final test access. Save
all failures, examples and outputs; no outcome-based terminal selection.

Large artifacts remain ignored under `local/`. One verified unused NVMe is
mounted at `local/nvme`; the other three SSDs are untouched. Latest space:
EBS 300 GiB free, NVMe 1.4 TiB free. EBS backups and scratch share this instance;
no external backup location is approved. The reviewed backup supervisor retains
two verified complete states per run; see `docs/CHECKPOINT_BACKUP.md`.
