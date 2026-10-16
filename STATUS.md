# Current state — 2026-10-16 22:51 UTC

Real-asset Trainium qualification; **no scientific C/R pair has launched**.
The owner authorizes sustained autonomous work, review agents, and this public
repository: https://github.com/dario-fumarola/useful-thoughts-trainium . Work is
isolated in `/home/ubuntu/dario`; original repositories remain unchanged. Weeks
are available. The prospective 14-day science schedule is a planning limit, not
an instance shutdown deadline (see DECISIONS.md).

## Assets and completed checks

- Released Qwen3-1.7B-Base revision
  `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, loaded from actual weights with
  the original embedding tie. Both proposed arms use FP32.
- Audited FineMath: **329,535 train / 18,360 dev / 18,239 test** records in
  `local/data/finemath/production-500000.audit-v2.records.jsonl`.
- Full-cap FP32 sampling/replay passes the unchanged .001 bound, including the
  exact trajectory that failed BF16. No official benchmark test outcomes read.
- C32 completed 12 real updates; ten steady updates averaged **10.17 seconds**.
  Sampled replicas agree bitwise, actual weights change, initialized optimizer
  fits (sampled peak 19.327 GiB). Checkpoints 3/6/12 are COMPLETE. Step 3 has a
  verified full EBS copy at `local/backups/qwen-c-fp32-001/step-0000003`.
- Accurate CPU Float64 gradient-norm reduction passes the original device norm
  bound. However, the strict three-step CPU per-value model comparison still
  **fails on 754 / 1,720,574,976 coordinates**, including after correcting CPU
  clipping. Aggregate cosine and relative-L2 criteria pass but do not override
  that failure. A tiny identical-gradient hardware AdamW test passes. Next:
  Ordinary C32 checkpoint1 now locates the discrepancy before Adam: 689 of
  those selected coordinates have opposite first-gradient signs (controls: 0).
  The full first-gradient vectors have cosine .9999999644 and relative L2
  error .0002707; selected errors remain unresolved. An independent FP64
  arithmetic reference is being prepared. No failed criterion has been changed.
- C32 fresh-process resume finished: full model, all 32 master/moment shards,
  counters and metadata pass the unchanged numerical bounds. **Exact device
  RNG fails on every rank**, each by exactly one XLA MarkStep seed transition.
  Overall resume is therefore not qualified yet.
- Development-only math inference is retained, including failures. CPU local
  controls/crossed readers and secondary likelihood paths pass their plumbing
  checks; their full-cap accelerator execution remains pending.

## Repairs and current jobs

1. **R normalization:** the original backward first exhausted memory. Actual
   `XLA_COMPILATION_CACHE_SIZE=2` repairs residency; the legacy Neuron limit
   does not. The completed backward then exposed a numerical defect: device
   int64 `mask.sum().clamp_min(1)` returns -1, reversing and scaling gradients.
   Float32 counting preserves the intended objective and passes six reduced
   production-shape cases. No scientific R update used the faulty expression.
2. **Repaired full-model backward PASS:** completed at 22:37 UTC, source
   `8b44cc1`, fixed original 1,024-action trace. Actual driver loss materialization
   and backward agree with the independent CPU reference under unchanged bounds:
   loss error 4.66e-10, norm error 1.33e-6 (allowed 6.28e-5), signed 9,920-value
   sample cosine .9999999940 and relative L2 error .00010923. All 310 parameter
   gradients are present. Worker 636493 and its supervisors exited 0; core 2 is
   released. Receipts: `local/qualification/p2-full-fp32-countfix-signed-001`
   and adjacent `.comparison.json` / `.execution_receipt.json`. This qualifies
   a fixed-trajectory backward, not a distributed R update or resume.
   Scalar controls also confirm integer upper-bound narrowing; see
   `validation/int64_clamp_mechanism_20261016.json`. Diagnostic core 3 is released;
   production compiler flags are unchanged.
3. **Checkpoint RNG:** installed Neuron rendezvous/export does not advance the
   measured seed; an explicit sync does. The old resume restores RNG before
   that sync, causing the observed extra transition. A bounded hardware tiny
   test reproduces old-order failure and fixed-order exact next-two-draw,
   CPU/device RNG, model and optimizer equality. Root has integrated restoration
   after the load sync and a checkpoint observation fence. The integrated helper
   passes the actual tiny hardware test, including both next random tensors and
   optimizer state bitwise. Source review and **233 CPU tests plus toy smoke**
   pass (`local/checks/tests.Yr1uiR`). No per-record reseeding or relaxed criterion.
4. **Cores 32–63:** common-source C12 baseline completed and all workers exited
   0. Ten steady updates averaged **10.007 seconds**. Checkpoints 3/6/12 are
   COMPLETE; verified EBS copying/retention is supervised. Fresh-process resume
   is active (driver **731270**); actual restored model, all 32 optimizer shards,
   metadata, counters and CPU/device RNG pass raw-byte exact comparison.
   Continued-update comparison is pending. Distributed agent owns exact PIDs and
   receipts under `local/qualification/qwen-c-fp32-final-003*` (six-hour bound).
   Output: `local/nvme/qualification/qwen-c-fp32-final-003`. Source `3aaaf8a`,
   active-code hash `244e2f13…31287ff7`. The new CPU3 comparison has exactly the
   same 754 failures; it has not been reclassified as a pass.
5. **Both full-cap evaluator suites PASS and cores 0/1 released.** Local:
   36 device/36 CPU rows, all 192 token comparisons, max error .00023079.
   Secondary: 8 rows, 90 token comparisons, max error .000070572, zero greedy
   token/ranking mismatches. Both meet the unchanged .001 bound. Receipt:
   `validation/evaluator_fp32_fullcap_20261016.json`. No official test outcomes
   were accessed.
6. **R32 attempt 002 stopped before model loading.** Its 31 peers joined C's
   default TCPStore port 12355 while R rank 0 failed to bind it. Stopping one
   blocked child exposed the original EADDRINUSE exception and reaped every R
   process (exit 1, no output/model/update). Preserve the adjacent bootstrap and
   stop receipts; do not reuse this attempt's name. One new bounded attempt 003
   is authorized with the same source/settings and Neuron endpoint 64194, plus
   explicit `MASTER_ADDR=127.0.0.1`, dedicated `MASTER_PORT=64231`. No other
   change or automatic retry. Data agent owns its launch/preflight receipts.
7. Independent CPU reader comparison to all twelve existing full-cap device
   development notes passes, including five natural stops: max absolute logp
   error **8.49e-5** at the unchanged .001 bound. Receipt:
   `validation/p2_full_reader_reference_20261016.json`.

## Independent CPU diagnostic

`local/qualification/qwen-c-fp64-firstbatch-001` is running (32 threads, hard
2-hour bound). It promotes differentiable RMSNorm/attention/loss arithmetic to
FP64 with exact released FP32 weights and frozen native-CPU FP32 RoPE constants;
plain `model.double()` would silently retain FP32 operations. Same first 128
ordered TRAIN records, no optimizer. Four tiny formula/dtype fixtures and
independent source review pass. This is a derivative diagnostic, not a new gate.

## Immediate recovery and next action

```bash
tail -20 local/qualification/qwen-c-fp32-final-003.resume.log
cat local/qualification/p2-full-fp32-countfix-signed-001.comparison.json
cat local/qualification/qwen-c-fp64-firstbatch-001.progress.json
source scripts/env.sh
bash scripts/run_checks.sh
```

Complete the full-model RNG repair check, diagnose the C pointwise failure,
then run distributed R updates, actual restored-state readback, first resumed
stochastic tapes and complete checkpoint comparisons. Require ten complete
steady updates per arm before timing-selecting terminal exposure and freezing.
The science recipe and final outcomes have not been selected using these probes.

Large artifacts remain ignored under `local/`; scratch is the one verified unused
NVMe mounted at `local/nvme`. The other three SSDs are untouched. Both scratch
and EBS backups are on this instance; no external backup location is approved.
Reviewed `scripts/watch_checkpoint_backups.py` supervises verified EBS copies
and retains two recoverable saves for multi-day science. See
`docs/CHECKPOINT_BACKUP.md`.
