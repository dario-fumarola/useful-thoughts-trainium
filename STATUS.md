# Current state — 2026-10-16 22:27 UTC

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
  fits (sampled peak 19.315 GiB). Checkpoints 3/6/12 are COMPLETE. Step 3 has a
  verified full EBS copy at `local/backups/qwen-c-fp32-001/step-0000003`.
- Accurate CPU Float64 gradient-norm reduction passes the original device norm
  bound. However, the strict three-step CPU per-value model comparison still
  **fails on 754 / 1,720,574,976 coordinates**, including after correcting CPU
  clipping. Aggregate cosine and relative-L2 criteria pass but do not override
  that failure. A tiny identical-gradient hardware AdamW test passes. Next:
  recover the actual first-step reduced/clipped gradient from ordinary C32
  checkpoint moments, without adding training hooks.
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
2. **Core 2, active full-model repaired check:** worker **636493**, PGID 636491,
   tmux `dario-p2-full-fp32-countfix-signed-001`, log
   `local/p2-full-fp32-countfix-signed-001.log`. Immutable source `8b44cc1`.
   Original saved 1,024-action trajectory, actual driver loss materialization,
   original CPU norm bound plus prospectively fixed signed-gradient checks.
   The repaired loss matches CPU within **4.66e-10**; backward is cold-compiling.
   Hard deadline **23:44:04 UTC**, plus 30-second termination grace. Data agent
   monitors this job. R32 remains off until the actual gradient comparison passes.
3. **Checkpoint RNG:** installed Neuron rendezvous/export does not advance the
   measured seed; an explicit sync does. The old resume restores RNG before
   that sync, causing the observed extra transition. A bounded hardware tiny
   test reproduces old-order failure and fixed-order exact next-two-draw,
   CPU/device RNG, model and optimizer equality. Root has integrated restoration
   after the load sync and a checkpoint observation fence. Source review and
   **229 CPU tests plus toy smoke** pass (`local/checks/tests.pYJ2nn`). A rerun
   using the integrated helper precedes new C/R qualification. No per-record
   reseeding or relaxed RNG criterion.
4. **No other accelerator training job is active.** Cores 32–63 are free after
   the old C baseline/resume and bounded tiny diagnostics. Prepared older-source
   C/R launchers remain held; refresh them to the reviewed common source first.

## Immediate recovery and next action

```bash
tail -20 local/p2-full-fp32-countfix-signed-001.log
cat local/qualification/qwen-c-fp32-001.resume_compare.result.json
source scripts/env.sh
bash scripts/run_checks.sh
```

Qualify the integrated RNG repair, diagnose the remaining C pointwise failure,
then run common-source C/R updates, actual restored-state readback, first resumed
stochastic tapes and complete checkpoint comparisons. Require ten complete
steady updates per arm before timing-selecting terminal exposure and freezing.
The science recipe and final outcomes have not been selected using these probes.

Large artifacts remain ignored under `local/`; scratch is the one verified unused
NVMe mounted at `local/nvme`. The other three SSDs are untouched. Both scratch
and EBS backups are on this instance; no external backup location is approved.
Reviewed `scripts/watch_checkpoint_backups.py` supervises verified EBS copies
and retains two recoverable saves for multi-day science. See
`docs/CHECKPOINT_BACKUP.md`.
