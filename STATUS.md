# Current state — 2026-10-16 20:20 UTC

Stage: real-asset Neuron qualification; scientific C/R pair not launched.
Owner now authorizes sustained autonomous work, scientific judgment and review agents; weeks are available. No actual instance cutoff has been specified. A finite job planning horizon will be declared before a scientific launch.

Workspace/private remote: `/home/ubuntu/dario`, https://github.com/dario-fumarola/dario . Original repos unchanged. Released Qwen3-1.7B-Base pinned to `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` in `local/assets/qwen3-1.7b-base`.

Verified real data: 329,535 train / 18,360 dev / 18,239 test; `local/data/finemath/production-500000.audit-v2.records.jsonl`; receipt `validation/data_production_v2_20261016.json`. URL-identity bug corrected before freeze; original audit retained. Latest full CPU checks: 137 tests + toy smoke passed in `local/checks/tests.9X6EoI`.

Hardware evidence so far:
- P1-005 passed with released tie/fingerprint preserved: FP32 CPU/device selected-logp error 2.57e-5; BF16 error .14507 under predeclared .15 bound. Original XLA transfer tie break fixed; one proven orphan cache lock preserved/renamed after owner death.
- Short BF16 traces passed, but full1024action BF16 failed fixed .2 bound (max .21841). FP32-head-only repair also failed (.22614), including exact failing action replay. No BF16 long-run qualification. Full-FP32 random and old-action probes now running with unchanged tighter .001 bound; all failures retained.
- Tiny FP32 distributed update/resume passes numerically, not bitwise (`distributed-005`). Full Qwen C first4rank update and complete checkpoint passed:1.50265loss,6.26648preclipnorm,~99%sampledFP32valueschanged; all310Adamstates initialized/mastershards FP32. Checkpoint1~25.64GiB; fullQwen resume still pending. Peak~22.42GiB/core atworld4; R needs more sharding.
- GSM8K TRAIN-only direct and initial thought evaluator paths run; structural reader defect found and repaired before final freeze. No official-test outcomes inspected.

Active work/jobs (bounded diagnostic runs):
- Core0 free (P1-005 completed).
- Core1 fullFP32 512prefix/1024action cache/replay/backward: PID374089, `local/p2-full-fp32-001.log`; timeout3h. BF16 attempts failed before backward.
- Core2 fullFP32 exact-old-action numeric probe: PID389250, `local/qualification/p2-fixed-trace-fullfp32-001`. Five TRAIN evaluator items finished; conventional4shot protocol staged, awaiting precision qualification. No final test accessed.
- Cores4–7 Qwen C2step qualification: parent350424, workers350441/350444/350445/350446; `local/qualification/qwen-c-001*` immutable source snapshot. Checkpoint1 COMPLETE; update2 compiling new backward variant. Resume next. Do not overlap.

Next exact command:
```bash
tail -20 local/p2-full-fp32-001.log
```
Continue P1–P4 qualification and measure ≥10 complete target-shape optimizer steps before freezing terminal exposure. Main production driver exists but is not qualified. Latest complete checkpoint is qualification-only `local/qualification/qwen-c-001/step-0000001`; no scientific checkpoint yet. Checkpoint backups remain local; no external destination approved. Four unused NVMe devices inspected read-only (7.6TB total, no signatures), not modified.

CPU checks: `source scripts/env.sh && bash scripts/run_checks.sh` (script forces CPU-only Transformers discovery).
