# Current state — 2026-10-16 19:30 UTC

Stage: real-asset Neuron qualification; scientific C/R pair not launched.
Owner now authorizes sustained autonomous work, scientific judgment and review agents; weeks are available. No actual instance cutoff has been specified. A finite job planning horizon will be declared before a scientific launch.

Workspace/private remote: `/home/ubuntu/dario`, https://github.com/dario-fumarola/dario . Original repos unchanged. Released Qwen3-1.7B-Base pinned to `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` in `local/assets/qwen3-1.7b-base`.

Verified real data: 329,535 train / 18,360 dev / 18,239 test; `local/data/finemath/production-500000.audit-v2.records.jsonl`; receipt `validation/data_production_v2_20261016.json`. URL-identity bug corrected before freeze; original audit retained. 121 CPU tests + toy smoke passed in `local/checks/tests.2VOisY` (later additions may raise count).

Hardware evidence so far:
- Real-weight FP32 CPU/device selected-logp error 2.57e-5; XLA transfer broke the tie, now explicitly restored and fingerprint matches. Final tied BF16 check pending; p1-004 timeboxed out in compilation, p1-005 waiting on a stale cache lock.
- Short real BF16 cached/replayed traces passed, max action-logp errors .02591/.05649. `local/qualification/p2-bf16-001` contains raw IDs, probabilities, evidence and text. Not a full-cap qualification.
- Two-rank tiny FP32 mean update and fresh-process resume pass numerical tolerance, not bitwise; `local/qualification/distributed-005`, `docs/DISTRIBUTED_QUALIFICATION.md`. Full Qwen update/resume remains pending.
- GSM8K TRAIN-only direct and initial thought evaluator paths run; structural reader defect found and repaired before final freeze. No official-test outcomes inspected.

Active work/jobs (bounded diagnostic runs):
- Core0 P1-005: PID342172, `local/p1-005.log`, `local/qualification/p1-005`; timeout90min. Do not duplicate; resolve provably stale cache lock.
- Core1 full512prefix/1024action cache/replay/backward: PID342155, `local/p2-full-001.log`, `local/qualification/p2-full-001`; timeout2h.
- Core2 evaluator agent: `local/evaluation/dev-thought-002.log`, 5 fixed TRAIN development items; no scientific final test.
- Cores4–7 distributed agent: starting real-Qwen C2step update/checkpoint/fresh-process resume; `local/qualification/qwen-c-001*` immutable source snapshot. Check exact PID/output before launching anything overlapping.

Next exact command:
```bash
tail -20 local/p2-full-001.log
```
Continue P1–P4 qualification and measure ≥10 complete target-shape optimizer steps before freezing terminal exposure. Main production driver exists but is not qualified. No scientific checkpoint yet. Checkpoint backups remain local; no external destination approved. Four unused NVMe devices inspected read-only (7.6TB total, no signatures), not modified.

CPU checks: `source scripts/env.sh && bash scripts/run_checks.sh` (script forces CPU-only Transformers discovery).
