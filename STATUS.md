# Current state — 2026-10-16 21:02 UTC

Stage: real-asset Neuron qualification. No scientific C/R pair has launched.
The owner authorizes sustained autonomous work, review agents, and public GitHub
publication. Weeks are available; no actual instance cutoff is specified. Declare
a finite planning horizon before the scientific launch. Prospective policy now
allows a 14-day horizon and a timing-selected two-seed schedule; see DECISIONS.md.

Workspace: `/home/ubuntu/dario`. Public repository:
https://github.com/dario-fumarola/useful-thoughts-trainium . Original repos remain
unchanged. Released Qwen3-1.7B-Base revision:
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.

Verified data: **329,535 train / 18,360 dev / 18,239 test** records in
`local/data/finemath/production-500000.audit-v2.records.jsonl`.
Receipt: `validation/data_production_v2_20261016.json`. A URL identity bug was
corrected before freeze; the original audit is retained. Latest CPU checks:
184 tests and toy smoke passed in `local/checks/tests.QJY1yz`.

## Actual evidence

- Released weights, ties, and fingerprints pass forward qualification. FP32
  CPU/device selected-logp error is 2.57e-5.
- Full-cap BF16 cache/replay failed the fixed .2 bound (max .21841). FP32 head
  alone also failed. Full FP32 passes the exact old failing trajectory with max
  2.84e-5 under the original tighter .001 bound. Three fresh full-cap FP32 traces
  also pass; the single-thought backward is still compiling.
- Tiny distributed FP32 updates and numerical resumption pass. Full Qwen C on
  four ranks completed one real update and a complete checkpoint, but update two
  exceeded memory. No full-model resume pass yet. Qualification moved to 32 ranks
  before any scientific training.
- Five held-out GSM8K TRAIN items ran on the accelerator with fixed four-shot
  prompts in direct, thought, and empty-format modes. Outputs and failures are
  retained; these are interface checks, not benchmark estimates. No official test
  outcomes have been inspected. Secondary likelihood evaluation has CPU evidence
  only (`validation/secondary_likelihood_cpu_20261016.json`).
- CPU partial-product diagnostics show semantic reader sensitivity on 16/16 fixed
  cases but a substantial insertion cost versus raw silence. K=1 is whitespace on
  this synthetic grid; full-target sensitivity is not K=1 sensitivity. Receipt:
  `validation/reader_interface_cpu_20261016.json`. New local-content/crossed-reader
  code has CPU tests; released-model CPU plumbing is running on two DEV records.
- Final evaluation now binds terminal model identities, transitive code hashes,
  canonical gold projection, and scored condition provenance. No final experiment
  or evaluation receipt has been frozen yet.

## Active bounded jobs

- **Core 1:** full-FP32 prefix512/cap1024 cache/replay/backward, PID374089,
  `local/p2-full-fp32-001.log`, hard limit three hours from about 20:05 UTC.
  Fresh trace numerical checks passed; backward is pending.
- **Cores 32–63:** FP32 Qwen C, 32 ranks, global batch128, 12 updates,
  launcher403728, tmux `dario-qwen-c-fp32-001`. Immutable source `fe667bf`.
  State/log: `local/qualification/qwen-c-fp32-001.state.json` and adjacent `.log`.
  Output: `local/nvme/qualification/qwen-c-fp32-001`. Full-FP32 forward compiled
  at20:57:58 UTC; first backward is compiling. The independent CPU first-update
  reference is CE1.34796970 / preclip norm1.62742829; device comparison is pending.
- **Cores 0–31 reserved:** R32 launcher prepared by data agent, not running.
  It requires the full-cap FP32 backward to pass and its process to exit first.
  Same immutable source and pilot records as C32; 14 updates, eight-hour bound.
  One-shot watcher PID437606/tmux `dario-r-launch-gate` is armed until23:20 UTC;
  state `local/qualification/qwen-r-fp32-001.gate-watcher.json`. No retry loop.
- Independent CPU FP32 first-three-update C reference is running at low priority
  while accelerator graphs compile. Reviewer is finishing local CPU qualification.
  Core2 is free; do not overlap reserved accelerator layouts.

## Next actions and recovery

```bash
tail -20 local/p2-full-fp32-001.log
cat local/qualification/qwen-c-fp32-001.state.json
```

Require real C/R updates, numerical full-model resume, and at least ten complete
steady updates per arm before selecting terminal exposure and freezing the first
pair. Latest complete checkpoint is qualification-only:
`local/qualification/qwen-c-001/step-0000001` (~25.64 GiB). It does not establish
that the failed four-rank layout can resume.

One verified unused instance SSD is mounted at `local/nvme` (1.7 TiB free at
preparation); the other three were untouched. Provenance and preparation receipts
are in `validation/`. Store large intermediates there; copy scientific terminal
and daily latest-complete checkpoints to EBS. Both are local to this instance;
no external backup destination has been approved.
The verified-copy utility is `scripts/backup_checkpoint.py`; the distributed
agent will copy C32 checkpoint3 to `local/backups/` once COMPLETE exists. Before
multi-day science, arrange at least daily complete saves and verified EBS copies.
Current C qualification reward-related metric zeros are unmeasured placeholders,
not actual clean/thought likelihoods.

CPU checks: `source scripts/env.sh && bash scripts/run_checks.sh`.
