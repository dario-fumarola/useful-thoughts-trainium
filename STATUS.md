# Current state — 2026-10-16 21:45 UTC

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
189 tests and toy smoke passed in `local/checks/tests.ZRihpt`.

## Actual evidence

- Released weights, ties, and fingerprints pass forward qualification. FP32
  CPU/device selected-logp error is 2.57e-5.
- Full-cap BF16 cache/replay failed the fixed .2 bound (max .21841). FP32 head
  alone also failed. Full FP32 passes the exact old failing trajectory with max
  2.84e-5 under the original tighter .001 bound. Three fresh full-cap FP32 traces
  also pass. The original full-cap single-thought backward failed runtime memory
  allocation at21:09:44 UTC, before a gradient norm was obtained. R did not launch.
  The bank was full with23.968GB, including5.295GB retained program spill storage.
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
  `validation/reader_interface_cpu_20261016.json`. Local-content/crossed-reader
  code also completed released-model CPU plumbing on two DEV records (all36
  score rows, no errors). This does not qualify accelerator execution.
- Final evaluation now binds terminal model identities, transitive code hashes,
  canonical gold projection, and scored condition provenance. No final experiment
  or evaluation receipt has been frozen yet.

## Active bounded jobs

- **Core 1:** unchanged-source warm-cache memory diagnostic, PID448724,
  tmux `dario-p2-full-fp32-n2-001`, `local/p2-full-fp32-n2-001.log`.
  Same three full-cap traces/backward and fixed precision/tolerance; only
  `NEURON_NUM_RECENT_MODELS_TO_KEEP=2` is set. This legacy setting does not bound
  residency in the installed stack (nine programs observed). All three traces
  pass, backward is cold-compiling. Independent watchdog503227 safely paused
  only the old GNUtimeout448721, with a finite replacement deadline22:44 UTC;
  worker/compiler continue. Receipt: adjacent `.supervision-active.json`.
- **Core 2:** same-source full-cap diagnostic, PID494794,
  tmux `dario-p2-full-fp32-lru2-001`, `local/p2-full-fp32-lru2-001.log`.
  `XLA_COMPILATION_CACHE_SIZE=2` actually bounds steady residency to two programs;
  it still retains substantial scoring-program DMA storage. All three traces
  are bitwise-identical to the other diagnostic and pass replay. Backward waits
  on the live core1 compiler cache lock; do not remove it. Ninety-minute bound
  from21:25 UTC. A third transient load before eviction remains a memory risk.
- **Cores 32–63:** FP32 Qwen C, 32 ranks, global batch128, 12 updates,
  launcher403728, tmux `dario-qwen-c-fp32-001`. Immutable source `fe667bf`.
  State/log: `local/qualification/qwen-c-fp32-001.state.json` and adjacent `.log`.
  Output: `local/nvme/qualification/qwen-c-fp32-001`. Full-FP32 forward compiled
  at20:57:58 UTC; first/later-accumulation backward and finite-gradient graphs
  passed. Baseline finished12/12updates and exited0. Ten steady steps3–12 average
  10.1728seconds (range10.1029–10.3651). First update completed21:20:40,
  CE1.34797621 versus CPU1.34796970.
  All rank samples agree bitwise and99.09% of sampled values changed. Device
  preclip norm1.62842071 initially failed against ordinary CPU FP32 vector_norm;
  an independent sum of ALL gradient squares in Float64 gives1.62834683 and
  passes the unchanged bound. Both outcomes are preserved in
  `validation/c_fp32_gradient_norm_reference_20261016.json`. Initialized optimizer
  memory passed; sampled peak19.315GiB, no sampled runtime errors. The original
  three-update CPU comparison still FAILS its mandatory pointwise rule on754 of
  1,720,574,976 entries (max2.0014e-6). Aggregate update cosine.999999758 and
  relative L2error.00069529 pass their separately predeclared bounds; they do not
  override the pointwise failure. Corrected-clipping comparison remains pending.
  All three checkpoints3/6/12 are COMPLETE. Step3 EBS copy/readback/hash PASS
  finished21:43:38. Fresh-process FE resume fromstep3 began21:43:57,
  tmux `dario-qwen-c-fp32-resume-001`, same world32 and unchanged strict criteria.
- **Cores 0–31 reserved:** R32 launcher prepared by data agent, not running.
  It requires the full-cap FP32 backward to pass and a qualified memory policy.
  Same immutable source and pilot records as C32; 14 updates, eight-hour bound.
  Original watcher exited `P2_FAILED_NO_R_LAUNCH`; it is not armed. State:
  `local/qualification/qwen-r-fp32-001.gate-watcher.json`. No retry loop.
- Independent CPU FP32 first-three-update C reference has finished; full weights
  and Adam states are hashed and preserved. A separate corrected-norm/clip
  three-update lineage began21:31:49 and is on update2; original results remain
  intact. Diagnose any remaining pointwise errors before numerical qualification.
  A native
  CPU full1024-action PG backward also finished: full Float64-reduced norm
  .1056467963; hardware comparison is still pending. Receipt:
  `validation/p2_full_gradient_cpu_reference_20261016.json`. Reviewer prepared full-cap
  local/secondary accelerator suites and
  waits for cores62–63 after C32 and resume exit. Do not overlap
  reserved accelerator layouts.

## Next actions and recovery

```bash
tail -20 local/p2-full-fp32-n2-001.log
tail -20 local/p2-full-fp32-lru2-001.log
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
The verified-copy utility is `scripts/backup_checkpoint.py`; C32 checkpoint3 is
verified at `local/backups/qwen-c-fp32-001/step-0000003`. Before
multi-day science, arrange at least daily complete saves and verified EBS copies.
Current C qualification reward-related metric zeros are unmeasured placeholders,
not actual clean/thought likelihoods.
Main commit `8749d87` now binds runtime cache settings to frozen per-arm policies
and resumes, emits null for unmeasured C rewards and unaudited replay errors, and
saves complete checkpoints at most12h plus one complete update apart. Independent
review and189 CPU tests/toy smoke pass. Running immutable qualifications are
unaffected. New-source C32 warm qualification/resume will follow the existing
FE baseline/resume; R32 must use the reviewed source and actual qualified cache
policy. New-source device execution remains pending.

CPU checks: `source scripts/env.sh && bash scripts/run_checks.sh`.
