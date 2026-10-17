# Status — 2026-10-17 23:11 UTC

The owner stopped the original 256-update R study and redirected work to the
reward interface, genuine candidate oracles, and iteration speed. The old run
order in CODEX_NEXT_DAY.md and older handoffs is superseded. Follow NEXT_STEPS.md.

## Active work

- **interface-tax-dev32-v1 complete**, exit0 at22:34:13UTC,896/896 scores
  succeeded. Output `local/nvme/diagnostics/interface-tax-dev32-v1`, control
  `local/diagnostics/interface-tax-dev32-v1`; all owned processes gone.
  Public `validation/interface_tax_dev32_20261017.json` and figure; independently
  reviewed all400 contrasts/intervals against originals to1e-12.
  Original-interface token parity max3.8147e-5 nats. Empty−raw tax: original
  −6.4247, noheader−6.6520, reanchor32−1.0797, reanchor64−1.4121.
  Reanchor32 own−empty−1.9057, CI[−2.6787,−1.1928]; own−other−.4319,
  CI[−.9675,+.0582]; own−raw repeated-prefix−1.6273. Interface repair helps,
  but original G1 thoughts still hurt on average. K16/strata/control results
  remain in the report. EBS archive `local/backups/diagnostics/interface-tax-dev32-v1/VERIFIED.json`:
  16files/2,026,409B, originals/selected inputs/first4 examples/sealed sources.
- **candidate-oracle-dev32-cap256-v1 RUNNING**, launched22:44:46UTC on
  logical32/device8. tmux `dario-candidate-oracle-v1`; launcher2157411,
  generation timeout2157416/worker2157417. Control `local/diagnostics/` and
  output `local/nvme/diagnostics/` under that experiment ID. Overall finite
  deadline2026-10-18T00:44:46.913949+00:00. Protocol SHA2d433f847a091f67e70e0205142d7ba721f009eeeb4372032f64b7b81b79b677.
  Original prefix-only B0 writer,16 draws per fixed pilot prefix (four
  independent G4 groups), cap256/T.7; original+reanchor32 readers. One standalone
  cache8 process, private tmpfs; firstgroup FP32 cache/replay passed max1.097e-5
  nats. First group sampling163.65s + audit49.03s; next twoG4 groups12.50/12.52s
  (not a full-run average; group wall counted once).
  All512 candidates precede future access;4256 reader contexts include controls
  and same-interface empties. Oracle uses future selection and is distinct from
  mean16 utility; max16-control comparisons expose selection-only effects.
  Search is exploratory; cap256 does not bound the1024-token oracle.
- **replay-padding-dev2-runtimefix-v1 RUNNING**, launched22:48:48UTC on
  logical0/device0, tmux `dario-replay-padding-fix-v1`. Control/output under
  local/{diagnostics,nvme/diagnostics}/sameID; read actual launcher/timeout PID
  files there. Protocol7b1f008901b7ddf877fe9747809266fb54c9ef317459f2fd1b51c10fd314ab8d;
  deadline2026-10-18T00:48:48.339414+00:00. Same cases/settings/criteria and
  oldRLRU2 as prior diagnostic; only original-driver device registration added.
  Prior replay-padding-dev2-v2 failed before its first backward (torch.xla
  module unregistered); all outputs/exact10-file source snapshot retained.
  Its full first forward269.6s is cold compile/setup, not a steady-state cost;
  the repaired run has passed both signed full-gradient empty1 comparisons
  (relativeL2 1.759e-6,cosine .9999999999985). The real89-action comparison
  remains pending. These are isolated stages, not distributed throughput.
  This is a runtime helper repair,
  not a new scientific qualification or numerical backend change.
- **anchors-train1024-v1 COMPLETE**, exit0 at22:58:32, all1024 raw TRAIN
  rows finite/ordered,237 qualify at frozen surprisal>=1.7; first128 selected
  without fallback. DEV width parity6.676e-6. Public report
  `validation/anchors_train1024_20261017.json`; verified EBS33files/1,382,445B.
  This is target-based TRAIN selection, not held-out utility.
- **prompt-interface-dev32-v1 COMPLETE**, exit0 at22:59:20,864 finite rows.
  Published wording empty−raw−6.223 nats; plain-newline+reanchor32−1.085,
  own−empty−1.175, own−other−.175 with interval spanning zero. Matched
  plain empty−raw-repeat32+.273, interval spans zero. Actual reader parity
  max4.578e-5. Independently audited original summary/coverage/contrasts.
  Public paired report `validation/prompt_interface_dev32_20261017.json`;
  verified EBS31files/2,011,921B includes fixed inputs/examples/sealed sources.
  No new writer, exact-paper serialization, or capability claim.
- **Released marker embeddings inspected**: all25 defined non-EOS added IDs
  151644–151668 are exactly identical at B0, including think/end-think/chat
  boundaries; equality extends through151736 in the inspected tail (93rows).
  They are nonzero. Input identity aliasing explains identical scores when
  the published delimiter typo is corrected. The tied output head shares
  those rows. No inference about training history or sole failure cause.
  Reproducible scalar report `validation/B0_marker_embeddings_20261017.json`.
- **reader-support-dev8-v1 COMPLETE**, exit0 at23:08:40;128 contexts/2048
  token positions, no failures, full-support parity2.766e-5. Same-logit
  defined-support renormalization changes original empty−raw tax by only
  .000549 nats out of5.678. Maximum undefined mass .002060. This rejects
  undefined-head mass as an explanation on these fixed8 development docs;
  no masking/weight/policy change. Public report
  `validation/reader_support_dev8_20261017.json`; verified EBS32files/2,180,375B.
- **candidate-corpus-train128-g16-v1 GENERATING**, released23:07:20,
  deadlineOct18 01:07:20UTC. Eight shards on distinct physical devices
  1/3/4/5/6/7/9/10, logical4/12/16/20/24/28/36/40. Standard control/output
  directories; per-shard PID/log/exit files under control/shard-XX, tmux
  dario-corpus-00..07. Exact protocol a02a536613846765f30d7ed01ef69a05109698ebcbc3dd23ea64979a37d07f9e.
  All2048 original-writer candidates (128 TRAIN docs,G16,cap256,T.7,FP32)
  must close before global donors/controls and new future scoring. Only
  generation released;17024 reader contexts planned. No student training yet.
- Old shared backup and node services were deliberately stopped at22:36:14,
  after verified R63 backup; all six owned helper/producer/compressor/timeout
  identities are gone. Node stream is a closed INTERRUPTED prefix, not a full
  deadline capture. Backup handler's FAILED_STOP_REQUESTED state is expected
  from deliberate TERM; owner-cancellation receipt records the reason.
- Agents: distributed verifies R closure/backup and profiles source costs;
  data monitors DEV candidates and support closure; review audits new reports,
  TRAIN shards and the next short warmstart design. No old matrix releases.

## Old study stopped, evidence retained

R1234 STOP written22:27:20UTC. It completed **step63** at22:28:48, saved actual
state (COMPLETE22:29:38, SHA7c1e61a1fd4294d26e9d9a278d141de5d76ac0b1a44efeea3c8faaf499a9c428),
and driver exited0. 33 payloads total27,542,553,263B; counters8064 records,
4,136,832 source tokens,1,836,832 sampled actions. All original metrics/notes
remain. EBS backup readback and whole-group cleanup are verified. Exact public stop
report `validation/science_R_seed1234_stopped_20261017.json`; small run/log/node
archive93files/491,348,596B in local/backups/run-artifacts. Separate actual
checkpoint EBS BACKUP SHA82e8781d9e74873509f97573a21fab4ec08bf1b549c3fc1e253649d9067527cb. No step256
or scientific primary R/C result exists; R5678 was never launched.

C1234@64 local DEV and C5678@64 held development both exited143 at22:27:21 on
owner stop. Their original partial outputs remain, with OWNER_STOP.json and
exit receipts; verified interrupted EBS archives retain14files/6,012,750B
(local C1234) and42files/4,723,730B (held C5678). C5678 direct/empty each completed/scored128; thought has10
unscored predictions. Do not describe either interrupted suite as complete.
The old final plan remains PREPARED_NOT_LOCKED and on hold. No official test
outcomes will guide interface development.

## Established observations

- **B0 DEV512**, G1 per item: raw K1 mean−1.118 nats, empty format−7.361,
  own thought−11.203. Own−empty−3.842; own−other−.284 (95% conditional CI
  [−.641,+.059]); shuffled own−7.327. K4/K16, controls, strata and failures
  are in `validation/local_dev_B0_20261017.json`; original bundle and verified
  EBS archive retained. **K16 is a future horizon, not16 candidate thoughts.**
- Both C seeds finished256,32,768 unique TRAIN records/16.81M source tokens.
  Actual C steps3–256 average~10s. NVMe64/128/256 and verified EBS128/256 remain.
- GSM8K TRAIN development128: B0 direct/empty/thought113/107/63; C1234@64
  111/108/60; C1234@256109/109/91. C256 thought improved vs B0 thought but
  still trails its direct/empty by18. All128 C256 thoughts hit1024 cap;
  format-error recovery and~91s/item cost prevent a reasoning-only claim.
  Exact reports and fixed-order examples are preserved in validation/local EBS.
- FP32 original implementation, signed gradient and recovery checks passed.
  Those engineering checks do not show useful thoughts. BF16 replay mismatched;
  the current loss has no PPO clipping. Zero-initialized logit mixing eliminates
  both insertion tax and this stopped-reader writer reward until opened.

## Next command and validation

Observe candidate generation count/parity and replay phases in their run logs.
Both wrappers are finite and have no retry. Data agent supervises candidates;
distributed agent supervises replay. Root owns signals and final archive/report.
Do not edit either source/dependency closure while its job is active. Read
candidate oracle only after complete score coverage; preserve all cases/costs.

Latest required `bash scripts/run_checks.sh`:256 tests + CPU toy smoke passed,
`local/checks/tests.uPXJBj` after the new TRAIN corpus/support/statistics scripts. Candidate semantic tests6 and replay tests4 cover
stopping/seed/future barriers/document reduction/padding-gradient behavior;
these are not hardware outcomes. First interface pilot has actual FP32 parity.

Public repo https://github.com/dario-fumarola/useful-thoughts-trainium.
Keep weights/corpus/raw text/full topology/cache/secrets out of Git. Existing
hardware only; no paid teachers, new capacity, IAM/network/stack changes.
The Oct17 morning review ZIP is an older snapshot; do not rebuild every step.
