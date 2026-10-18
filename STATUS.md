# Status — 2026-10-18 02:49 UTC

The owner stopped the original R study and authorized the new direction. Old
launch orders, including CODEX_NEXT_DAY.md, are superseded. Do not resume them.

## Closed results

- Latest unchanged-B0 self-note probe is negative: own86/128 paths=67.19%,
  direct24/32=75%, empty19224/32=75%. Own−direct−7.81pp CI[−18.75,+2.34];
  numericLL−.238 CI[−.997,+.487]. Four paths are averaged within each question
  to estimate a single sampled-note policy, never best-of4. Same known TRAIN32
  is exploratory reuse. Equal token caps are not equal FLOPs/latency.
  All128 notes,416 fresh answers+416 finite LL,32 exact reused direct cells,
  original failures and fixed examples retained. Accuracy/LL/source/closure
  independently checked, including final public projection.
  Report: validation/gsm8k_self_note_train32_20261018.json
  SHA c13574ea9720d63442bf7d7a91ac36d3da66c515687cbd2d07c5cabec70d43ee.
  Summary: local/nvme/diagnostics/gsm8k-self-note-reader-train32-g4-v1/summary.json
  SHA047ca3837a8d5ac85752be4086c068d55cc1bfbe08cc6e210174317d75165b18.
  EBS: local/backups/diagnostics/gsm8k-self-note-train32-g4-v1/ARCHIVE.json
  SHAeed804d9b2f1316c270df04976760425951ebc369a62d64875001240b061e876,
  210 files/21,525,145B, all source/copy SHA readbacks verified.
- Previous TRAIN32 external calculation pilot remains unchanged. All3 strict
  verified-over-empty wins are format/cap repairs, not wrong-number corrections.
  Manual carnations note6736 missed an available bulk discount despite correct
  equations. Accepted mechanism annexv2 preserves this and all6 cases:
  validation/gsm8k_calculation_train32_mechanism_20261018.json
  SHA59ee01b3a0e46cbedf6522da8b02d94bb61e551baf1f6774737c1c08ec40664c.
  Original report SHA25ee9268…5c06 and frozen notes/scores remain unchanged.
  EBS annexv2 ARCHIVE SHAdd58e690…3b3; v1 retained in Git/EBS.
- Earlier matched77-document warmstart and all9 GSM8K TRAIN128 cells are
  negative, published at849e443. Both64-update terminal checkpoints are fully
  verified on EBS. Reader adaptation also failed. See README linked reports.
  Original R1234 stopped63/backed up; R5678 never launched; old C seeds finished.
  Original primary R/C comparison remains incomplete. No official TEST read.

## Runtime and checks

No live accelerator jobs. Self-note writer/answers/LL all4 shards each exit0;
all original PID/start identities, groups and sessions closed, devices clear.
Last device exit02:37:01UTC. Used cores28/32/36/40, devices7–10. Recheck topology,
processes and ports before a new allocation; no automatic relaunch.
Writer protocoldd258a79…8ec / source3cedd5f9…831d; reader protocolc594ab02…3c1 /
source9c7342ce…bc5. Native/device LP errors≤5.913e−5, all-action replay≤8.107e−6.
Latest required492 tests + toy_smoke + compileall PASS local/checks/tests.mXGFmo.
No model/runtime source has changed since that check. New data code will need
focused review and required checks before use.
Root EBS111GiB free, NVMe736GiB last measured02:43. Reserve35GiB for checkpoint
backup. No new cloud resources, paid APIs, stack upgrades or official TEST.

## Active work and exact next action

Data agent implements NEW train/calculation_boundaries.py and focused tests;
source/fixture work only until root/review accept it. No full corpus prep yet.
Source: local/data/finemath/production-500000.audit-v2.records.jsonl,
SHAbd81972fff861f497ce61479079d6904e1baef0a335dfc848cc33e4d90d069ff.
TRAIN/DEV only. Existing prefix512 + future16 chunks. Safe exact-Fraction AST,
2–6 numeric operations including multiplication/division/power; enough preceding
context; complete causal token cut after '='; reject already-visible whole
answer; select one boundary/document by fixed hash BEFORE RHS checks, no fallback.
Then verify immediate source RHS and preserve every rejection. Correct note
computes a proper subexpression from past only, not the whole answer. Freeze a
same-token-length numeric corruption before scores. Keep original source chunk,
natural continuation and numeric span. Hash-order TRAIN256/DEV64 feasibility
pools before any model score. This is explicit external arithmetic supervision.

Review agent checks source/schema/causality and fixtures when supplied. Distributed
proposed a lean new reader wrapper using unchanged setup_model/score_continuations:
raw/empty/correct/corrupt/shuffle/unrelated under prepend_plain, width1024/future16,
scalar LL derived from the same natural16 LPs when exact leading-span alignment
is proved. No implementation/release until data schema arrives; no hardware yet.
Later raw learning control must use FULL ordinary source CE on accepted chunks,
not the old future16-only control. Do not train merely because labels exist.

NEXT: inspect data agent's new source/fixture results, obtain independent review,
fix real defects, then run `source scripts/env.sh && bash scripts/run_checks.sh`.
Only after acceptance seal a new prospective data protocol and run full CPU prep.
Inspect fixed examples/quality before B0 reader qualification/scoring. No long RL.
