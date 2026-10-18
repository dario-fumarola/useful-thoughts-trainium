# Status — 2026-10-18 03:02 UTC

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
Latest required599 tests + toy_smoke + compileall PASS local/checks/tests.sBcbKy
on the final source bytes. Focused curator87/probe20 also PASS; independent
source review accepts both. Initial599 pass0WlU0W preceded a final probe guard
edit, so the required suite was repeated on the settled bytes. No model/runtime
helper changed. No accelerator qualification is claimed by these CPU checks.
Root EBS111GiB free, NVMe736GiB last measured02:43. Reserve35GiB for checkpoint
backup. No new cloud resources, paid APIs, stack upgrades or official TEST.

## Active work and exact next action

CPU corpus preparation is ACTIVE once, tmux dario-arithmetic-data-v1.
Launcher2446357/start12706533; timeout/PGID2446360;
worker2446361/start12706534. Deadline2026-10-18T04:00:00Z plus15s external grace.
Actual USE_TORCH_XLA=0, OMP/MKL1. Output:
local/nvme/data/finemath-arithmetic-boundaries-v1.
Controls/log: local/diagnostics/finemath-arithmetic-boundaries-v1.
Source full SHA gate passed before output creation. At03:00:38, elapsed65s,
24,289 audit lines/99 accepted; preparation counts only, no model outcomes.
Data agent owns monitoring/closure/count audit/fixed first4 selected per split.
Do not edit frozen sources or retry automatically.

Accepted source:
train/calculation_boundaries.py
SHA177363c6bb2e56226ac22496c2b558eb0f0d0c56357ebf602a1590a79e6ee6af.
Source records: local/data/finemath/production-500000.audit-v2.records.jsonl,
SHAbd81972fff861f497ce61479079d6904e1baef0a335dfc848cc33e4d90d069ff.
TRAIN/DEV only. The final selector hashes a search start128..192 and returns at
the FIRST eligible causal '=' cut; no later text is decoded to choose a task.
This replaces the prototype full-window candidate minhash before any full prep
or model outcome. Prefix-only safe Fraction arithmetic supplies a simplifying
proper intermediate; final answer is not in the note. Exact source RHS quality
checks happen afterward, with no alternate task if the chosen label fails.
Original528-token chunks, source spelling, numeric target and whitespace mask,
matched corruption and every rejection are retained. Exact decimals are rendered
without rounding; source mismatch can reflect source rounding, not factual error.
Hash-order accepted TRAIN256/DEV64 before any model score. Actual counts unknown.

Reviewed reader source train/calculation_boundary_probe.py
SHA7913b8512a6917416c18eff4c0e8b62902f78617d04bf9f03d5293daf8b6f36f.
Raw/empty/correct/corrupt/shuffle/unrelated under unchanged prepend_plain,
width1024/future16. One natural16 score yields nonwhitespace numeric LL primary,
full scalar joint LL and natural16 secondary; absolute arms/split CIs retained.
Complete1920 cells required, no shrink. Prospective native CPU refs cover first
TRAIN raw/second TRAIN correct/first DEV raw/second DEV correct; each device
must match full-head next-LP within.002 before scoring. This is external exact
arithmetic feasibility, not self-generated utility or unrestricted FineMath.

Distributed saved runtime proposal only, NO model load/launch:
local/diagnostics/finemath-calculation-boundary-probe-v1/runtime.prepared.json
SHA9025c356…0e09d. Proposed cores28/32/36/40, ports65061–68,06:00UTC deadline;
output local/nvme/diagnostics/finemath-calculation-boundary-probe-v1.
Actual devices/ports were free during preparation; recheck at release.

NEXT: monitor CPU prep. Require complete manifest/counts/source hashes and
inspect fixed examples before releasing reader prepare/native CPU qualification
and hardware scoring. Do not train merely because labels exist. If useful,
short fresh-B0 noteBC+full-sourceCE versus full-sourceCE is the proposed next
learning experiment; same source chunks/order, extra note labels/compute disclosed,
freshG1 DEV with fixedB0 reader and downstream checks regardless of LL sign.
No training protocol has been released. No long RL.
