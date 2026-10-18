# Status — 2026-10-18 03:35 UTC

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
Latest required739 tests + toy_smoke + compileall PASS local/checks/tests.WLpLnu
on final curatorv2, scalar-endpoint and future packer sources. Focused curator122/probe100/packer25 PASS;
independent exact-source review accepts both. Earlier599/710 runs predate final
additions/fixes; they are not current qualification. Shared model/runtime helpers
unchanged. CPU checks do not certify accelerator execution.
Root EBS111GiB free, NVMe736GiB last measured02:43. Reserve35GiB for checkpoint
backup. No new cloud resources, paid APIs, stack upgrades or official TEST.

## Active work and exact next action

CPU v1 preparation CLOSED exit0 at03:14:42UTC; original worker gone.
Output local/nvme/data/finemath-arithmetic-boundaries-v1, complete347895 audits.
Accepted1304=TRAIN1220/DEV84; fixed256/64 pools and964/20 reserves.
ManifestSHA95ad48f4d3a889434bc9218ae259ae6552770ae7805b9f5a0303a44b7cb1f1af.
AcceptedSHA195f113f…be95; auditSHAa2fd9bcd…71c0f. Data/review full arithmetic,
source and alignment audits PASS, but this does NOT clear the issue below.

BLOCKED DATA v1, no model qualification: root's fixedfirst4 TRAIN panel found
whole132600 already in selected#1 prefix as Rs.132600. The copy filter reads
.132600 as0.1326 and misses the currency-abbreviation form. Data/review quantify
all1304 and inspect grouped-number ambiguities. Preserve completev1 +exact
source177363/tests d7b2 before fixes. Newv2 output/source identity required; do
NOT overwritev1, prepare readers or load models on known-invalid noncopy data.
Curator CPU worker is closed, so no active-source mutation risk. No automatic
rerun: source/fixtures must be reviewed first. Controls/panels in
local/diagnostics/finemath-arithmetic-boundaries-v1.

ACTIVE v2 CPU preparation, tmux dario-arithmetic-data-v2, launched03:24:45UTC.
Launcher2448590/start12857626; timeout/PGID2448594;
worker2448595/start12857629. Deadline05:00UTC plus15s external grace.
Output local/nvme/data/finemath-arithmetic-boundaries-v2;
control/log local/diagnostics/finemath-arithmetic-boundaries-v2.
Actual USE_TORCH_XLA=0, OMP/MKL1, full source SHA gate passed. At03:33:38,
203427 audit rows/639 accepted; no errors. Data agent owns monitoring/closure.
No source mutation, automatic retry or model/hardware release.

Reviewed curatorv2 source0d32b6eb34ba24b1ff94814a66fce02b8251639e29652b8fe0bea6889c1f4a55;
tests5c365d856fbc5377baaa627df418d0273bda124023861adfcdf5d43c775f86d4.
V1 exact source177363/tests d7b2 preserved in v1 control/frozen-source, plusGit.
All1304 v1 source/arithmetic alignments PASS but three confirmed copy misses:
Rs.132600, Rs.38000, C.25.2, all TRAIN; first is one of selected256. Ambiguous
comma lists are NOT counted as confirmed leaks. V2 rejects contextual decimal
ambiguity, validates maximal Western/Indian groups without splitting, preserves
standalone .5 decimals, exact original salt/search starts and source-RHS rule.
Data/review accepted source/fixtures before this fresh CPU run. V1 is blocked,
not overwritten or retroactively filtered into a qualified dataset.

Source records local/data/finemath/production-500000.audit-v2.records.jsonl,
SHAbd81972fff861f497ce61479079d6904e1baef0a335dfc848cc33e4d90d069ff.
TRAIN/DEV only. Hash start128..192 then first eligible causal '=' cut; return
immediately, no suffix inspection to choose task. Chosen-RHS quality failure
never chooses another task. Notes are externally computed proper subexpressions,
not the whole result. Original528 tokens, numeric span/whitespace mask, matched
corruption and all rejections retained. Exact source mismatch can reflect
rounding; unsupported math is not a source-error claim. Actualv2 counts unknown.

Reviewed reader sourcef8a54017523d76da49032ffc1edf97bc530a19bf66b67668973c28f659f1a95f;
testsd13e23f6591a1af755948d0ac5a0c5291488c17ee8b901502e3d4c78fd0c5eb7.
No protocol/model prepared yet. Full1920 LL cells: TRAIN256/DEV64×six modes
(raw/empty/correct/corrupt/shuffle/unrelated), unchanged prepend_plain1024/16.
One natural16 pass yields nonwhitespace numeric, full scalar and natural16 LL.
Separate384 DEV greedy initial-scalar completions, cap32/EOS, exact Fraction
value comparison, all unsupported/incomplete/errors in denominator. Complete
initial value before a later cap may count; preserve every actual stop/action.
Parser is a declared narrow ASCII decimal/fraction grammar, not general math.
Native CPU refs cover TRAIN raw/correct and DEV raw/correct; actual noncached
and cached device full-head LP gates≤.002 before any rows. Generation masks
undefined tokenizer IDs; LL/native gates retain full released head support.
Source-form LL differs from value correctness (e.g.8.3/2 and4.15 are equal).

Obsolete runtime proposal preserved, NO model load/launch; rebind before use:
local/diagnostics/finemath-calculation-boundary-probe-v1/runtime.prepared.json
SHA9025c356…0e09d. Proposed cores28/32/36/40, ports65061–68,06:00UTC deadline;
output local/nvme/diagnostics/finemath-calculation-boundary-probe-v1.
Actual devices/ports were free during preparation; recheck at release.

NEXT: monitor v2 CPU prep; require complete manifest/counts/source hashes and
inspect fixed examples plus independent maximal numeric-span copy audit before
reader prepare/native CPU reference/hardware release. Rebind runtime proposal
to correctedv2 data and final source. No fullsource/sampler/checkpoint changes.

Future packer only: new train/calculation_warmstart_data.py source1595c1ca…b9ed
and tests1b851f84…3c91 independently reviewed;25 focused/739 required PASS.
No current driver edits, dataset prep or training. Fixed
instruction/two partial-calculation format demos; exact past; note+native newline
labels only. Separate source-only pack supervises all527 original targets.
Proposed next pilot (NOT RELEASED): BC+fullsourceCE vs fullsourceCE, same512TRAIN
2epochs/global8×128updates, extra note labels/compute disclosed, freshG1 DEV
fixedB0 reader plus downstream. Decide after actual reader feasibility. No longRL.
