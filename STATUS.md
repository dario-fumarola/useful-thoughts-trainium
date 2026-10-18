# Status — 2026-10-18 02:14 UTC

The owner stopped the original R study and authorized a new direction. Old
launch orders, including CODEX_NEXT_DAY.md, are superseded. Do not resume them.

## Closed evidence

- The matched77-document warmstart study is negative. Fresh DEV64/B0 reader
  own−empty K16: B0−1.101, rawCE64−.956, writerBC64−.649 nats. BC−raw change
  +.307 CI[−.209,+.908]. All1152 rows/330 means/100 CIs independently checked.
  Both64-update terminal states (27.53GB each) fully verified on EBS. Same512
  document occurrences/order, unequal labels/compute. Raw’s only allocator
  repair was NEURON_RT_CONTIGUOUS_SCRATCHPAD=0, with both failures preserved.
- Reader adaptation also failed; raw K16 drift−3.887 nats. All9 GSM8K TRAIN128
  direct/empty/thought cells closed: B0 113/107/63, BC 92/92/61, raw 56/112/76.
  Raw direct59 format errors (57 without hash4) prevent an isolated arithmetic
  interpretation. Frozen parser, all failures/examples/costs preserved.
  These reports are public at commit849e443.
- Calculation TRAIN32 pilot is closed: all32 local4B notes,320 answer attempts
  and320 finite numeric-LL cells. Correct/32: raw24, empty27, generated29,
  verified30, corrupted5; filler9/shuffle1/other0/vshuffle9/vother1. Verified−empty
  accuracy+3/32 CI[0,7]/32; numericLL+.945[.258,1.591] nats. Corruption shows
  vulnerability, distinct from improvement. Teacher final-answer inclusion in
  fixedfirst4 prevents an intermediate-only interpretation. Independent audits
  match every count/row and11 paired CIs exactly.
  Public report: validation/gsm8k_calculation_train32_20261018.json
  SHA25ee92684dad4b592353bb47021a7ad790733c43df7969fd5c2197ab9b805c06.
  EBS: local/backups/diagnostics/gsm8k-calculation-train32-v1/ARCHIVE.json
  SHA647c55ac91963a0a5af57034130484f2a649e534975ab953d28ae3ed8784cdbf,
  206 files/9,879,515B, all readback verified. Private sources/notes/examples kept.
  All teacher/answer/LL jobs exit0 and original identities/groups/sessions gone.

## Active work — reviewed self-note probe released for qualification

Implement/review the bounded unchanged-B0 self-note experiment before training.
Data agent owns train/calculation_self_notes.py and focused tests: same32 known
TRAIN questions; separate plain3shot partial-calculation prompt (demoIDs0/1/2),
G4/T.7/cap64; first decoded newline/EOS/cap ends generation. Retain all sampled
actions/raw bytes; reader content is declared first-line substring re-encoding.
All128 paths required. Native CPU/device next-LP ≤.002 and cached replay≤.001.

Root owns train/self_note_pilot.py/tests. 416 fresh reader cells per phase
(own/shuffle/other×128 + empty192×32) plus32 reused exact direct256 cells.
Locked4shotQA+[198]+note+[198], strict existing parser, readercap192. Average
all4 paths within each question before bootstrap: Monte Carlo estimate of a
single sampled-note policy, not an ensemble/oracle. Primary own−direct256.
Known TRAIN32 reuse is exploratory, not blinded confirmation. Equal tokencaps
are not equal FLOPs/latency. Prior gold is hash-bound to all4 original LL receipts
and parsed only after fresh answer closure. No official TEST.

Distributed owns runtime/closure/summary/archive. Both exact sources were
independently reviewed;492 tests + toy smoke PASS local/checks/tests.mXGFmo.
Root releases prepare/nativeCPUreference/qualified finite generation, then the
reader pipeline after strict128-note closure. Hardware checks are not yet run.
Runtime plan:
local/diagnostics/gsm8k-self-note-train32-g4-v1/RUNTIME_PLAN.json.
Proposed cores28/32/36/40 (devices7–10), deadline06:00UTC. Reserved valid port
blocks65031–38 writer /65041–48 answers /65051–58 LL; rebind-check at release.
Writer output: local/nvme/diagnostics/gsm8k-calculation-self-notes-train32-g4-v1.
Reader output: local/nvme/diagnostics/gsm8k-self-note-reader-train32-g4-v1.
No shared runtime/training helpers changed. Exact worker identities and actual
protocol hashes must be recorded after launch. No automatic retry.

## Next actions

1. Run prepared writer/nativeCPUreference and bounded device qualification;
   first4 CPU/device≤.002 and all-action cache/replay≤.001 required. Close all128
   notes and workers before reader preparation.
2. Preserve source hashes: writer3cedd5f9868a255b5bd5691647a8008d4ec0637928130eea97812a4258bf831d;
   reader9c7342ceeea83895cc21aadc481f0f0b56de079eec76a94c86ef930618eb1bc5.
3. Close all416 answers/groups before late gold, then all416 finite LL. Review
   fixed examples and actual content/accuracy/cost before any larger training.
4. Publish reviewed new sources and the posthoc mechanism annex. All3 earlier
   verified−empty wins are2 format repairs +1 avoided cap, no wrong parsed
   numeric answer corrected. Original report/scores unchanged. Separate EBS
   mechanism annex ARCHIVE SHA04c0112e…8509 is verified.
   Latest full492+toy_smoke PASS local/checks/tests.mXGFmo; native checks separate.

Original R1234 stopped at63/backed up; R5678 never launched; old C256 both
finished. Original primary R/C comparison incomplete. Preserve all old evidence.
