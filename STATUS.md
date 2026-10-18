# Status — 2026-10-18 01:18 UTC

The owner stopped the initial R study and redirected work to useful content,
reader interfaces and iteration speed. Old256-step/CODEX_NEXT_DAY launch orders
are superseded. Follow NEXT_STEPS.md; do not restart old launchers.

## Current scientific result

Writer-only behavior cloning completed64 updates from B0 on77 future-selected
TRAIN labels. Fresh G1 DEV64 notes with an unchanged B0 prepend reader remain
harmful atK16: own−empty−.649 nats CI[−1.117,−.176], own−other−.636
CI[−1.129,−.136]. All384 rows finite, raw-reference vectors exactly equal.
63/64 notes cap256,1EOS,0empty; mean252.56 tokens. Paired change versus B0
own−empty is+.451 CI[−.150,+1.128], not an established improvement. All64 notes,
controls/scores/fixedfirst4 examples and source metadata are SHA-verified on EBS.
Report: validation/warmstart_prepend_dev64_writerBC64_20261018.json.
This is not a successful warm-start or a reason to relaunch long RL.

Secondary reader crosses also fail: own−empty K16 is−2.502 for B0 notes and
−3.253 for BC64 notes with the adapted reader; both intervals exclude zero.
Adapted raw K16 falls−3.887 nats versus B0 CI[−5.235,−2.546]. All768 new
scores and fixed examples are verified on EBS; report:
validation/warmstart_reader_cross_dev64_20261018.json. No reader rescue.

Training itself passed64 finite-gradient/real-sampled-change/replica-slice checks,
not a full-weight bitwise equality or numerical-resume audit.512 exposures,
262144 prefix tokens,104606 note/close labels,zero true-future labels. Full27.53GB
step64 checkpoint copied and readback-verified to EBS00:53:41;16 remains onNVMe.
Run writer-bc-prepend-train77-64-v2 exited0 at00:48:16; all12 observed owned
process identities gone and devices2/3 empty. Full execution receipt:
validation/writer_bc_train77_64_20261018.json.
Model64SHA06dcba4e323344603434bae896492c2d1027bb88ecbc72f673f5351147f0b882.
Manifest64SHA9900a0fb99d1460912c535b1922039c235a8bf3147311c997984684e759189fb.

## Live jobs

- **Raw control RUNNING**: raw-control-prepend-train77-64-noncontiguous-v3,
  latest observed update27 at2026-10-18T01:13:35+00:00, loss0.358475.
 64updates/global8/LR1e-5/FP32/seed20261018; exact same77 IDs/order/CE math.
 Cores8-15/devices2/3, ports65501/65502. Control directory:
 local/diagnostics/positive-writer-prepend-train77-64-v1/raw-control-noncontiguous-v3.
 Launcher2314497/timeout2314503; tmuxdario-raw-control-train77-v3, train.log/STOP.
 Deadline06:58:14.563002UTC. New protocolSHA89639736dec69f9f07e9814b0339b7f3f2328d50ee057af3ecf25194227d4690.
 Sole repair NEURON_RT_CONTIGUOUS_SCRATCHPAD=0, bound as extra allocator field;
 old runtime_policy helper excludes it. Actual Python parent+all8 workers
 attested flag0/page-size unset/LRU2/compiler flags unchanged. First update
 exactly reproduces failed attempts except cost/time; update2 passes. Largest
 5s-sampled bank usage23.326/24GiB, not an instantaneous peak or full-run fit
 guarantee. First-two observer closed; private review SHA3b8727b3…c801bf8.
 Checkpoint16 all9 SHA readback verified01:11:01, spec4e7ff6f4…a70418,
 modelSHAe7ede707…9cb6fe. Preserve healthy64 and final64 EBS backup. Unequal
 writer/raw CE-label counts and actual compute remain explicit.
- **GSM8K correctness**: agent data owns16 single-core shards, B0thought256 and
 writerBC64 direct/empty/thought256,4shards each. Actual release01:04:11-13;
 deadline05:03:56UTC. Assigned ONLYcores0-7,16-23/devices0/1/4/5,ports65101-65132.
 Output rootlocal/nvme/evaluation/warmstart-gsm8k-train128-terminal64-v1;
 control rootlocal/diagnostics/warmstart-gsm8k-train128-terminal64-v1.
 Reviewed plan/review atlocal/evaluation/warmstart-gsm8k-train128-terminal64-v1.
 Same fixed TRAIN128/prompt/parser, greedy/FP32/thought256/answer256. B0direct113
 andempty107 reuse independently validated; new B0thought256 required. Rawcontrol
 three modes wait for actual64. Score only closed whole-cell128 rows; keep errors,
 truncation and all original predictions. No official final/test use.

## Next actions

1. Observe raw-v3 finite losses/updates/replicas and16/64 exports; no automatic retry.
   Complete final EBS copy with scripts/backup_checkpoint.py and35GiB reserve.
2. For completed control64, use existing train.warmstart_evaluate prepare/generate
   on fixedselection local/diagnostics/warmstart-dev64-cap256-v1/selection.json,
   fourB4/G1 shards, explicit model.pt SHA, then all-shard closure/global controls.
   Read those notes with unchanged B0 via train.warmstart_prepend_evaluate.
3. Full predeclared three-arm comparison: python -m train.warmstart_compare
   --b0 <B0prepend> --writer <writer64prepend> --control <raw64prepend>
   --output <fresh.json>. K16 primary/K1 secondary; no best-checkpoint selection.
4. Finish correctness regardless primary LL sign. Use its
   mechanism evidence and real-data/target-quality audit to choose next training.
   Current labels and recipes stay unchanged; no broader teacher dataset yet.
5. Prepare gsm8k-calculation-train32-v1: hash-selected TRAIN questions disjoint
   from all earlier dev/demos, no training or B0-failure selection. Reviewer
   prepares question-only verified partial calculations/corruptions; distributed
   implements one native4B greedy note/question; root implements fixed-B0
   prepend reader with raw/empty/content controls, forced numeric-answer LL and
   greedy correctness. Freeze all notes before opening gold. This is curated
   task feasibility, not FineMath transfer. No device release yet.

## Retained evidence and checks

- Prepend reader removes empty-format tax; original G1 still hurts. TRAIN128/G16
  rescored SAME2048 candidates: K16best16−empty+1.026, max-own−max-other+.535 and
  max-own−max-shuffled+.273; conditional TRAIN oracle, not deployed utility.
- Local4B teacher TRAIN16: prefix-only own−empty−.698, hindsight+2.858. Hindsight
  sees target; includes copying/hints and wrong math. High FineMath score does
  not certify truth. Actual finite3.1416 counterexample/source/outputs archived.
- Both raw attempts v1parallel/v2serial failed after1 with eight64MiB contiguous
  scratchpad growth failures. Serial last pre-error sample9.60GiB/bank, same
  failed NEFF bytes as successful writer. Source/CPU precision was not changed.
  Both failures+kernel/25-sample observer/closure limits have verified archives
  and separate public receipts. Runtime-only v3 repair follows official docs.
- Latest CPU suite417+toy smoke PASS (local/checks/tests.8ffa1r), independently
  reviewed paired comparator; actual device/replay/collective evidence is separate.
- Original R1234 stopped63/exit0/fullstate EBS; R5678 never launched. Both old C256
  finished; old R/C comparison incomplete. Original source/results retained.
  Official final suites PREPARED_NOT_LOCKED/unused. No paidAPI/newinstances/IAM changes.
