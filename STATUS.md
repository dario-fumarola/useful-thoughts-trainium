# Status — 2026-10-17 23:56 UTC

The owner stopped the initial R study and redirected work to the reward
interface, genuine candidate availability and iteration speed. Older256-step
orders/CODEX_NEXT_DAY.md are superseded. Follow NEXT_STEPS.md.

## Findings

- Original **B0 DEV512/G1**: raw K1−1.118 nats, empty−7.361, own−11.203;
  own−other−.284 with interval spanning zero. K16 is future length, not16 draws.
  Full report `validation/local_dev_B0_20261017.json`; all originals preserved.
- Fixed **DEV32 reader interventions**, original saved G1/cap1024 notes:
  original empty−raw−6.425, reanchor32−1.080. Prepending the note before the
  entire original prefix nearly removes empty-format cost: K1+.00913
  CI[+.00261,+.01701], K16+.03205 CI[−.0514,+.1562]. Content still does not help:
  prepend own−empty K16−1.783 CI[−3.041,−.740], own−other−1.647
  CI[−2.960,−.572]. This is a reader repair, not a trained policy or reasoning
  result. All704 new scores/parity pass; public prepend report independently
  checked320 contrasts/CIs. Verified EBS33files/1,830,327B.
- Actual released **think/end-think/chat input rows are exactly identical**
  (25 defined non-EOS added IDs; equality extends through151736). Nonzero
  vectors, tied output head. This explains identical scores for the published
  delimiter correction; no training-history/sole-cause inference.
- **Undefined output mass is not the tax** on fixed DEV8: renormalization
  changes original empty penalty by.000549 of5.678 nats. No masking fix adopted.
- **DEV32 genuine G16/cap256 oracle COMPLETE**,512 notes/4256 scores/no failures.
  Reanchor32 K1 mean16−empty−1.612; best16+.330 CI[+.051,+.684]. K16 mean16−4.627,
  best16+.395 CI[−.418,+1.289]. Max16 own−max16 other K16+.071
  CI[−.808,+1.069]; max16 shuffled+.405 CI[−.340,+1.269]. All max-control
  intervals span zero. These are future-selected exploratory candidates,
  not prefix-only deployed utility. EBS44files/14,874,011B, all512 notes and
  fixed first4-document examples retained; public oracle report.
- **TRAIN128/G16 original-writer corpus COMPLETE**,2048 notes/17024 scores.
  First128 of237 qualifying TRAIN1024 anchors (raw surprisal>=1.7); no DEV
  training. Reanchor32 K16 mean16−empty−5.501, best16−.661. A prospectively
  coded strict positive-label rule retains31 documents (23caps/8EOS): highest
  ownK16, nonempty, beating same-empty/filler/shuffle/other; raw wins are not
  required and raw deltas remain. These31 are future-selected training labels,
  not demonstrated useful reasoning. No student has trained. Verified EBS
  154files/46,373,554B includes all128 records and separate derived-label subtree.
- **Fresh DEV64 B0/G1 COMPLETE**, outside allDEV32pilot IDs,16/quartile.
  Cap256,64 notes/448 finite rows,45caps/19EOS/1empty; all4 actual B4/G1 replay
  audits pass max1.574e-5, raw16 parity to original B0 max9.06e-6.
  Reanchor32 own−empty K1−1.335 CI[−1.786,−.902], K16−4.564
  CI[−5.735,−3.437]. Own−other K16−1.920 CI[−3.066,−.813]. Public report;
  verified EBS89files/2,081,308B. This is a baseline, not a student result.
- **Padding replay profiler PASSED**: synthetic empty and real89-action note,
  both advantage signs, all310 tensors/1,720,574,976 gradients. Maximum
  relativeL2 1.826e-6/cosine>.999999999998; reader error4.292e-6. Real-note
  immediate-repeat replay forward+backward full1024~4.244s vs bucket128~.461s.
  Isolated LRU2 stages, not end-to-end training speed. Cold/setup/exports remain
  separate. Exit0, verified EBS23files/1,043,778B. Prior runtime setup failure
  preserved; only original-driver XLA registration repaired in profiler.

## Active work and next commands

- **Prepend TRAIN rescore COMPLETE**, all8 exit0/groups gone,8704 finite rows.
  SAME2048 candidates, no new sampling. K16 mean16−empty−1.07156;
  oracle best16−empty+1.02558 CI[+.84075,+1.22551]. Max16-own−max16-other
  +.53506 CI[+.3251,+.7493]; −max16-shuffled+.27346 CI[+.0881,+.4658].
  All128 docs retained. Public validation/candidate_prepend_train128_20261017.json;
  verified EBS130files/63,429,211B. Future-selected TRAIN evidence, not G1 utility.
- **New dataset positive-prepend-writer-train128-v1**,77/128 strict winners,
  nonempty and above raw/empty/same-index filler/shuffle/other K16. Highest own
  selected before filtering, no fallback.50caps/27EOS; note mean203.026tokens,
  median256. Dataset local/nvme/diagnostics/positive-prepend-writer-train128-v1,
  examples SHAee2b384dc6fc6b032745a1340603f78eda06beb08d88ae0a461c1c9e70b9ffe7,
  metadata SHAa19149a3cb63a5fbb09fe151b062a1e20ace213b72b290599607daa70052906b.
  All77 writer tensors unchanged when every future is replaced; zero future
  input/labels, width832/T258. Old31 joint-reader labels remain separate/unused.
- **writer-bc-prepend-train77-64-v2 RUNNING**, released23:52:12UTC,
  finite deadlineOct18 05:52:12. FP32/LRU2,64updates/global8/LR1e-5, one ZeRO
  reduction owner, writer note+canonical-close CE only; raw_control later uses
  the same77 ID/order/512 processed examples (6 full passes+50). Actual first
  update/hardware qualification pending; do not infer it from CPU tests.
  Protocol local/diagnostics/positive-writer-prepend-train77-64-v1/protocol.json
  SHAe5c7e9e983efc8a2f97167afc059d503ae38855577b5561a55fb9706c2879bd4.
  Control subdir warmstart-contiguous-v2: launcher2221770/timeout2221774,
  train.log and private STOP path; tmux dario-writer-bc-train77-v2. Output at
  local/nvme/diagnostics/writer-bc-prepend-train77-64-v2. Contiguous logical
  cores8-15 on adjacent devices2/3, ports65371/65372.
  Watch first finite loss/gradient/actual update/replica agreement, then16/64
  complete checkpoints. After exit and ownership closure, release raw_control
  under this exact pair protocol on the same cores. No automatic relaunch.
  V1 failed before any update at first rendezvous, exit1 at23:50:55: sparse
  core-list remapping succeeded but collective graph had no device paths
  (no_hier/no_mesh). All observed owned PIDs absent, no metrics/checkpoints.
  V1 source/output/log preserved; V2 changes launch topology only, same exact
  source/data/pair protocol. A supported parser is not a qualified collective.
- **hindsight-teacher-train16-v1 RUNNING**, released23:47:20UTC,
  finite deadlineOct18 01:47:20. Qwen3-4B pinned1cfa9a72, FP32, logical0/device0,
  ports65351/65352; tmux dario-hindsight-teacher-v1. Local control sameID,
  launcher2210950, generate.log; output local/nvme/diagnostics/sameID.
  Both actual B4 native-CPU/device next-logp checks passed before any notes:
  prefix-only max.00044632, hindsight.00035477 (criterion.002).
  Both first cached/replay audits passed: prefix-only2.2054e-5/hindsight3.1412e-5
  (criterion.001).20/32 notes saved at23:56; wait for all32/exit0 before scores.
  Protocol bacf19a47b4304d9a1852339285b59cd2b6d0199d3eb5a49a268ae6970f50961.
  Earlier CPU reference publication failed only on loading-info set serialization;
  failed source/output preserved separately, then exact inputs re-prepared and
  reference completed. Teacher utility is not measured yet.
- **warmstart-prepend-dev64-B0-v2 RUNNING**, released23:55:44, finite1h deadline
  Oct18 00:55:44. Reuses all64 closed B0 G1 notes/global controls, no generation;
  384 contexts, four16-doc shards on logical44/48/52/56, ports65401..65408.
  Output under local/nvme/diagnostics/sameID; controls under local/diagnostics/
  sameID/score-shard-0..3 (PIDs/log/exit). Protocol
  e0a862fea81f4465df8b2736367e9db98b4f0d2df85d4dcfa9c92efde52bfa0d.
  New evaluator independently reviewed; K16 primary/K1 secondary fixed before
  new scores; prior prepared-only K1 protocol retained as pre-review. Same
  students/fixedB0reader later, adapted readers secondary. On all4 exit0/groups
  gone: `python -m train.warmstart_prepend_evaluate summarize --output
  local/nvme/diagnostics/warmstart-prepend-dev64-B0-v2`.
- **Teacher scorer ready**, train/teacher_prepend_probe.py source5bf158f2,
  independently reviewed; data agent will prepare176 contexts only after teacher
  closed32/exit0. Root alone launches/signals. Review maintains EBS/public reports.
  Keep active teacher/SFT/evaluator dependencies fixed.
- Latest complete CPU checks388+toy smoke (`local/checks/tests.XEKEpb`). Actual
  reader/parity/full-gradient evidence above is separate from new SFT execution.

## Old study retained

R1234 stopped63/exit0/actual27.54GB checkpoint and verified EBS backup. R5678
never launched. Both C seeds completed256; the original R/C comparison is
incomplete. Two old DEV suites stopped143 with partial archives. Old backup/node
services intentionally retired after R63 backup. All old source/results remain;
see `validation/science_R_seed1234_stopped_20261017.json` and prior reports.
Official final tests remain on hold, PREPARED_NOT_LOCKED; no final outcomes used.

Public repo https://github.com/dario-fumarola/useful-thoughts-trainium.
Keep weights/raw corpus/examples/full topology/cache/secrets out of Git.
