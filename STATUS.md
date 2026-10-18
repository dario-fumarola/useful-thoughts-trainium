# Status — 2026-10-18 00:28 UTC

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
- **writer-bc-prepend-train77-64-v2 RUNNING**, FP32/LRU2,64updates/global8/
  LR1e-5, zero true-future writer input/loss. Update16 passed at00:25:15;
  full16 checkpoint closed00:26:52, all9 files SHA-readback verified00:28:00. All16 metrics
  have finite loss/gradient, actual sampled changes and equal sampled parameter
  slices across8 replicas; these checks are not a full-weight equality audit.
  Step1 took994.867s, initialized-optimizer step2 took567.618s; steps3-16
  took25-27s. Cold costs remain separate from this short-loop warm timing.
  Protocol local/diagnostics/positive-writer-prepend-train77-64-v1/protocol.json
  SHAe5c7e9e983efc8a2f97167afc059d503ae38855577b5561a55fb9706c2879bd4.
  Control subdir warmstart-contiguous-v2: launcher2221770/timeout2221774,
  train.log and private STOP path; tmux dario-writer-bc-train77-v2. Output
  local/nvme/diagnostics/writer-bc-prepend-train77-64-v2. Contiguous logical
  cores8-15/devices2/3, ports65371/65372; deadline05:52:12UTC. Let healthy
  training finish64. Keep16/64 NVMe; verify EBS final64 backup. No automatic
  relaunch or DEV outcome selection at16. V1 sparse launch failed at initial
  rendezvous before any updates; all original evidence remains preserved.
- **raw-control-prepend-train77-64-v1 FAILED**, exit1 at00:19:14 after one
  published complete update. First loss2.307752/gradient27.202129, sampled
  updateL2.00102786, sampled slices agree all8,128future/0note labels. During
  initialized optimizer loading for update2, kernel logs record eight64MiB
  contiguous HBM scratchpad allocation failures across devices4/5. Shared NEFF
  hashes equal writer's successfully executed optimizer graphs. Why the region
  was unavailable is unresolved: no failure-time HBM usage was measured;
  total exhaustion, fragmentation and concurrency contribution are not proven.
  All12 observed owned PIDs absent/devices4/5 empty at00:22:58. No checkpoint;
  partial step2 is not a resumable or matched control result. Failed source,
  metric/log/exit/control preserved in verified EBS archive; public receipt
  validation/raw_control_train77_failed_attempt_20261018.json.
  **Next control attempt:** after writer64/exit0/group closure, restart from B0
  as raw-control-prepend-train77-64-v2 on the exact released writer cores8-15,
  same immutable pair protocol/source/data/order/optimizer/LRU2. New output,
  ports/private TMP/deadline; serial placement is the minimal next test. No
  blind relaunch loop or changes to the healthy writer. Control matches IDs,
  updates and optimizer; unequal CE-label counts/compute remain disclosed.
- **Fresh prepend DEV64 B0 COMPLETE**,384 finite rows/0fail, all4 exit0/owned
  groups gone. Raw64×16 reference vectors exactly match. K16 empty−raw+.02987
  CI[−.00199,+.06548]; own−empty−1.10055 CI[−1.72811,−.56653], own−other−1.02279
  CI[−1.67546,−.46486]. Same closed64 G1 notes/global controls; no sampling here.
  Independently reproduced all60 contrasts/CIs and60 absolute cells. Public
  validation/warmstart_prepend_dev64_B0_20261017.json; EBS81files/2,454,075B.
  Output local/nvme/diagnostics/warmstart-prepend-dev64-B0-v2/summary.json.
  K16 primary/K1 secondary fixed before scores; pre-review K1 preparation retained.
- **Local teacher COMPLETE**,32 notes/exit0 at23:57:08, allowned groups gone.
  Qwen3-4B FP32/native CPU forward max.000447; both replay max3.142e-5 pass.
  Prefix-only16EOS, noempty; hindsight15EOS/1cap, noempty. Public generation
  report validation/hindsight_teacher_generation_20261017.json; verified EBS
  37files/6,082,447B includes allprompts/notes/CPUrefs/fixedfirst4 examples.
  Earlier CPU reference publication failure (set serialization) remains separately
  retained; no numerical failure. B0 remains unchanged Qwen3-1.7B-Base.
- **Teacher prepend reader COMPLETE**,176 finite rows/0fail/exit0, raw parity
  7.868e-6. Prefix-only K16 own−empty−.69811 CI[−1.11234,−.33965]. Hindsight
  own−empty+2.85784 CI[+.72989,+5.30923], own−other+3.22091, own−shuffled+2.80523
  with positive intervals. Hindsight saw the continuation: target-derived label
  information, not deployed useful thought.7/16 copy exact4grams,3/16 exact8grams;
  gains also occur without those matches, which does not exclude paraphrase/hints.
  Root/auditors inspected actual notes: some high gains reflect arbitrary future
  details or erroneous math. Do not equate continuation LL with correctness.
  Public validation/teacher_prepend_train16_20261017.json; EBS61files/7,735,157B.
  Output local/nvme/diagnostics/teacher-prepend-train16-v1/summary.json.
- **Quality audit saved**, validation/teacher_anchor_quality_20261018.json,
  verified EBS43files/1,357,152B. Actual cached author assertion falsely calls
  finite3.1416 irrational despite FineMath int_score5; correct teacher note
  K16own−empty−.47978 versus false hindsight+1.56470. At least7/16 fixed panel
  targets are manually classified scaffolding (posthoc/nonblinded, no population
  frequency claim). Prefix-only calculation-boundary curriculum remains an
  unimplemented prospective idea. No current labels/gates changed.
- **Cold compile audit**, control warmstart-contiguous-v2/hlo-review/conclusion.json
  SHA3a15fc52a1f3d141088e17452dcfb4880f3199e4c6a4ca6f751805d1e56344ab.
  Eight optimizer HLOs differ only by rank-specific310 padding configurations;
  raw_control uses the same exact8 cache keys. CE IDs/labels/positions/masks are
  HLO parameters, no array constants. No first-batch data-value specialization
  observed. Initial and initialized-state optimizer HLOs contain rank-specific
  pad offsets; raw uses the same eight keys at each state. Both cold stages
  and later25-27s writer updates are now measured.
- **Next**: finish writer64 and its verified backup, run serial fresh-B0 control64,
  preserve full16/64 checkpoints, then evaluate both64 models with fresh G1 and
  fixed B0 prepend reader; compare paired student/B0/raw-control utilities.
  No best-checkpoint selection. Data audits prefix-only anchor quality locally;
  distributed watches live compilation; review maintains EBS/public receipts.
  Keep active teacher/SFT/evaluator dependencies fixed.
- Latest complete CPU checks417+toy smoke (`local/checks/tests.8ffa1r`),
  including29 paired-analysis cases and independent review. New command
  `python -m train.warmstart_compare --b0 <B0prepend> --writer <writer64prepend>
  --control <raw64prepend> --output <fresh.json>` requires complete planned64
  checkpoints and identical fixed-reader DEV64/policy; computes paired utility
  changes before any student outcomes are available. Actual
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
