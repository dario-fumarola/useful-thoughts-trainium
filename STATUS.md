# Status — 2026-10-17 23:39 UTC

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

- **candidate-prepend-train128-g16-v1 RUNNING**, released23:36:25UTC,
  deadlineOct18 00:36:25. Rescores the SAME2048 saved TRAIN notes/global controls,
  no sampling; prepend_plain=note+[271]+full512prefix,8704 total contexts.
  Eight shards on logical4/12/16/20/24/28/36/40 (devices1/3/4/5/6/7/9/10),
  ports65501..65516. tmux dario-prepend-score-00..07. Standard NVMe output and
  local control directories; PID/log/exit under control/score-shard-XX.
  Shard0 launcher2202681/timeout2202690; shard7 launcher2202733/timeout2202741.
  Protocol e19600403a71164f4980d7d4908e39672f32778b1b8d27df28cc6f939f7b8b19.
  Keep source/dependencies fixed. Observe counts/parity, no partial oracle reads.
  After all8 exit0/groups gone: `python -m train.candidate_prepend summarize
  --output local/nvme/diagnostics/candidate-prepend-train128-g16-v1`.
- **Local teacher preparation, CPU only**: Qwen/Qwen3-4B snapshot revision
  1cfa9a7208912126459214e8b04321603b3df60c downloaded and byte-hashed,
  8,060,925,157B/13files, local/nvme/assets/qwen3-4b-teacher. Separate teacher,
  B0 remains Qwen3-1.7B-Base. No paid service/stack change. Distributed agent
  prepares train/hindsight_teacher.py and CPU next-logp references for first16
  frozen TRAIN docs, prefix-only vs hindsight (future16 supplied),32 notes
  planned. No teacher hardware/generation released. Require actual CPU/XLA
  next-logp parity and cached/replay before generation. Save precise prompts,
  token support/stop/seed/source lineage. No future in deployed student input.
- **Short SFT infrastructure PREPARED, NOT LAUNCHED**: masked full-support T1 CE,
  explicit causal masks, canonical two-token close and separate future16 mean;
  eight-rank ZeRO/checkpoints, proposed64steps/LR1e-5. Raw-control objective is
  future16-only on the identical selected TRAIN/order, distinct from old C.
  Reanchor32 joint-SFT recipe is not selected for launch while reader/candidate
  measurements finish. Actual teacher/data/interface choice remains open.
- Latest complete CPU checks317+toy smoke (`local/checks/tests.2NxtGc`), plus
  actual hardware reader/parity/full-gradient evidence above. CPU tests do not
  qualify new teacher or SFT hardware. Root owns launches/signals; data monitors
  prepend, distributed prepares teacher, review verifies public reports/archives.

## Old study retained

R1234 stopped63/exit0/actual27.54GB checkpoint and verified EBS backup. R5678
never launched. Both C seeds completed256; the original R/C comparison is
incomplete. Two old DEV suites stopped143 with partial archives. Old backup/node
services intentionally retired after R63 backup. All old source/results remain;
see `validation/science_R_seed1234_stopped_20261017.json` and prior reports.
Official final tests remain on hold, PREPARED_NOT_LOCKED; no final outcomes used.

Public repo https://github.com/dario-fumarola/useful-thoughts-trainium.
Keep weights/raw corpus/examples/full topology/cache/secrets out of Git.
