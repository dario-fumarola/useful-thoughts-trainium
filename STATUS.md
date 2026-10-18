# Status — 2026-10-18 01:23 UTC

The owner stopped the original R study. Its old launch orders, including
CODEX_NEXT_DAY.md, are superseded. Follow NEXT_STEPS.md; preserve old evidence.

## Result so far

Writer-only cloning on 77 future-selected TRAIN notes completed 64 updates.
Fresh notes on the fixed DEV64 remain harmful with the unchanged B0 reader:
K16 own minus empty −0.649 nats (95% CI −1.117 to −0.176), and own minus unrelated
−0.636 (−1.129 to −0.136). Of 64 notes, 63 hit cap 256. The paired change versus
B0 is +0.451 (−0.150 to +1.128), which does not establish improvement.

Adapting the reader also fails: own minus empty is −2.502 for B0 notes and
−3.253 for cloned notes. Adapted raw K16 drops 3.887 nats. Reports and verified
EBS archives retain every note, control, score and fixed example:

- validation/warmstart_prepend_dev64_writerBC64_20261018.json
- validation/warmstart_reader_cross_dev64_20261018.json
- validation/writer_bc_train77_64_20261018.json

The final writer checkpoint (27.53 GB, all 9 files) is verified on EBS under
local/backups/diagnostics/writer-bc-prepend-train77-64-v2/step-0000064.
Model SHA: 06dcba4e323344603434bae896492c2d1027bb88ecbc72f673f5351147f0b882.
Training checks cover finite gradients, real sampled changes and replica slices;
they do not prove full-weight bitwise equality or numerical resume parity.

## Running work

1. **Matched raw control:** raw-control-prepend-train77-64-noncontiguous-v3.
   Latest observed update 45 at 2026-10-18T01:22:37+00:00, loss 0.067124. Same B0, 77 IDs/order, 64 updates,
   global batch 8, seed 20261018, LR 1e-5, FP32. The only repair after two
   archived allocator failures is NEURON_RT_CONTIGUOUS_SCRATCHPAD=0.
   New protocol SHA 89639736dec69f9f07e9814b0339b7f3f2328d50ee057af3ecf25194227d4690;
   spec 4e7ff6f4a8bd0b75246229649f8617c095dcd572d4cb720b6c0c6ad7d3a70418.
   Actual parent and all 8 workers attest this allocator flag; old runtime_policy
   does not capture it. Checkpoint 16 passed all-file SHA readback at 01:11:01.
   Cores 8–15 / devices 2–3; tmux dario-raw-control-train77-v3.
   Launcher 2314497, timeout 2314503; deadline 06:58:14 UTC.
   Output: local/nvme/diagnostics/raw-control-prepend-train77-64-noncontiguous-v3.
   Control: local/diagnostics/positive-writer-prepend-train77-64-v1/raw-control-noncontiguous-v3.
   Keep healthy work running. After COMPLETE and exit/closure, verify and copy
   final 64 using scripts/backup_checkpoint.py with a 35 GiB reserve.
   Writer and control have unequal CE label counts and measured compute.
2. **GSM8K TRAIN128 correctness:** agent data owns B0 thought-cap256 and cloned
   writer direct / empty / thought-cap256. Direct, empty and B0 thought are
   closed; writer thought is progressing on device 5. B0 direct113/128 and
   empty107/128 are independently validated reused references. Fixed prompts,
   strict parser, greedy FP32, answer cap256; all failures stay in denominator.
   Output: local/nvme/evaluation/warmstart-gsm8k-train128-terminal64-v1.
   Control: local/diagnostics/warmstart-gsm8k-train128-terminal64-v1.
   First four cells deadline05:03:56 UTC. Raw three modes wait for actual64.
3. **New calculation pilot:** 32 hash-selected TRAIN questions disjoint from all
   139 earlier dev/demo IDs and normalized questions; no model-outcome selection.
   Questions and 32 verified partial notes / length-matched one-number corruptions
   are frozen in local/diagnostics/gsm8k-calculation-train32-v1. Root independently
   read all triples and checked all 61 intermediate equations, without gold.
   Agent distributed owns native Qwen3-4B CPU reference then four greedy-note
   shards on cores28/32/36/40 only, ports65601–65608, deadline03:30 UTC.
   Output: local/nvme/diagnostics/gsm8k-calculation-teacher-train32-v1.
   Control: local/diagnostics/gsm8k-calculation-teacher-train32-v1.
   A native tokenizer container mismatch was fixed before any prepared output or
   device allocation; exact token IDs matched. New source is frozen after 445 tests.
   All notes must close before reader prepare.

## Exact next steps

- Observe raw control, then back up final64 and run fresh G1 DEV64 with existing
  train.warmstart_evaluate / train.warmstart_prepend_evaluate. Use original fixed
  selection local/diagnostics/warmstart-dev64-cap256-v1/selection.json, explicit
  terminal model SHA and unchanged B0 reader. No checkpoint selection.
- Run train.warmstart_compare on closed B0 / writer64 / raw64 prepend outputs.
- After teacher closure, prepare train.calculation_pilot with its teacher bundle,
  verified_notes.jsonl and the unchanged held-dev128 prompt.json. Raw is complete
  four-shot QA; all note conditions prepend note + [271] + that identical QA.
  Run all four answer shards (10 conditions ×32 questions, greedy answer cap256)
  before project-gold. Then score forced canonical numbers after '\n#### ',
  independently of free-running accuracy. Retain all errors and controls.
- No further training until useful calculation is demonstrated. This is curated
  QA feasibility with external assistance, not FineMath transfer or self-discovery.

Latest CPU suite: 445 tests + toy smoke PASS, local/checks/tests.Cmh8UX.
Native CPU/device parity and actual cached replay still need new hardware evidence.
Original R1234 stopped at63 and was backed up; R5678 never launched. Old C seeds
finished256; original R/C primary is incomplete. Official final tests remain unused.
