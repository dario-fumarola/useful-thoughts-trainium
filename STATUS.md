# Status — 2026-10-18 01:52 UTC

The owner stopped the original R study and authorized a new direction. Old
launch orders, including CODEX_NEXT_DAY.md, are superseded. Do not resume them.

## Completed comparison: negative

All three fresh G1 DEV64 cells are closed, with an unchanged B0 prepend reader,
all 1152 finite score rows, exact raw-vector parity, saved controls and examples.

| Writer | K16 own minus empty | Own minus unrelated | Cap256 |
|---|---:|---:|---:|
| B0 | −1.101 | −1.023 | 45/64 |
| Raw CE64 | −0.956 | −0.939 | 50/64 |
| Note BC64 | −0.649 | −0.636 | 63/64 |

BC minus raw in own-minus-empty is +0.307 nats, CI [−0.209,+0.908]; no clear
improvement, and both policies still hurt absolute prediction. Independent
arithmetic audit matches all 330 mean cells and 100 paired CIs exactly.
Public result: validation/warmstart_dev64_three_arm_20261018.json.

Both training arms completed 64 updates and exited 0. Same 512 occurrences,
order and optimizer; 77 TRAIN documents; unequal labels (BC 104606, raw 8192)
and actual compute. All 9 files of both 27.53GB checkpoints are verified on EBS:
local/backups/diagnostics/{writer-bc-prepend-train77-64-v2,raw-control-prepend-train77-64-noncontiguous-v3}/step-0000064.
Recovery step16 checkpoints remain verified on NVMe only. Raw allocator repair was
NEURON_RT_CONTIGUOUS_SCRATCHPAD=0 alone; actual environment is independently
attested because old runtime_policy omits this key. Reports preserve both failed
attempts and the successful condition. Training integrity is not full-weight
bitwise equality or numerical-resume qualification.

Adapted-reader crosses also fail (raw K16 drift −3.887 nats). Full GSM8K
TRAIN128 direct/empty/thought256 results: B0 113/107/63, BC 92/92/61,
raw 56/112/76. Every thought arm loses to its same-model empty wrapper.
Raw direct has 59 format errors (57 lack the hash4 marker), so do not interpret
its strict score as isolated arithmetic loss or relax the parser after outcomes.
Full report: validation/warmstart_gsm8k_train128_terminal64_20261018.json.
All nine cells and processes are closed; verified EBS archive has 353 files.
Official tests remain unused.

## Active work

1. GSM128 and raw DEV64 pipelines are complete; agents data/review own no live
   hardware jobs. Their closed reports/archives need the final public commit.
2. **Calculation pilot:** agent distributed owns reader answers then likelihood.
   32 new hash-selected TRAIN questions are disjoint from 139 earlier dev/demo IDs
   and normalized questions. Root verified all 61 equations in manual partial
   notes, each with a same-length one-number corruption. No note selection.
   Question/verified inputs: local/diagnostics/gsm8k-calculation-train32-v1.
   Actual local 4B teacher32 is closed/exit0/cleanup: all 32 EOS, zero caps/empty,
   2244 content tokens. Native CPU/device and cached replay checks passed.
   All first4 teacher notes include final answers despite the omission instruction;
   retain them, report this failure, and do not confuse teacher answer copying
   with intermediate reasoning. Verified manual notes omit final requested values.
   Reader launched01:45:56 on cores28/32/36/40 (devices7–10), ports65011–65018.
   Workers2407546/2407549/2407554/2407556; hard deadline04:30UTC.
   Output: local/nvme/diagnostics/gsm8k-calculation-reader-train32-v1.
   Control: local/diagnostics/gsm8k-calculation-reader-train32-v1.
   Protocol SHA2a85462169f4904470bf97398602bff3c31fe00221a6cbfbe047fbb18c5b5ca1.
   Ten conditions,320 answer attempts, cap256, FP32 greedy. Raw is full four-shot QA;
   other modes append [198]+note+[198] after Answer:; empty keeps both separators.
   This prospective interface was independently reviewed before reader outcomes.
   All four answer shards must close/exit/clean up BEFORE project-gold. Then four
   LL shards on same cores, ports65021–65028, same deadline; all 320 finite before
   summarize. Free-running accuracy and forced numeric LL after '\n#### ' are
   separate metrics. No training or official TEST. Teacher protocolff1dc323…98bd7.

## Next actions

- Let the finite calculation pipeline finish. Do not rush healthy compilations, retry failed
  jobs automatically, alter sources, select checkpoints, or drop failed answers.
- Publish the completed reports and collect the calculation-pilot evidence.
  Read fixed examples and check actual semantic benefit before more training.
- New QA feasibility with external calculations is not FineMath transfer or
  self-discovery. Teacher-generated final answers are a separate assistance cost.
- Latest full CPU suite: 445 tests + toy smoke PASS, local/checks/tests.sKjRbj.
  Actual device evidence is separate. All shared runtime/training sources unchanged.

Original R1234 stopped at63 and was backed up; R5678 never launched. Both
old C256 runs finished; original primary R/C comparison is incomplete. All old evidence retained.
