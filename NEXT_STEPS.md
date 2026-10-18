# Current queue — matched useful-computation study

1. The frozen128-model DEV question is **complete and independently audited**:
   `validation/partial_state_v6_DEV512_frozen128_20261018.json`. All18432answers,
   2048notes, numerical vectors, paired intervals and complete EBS archives pass.
   Student answers285/238/140/151 versus compact CoT287/235/118/48 of512;
   interpolation/structure tie, range+4.30pp, composition+20.12pp. Same-reader
   raw/empty/unrelated contrasts are positive on all four panels. All notes stop
   naturally. This is one-seed synthetic evidence; split used two models and
   twice the training updates. No matched-cost or real-text pretraining claim.
2. Source57d3220 is integrated and immutable in
   `local/worktrees/partial-state-science-v7`; full checks1601passed.
   The design was fixed before new learning:
   `validation/partial_state_v7_matched_design_20261018.json`.
   Each seed compares answer-only640, compact-CoT640, full-trace-CoT640, and
   reader320+writer320 from fresh B0 on the same5120expressions. Both CoT baselines
   stay. Seeds2026101806/07; constant1e-5, BF16 storage/FP32 masters, world4×micro8.
   Reader gets one teacher and one plain view per expression over two passes.
3. First-seed reader320 and writer320 are complete, independently audited,
   full terminal protected on EBS, and frozen. Actual reader64 fresh restore and
   ordered65–320 pass. No new DEV answers inspected. Prepare full fixed-panel
   generation on physical8(writer)/9(reader) while training stays on physical3.
   Preparation log: `local/analysis/partial-state-v7-seed1806-evaluation-preparation-20261018.log`.
   Direct640 is active in tmux `dario-ps7-direct640`, timeout PID144944/start19762841,
   controls `local/diagnostics/partial-state-v7-direct-TRAIN5120-seed2026101806-v1`;
   output same basename under `local/nvme/runs`. Deadline Oct19 02:30UTC.
   Next finish its full state audit, then compact/full CoT640, then seed2.
   Baseline controllers are prepared, unreleased; outputs use `local/nvme-study/runs`.
   Unique ports/private TMPDIR, one training arm at a time, memory/disk floors,
   no automatic retry. All17source hashes must match immutable source57d3220.
   Fixed320/640 protocols live in
   `local/diagnostics/partial-state-v7-unreleased-protocols-before-DEV-review`.
4. Terminal320/640 is primary. Save64/160/terminal (plus320/480 for640); no best
   checkpoint selection. Evaluate all four512-item panels separately, retaining
   every error, cap and unsupported answer. New inference budget96: writer64+
   reader32 or single-model96. Measure actual tokens/chip-time/latency, supervised
   labels and two-model capacity. Compare own−raw/empty/unrelated and split versus
   answer-only and both CoT baselines; report both seeds. Existing DEV is explicitly
   developmental after the precursor inspection. Official TEST remains unopened.

The north star and necessary components are in `docs/RESEARCH_ROADMAP.md`.
Reader use and student-note benefit now have fresh evidence. Matched-cost value
is the next unresolved link. Both writer reliability and the reader's remaining
calculation still fail; the descriptive post-hoc decomposition is saved separately.
Prefer the simpler solver if it matches. Broad general SFT/RL stays held while
this question is resolved, followed by a source-matched real-document study.

Qwen3-8B-Base metadata/static sizing is in
`validation/qwen3_8b_capacity_assessment_20261018.json`:15.26GiB BF16 weights,
30.51GiB weights+gradients versus24GiB/logical core. Full replicated SFT cannot
reuse this layout unchanged. Adapters/inference/sharding remain unqualified;
no8B weights or run launched. The installed gradient hook is not gradient sharding.
Continue1.7B for the matched-cost test because useful fresh computation already exists.
Papers and objective corrections are in `docs/REFERENCE_NOTES.md`.

All prior jobs are closed. Frozen v6 sources and all outcomes stay intact;
old R1234 remains stopped63. The complete matched developmental64 result is
negative overall and original terminal320 remains interrupted. Short world4
save3/restore/update6 is verified; cache256 is faster on that topology, without
proving the originalworld32 OOM fixed or qualifying the old RL loop.

Keep35GiB EBS reserve; EBS is same-instance storage. Full fitting states remain
on NVMe, frozen models/output/numerical evidence on read-back verified EBS.
Three inactive EBS qualification duplicates were retired after fresh complete
readbacks of both copies; original full NVMe states and all EBS metadata remain.
Receipts: `local/analysis/retire-{one,second,third}-inactive-EBS-duplicate-20261018.json`.
Four older NVMe alias pairs already shared inodes: fresh readback reclaimed0bytes,
not102GiB. The bounded lossless archive was fully restored to original .pt paths
(all41payloads rehashed). Three previously unformatted/unmounted local instance
SSDs were identified; one is now initialized/mounted at `local/nvme-study`, adding
1.7TiB. Real write/readback and no-signature/no-device-user preflight receipt:
`local/analysis/extra-local-store-initialization-20261018.json`. No cloud changes.
A and old science stay on the original disk; future outputs use the extra store.
Retain full terminals and all results; EBS same-instance backups still require
35GiB reserve and explicit full-state/model retention. No destructive disk moves.
Root owns work; subagents unavailable due account usage limits.
