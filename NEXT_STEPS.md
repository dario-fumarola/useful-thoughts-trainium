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
3. All training and all eleven evaluation controllers close. Independent audit
   is ACTIVE in `dario-ps7-whole-question-audit`:
   `local/diagnostics/partial-state-v7-whole-question-close-and-audit-v1/audit.log`.
   At 01:45 UTC, twelve of twenty public gzip payloads are written (B0 and first
   seed complete); second seed is being verified. Do not read partial results,
   duplicate the command or change any source. Final report:
   `validation/partial_state_v7_matched5120_complete_20261018.json`.
   Wait for controller completion and final report, then interpret both seeds,
   all four panels, both CoT controls, unchanged B0 and actual costs. No TEST/RL/8B.
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
Four inactive EBS qualification duplicates were retired after fresh complete
readbacks of both copies; original full NVMe states and all EBS metadata remain.
Receipts: `local/analysis/retire-{one,second,third,fourth}-inactive-EBS-duplicate-20261018.json`.
Four older NVMe alias pairs already shared inodes: fresh readback reclaimed0bytes,
not102GiB. The bounded lossless archive was fully restored to original .pt paths
(all41payloads rehashed). Three previously unformatted/unmounted local instance
SSDs were identified; one is now initialized/mounted at `local/nvme-study`, adding
1.7TiB. Real write/readback and no-signature/no-device-user preflight receipt:
`local/analysis/extra-local-store-initialization-20261018.json`. No cloud changes.
A and old science stay on the original disk; future outputs use the extra store.
Retain every original full checkpoint on local SSDs and all scientific outputs.
EBS keeps every terminal model and the latest full resumable state. Retire only
older EBS optimizer duplicates after fresh both-side full hashes and verification
of latest full recovery; preserve all metadata/receipts. Current index:
`validation/partial_state_v7_checkpoint_retention_20261018.json`. EBS is same-instance;
keep35GiB floors. Do not change frozen checkpoint paths or live source.
Root owns work; subagents unavailable due account usage limits.

Completed-run memory inspection is read-only: within-update bins are flat,
then later full-save boundaries retain about20GiB across four workers. See
`validation/partial_state_v7_checkpoint_host_memory_observation_20261019.json`.
Cause remains unresolved; separate inspection/transfer/serialization/allocator
factors before a much longer loop. Current bounded640jobs stay below measured
node limits. Candidate official XLA serialization needs actual byte/recovery/
memory qualification; no fix has been applied to frozen science source.

Export-memory question closed. Tensor followup and actual SFT show bounded cleanup
memory with exact matched saved/recovered state. Original uninterrupted-parity
gate fails by a 0.000014 loss difference after resume; retain it. Separate control
restart without cleanup reproduces the same drift, and both resumed model,
optimizer, RNG and loss/norm trajectories are byte-identical. Do not demand FP32
replay to hide an existing numerical restart limit. Candidate fb72065 remains
isolated until the frozen math audit closes; no scientific source changes.
Public reports: checkpoint_export_SFT_recovery_v8_20261019.json and
checkpoint_export_restart_attribution_v8_20261019.json; verification source and
all 28 update metrics are public, exact originals and recovery protected.
No more memory probes are queued. This qualifies bounded short world-4 mechanics,
not original world-32 OOM or indefinite continual training.

Real-document preparation is COMPLETE, not a learning release. Source-verified
raw compound anchors: 5,885 TRAIN / 334 DEV. Prefix-only nonroot labels fitting64
actions: 5,807 / 329. Fixed hash selects5,120/256, all source/data/code EBS readback.
Public readiness: `validation/real_prefix_partial_data_v9_20261019.json`.
Output: `local/nvme-study/data/real-prefix-partial-labels-v9`. All83length exclusions
and the full candidate inventory remain; 44selected DEVexpressions appear inTRAIN.
276expression-novel DEVrows are available before any learning; freeze a novel
cohort under the next actual protocol if useful, preserve current data identity.
Do not rerun either closed data pipeline. Writer projection receives prefix IDs
only; teacher labels derive from prefix and omit root, full traces separate.
After matched-question readout, choose the next model/method and declare training,
interface, cost and held endpoints before hardware qualification/learning.
Thinking can improve learning and/or inference; keep the two hypotheses separate
as clarified in `docs/REFERENCE_NOTES.md`. Current fixed comparison is unchanged.
