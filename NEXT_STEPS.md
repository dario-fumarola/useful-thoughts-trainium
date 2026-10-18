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
3. All5first-seed models independently pass full updates/states and are frozen.
   Second reader320 also passes full audit/EBS backup and is frozen. Second writer
   runs in tmux `dario-ps7-writer2`, timeout PID233263/start20113193,
   physical3/logical12–15, deadline Oct19 06:30UTC; last197,full64/160.
   Controls/output basename `partial-state-v7-writer-TRAIN5120-seed2026101807-v1`
   under diagnostics and `local/nvme-study/runs`. After320closure: generic full
   audit, freeze actual writer2 with checkpoint.identity, preserve binding onEBS.
   Then release seed2 direct640,compact640,full640 sequentially, one training arm.
   All controllers/protocols prepared; verify hashes and actual release. Fresh
   short TMPDIRs are p7s2r/w/a/c/f; old ps7r2 belonged to prior recovery and stays intact.
   Source remains immutable57d3220. No automatic training retry or outcome selection.
   A/compactS1receipts copied320digest after verifying all5states: preserved originals
   plus explicit640correction annexes pass strict identity. Every other640template fixed.
   First-seed Wnotes and Aanswers complete/EBS readback; compact3/full-CoT1panels closed.
   Reader/device9 and B0/device4 wait on active longer cache compilation; node memory safe.
   Compact/device6 and full-CoT/device7 evaluations advance. No new outcome review.
   After W2freeze execute `local/analysis/prepare_partial_state_v7_seed1807_evaluation_20261018.py`,
   then `prepare_v7_second_seed_eval_controllers_20261018.py` with frozen57PYTHONPATH.
   Launch W2/R2 evalcontrollers on physical10/11 (ports64801/64821,shortw7b/r7b).
   Single-model evalprep utility accepts seed2026101807; use physical0/1/2,ports64901/64921/64941.
   Registry: `local/diagnostics/partial-state-v7-matched5120-evaluation-registry-20261018.json`.
   Public independent audit requires all11controllers closed (10primary+1shared B0)
   before any gold/accuracy/state review. B0direct/cot/empty is one shared auxiliary,
   not two seed replicas; primary training comparisons remain fixed.
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
