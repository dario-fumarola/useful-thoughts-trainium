# Uniform-LM native qualification

The unchanged exact-recovery requirement now passes: all replayed losses/norms, every final model tensor and every FP32 optimizer shard match the uninterrupted reference. Initial native training loss differs from CPU by 0.000678694 nats/token. Actual warmup/decay LRs, single gradient reductions, finite FP32-master updates and replica checks pass. This qualifies execution, not held-out model quality.

The first strict recovery comparison remains failed. Its saved state loaded exactly, but the fourth loss differed by 0.00001633 nats/token. Equal restored counters collapsed 319 FP32 scalar inputs into 10, with unchanged arithmetic counts. Disabling the device-data cache only during mutable state transfer/materialization restores 319 inputs and exact continuation; the prior cache setting is restored afterward. CPU lifecycle/error tests and the full 2,029-test suite pass. No numerical tolerance was widened. [Original failure](strict-recovery-failure.json) · [HLO comparison](scalar-state-comparison.json) · [Exact recovery](native-qualification-counter-cache.json) · [Shared repair](../../train/runtime_compat.py).

The separate qualification-only scoring job covers all 527 original targets on 1,024 sources. Its first global-batch NLL differs from CPU by 0.000674747 nats/token; all eight actual B0 signatures agree. Warm batches take at most 1.111 seconds. [Scoring check](likelihood-qualification.json).

All completed reservations, including unsuccessful attempts, total 4.438494 chip-hours before scientific fitting. The measured complete projection is **17.832178 chip-hours**, including four fixed fits and 15,360 scoring documents, within the unchanged **18** ceiling and at most two existing chips. The first fixed fit has started. Scientific recipes, source allocation, endpoint and selection rule remain unchanged. [Projection](measured-workload.json).

To retain every original and repaired qualification state plus the four planned terminals, full-checkpoint scratch on the existing RAM filesystem extends from the proposed 180 to 256 GiB. The 35 GiB free-disk and 512 GiB host-memory floors remain unchanged. [Workspace record](checkpoint-workspace-amendment.json).

The current native source is 4f9166b. The fixed comparison remains ordinary uniform next-token training: four hyperparameter settings, one source order, no thought objective. All four terminal checkpoints must finish before DEV selection; the selected existing checkpoint freezes before separate CONFIRM scoring.
