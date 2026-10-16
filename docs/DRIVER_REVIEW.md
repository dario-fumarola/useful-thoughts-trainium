# Independent production-driver review — 2026-10-16

Scope: `train/neuron_driver.py`, `train/neuron_backend.py`, frozen experiment
bindings, source exposure, reward, normalization, and diagnostic accounting.
This is source review plus the separately recorded CPU/data checks, not evidence
that the released-model production step ran on Neuron. Checkpoint implementation
and distributed numerical qualification are reviewed by the distributed agent.
The implementation is being edited concurrently; resolve the findings below
against the current file before treating this review as closed.

## Findings requiring resolution

1. **C branch references undefined reward variables.** In `worker`, immediately
   after the vanilla backward synchronization, the finite-reward check accesses
   `thought` and `clean`. Neither is defined in the vanilla branch. This raises
   `UnboundLocalError` on the first C micro-step. Move the reward check into the
   thought branch after materializing `thought`/`clean`, before replay. Reported
   directly to the driver owner; not edited by this reviewer.

2. **The frozen recipe is only partially enforced.** Production reads batch, cap,
   topology, paths, and terminal steps from the frozen spec. The implementation
   fixes G=4, temperature=.7, K=1, sequence normalization, LR=1e-6, betas=(.9,.95),
   weight decay=.1, clipping=1, no reader/NTP/KL auxiliary, one micro-prefix per
   rank, and one fresh update. Those values match the current proposal. A changed
   proposal can nevertheless hash/qualify successfully while the driver silently
   uses these constants; arbitrary CLI seeds are also accepted. Before a scientific
   launch, validate the supported fixed recipe and allowed seeds explicitly, or
   read every supported setting from the frozen proposal. Reject unsupported
   settings rather than broadening the experiment.

3. **Nonfinite gradients are detected after the optimizer has stepped.**
   `found_inf` is initialized to zero and never updated. Finite scalar loss does
   not prove finite backward gradients. A nonfinite backward can therefore mutate
   parameters before the global-norm check raises. No subsequent checkpoint is
   written, so earlier complete checkpoints remain usable. Prefer a rank-agreed
   finite-gradient guard before the update, while retaining one reduction/clipping
   owner. A rank-local exception before other ranks enter a collective also needs
   a bounded failure/termination path rather than leaving peers indefinitely hung.

## Semantics that agree with the stated experiment

- Each seed uses the same fixed `seed_order` for C/R, distinct distributed record
  indices, and no dataset wrap. Only training-split records enter updates.
- The writer API receives only the prefix. The true future enters causal reader
  scoring and C targets. Cached sample state is recreated for every rollout.
- Reward uses the first real continuation token. Reader and clean evidence are
  detached; raw clean evidence is retained. Signed LOO uses every one of four
  samples; the common clean baseline cancels algebraically.
- Each replay contribution divides by its sampled-action count, four samples,
  and accumulation once. ZeRO owns rank averaging and clipping. C supervises all
  next tokens in prefix+one future token; the other 15 stored tokens are unscored.
- Reader content is compacted before the close delimiter, with contiguous real
  positions. The sampled first terminator enters policy loss and is removed from
  reader content; capped samples have no invented stop.
- The owner's concurrent changes use the existing frozen `spec_hash` directly
  and convert loss/reward/cap diagnostics to appropriate global means. Earlier
  snapshots of those two issues are superseded by the current implementation.

## Diagnostic improvements

The saved `dev-trace-*` is a training record, not development evidence. Rename it
as a training trace and render content/valid actions separately from the full
padded action tensor. Retain raw IDs/masks and log which terminator occurred.

`source_records` and `source_tokens = records*513` correctly describe fresh
source exposure. They are not total compute tokens: R repeats prefixes and runs
fixed padded decode/replay work. Keep semantic sampled actions distinct from
executed decode slots. Per-step timing covers the update; separately record
startup/compilation, checkpoint/evaluation, and final job wall time for resource
comparisons. Include an explicit terminal-versus-budget-stopped run status.

## Data and operational bindings

Use `local/data/finemath/production-500000.audit-v2.records.jsonl`, SHA-256
`bd81972fff861f497ce61479079d6904e1baef0a335dfc848cc33e4d90d069ff`.
The [data receipt](../validation/data_production_v2_20261016.json) verifies all
366,134 records, exact tokenizer/source/question hashes, and 48 source anchors;
329,535 train records support the maximum 2,048*128 fresh source exposure.
The earlier v1 asset is retained solely as evidence of the corrected URL-query
identity bug. No additional corpus expansion is needed for this experiment.

The user explicitly states weeks are available and gives no hard shutdown
deadline. An agent-selected 24-hour finite job/planning horizon is a legitimate
pre-freeze resource budget; describe it as that budget, never as an instance
expiry or user-imposed weekend cutoff. Terminal scientific checkpoints remain
fixed before final-test outcomes are inspected.
