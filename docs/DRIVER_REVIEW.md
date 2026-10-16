# Independent production-driver review — 2026-10-16

Scope: `train/neuron_driver.py`, `train/neuron_backend.py`, frozen experiment
bindings, source exposure, reward, normalization, and diagnostic accounting.
This is source review plus the separately recorded CPU/data checks, not evidence
that the released-model production step ran on Neuron. Checkpoint implementation
and distributed numerical qualification are reviewed by the distributed agent.
The implementation was edited concurrently. The findings below are now resolved
in the source, with the explicitly CPU-only control-flow checks described here;
released-model accelerator qualification remains separate.

## Findings resolved in source and CPU control-flow checks

1. **Undefined C-branch reward variables — fixed.** The finite-reward check now
   lives inside the thought branch after `thought`/`clean` materialization and
   before replay. The vanilla branch only checks its own loss. The actual
   `worker()` function completes a one-step CPU control-flow fixture without
   calling a thought sampler/reader or accessing thought-only variables.

2. **Partial frozen-recipe enforcement — fixed for the supported plan.**
   Production reads batch, cap,
   topology, paths, and terminal steps from the frozen spec. The implementation
   fixes G=4, temperature=.7, K=1, sequence normalization, LR=1e-6, betas=(.9,.95),
   weight decay=.1, clipping=1, no reader/NTP/KL auxiliary, one micro-prefix per
   rank, and one fresh update. `validate_fixed_recipe` now rejects unsupported
   values and undeclared seeds before the production qualification barrier.
   Regression tests use the actual proposal and both predeclared seeds, then
   change the algorithm, NTP auxiliary objective, temperature, loss normalization,
   learning rate, and seed. The algorithm mutation initially exposed an omitted
   check; the owner added it and the rejection test now passes.

3. **Pre-update finite-gradient check — added.** The driver now checks every
   present local gradient, reduces a scalar bad-gradient flag with MAX, and raises
   before `optimizer.step` when any rank reports a failure. This scalar flag does
   not reduce gradients or add a second clipping owner. CPU fixtures demonstrate
   a finite forward loss with an injected infinite backward gradient leaves
   parameters unchanged and never calls the optimizer. An injected remote-rank
   failure also blocks an otherwise finite local update. Separate fixtures show
   nonfinite C loss stops before backward and nonfinite reader reward stops before
   policy replay. These are actual driver-control-flow tests using CPU runtime
   boundaries; they do not verify Neuron collectives or numerical behavior.

All 12 tests in `tests/test_driver_recipe.py` passed in 1.58 seconds; output is in
`local/driver_recipe_tests.fixed.log`. No driver source was edited by this
reviewer. The subsequent full CPU suite passed 135 tests in 11.56 seconds and the
toy smoke check; logs are in `local/checks/tests.KSKVkU/`. A rank-local failure
partway through a micro-step still depends on the
worker manager's bounded peer cleanup; that operational path is not exercised by
the CPU fixtures.

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
- The existing frozen `spec_hash` is used directly. Metrics divide accumulated
  loss by world size and per-prefix reward/cap/advantage sums by global prefix
  batch, while actual action counts remain sums. These are source observations,
  not claims of measured distributed production behavior.

## Diagnostic improvements

The saved `dev-trace-*` is a training record, not development evidence. Rename it
as a training trace and render content/valid actions separately from the full
padded action tensor. Retain raw IDs/masks and log which terminator occurred.

`source_records` and `source_tokens = records*513` correctly describe fresh
source exposure. They are not total compute tokens: R repeats prefixes and runs
fixed padded decode/replay work. The source now records executed padded decode
slots separately from semantic sampled actions. Per-step timing covers the
update; separately record
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
