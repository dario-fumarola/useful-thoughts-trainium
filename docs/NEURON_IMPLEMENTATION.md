# Production implementation: smallest useful path

## State at delivery

Instance update, October 16: full FP32 compute is now the production candidate.
Long-trace BF16 and BF16-trunk/FP32-head replay exceeded the fixed numerical
criterion; full FP32 passes the exact failed trajectory at the tighter criterion.
See `DECISIONS.md` and `IMPLEMENTATION_REVIEW.md`. The implementation below remains
subject to full-model distributed, resume and complete-step qualification.

`train/neuron_driver.py` and the cached backend now exist and are undergoing
real-asset qualification. A real four-rank update/checkpoint completed, but its
next optimizer graph exceeded memory; 32-rank FP32 qualification is running.
Long FP32 cache/replay checks pass, while the full-cap backward and distributed
resume/timing evidence remain pending. See `STATUS.md` for live state.

The remainder specifies the required behavior. CPU reference tests and bundled
legacy examples do not certify the current accelerator path.

## Reuse map

All historical paths below are under `legacy/ema-trainium-pre-thinking/`.

| Need | Starting material | What not to inherit |
|---|---|---|
| XLA initialization and graph boundaries | `pre-training/train/train.py` | Historical corpus offset, resume path, scratch initialization |
| Optimizer/master state | `train.py` optimizer builder | Multiple gradient reducers or unverified BF16-only updates |
| Cache storage | `pre-training/train/think_gen.py` | Echo/insertion/branch attention geometry |
| Trunk/head loss handling | `train.py` and late trainers | Full vocabulary logits for all padded tokens at once |
| Loading behavior to avoid | `pre-training/train/model_init.py` | `from_config` plus fresh initialization |
| Benchmark item handling | `benchmark/score.py`, `metrics.py`, `fetch_data.py` | CPU-only assumption and old-model loader |
| Correct active objective | `weekend_thinking/core.py`, `reference.py` | Any target-aware draft selection at deployment |
| Data order / resume | `weekend_thinking/order.py`, `checkpoints.py`, `session.py` | CPU RNG as a substitute for XLA per-rank RNG |

Search just the relevant function. The old runbook and journal are evidence, not
startup instructions. The target patch is narrow; it does not certify old training.

## Driver interface

```text
python train/neuron_driver.py \
  --frozen local/experiment.json \
  --qualification local/qualification.json \
  --arm vanilla|thought --seed 1234 \
  --output local/runs/<unique-run-id> \
  [--resume <complete-checkpoint>] [--stop-file local/STOP]
```

This interface exists, but a scientific launch still requires the actual frozen
assets and qualification receipt. A short `--qualify` path may run first. It must
have a hard step limit and identify its artifacts as qualification, not science.
Long training must verify the frozen assets and actual hardware receipt, actual
deadline, and output directory before reserving resources. Reuse of an output
without an explicit compatible resume is an error.

## Three operations and their boundaries

1. `sample(prefix_ids, format, group_size, cap, temperature)` returns actions,
   sampled-action masks, behavior log probabilities, stop type and actual counts.
   No `future`, target, or Record object is accepted by the writer API.
2. `score(prefix_ids, actions, future_ids, format, horizon)` produces stopped
   log evidence. The true future enters only this operation.
3. `replay(prefix_ids, actions, format, temperature)` gives differentiable log
   probabilities under the same weights and sampling transform.

The typed contract is in `weekend_thinking/backend_contract.py`. It deliberately
separates inputs; it is not a claim that a Python type annotation prevents leakage.
Tests must change future tokens while keeping the prefix, model, and RNG fixed.

Disable dropout consistently. Keep parameters fixed from first rollout through
last reward/replay micro of an update. Do not reuse old samples for additional
updates. Use all four signed leave-one-out advantages; gradients do not flow
through the reader rewards. Completion-length normalization is the chosen
surrogate, not an unbiased raw expected-reward claim for varying lengths.

## Stage P1: real loading and an ordinary causal forward

Load pinned local safetensors with `from_pretrained`, no remote custom code.
Inspect the installed Transformers API; `dtype` versus `torch_dtype` compatibility
must be handled without silently fetching a different package/model.
Verify the expected model type/shape, zero missing/unexpected keys, tokenizer
bounds, and intended embedding/head ties before and after XLA transfer.
The new loader fails rather than accepting partial weights.

Use explicit causal labels from `causal_batch`; they are already shifted once.
Do not pass them to a helper that shifts them again. For C, prefix plus one future
token has its last input unscored and its last prefix token predicts the real
future. The additional 15 stored analysis tokens are not NTP training targets.

First compare a small fixed CPU/device set of next-token log probabilities.
Tolerances must be based on same-precision comparisons, not relaxed until a bad
run passes. Device BF16 evidence need not match CPU FP32 bit-for-bit.

## Stage P2: static cached generation

Use an ordinary causal cache and fixed prefill/decode shapes. Prefer a supported
cache in the already installed stack; adapt the legacy storage primitive only
when needed. Growing Python sequence shapes at every token repeatedly recompiles
XLA graphs. Cache positions and active masks should be tensor values, not Python
constants that create one executable per position.

Group size is four. Keep sampling and vocabulary operations on device. A full
vocabulary-sized host transfer per token is not an acceptable final path. Sampling
and replay must share temperature and vocabulary masks. Avoid untracked top-k,
top-p, repetition penalties, or normalization changes.

Natural stop: include the actually sampled first terminator in the policy loss,
exclude later padding, strip it from content, and append the fixed reader close
format. A capped thought has no invented stop action. Emit actual generated-token
counts and stop/cap rates. Executing padded decode steps after all samples stop
is a compute cost; do not count those as semantic generated tokens.

The reader must see the close delimiter immediately after actual thought content,
not after 1,024 visible padding tokens or a hidden positional gap. Pack content,
close format, and future into a statically sized right-padded tensor using
data-valued lengths/indices, and compare its valid-token positions to the CPU
reference. Masked padding must not alter RoPE positions of real continuation tokens.

For short qualification traces, compare cached behavior log probabilities against
teacher-forced replay at every active action. Include empty, naturally stopped,
and capped thoughts. Verify all future changes leave generated actions unchanged
under identical RNG. Qualify the exact maximum length before scientific training.

## Stage P3: memory and distributed mean update

At LNC2 on the expected node, plan against per-logical-core memory, not aggregate
HBM. FP32 weights, gradients, optimizer state, activations, cache, and temporary
vocabulary logits all matter. Use the existing compatible ZeRO-1 path and
activation checkpointing as needed; do not assume 1.7B replicated Adam fits.

Process one thought replay at a time when needed. Divide its contribution by G
and by accumulation exactly once. A mean-reduction optimizer handles world-size
averaging; do not divide by world size again. For group advantages, gather all
four scalar rewards for a prefix before replaying that group's samples.

Use trunk hidden states and compute the vocabulary head in bounded chunks.
Taking a few target positions after producing all [sequence,vocabulary] logits
is too late to save that memory. Keep graph boundaries stable per micro; measure
rather than accumulating a giant lazy graph across a whole update.

Exactly one component owns synchronization and global clipping. For documented
XLA ZeroRedundancyOptimizer, `optimizer.step()` already handles reduction.
Do not also call `xm.optimizer_step`/manual reduction. Confirm with installed
source, then test two ranks against a concatenated-batch reference. Check parameter
agreement, rank-distinct RNG/examples, and both update norm and changed-weight
fraction. A gradient value is not proof that FP32 master weights changed.

If partitioning the node, verify actual disjoint visibility. Use separate rendezvous
endpoints. A local process lock alone does not prove accelerator isolation.
Sequential full-node execution is the permitted fallback, preserving global batch.

## Stage P4: durable resume, timings, and a real evaluator

Save every optimizer shard plus model state, CPU/NumPy/device RNG per rank, fixed
record-order hash/cursor, step, learning rate, frozen spec, format, and counters.
Use a streaming device-supported save path for large states. The CPU helper uses
BytesIO only for tiny tests; do not scale that helper into a 1.7B memory spike.

After an actual barrier and complete shard writes, publish a manifest/COMPLETE
marker last. Hash verification checks file integrity, not whether every rank
saved at the same step; the adapter must establish that. Compare uninterrupted
versus resumed training at fixed settings. An unresumable optimizer is a new
lineage, not an exact continuation. Do not change topology on an exact resume.

Measure at least ten complete steady-state optimizer steps, plus checkpoint and
evaluation throughput. Include generation, scoring, replay, collectives, optimizer,
and device synchronization. No estimated times from FLOPs or only backward speed.
Verify the compiler cache is reusable and no continuous shape churn remains.

Record `weekend_thinking.runtime_policy.runtime_policy()` in each arm's resolved
site configuration as `runtime_policy_by_arm.vanilla` and `.thought`. This captures
the actual allowlisted environment, including unset values; it does not assume
that a documented legacy cache variable works in this installed runtime. The
driver binds the policy into qualification specs, run records and every saved
optimizer shard, and rejects changes on resume. If the arms use different cache
policies, measure both policies' complete-step costs; do not reuse timings from
an unmeasured policy.

Set `checkpoint_interval_seconds` in the site file and the matching driver flag
(default43200). All ranks synchronize the save decision. Besides the existing
quarter/half/terminal saves, a complete save occurs after12hours at the next
update boundary; a slow single update can extend that interval. Copy the latest
complete state to EBS with verified readback. Existing immutable qualification
jobs retain their original schedule. Unmeasured C reward metrics and unaudited
cache/replay errors are null, rather than apparent numerical measurements.

Before the long pair, execute one genuine task inference item through each
planned mode on the accelerator. Otherwise Sunday may arrive with training
checkpoints but no evaluator. Do not postpone all evaluator integration until then.

## Failure handling

If unsupported APIs, numerical instability, or memory constraints block the path,
record the smallest failing case and continue independent preparation/evaluation.
Use the fixed fallback rules before viewing outcomes; do not reduce L/G/K silently.
Never mark CPU fixtures or old logs as new Neuron evidence. An explicit unresolved
backend is preferable to a weekend-long unqualified job.
