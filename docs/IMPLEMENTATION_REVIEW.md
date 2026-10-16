# Independent implementation review — 2026-10-16

Scope: current causal/reference semantics, installed Transformers/Qwen3/cache and
XLA ZeRO-1 sources, and the first `train/neuron_backend.py` draft. No accelerator
was reserved or run for this review. Findings below are review guidance, not a
qualification receipt. User authorization now allows weeks of work and deliberate
adaptation; the inherited weekend cutoff is no longer an engineering constraint.

## Repair before accelerator cache qualification

- **Installed abstract cache API:** `CacheLayerMixin` requires `get_max_length()`;
  the old `get_max_cache_shape()` alias does not satisfy it. The first draft's
  `_CausalCacheLayer` cannot be instantiated until that method exists.
- **Empty headers:** every token-list tensor must explicitly use `torch.long`.
  `torch.tensor(())` otherwise creates float32 and promotes concatenated IDs.
  Empty-header qualification is a useful real boundary case.
- **Checkpointing:** installed `GradientCheckpointingLayer.__call__` checkpoints
  only when its `gradient_checkpointing` and `training` flags are both true.
  `model.eval()` silently disables stock trunk checkpointing. Use explicit
  checkpoint wrappers that work in eval, or qualify a replay-only train mode
  after verifying every stochastic layer is disabled. Sampling must run in eval:
  stock training/checkpointing wrappers remove caches, even under `no_grad`.
- **Ties:** the actual P1-002 artifact reports loss of the embedding/head identity
  after `.to(xla)`. Retie immediately after transfer and before optimizer creation;
  recheck parameter count and identity. A numerically matching forward does not
  excuse training two independent copies of the published tied parameter.
- **Layer types:** reject non-`full_attention` entries in `config.layer_types`.
  `config.has_sliding_layers` is not the installed Qwen3 indicator; the model
  trunk computes `has_sliding_layers` from the list.

These issues were reported to the implementation owner as the draft appeared;
they may already be repaired by the time this file is read.

Follow-up review of the next draft confirmed the cache method, integer token
tensors, full-attention guard and explicit eval-mode checkpoint wrappers are
repaired. The wrappers bind each layer in the closure correctly. The installed
CPU Qwen3 suite passed all four checks in 2.25 seconds, including compact reader,
cached/replay log probabilities and full parameter-gradient comparison.

## Smallest correct production geometry

The current code's basic token geometry is sound: the final writer-prompt hidden
state predicts action zero; replay feeds prompt plus `actions[:-1]`; the first
sampled terminator contributes to the policy loss and is removed from reader
content. Capped thoughts do not receive an invented stop action.

Use a fixed causal cache of width `prompt_width + cap`, one fixed prompt shape
and one single-token decode shape. The installed ordinary `StaticLayer` already
has a tensor-valued cumulative position; alternatively retain the narrow custom
storage primitive. Qwen3 attention calls `cache.update(k, v, layer_idx)` without
positions, so custom storage needs its own shared tensor-valued position holder.
Explicit `position_ids` and `attention_mask={"full_attention": mask}` avoid default
mask/cache-length inference. Keep legacy echo/branch masks and host top-k shaping
out of this path.

Reader content must remain contiguous: header, raw prefix, open delimiter, actual
content, close delimiter, then teacher-forced future. Vectorized gathered writes
into a right-padded fixed width are appropriate. Earlier valid causal queries
cannot attend trailing padding, so an explicit per-row padding mask is optional
for this particular layout; it becomes necessary if padding ever moves between
real tokens. Gather only the valid prediction positions before the vocabulary
head. K=1 can omit the future token from the input entirely and use it only as a
label; including the unused final token is causally correct but wastes a row.

Keep all four rewards before any backward contribution, keep parameters fixed
across the complete group/update, detach rewards and advantages, and divide each
one-completion replay loss once by group size and accumulation. ZeRO-1's installed
`step()` performs mean reduce-scatter and global clipping; do not also average
by world size, clip rank-local full gradients, or call `xm.optimizer_step`.
Use `save_master_weights=True` and save every rank's optimizer shard.

## Memory and numerical checks that matter

The released local config has 28 layers, 8 KV heads, head dimension 128, hidden
size 2048 and 1,720,574,976 unique tied parameters. FP32 parameters plus full FP32
gradients alone occupy about 12.8 GiB. A four-sample BF16 cache at a roughly 1,580
token writer width is about 0.68 GiB; its FP32 form doubles that. Plan against
measured memory per logical core, including optimizer shards and transient
attention/vocabulary buffers. None of these estimates certifies fit.

Replay one completion at a time. Chunked vocabulary projection must itself be
checkpointed or backpropagated in bounded stages: merely collecting all chunks'
log probabilities before a single backward can retain their full vocabulary
activations. The draft's checkpointed scalar chunk losses follow the useful old
pattern. Trunk checkpointing still needs the mode repair above.

All softmax/log-probability and Gumbel sampling transforms should operate in FP32;
BF16 matmul compute can remain under the qualified autocast. Replay must use the
same temperature, vocabulary support, stop convention and model weights. Check
selected action log-probabilities at every active action, reporting max and mean
errors and separating stop actions. Also examine full-distribution differences
at a few fixed short trace positions. A large tolerance is not permission for a
meaningfully different behavior policy. Real FP32 parameter changes, changed
fraction and update norm are required alongside finite gradient norms.

The tokenizer/config supports IDs through 151668, while the published output
head has 151936 rows. Before any rollout outcomes, the implementation owner chose
to ban exactly IDs 151669 through 151935 from the writer policy. This makes the
writer's support correspond to the tokenizer's defined token inventory; those
model-only rows do not have a reliable text representation. This is the fixed
vocabulary transform already supported by the reference primitives. Freeze its
IDs/hash and apply it identically to cached sampling, teacher-forced replay and
checkpointed one-completion replay, and to C/R downstream generation. The reader
clean/thought reward and C's NTP loss retain the released raw full-vocabulary
normalization. No auxiliary loss or reward change is implied. Do not add
unrecorded top-k, repetition or frequency penalties.

A separate tiny installed-CPU Qwen3 check with six banned rows verified no banned
active actions, cached/replay max log-probability error `2.3841858e-7`, zero scalar
loss error and max full parameter-gradient error `1.4901161e-8` against an
unchunked masked reference. The trace included an immediate additional-stop
sample and capped samples. This is CPU implementation evidence only.

## Scientific interpretation and inexpensive diagnostics

The chosen per-completion `1 / sampled_length` loss is an action-length-weighted
surrogate. It is not generally an unbiased gradient of raw expected sequence
reward. Independent LOO baselines do not repair the action-dependent denominator;
the common clean baseline nevertheless cancels exactly in the actual LOO values.
Preserve the declared recipe for its first identity. Log length/stop distributions,
reward spread, advantage-versus-length, and policy-loss/gradient scale. A summed
or fixed-cap loss is a separate later experiment, not an invisible correction.

Detached reward blocks reader gradients during a step, but shared parameter
updates still change the reader between steps. Improving local self-reader gain
therefore does not establish useful mathematical computation. Keep absolute
clean and thought evidence, use held-out prefixes, and do the planned initial/
trained writer × initial/trained reader audit. Record own, empty, filler,
shuffled and other-note controls by fixed item; retain bad outputs as well as
good ones. Stratify K=1 diagnostics by token class/surprisal without selecting
training anchors using those diagnostics. A whitespace/punctuation likelihood
improvement is a measured effect, not a math capability result.

Before scientific training, use a small frozen prefix-derived arithmetic reader
probe with correct calculation, wrong calculation and matched format. This
checks whether the chosen base prompt/readout can use semantic notes. It is a
diagnostic, not a requirement that early sampled notes already help. Lock GSM8K
development prompts/parser before final testing; require the C/R thought task
contrast and direct controls. Compare stop/cap rates and real token/core-time
budgets, since a nominally identical cap may hide different inference spending.

The [RLP v2 method](https://arxiv.org/html/2510.01265v2#S2) explicitly treats reward
as stopped, uses group-relative advantages, and gives a length-normalized clipped
surrogate. One fresh update has the same initial unclipped policy gradient as
REINFORCE. This implementation remains a declared simplified condition rather
than a reproduction of all published settings.

## Qualification priorities

1. Fix cache construction, empty header, ties and checkpoint-mode behavior; run
   short CPU geometry/parity checks with the installed Qwen3 implementation.
2. Run short device cached/replay traces, then exact-cap traces. Include empty,
   naturally stopped and capped cases; changing future tokens must leave draws
   unchanged when prefix, weights and RNG are restored.
3. Verify a two-rank mean/clipped update against a concatenated-batch reference,
   actual master updates and checkpoint/resume agreement.
4. Measure ten complete target-shape updates and run one genuine downstream
   inference item before committing a long exposure. Revisit engineering choices
   using evidence, while keeping scientific recipe changes explicitly identified.
