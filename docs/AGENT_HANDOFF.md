# Handoff to the coding agent

October18 update: the owner stopped the original R study. The active task is
verified intermediate-only computation, a trained then frozen reader, a separate
writer and a compact CoT baseline. Read [the current queue](../NEXT_STEPS.md) and
[status](../STATUS.md). The original driver instructions below are historical;
they do not authorize restarting the retired study or override the active plan.

October18 21:24UTC live handoff: all TRAIN128 fitting jobs are closed, native CPU/
Trainium writer actions exactly match128/128, and frozen-reader/compactCoT both
answer128/128. This establishes fitting mechanics, no architecture advantage.
Fresh four512DEV panels run on immutable partial-state-held-v6/3301d22:
Writer and all four compactCoT panels are closed. Same frozen reader10/logical40–43
has closed three panels and is generating final composition. ControllerPID81024;
stage3 timeout owners124166/124168/124169/124171, Python124167/124170/124172/124173.
Controllers:
local/diagnostics/partial-state-v6-fixed-DEV512-{writer,cot,reader}128-v1.
Bound01:30UTC Oct19,3h per child,512GiB floor, no retries. Reader preparation
waits for each complete512writer bundle; never selects by utility. All2048writer
notes close20:16UTC; their utility remains unread. Long-cache compilation passed
20:56UTC. Monitor the remaining reader job; do not inspect partial outcomes.
Only after all three four-panel controllers close, run:
`source scripts/env.sh; USE_TORCH_XLA=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/audit_partial_state_panels.py --root /home/ubuntu/dario`.
Audit actual complete answers, controls, vectors, intervals and archives, then
report all panels separately. Official TEST untouched. Future public branch
partial-state-study-v7 is CPU-only/unreleased; current jobs never use it.
Budget proposal is A640/Bpartial640 versus R320+W320, total640updates per system;
full-root CoT is an extra prepared variant, not a replacement for compact CoT.
No larger learning release until fresh diagnosis, memory/disk/recovery review.
Main proof ladder and paper notes incorporate the owner's north-star/7–8B steering.
Official8B metadata/static sizing is preserved; no weights or run launched. The
installed gradient hook retains a full buffer, so it does not repair replicated fit.
Local v7 protocol drafts were validated before fresh outcome review at
local/diagnostics/partial-state-v7-unreleased-protocols-before-DEV-review; not released.

## Your job

Implement and qualify one production driver for the frozen experiment in `configs/weekend.json`. Use the bundled `legacy/ema-trainium-pre-thinking` repository snapshot and the already working instance environment. The CPU reference in this kit defines semantics; it is **not** an efficient accelerator trainer. Do not run it on every Neuron core and assume the hardware will fix the full-prefix generation cost.

Do not rewrite the paper, invent new metrics, launch extra models, call Bedrock, replace the objective with best-of selection, or silently reduce the thought cap. Do not change any cloud resources without the owner's instruction. Cap initial integration/qualification at approximately six hours and report specifically what remains unqualified.

## First read

Read the weekend plan, this handoff, `validation/source_manifest.json`, and these existing source files (all relative to `legacy/ema-trainium-pre-thinking/`):

- `pre-training/train/train.py`: distributed runtime, syncfree/ZeRO-1 optimizer, chunked vocabulary loss, checkpoint patterns.
- `pre-training/train/model_init.py`: initialization and historical embedding/head caveat.
- `pre-training/train/think_gen.py`: static-cache storage primitive, not a license to reuse its old insertion/echo geometry.
- `pre-training/train/think_oracle.py`, `think_bestof.py`, `think_hintmix.py`: known target and objective issues; do not inherit the whole recipe.
- `benchmark/score.py` and `benchmark/README.md`: existing task/item handling, currently CPU and old-model oriented.

The uploaded `model_init.build_model()` always initializes from scratch. A model-name change is not a mature-checkpoint experiment. Use actual pretrained weights and verify load keys, values, intended ties, tokenizer IDs and CPU/XLA forward agreement.

## Minimal implementation surface

Create the NEW active `train/neuron_driver.py` rather than overloading old run numbers. It must support `--arm vanilla|thought`, a locked plan, an explicit local model snapshot, record manifest, seed, output directory and terminal step. Never default to a historical resume checkpoint or old corpus offset.

Three backend operations are sufficient:

```python
sample(prefix_tokens, group_size, max_actions, format, temperature)
    # -> sampled_actions[B,G,L], stop_mask, behavior_logps
score(prefix_tokens, sampled_actions, future_tokens, format, K)
    # -> stopped-gradient true-future log evidence[B,G]
replay(prefix_tokens, sampled_actions, format, temperature)
    # -> differentiable sampled-action logps[B,G,L]
```

Only `score` receives real future tokens. `sample` and `replay` must not. Generation, reward scoring and policy replay use the same unchanging weights within an optimizer update. Disable dropout consistently; `.eval()` does not disable autograd. One new update per rollout group: no stale/off-policy sample reuse.

Use `core.loo_advantages` and the sequence-normalized signed `core.policy_loss`. Keep gradient magnitudes correct under accumulation: divide each micro's contribution by the accumulation count; the distributed reducer averages ranks once. If looping over individual thoughts, include the 1/G factor. Do not divide a second time by global world size after a mean-reduction optimizer.

The minimal update needs no EMA copy: a scalar baseline shared by a context's samples cancels exactly from LOO advantages. Still compute/log same-model clean evidence and preserve an immutable starting model for bounded cross-reader evaluation. A moving shared reader can change the reward; baseline cancellation is not a guarantee of genuine thinking.

For vanilla, train all next-token labels on the same raw prefix and K=1 future. The last input token is unscored; the last prefix token predicts the true next token. Do not accidentally train the additional stored 15-token analysis future.

## Trainium execution constraints

1. **Qualify the actual layout.** At default LNC2 on trn2.48xlarge, expect 64 logical cores. Verify visibility for each intended partition and absence of overlap. Logical 24 GB/core, not node-wide memory, governs replicated fit. If two partitions do not work promptly, use a sequential full-node plan and preserve global batch.
2. **Preserve a valid FP32 update path.** FP32 parameters with BF16 autocast is the proposed route. BF16-only optimizer updates without an FP32 accumulation/master path are unacceptable. Do not accidentally enable environment-wide parameter downcasting. Check actual parameter changes.
3. **Use one reducer owner.** With XLA `ZeroRedundancyOptimizer`, `optimizer.step()` already manages reduction and clipping. Do not add `xm.optimizer_step`, `xm.reduce_gradients`, DDP all-reduce or another manual gradient sum on that path. Check this against the installed version, not a remembered API.
4. **Do not assume 1.7B fits with replicated Adam.** Use the already supported ZeRO-1 route from `train.py`, activation checkpointing and vocabulary-head chunking. Sharding optimizer state does not shard all parameters and activations. Record actual peak usage.
5. **Cache ordinary causal generation.** Reuse the static-cache storage primitive only after checking the installed Transformers cache API. The CPU reference's full-prefix loop is for correctness. Use fixed prefill/decode shapes; feed changing positions/masks as tensor values rather than recompiling per token. Do not port old echo/insertion masks into this new causal sequence.
6. **Keep large tensors on the device.** Sample with a device-side transform matched to replay; do not copy 150k-vocabulary logits to CPU every step. Stop masks and lengths can remain device tensors until the end of a group. If a token-ID transfer is unavoidable initially, measure it rather than claiming optimized throughput.
7. **Bound memory for loss.** Obtain trunk hidden states and process the vocabulary head in fixed chunks, as in the existing trainer. Do not materialize all `[B,G,context+thought,vocab]` logits or retain every sampled group's backward graph at once.
8. **Use stable graph boundaries.** Synchronize per accumulation micro where appropriate. Repeatedly compiling a new graph at every step or closing the entire accumulation loop into one giant graph fails qualification.

Public serving support does not qualify training-to-serving weight transfer. Do not build a new vLLM/Neuron weight-swap system during this experiment.

## Required acceptance checks

These checks concern implementation, not whether early research curves look promising:

**CPU/core:** run all packaged tests, the toy smoke, and the source repair check. They already pass in this working container, but rerun after any changes.

**Single accelerator:** load the correct base and compare a fixed list of token log probabilities to CPU; compare intended ties and sampled fingerprints before/after device transfer. Use tolerances appropriate to the actual precision. Establish those tolerances from same-precision references before looking at research outcomes. A failed check is not waived because the loss looks plausible.

Compare cached generation log probabilities with uncached teacher-forced replay for the same sampled tokens, positions, stop mask and temperature. Include truncated and naturally stopped thoughts, empty-format and raw-silence controls. Changing future tokens must never change prefix-only generation when the RNG and model are fixed.

**Distributed:** compare a small two-replica update against a one-replica update on the concatenated batch. Verify equal parameters after update, correct global mean gradient and clipping, different sample streams across ranks, and correct global data indices. Unit tests of arithmetic do not replace this real collective check.

**Resumption:** run a few fixed tiny updates uninterrupted versus save/resume. Store model, optimizer shards on all ranks, CPU/NumPy/device RNG, global data cursor, step, LR schedule, source hashes and format/temperature. Verify numerical agreement within the established execution tolerance. Model-only recovery is a new lineage, not exact resume.

**Reader control:** use non-evaluation examples with prefix-derived intermediate calculations, incorrect calculations and matched format. Do not use target-copy success as proof that semantic hints work. These are diagnostics of the interface, not a gate that local CE must improve before downstream evaluation.

**Timing:** ten or more complete steady-state optimizer updates at target shapes. Include generation, score, replay/backward, collective reduction and optimizer step. Measure checkpoint duration and evaluation throughput separately. Use the budget script to lock feasible exposure before the first scientific run.

## Data and prompts

Pin snapshot commits; do not let `main` silently move between arms. Keep raw-text document boundaries and exact train/dev/test manifests. Derive a fixed training-record permutation for each training seed, shared by C and R within that seed. Persist the permutation/hash and cursor. This gives replication meaningful data-order variation rather than merely repeating deterministic CPT. Audit benchmark overlap and near-duplicates; the packaged preparation script only removes exact duplicates and groups page versions.

Load the JSON token lists with `Format.from_dict(obj["format"])`. Use the resolved prompt format from `make_format.py`, or record a clearly different choice before the run. The RLP PDF contains a prompt on printed page 21; the web HTML omitted it. The printed delimiter line itself appears inconsistent, so this kit uses a paraphrased instruction and explicit `<think>` / `</think>` markers. Do not claim byte-identical prompt replication. Resolve IDs with the actual tokenizer; no new random embeddings or hardcoded remembered token IDs.

At K=1, only the first future token supplies reward. K=4/16 evaluation is a declared generalization check. There is no target-aware anchor search or answer-aware deployment selection. Save a few fixed development traces for debugging, not a changing collection of especially attractive examples.

## Output contract

Every checkpoint has a JSON sidecar: run ID, source/format/model/data hashes, dtype, sharding/reducer path, LR/scheduler horizon, world size, accumulation, source-record cursor, actual source/generated/scored tokens, optimizer step, wall time and checkpoint hashes. Store rank-reduced statistics, not only rank-zero local values.

Every evaluation item has at least:

```json
{"run_id":"...","checkpoint_sha256":"...","task":"...","mode":"direct|one_thought|...","selection_policy":"single_greedy|single_sample|oracle_diagnostic","item_id":"...","document_id":"...","score":0.0,"n_scored_tokens":1,"n_generated_tokens":0,"decode_cap":1024,"status":"scored"}
```

Answer extraction errors and truncations remain in denominators. Never silently drop unmatched or failed items. For multiple generated samples, aggregate per item before paired bootstrap. Report token-normalized raw CE separately from per-document mean thought gain. Resource-matched is not automatically FLOP-matched.

Terminal-step checkpoints are primary. A selected best development checkpoint may be reported separately with its selection rule. Do not choose the primary checkpoint using the final test set.

## Boundaries and escalation

You may repair concrete implementation failures and optimize the execution path while preserving reference semantics. You may not change model, data distribution, objective, thought length, horizon, treatment/control exposure, number of updates, or success criterion after seeing outcomes. Any such change gets a new run identity and a human decision.

If the six-hour integration allowance expires without qualification, give a precise report of the remaining failure. Run only verified baseline/evaluation or short diagnostic work. Never use an unverified 48-hour launch to make the instance look busy.


## Local packaging additions

Use `scripts/start_here.sh` once after extraction and `scripts/run_checks.sh` after
changes. CPU resumption is demonstrated by `scripts/reference_session.py`; its
BytesIO checkpoint writer is for tiny tests, not large production shards. Stream
large XLA shards with the installed backend's supported save API. The generic
COMPLETE manifest may be reused after all actual ranks have durably written.

`freeze_experiment.py` binds real local assets and source. The qualification
template deliberately starts pending. Never fill a passed field from a template
or a synthetic test. After code changes, re-run only relevant checks and refresh
the evidence; do not keep a receipt bound to different code.

A subtle estimator limit: dividing each completion by its own sampled length is
a token-averaged surrogate. For variable stopping lengths it is not, in general,
an unbiased estimator of the expected sequence reward; an action-dependent
normalizer also limits the usual zero-mean baseline argument. Common-baseline
cancellation in the actual LOO numbers still holds exactly. The reference retains
the previously proposed surrogate; a fixed-cap normalization is a named later
ablation, not a silent correction.
