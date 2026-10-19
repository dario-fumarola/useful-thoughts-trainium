# Handoff to the coding agent

October18 update: the owner stopped the original R study. The active task is
verified intermediate-only computation, a trained then frozen reader, a separate
writer and a compact CoT baseline. Read [the current queue](../NEXT_STEPS.md) and
[status](../STATUS.md). The original driver instructions below are historical;
they do not authorize restarting the retired study or override the active plan.

October19 02:22 UTC handoff: current owner steering asks one small variable-depth
writer intervention before a broader dataset/architecture pivot. The complete
v7 result is published in a0ae5ec,review ZIP saved and linked to owner. FullCoT
beats split all panels/both seeds; two-line stopping is a curriculum hypothesis.
New source5af3431 at local/worktrees/variable-depth-writer-v11,1614CPUchecks pass.
Public prospective design validation/variable_depth_writer_v11_design_20261019.json.
Disposable four-core qualification now active via
local/diagnostics/variable-depth-writer-v11-v1/qualify.py (tmux variable-depth-v11-qual).
Fresh1–6/fullsave/close/fullrestore6/updates7–8; inspect supervisor/stage logs.
No pilot model released yet. Then fixed640 versus varied640(one–five steps),
160updates/eight passes each, frozen v7 reader/fullCoT references and B0. New
DEVdepth2/6 n256 plus targeted existing composition512. Strict gates in JSON.
Both writers128actions+reader32 versus single160; all failures retained.

Real-data CPU implementation remains in worktree real-trace-cpt-v10 (1621tests),
its hardware plan HELD before any accelerator launch. Do not accidentally use
it as release. Keep real-document usefulness separate from synthetic mechanics;
owner wants both bounded questions,math repair first. Follow NEXT_STEPS.md.

Main report:validation/partial_state_v7_matched5120_complete_20261018.json.
Posthoc:validation/partial_state_v7_matched5120_descriptive_20261019.json.
Figures+CSV beside descriptive. Original implementation immutable57d3220 source
worktree stays intact. Audit completed; candidate new training source may now
change only under NEW identity. Main generation wasFP32, trainingBF16; note this
for cost/precision. Never turn B0direct/emptyzero into lack of math:all512outputs
unsupported eachpanel; B0CoT188/273/91/259with263/157/315/159unsupported. FullCoT
gains overB0CoT on first3panels; composition contrast inconclusive bothseeds.
Recorded decode times are separate concurrent calls, not end-to-end service.

Next phase is real single-model data augmentation, not another split run. Strong
raw continuation/full-trace reference, optional post-document explanatory arm
keep training and deployment hypotheses separate. First inspect existing masked
CE/packing API, then adapt one current SFTengine (constantLR,BF16params/grads,
FP32masters,one reducer, full checkpoints) to native prefixes up to512 plus
traces. Considerwidth768,targetslots768,world8×micro4global32; only bounded
mechanics/native/save/actual-restart/memory qualification before learning.
Choose objective/heldnovel256/2seedbudget explicitly before science; do not launch
an undefined plan. Realdata5120/256ready,276noveleligibleDEVavailable. Rootuser
has full autonomy and weeks. Subagents quota-unavailable; avoid retrying.

Engineering export question CLOSED, no active jobs. Candidate fb72065,
1601CPUchecks. Actual14update memory49–139GiB control versus21–22cleanup;
all14full states/native gates/cursors pass. Saved12 all bytes identical.
Original strict uninterrupted gate FAIL is preserved (step14 loss drift0.000014).
Separate unchanged-exporter restart reproduces exact drift; both resumed model,
optimizer,RNG,loss/norm trajectories byte-identical. Cleanup preserves matched
recovery; no originalworld32OOM/indefinite guarantee. Public reports and sources,
all original states plus EBS terminal/control evidence saved. No more probes.

Real raw/partial DATA preparation is COMPLETE. Both pipelines closed and EBS
readback: `real-raw-compound-prefix-v9-v1` and `real-prefix-partial-labels-v9-v1`
under local/diagnostics. No model/training/interface/hardware release.
Source curator branch9d57df7 (1617CPUchecks), same347895auditedTRAIN/DEVrawdocs,
minimum2prefixoperations, unchanged source/numeric validation, TEST never decoded.
5885TRAIN/334DEVraw compounds; after prefix-only exact nonroot derivation and
64action budget5807/329eligible. Fixed document-hash selection5120TRAIN/256DEV,
no document overlap,83long labels excluded and retained,44DEVexpressions seen
inTRAIN. There are276eligible novel-expressionDEVrows overall, allowing a future
new256novel cohort before learning. Current data identity stays preserved.
Data: `local/nvme-study/data/real-prefix-partial-labels-v9`; exactTRAINsha
437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915,
DEVsha8ba676b1cb7cb1f46405d259911207e2d4467ffed922ef07ab2bf9e28738ce79.
Public readiness/four fixed computed examples:
`validation/real_prefix_partial_data_v9_20261019.json`. Native prefixes/16future
labels + nonroot partials + separate full traces/provenance are local+EBS.
Projection code public beside report; writer input excludes future/notes, poison
check passes. No data pipeline should be rerun. Pick next method after readout.

Re-read BoLT v2 sections3.3/B.1 and TPT section2. New reference-note clarification:
latent marginal/ELBO includes prior compatibility; exact hindsight-note matching
isn't universal necessity for a latent training objective. TPT trains doc then
thinking. Keep training value and deployed extra-computation value separate; do
not make the current channel result a verdict against every training scaffold.
Paper notes and real-data readiness are committed in4337765; readout publication complete in a0ae5ec.
Root owns work, subagents quota-unavailable. Keep <=60s meaningful updates,
bounded waits, and avoid long internal speculation while jobs await handling.

All original full checkpoints stay on local SSDs. EBS all terminal models plus
latest full optimizer state; older EBS optimizer duplicates retire only after full
both-side hashes and latest recovery verification. Current retention JSON public.
Extra local/nvme-study verified mountedSSD; old paths fixed. Keep35GiB floors.
A/compactS1closure320digest typos fixed only by preserved explicit640annexes;
all unstarted640templates corrected. S2temp paths p7s2r/w/a/c/f; oldps7r2recovery intact.
Root owns work; subagents unavailable due quota, do not retry spawning.

Interpretation later:split changes loss weighting and has two full models. A
positive result does not isolate architecture. Shared/balanced single-model CoT
is a cheap possible next ablation if it wins; if CoT matches, prefer it. Real-text
transfer follows; use the failed link to choose scale/data/objective after review.

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
