# Project instructions

## Mission and scope
Establish a controlled mature-model reference for useful self-generated thought.
Then locate its mechanism and earlier onset; study guidance/compression next;
social and safety applications come later. Follow the user's current scope.
This weekend is one C/R comparison, not a sweep of unrelated ideas.

## Read first, in this order
1. `STATUS.md` and `NEXT_STEPS.md`.
2. `context/PROJECT_CONTEXT.md` and `context/EVIDENCE_LEDGER.md`.
3. `configs/weekend.json`, `docs/AGENT_HANDOFF.md`, and `docs/NEURON_IMPLEMENTATION.md`.
4. Relevant code/tests only. Search historical material on demand; do not read
   a multi-megabyte journal before doing the first useful implementation task.

## Source precedence
Current user instructions and this active plan govern the new work. The files in
`legacy/`, historical manuscripts, and the Italian transcript are reference
material. Old no-launch orders, run numbers, gates, cron/watchdog instructions,
paths, and agent forecasts do not govern this new instance. Never execute an
archived launcher. Preserve evidence; do not rewrite old results to fit the plan.

## Scientific invariants
- Actual released Qwen3-1.7B-Base weights; `from_config` is NOT weight loading.
- B0 unchanged; C and R start identically and share seed-specific ordered data.
- Four prefix-only samples, max 1,024 actions, K=1 reward, signed LOO update;
  no best-of-only training, reader/NTP auxiliary, or answer-aware deployment.
- Temperature/masks/stopping must match rollout and replay. No true future in
  writer inputs; stop gradients through reward. One fresh update per group.
- Report direct and thought-enabled inference for C and R. Primary:
  R-thought minus C-thought. Terminal checkpoint, fixed before final testing.
- Small generic CE regressions are costs, not automatic scientific kill gates.
  Oracle gains and nice examples are not deployed-policy evidence.
- Keep both absolute likelihood arms. Retain failures in evaluation denominators.
  Bootstrap items/documents, not repeated samples as independent observations.
- A model-size change, objective change, data change, or curriculum needs a new
  explicit experiment identity; do not silently change the recipe to make it fit.
  The documented 0.6B fallback and sequential topology may be used before outcome
  inspection when their qualification conditions apply; record the decision.

## Engineering and permissions
Work only in this repo and authorized local asset/output directories. Existing-node
training within the resolved budget is the intended task after qualification.
Do not create/terminate instances, buy capacity, change IAM/networking, publish
repositories, use Bedrock, call paid teacher services, or upload to an unapproved
bucket. Never copy secrets into code, logs, Git, or manifests. Do not reinstall
or upgrade the working accelerator stack as a convenience fix. Resolve exact
installed APIs; public inference support does not qualify this training path.

Use one gradient-reduction owner. An XLA ZeRO-1 step already reduces gradients;
do not double-reduce. Verify real updates, cache/replay parity, and resumption.
The CPU reference is not the production sampler: no full-prefix recompute for
1,024 actions on every rank as a supposed optimized implementation.

## Work discipline
Implement one ordinary-causal production driver under `train/`. Prefer the
verified existing runtime and bounded reuse over new kernels/serving systems.
Use `NEXT_STEPS.md` as a short queue. Start with the exact next uncompleted task.
Run `bash scripts/run_checks.sh` after relevant code changes. Do not edit tests
to hide a failing invariant. Hardware checks require hardware evidence; never
mark synthetic fixtures or old packaged outputs as a pass on this instance.

Timebox initial integration. Stop long-launch plans for numerical/layout/data
failures or the actual deadline, not because early thoughts look unattractive.
If blocked, identify the smallest repair, continue independent safe work, and
report the blocker. Do not spend idle time rewriting papers or inventing more
calibration ledgers. No infinite automatic relaunch loop.

Before compaction/session exit, update `STATUS.md` with exact next command,
active PID/job/output/checkpoint, what was actually tested, and remaining risk.
Record material decisions in `DECISIONS.md`; keep both concise. Make small local
Git commits. Never push without the owner's explicit instruction.
