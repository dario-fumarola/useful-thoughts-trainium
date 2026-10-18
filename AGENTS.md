# Project instructions

## Mission and scope
Establish a controlled mature-model reference for useful self-generated thought.
Then locate its mechanism and earlier onset; study guidance/compression next;
social and safety applications come later. Follow the user's current scope.
On 2026-10-17 the owner stopped the initial C/R study and redirected work to
the initialization interface, candidate oracle, and iteration speed. The old
256-update run order and CODEX_NEXT_DAY.md launch instructions are superseded.
Do not resume old training/evaluation launchers. Read NEXT_STEPS.md first.

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
- The original study used four prefix-only samples, max 1,024 actions, K=1
  reward and signed LOO updates. Preserve that study's identity and results.
  New diagnostic/objective/interface experiments need explicit new identities;
  the owner authorizes adapting the research direction.
- Temperature/masks/stopping must match rollout and replay. No true future in
  writer inputs; stop gradients through reward. One fresh update per group.
- Report raw, empty-format and content controls before claiming useful thought.
  Improvements over a degraded comparison alone do not establish capability.
  Keep official final outcomes out of interface development.
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
The owner explicitly authorized maintaining this GitHub repository and making it
public as `dario-fumarola/useful-thoughts-trainium` on 2026-10-16. Commit and push
reviewed source, documentation, and small reports; keep local assets out of Git.
Do not create/terminate instances, buy capacity, change IAM/networking, use
Bedrock, call paid teacher services, or upload to an unapproved bucket.
Never copy secrets into code, logs, Git, or manifests. Do not reinstall
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

Use small measured experiments to repair the initial reward landscape before
another long training run. The 2026-10-18 owner direction requires one batched
full-span interface oracle on>=256 DEV documents, then>=5000 warmstart examples;
the unlaunched512-example pilot is superseded. STATUS entries are<=15 lines per
experiment; hashes/archives belong in JSON. Review once per question. Probes on
<=4 cores do not require a serial release gate; jobs over one accelerator-hour do. Profile actual stage costs. A zero mixer weight gives
zero interface tax but does not establish content utility or a writer gradient.
If blocked, identify the smallest repair, continue independent safe work, and
report the blocker. Do not spend idle time rewriting papers or inventing more
calibration ledgers. No infinite automatic relaunch loop.

Before compaction/session exit, update `STATUS.md` with exact next command,
active PID/job/output/checkpoint, what was actually tested, and remaining risk.
Record material decisions in `DECISIONS.md`; keep both concise. Make small local
Git commits and keep the owner-authorized remote updated.
