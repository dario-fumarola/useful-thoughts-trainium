# Decision log

## 2026-10-16 — handoff baseline
- Start with one mature-model reference condition; do not claim novelty from a
  mature-model success alone.
- Use the prior weekend specification, not a new objective invented during packaging.
- Keep the training/inference/deployment/efficiency claims separate.
- Reuse audited runtime components, not old launchers or entire damaged lineages.
- Package original source and project context so a second upload is unnecessary.
- Add local resumption/scoring/specification tools; no hardware qualification claim.
- Remove the sampler's unnecessary finite upper-tail Gumbel clamp; the intended
  temperature-softmax categorical policy remains unchanged. Device numerical parity
  still needs measurement. This is documented in CHANGELOG.md.

For each later scientific change, record: date, old/new choice, reason,
whether outcomes had been inspected, new experiment ID, and supporting artifact.
Routine formatting changes do not need a decision entry.

## 2026-10-16 — isolated owner-authorized workspace
- Clone the handoff into `/home/ubuntu/dario`, preserving the original sources and history.
- Owner explicitly requested a GitHub repository and updates: created private `dario-fumarola/dario`; this authorizes commits/pushes to that repository, overriding inherited no-push instructions for this scope. Keep assets, credentials, compiler output and run data ignored.
- Reuse existing accelerator environment unchanged; install missing test helpers only under ignored `local/test-deps/`.
- Current session is familiarization only; retain fixed scientific recipe and defer implementation/launch. No research outcomes inspected.

## 2026-10-16 — implementation phase
- Owner extends available time to weeks and explicitly authorizes review/implementation agents and evidence-based adaptation. Preserve one interpretable primary C/R experiment; do not treat six-hour integration or weekend cutoff as current instance expiry. Use bounded diagnostic attempts.
- Download actual released Qwen3-1.7B-Base at immutable commit. Preserve its declared embedding/head tie explicitly after XLA transfer; no independently trained matrices are retied.
- Before scientific outcomes, declare writer support as tokenizer-supported IDs only (mask267model-only rows), identically in sampling/replay and C/R deployment. Reader likelihood and C loss retain released full vocabulary.
- Install no accelerator upgrades. Use hash/version-gated process-local fix for an observed compiler artifact-path bug and explicit restored syncfree scalar-state device placement; preserve failures and numeric-resume limitations.
- Correct URL identity before freeze: preserve content-bearing query parameters. Use audited v2 FineMath asset; retain v1 artifact as evidence of the bug.
- Qualify task answer serialization on TRAIN items only; repeat the original question after thought close to prevent the reader from continuing into a new few-shot example. Preserve failed initial output.

## 2026-10-16 — long-trace precision diagnosis
- Full-cap BF16 cache/replay maxlogp error .21841 exceeds predeclared .2 bound despite tiny typical error; keep failed evidence. FP32 vocabulary head alone does not repair it: exact old trajectory still differs .22614. Do not raise tolerance. Test full FP32 compute at the already declared .001 bound, with both random rollouts and the fixed old failing trajectory, before selecting production precision.
- Replace per-action output lists with static on-device buffers and check only an alive scalar every32steps. Stop padded computation after all four thoughts terminate; keep1024output shape/cap, sampled-stop masks and raw IDs. Count padded work separately. CPU parity/stopping tests pass; new hardware path remains pending.

## 2026-10-16 — select FP32 production candidate before scientific outcomes
- Full FP32 passes the exact old failing actions: maxlogp error2.83718e-5 under the original tighter.001 bound; the same oldoutlier is3.576e-6. FP32head alone fails, so headquantization is not sufficient explanation; BF16 trunk/cache/full arithmetic causes the discrepancy in this stack.
- Select full FP32 parameters/compute for BOTH primary arms; update proposal identity to mature-reference-2026-10-16-v2-fp32 before any scientific launch or final evaluation. Objective,model,data,G4,L1024,K1,temperature,batch,LR remain unchanged. Complete timing and fullmodel resume remain required at selected precision/topology.
- Deep SSD preflight found nonzero highentropy middle blocks despite absent FS signatures; abort before any device write. Continue EBS qualification; investigate read-only before treating devices as disposable scratch.

## 2026-10-16 — initialize verified unused instance-local scratch
- Read-only boot/device evidence establishes fresh currentinstance, allfourinstance-store devices with zero kernelwrites/discards, nofilesystem/partitions/mounts/holders. Middleblockhighentropy is not a zero-read guarantee for discardedNVMe; it alone doesnotestablishuserfiles.
- With existing ownerhardwareautonomy, initialize ONLY `/dev/nvme0n1` as ext4 at `/home/ubuntu/dario/local/nvme`; repeat model,size,zero-write,no-signature/partition/mount guards immediatelybeforewriting. Receipt `validation/nvme_preparation_20261016.json`. OtherthreeNVMe andEBS untouched.
- Use scratch forlargeintermediatecheckpoint/compilerfiles. Scientific terminalcheckpoints and dailylatestcomplete backups also copiedtoEBS; instance-localNVMe is not externaldurablebackup. No cloudcapacity or accountchanges.

## 2026-10-16 — owner-authorized public repository
- Owner explicitly requests public visibility and a clearer name. Rename remote to `dario-fumarola/useful-thoughts-trainium`; publish after tracked-history credential scan and applicable third-party notice/source repair. No weights, corpus, credentials, checkpoints or raw local outputs enterGit.
- Preserve original research history and attribution. Public release is not a new scientific result or blanket relicensing of supplied historical material.

## 2026-10-16 — prospective execution budget with weeks available

- The owner replaced the original weekend constraint with weeks of available
  time. Adopt a 14-day local planning horizon from the first scientific launch;
  this is an autonomous job limit, not a claim about instance shutdown.
- Allow up to 240 hours for the two-seed training schedule, including the
  existing 15% reserve. Keep the remaining horizon for evaluation, verification,
  and recovery. Choose the largest predeclared step count in 256/512/1024/2048
  that fits actual complete-step timing and fresh data. Use timing only; neither
  qualification utility nor final outcomes select exposure.
- After C/R isolation and full-model resume qualify, C1/C2 can run consecutively
  on one partition while R1 runs on the other; then R2 uses the freed partition.
  Check this complete schedule against measured C and R times. If concurrency
  does not qualify, budget the sequential schedule explicitly.
- Before multi-day runs, arrange complete checkpoints at least daily, with
  verified EBS copies of the latest complete state and each terminal result.
  Current immutable qualification jobs keep their original checkpoint schedule.

## 2026-10-16 — runtime identity and checkpoint cadence before science

- Device-memory diagnostics show that the documented legacy Neuron cache limit
  does not control residency in this stack. The installed XLA LRU setting does;
  its full-backward adequacy is still pending. Bind the actual allowlisted cache,
  compiler and CPU-thread environment to each arm's qualification/specification,
  run and resume state. Permit separately declared arm policies only with their
  own measured costs. This changes operations, not the scientific objective.
- Before multi-day science, add synchronized time-triggered complete saves every
  12 hours at update boundaries, in addition to the original milestones. Record
  unmeasured C reward fields and unaudited replay errors as null. Qualify the new
  source's actual warm updates and resume before freezing; CPU tests alone do not
  establish that the added save-decision collective works on the accelerator.
- Preserve the original failed CPU-vector-norm comparison. A full-gradient
  Float64 reduction exposed a biased FP32 reference reduction and passed the
  original tolerance against device update1. Recompute the three-update CPU
  reference with accurate clipping as a separately identified lineage; do not
  change the declared bound or overwrite the original result.

## 2026-10-16 — repair observed device sequence normalization before R

- The installed XLA LRU limit allows the original full-cap backward to fit in
  memory. Its norm nevertheless differs from the independent CPU reference by
  roughly1024 times. Preserve that failed numerical result; R remains gated.
- A reduced device test establishes that the raw int64 mask count is correct,
  but its `clamp_min(1)` returns -1. Both signed losses and gradients reverse
  direction and lose the length normalization. Materializing loss first does
  not repair this defect. Use `mask.float().sum().clamp_min(1.0)` instead: counts
  through1024 are exactly representable. All six fixed-cap1024/head-chunk32
  cases (active lengths1/8/1024, both materialization paths) pass the original
  scalar/full-input/full-head gradient bounds. Separate short-shape failures
  remain retained. Receipt: `validation/p2_integer_length_clamp_20261016.json`.
- This is an implementation correction to the existing signed normalized
  objective, before any scientific R run. Require actual repaired full-Qwen
  loss/norm and signed-gradient sample agreement before launching R qualification.
  Keep the original norm bound; add prospectively declared sample cosine/L2
  checks because a norm cannot detect a sign reversal.
- Qualification R saves every rank/micro's rollout/reward tape after backward,
  and saves the first post-resume baseline checkpoint (R14:3/4/7/14). A resume
  also exports actual restored state before sampling, then verifies the first
  resumed actions and full post-update state under the original tight bounds.

## 2026-10-16 — preserve stochastic continuation across checkpoint observation

- The completed C32 fresh-process comparison passes full model/master/moment
  numerical bounds but fails exact device RNG on all 32 ranks. Each mismatch is
  exactly one installed XLA MarkStep transition. The old path resets the saved
  seed before its final load sync; that sync advances even a deterministic graph.
- Move CPU/device RNG restoration after all device loading and its final sync.
  Checkpoint publication/readback captures the incoming RNG at a settled boundary
  and restores it after all exports and barriers. Preserve unsigned 64-bit seeds;
  add no sync after resetting. Do not substitute per-record reseeding or weaken
  exact stochastic-resume criteria.
- Actual installed Neuron barriers differ from generic upstream XLA rendezvous:
  the measured no-payload barrier and checkpoint exports do not advance RNG.
  A bounded hardware negative control reproduces the old seed/next-draw failure;
  the corrected order reproduces both subsequent random draws and optimizer
  state exactly. Require the integrated helper and then full C/R resume evidence
  before scientific launch; the small probe alone is not full-model qualification.
