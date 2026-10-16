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
