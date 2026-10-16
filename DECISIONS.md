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
