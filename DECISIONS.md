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
