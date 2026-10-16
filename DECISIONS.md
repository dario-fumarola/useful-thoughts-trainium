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
