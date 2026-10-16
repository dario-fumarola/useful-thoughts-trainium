# Changes from the previous weekend kit

## Complete repo handoff — 2026-10-16
- Included relevant original source and historical evidence; no second upload needed.
- Added root AGENTS.md, opening prompt, concise persistent status/decision files,
  and project/evidence context that a fresh Codex session can use.
- Added explicit pinned-model and bounded streaming-corpus acquisition helpers.
- Added deterministic seed-specific source order with no silent corpus cycling.
- Added complete local checkpoint manifests, integrity verification, and exact toy
  CPU resume tests. Multi-rank/large-model resume is still unqualified.
- Added strict final-number answer scoring and all required downstream contrasts.
- Added normalized n-gram overlap screening, clearly not full decontamination.
- Added actual-asset specification freezing and source/evidence-bound qualification
  checks. These do not fabricate or independently prove hardware evidence.
- Added safe first-run/Codex entry scripts, file-integrity verification, and explicit
  instance-side implementation/operations work.
- Preserved old validation under `validation/prior_kit/`; current validation is separate.
- Removed the unnecessary upper-tail clamp in the Gumbel draw. Uniform RNG remains
  finite precision; sampling/replay must still be qualified on the device. The
  intended categorical policy and scientific hyperparameters are unchanged.
- Documented per-completion normalization as a heuristic when lengths vary; do not
  call it an unbiased estimator of raw expected reward. No silent objective change.

No Qwen evaluation, Neuron training, private cloud action, or new research result
was performed as part of this repo assembly.
