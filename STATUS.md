# Current handoff state

**Stage:** CPU-validated starter repo; production Neuron integration not yet done.
**No training is running from this delivered repo.** No cloud resource was changed.

## Done
- Bundled relevant original training/evaluation source and historical evidence.
- Preserved the weekend recipe and long-term proof ladder.
- Added deterministic record ordering, local checkpoint verification/resumption,
  strict answer scoring, asset helpers, and fixed-spec/evidence checks.
- Executed local tests; see `validation/VALIDATION.md` for the exact boundary.

## Not yet done
Actual instance inventory, pinned asset downloads/loading, production cached
Neuron sampler, multi-rank training/resume, complete-step timing, final task
inference, and approved durable backup. The presence of old paths is not evidence
that old checkpoints or corpus files survive.

## Next action
From the qualified instance environment: `bash scripts/start_here.sh`, then
`bash scripts/start_codex.sh`. Codex should begin item P0 in `NEXT_STEPS.md`.

## Session continuation fields
- Active job / PID / output directory: none.
- Latest verified scientific checkpoint: none in this delivery.
- Current blocker: production backend has not been implemented/qualified here.
- Exact next command after the bootstrap: inspect the new read-only host report
  and resolve `configs/site.local.json`; then implement P1, not a new research plan.

Replace this section with current facts as work proceeds; keep it under one page.
