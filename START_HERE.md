# Start here

## 1. Put this one folder on the instance

```bash
unzip trainium_thinking_repo.zip
cd trainium_thinking_repo
# Activate the already working Neuron Python environment here.
bash scripts/start_here.sh
```

The script is intentionally non-destructive. It verifies the delivered files,
checks dependencies, runs CPU tests, and records a read-only host report under
`local/checks/`. It does not replace Torch, XLA, Transformers, or Neuron packages.
If a helper dependency is absent, it reports that fact rather than installing a
new accelerator stack. CPU checks are not accelerator qualification.

After intentional source edits, a delivery checksum mismatch is expected. Use
`git diff` to inspect changes and `bash scripts/run_checks.sh` to rerun tests;
do not regenerate the original delivery manifest to conceal changes.

## 2. Start Codex from the repo root

```bash
bash scripts/start_codex.sh
```

It supplies the full opening task from `CODEX_START_PROMPT.md` and uses a local
session lock to prevent two coding agents from editing the same worktree at once.
It does not weaken your Codex permission settings or supply credentials.

Alternatively run `codex` normally and say:

> Read AGENTS.md and CODEX_START_PROMPT.md, then continue from STATUS.md. This is
> the already authorized Trainium instance. Implement and qualify the fixed
> experiment, preserving the research question and validation boundaries.

## 3. Codex resolves the actual machine and assets

`configs/site.example.json` is copied to ignored `configs/site.local.json` if
missing. Resolve model, corpus, runtime, data paths, core visibility, deadline,
and approved backup destination there. Paths in old logs are historical; do not
assume their disks, buckets, checkpoints, or credentials survive.

The important unknowns are the actual environment, available model/corpus,
capacity cutoff, and backup destination. Codex should discover what it can
locally. Ask you only for genuinely unavailable permissions or a decision that
changes the science. Do not ask you to choose every runtime knob.

## 4. Follow the bounded work order

**Inventory → assets → reference parity → distributed/resume checks → throughput
→ freeze the paired run → first seed → confirmation → evaluation and backup.**

`NEXT_STEPS.md` gives the next concrete deliverable at each stage.
`docs/NEURON_IMPLEMENTATION.md` names the code to reuse and what must be new.
The six-hour qualification allowance is a planning cap, not permission to waive
a failed check or a prediction that all integration will finish in that period.

No long job before a real `local/qualification.json` exists for the actual code
and fixed assets. The template contains only pending values. It is a compact
checklist, not another open-ended audit project.

## 5. End every session with a usable handoff

Keep `STATUS.md` short: completed work, active process/job identity, checkpoint,
remaining blocker, and the exact next command. Record scientific changes once
in `DECISIONS.md`. Save real evidence under `local/`; never overwrite the
packaged validation with claimed accelerator success.

The research chain lives in `docs/RESEARCH_ROADMAP.md`. It must survive even if
Codex's conversational context is reset.
