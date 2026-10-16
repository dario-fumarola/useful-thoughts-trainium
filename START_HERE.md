# Start here

## 1. Open the isolated repository

```bash
git clone https://github.com/dario-fumarola/useful-thoughts-trainium.git
cd useful-thoughts-trainium
# Activate a compatible Python environment, then run the CPU checks.
bash scripts/run_checks.sh
```

The existing instance workspace is `/home/ubuntu/dario`; use that checkout when
continuing its work. Read `STATUS.md` before touching running jobs. CPU checks
require PyTorch, NumPy, pytest and Transformers, and record results under
`local/checks/`. They do not qualify accelerator execution or install packages.

The original delivery manifest and `scripts/start_here.sh` are retained for
provenance. Their original checksums differ after intentional development. Use
Git to inspect current changes; do not regenerate that manifest to conceal them.
Do not replace the working Torch/XLA/Neuron stack to run these checks.

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
The owner has extended the work to weeks. Current finite job bounds and the
prospective science schedule are recorded in `STATUS.md` and `DECISIONS.md`.

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
