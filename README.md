# Useful thinking on Trainium
**Dario Fumarola · October 2026 · private research handoff**

Establish one useful, controlled mature-model thought-training result. Separate
training benefit from inference benefit; then study earlier onset, less guidance,
compression, and finally social or safety applications.

**This repo includes the previous kit, relevant original source, research context,
and tested local utilities. It is not a hardware-qualified Trainium trainer.**
The remaining engineering task is the optimized Neuron driver described in
`docs/NEURON_IMPLEMENTATION.md`. Codex should implement and qualify that path,
then run the fixed comparison—not restart the research from scratch.

## Start on the instance

Activate the instance's existing working Neuron environment first. From this
folder:

```bash
bash scripts/start_here.sh
bash scripts/start_codex.sh
```

The first command checks the package, runs local tests when dependencies are
present, and records the host environment. It does not install packages, fetch
models, use a NeuronCore, or launch training. The second opens your installed
Codex CLI with `CODEX_START_PROMPT.md`; normal permissions and authentication
remain in force. Neither command provisions a cloud resource.

Missing Codex? Start it through your usual installation/authentication process,
open this directory, and paste the contents of `CODEX_START_PROMPT.md`. Do not
put credentials into this repository.

**Read `START_HERE.md` for the exact sequence.** No second archive is needed for
the included implementation/context; model weights, corpus, and instance-local
credentials are deliberately not bundled.

## What to open

| File | Purpose |
|---|---|
| `AGENTS.md` | Short persistent instructions for Codex |
| `CODEX_START_PROMPT.md` | The first task to give Codex |
| `STATUS.md` / `NEXT_STEPS.md` | Current state and next bounded work item |
| `context/PROJECT_CONTEXT.md` | Goal, past experiments, interpretation corrections |
| `context/EVIDENCE_LEDGER.md` | What is supported, reported, or still uncertain |
| `docs/WEEKEND_PLAN.md` | Fixed primary comparison and experimental choices |
| `docs/AGENT_HANDOFF.md` | Objective semantics and required qualification |
| `docs/NEURON_IMPLEMENTATION.md` | Concrete production work, by dependency |
| `docs/OPERATIONS.md` | Run, stop, resume, storage, and handoff procedure |
| `docs/RESEARCH_ROADMAP.md` | What we must prove next, step by step |
| `validation/VALIDATION.md` | What was actually executed during packaging |

## Included implementation

`weekend_thinking/` contains causal-label and signed-policy primitives, the toy
reference loop, deterministic data ordering, verified local checkpoint files,
paired analysis, strict numeric answer scoring, hardware inventory, and an
asset/specification lock. `scripts/` exposes those functions plus explicit model
and corpus acquisition. `tests/` checks them without network or a Neuron device.

`legacy/ema-trainium-pre-thinking/` contains relevant original training/evaluation
source, the current manuscript sources, selected probe outputs, and the historical
journal/runbook. Treat it as evidence and reusable implementation material, not
as active instructions. Historical launch scripts are not active shell scripts.
The earlier manuscript and project-only call excerpt are under `context/`.

## Primary experiment

Start unchanged base **B0**, ordinary continuation **C**, and thought training
**R** from released **Qwen3-1.7B-Base** weights. C and R share ordered source
records within each seed. Evaluate both weights with direct and thought-enabled
inference. The primary downstream contrast is **R-thought minus C-thought**.
Always report **R-direct minus C-direct** and **R-thought minus R-direct**.

Proposed R: four prefix-only thoughts, cap 1,024 actions, next-token reward,
signed leave-one-out updates, no NTP or reader auxiliary. These are unproven
starting hyperparameters, not a guaranteed reproduction of RLP.

Do not reject a downstream benefit automatically because generic-text loss
increases slightly. Do not claim efficiency without an appropriate resource
comparison. A positive oracle-selected tail is not a deployed policy result.

## Validation boundary

The CPU tests and local save/resume checks run. Model/dataset network helpers are
not live-service qualified here. No Qwen forward pass, Neuron compilation,
cached accelerator decoding, distributed update, HBM measurement, multi-rank
resume, final benchmark, or cloud backup has been performed for this delivery.
**The production driver is the first substantial task, not a hidden finished
component.** All example scores/timings and toy checkpoints are synthetic.

This private snapshot has no remote repository configured. Do not publish it or
upload its data/checkpoints to an unapproved destination.
