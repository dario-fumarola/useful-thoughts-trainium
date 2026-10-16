# Initial assessment — 2026-10-16

## Scope
The owner requested autonomous work, an isolated `dario` workspace and GitHub repository, and for now familiarization and ideas. This session performs setup and reconnaissance only. Original repositories remain unchanged. Private remote: https://github.com/dario-fumarola/dario . The owner explicitly authorized GitHub setup and keeping it updated; this overrides the inherited no-push rule for this private repository only. No model/data/checkpoint uploads are authorized by that decision.

## Verified environment
- Host reports trn2.48xlarge, 16 devices, LNC2, 64 logical cores, 96 GB HBM per device. Per-core memory and partition isolation remain to be qualified.
- 192 logical CPUs, approximately 2 TiB host RAM, 430 GB available root disk at inspection.
- Existing Python: `/home/ubuntu/ema-trainium/.venv/bin/python`, Python 3.12.3.
- Installed metadata: torch 2.9.1, torch-xla 2.9.0, torch-neuronx 2.9.0.2.15.32035+de43f57c, transformers 5.15.1, neuronx-cc 2.27.5334.0+f702b353, numpy 2.5.2. Metadata is not runtime qualification.
- Existing assets: 83 MB mixed tokenized smoke corpus; 8.5 GB smoke checkpoint/optimizer pair at step 3. Neither is established as a released mature base or audited document-level math corpus.
- No active training process was observed. No accelerator computation was launched.
- Original ema revision: `8b907247320e544eba47332bc34754906c47ec4c`; handoff baseline: `06626f4`.
- Bootstrap verified 184 packaged files; 82 tests passed; CPU toy smoke completed. Evidence: `local/checks/bootstrap.4NOPif/`. These are CPU checks only.
- pytest 8.3.5 and helper dependencies installed solely into ignored `local/test-deps/`; shared accelerator environment unchanged.

## Scientific reading
The primary result is R-thought minus C-thought, with R-direct minus C-direct and R-thought minus R-direct also required. Start from actual released Qwen3-1.7B-Base; four prefix-only samples, 1024-action cap, one-token reward, signed leave-one-out update, one fresh update, no auxiliary. Hold data order and initialization paired. Terminal checkpoints are primary. Oracle selection is diagnostic only. Historical synchronization failures limit old conclusions; they do not establish that useful thought training cannot work.

## Working ideas, in dependency order
1. Qualify the existing strict local pretrained loader against installed Transformers and actual pinned weights. Establish CPU evidence, expected keys/ties, then device parity with predeclared tolerances.
2. Reuse the old scatter-cache storage and chunked vocabulary projection narrowly. Build ordinary causal cached decoding with tensor positions and fixed shapes; do not inherit old insertion geometry. Check every sampled action against teacher-forced replay, including stop and cap cases.
3. Inspect installed ZeRO-1 source before wiring updates. Exactly one reducer, FP32 update path, and a real two-rank reference comparison are essential after the historical synchronization issue.
4. Keep sample/replay APIs prefix-only; restrict future tokens to scoring. Pack reader close delimiters immediately after actual thought content. Preserve all signed samples and action masks.
5. Start data provenance and the evaluator before a long run. Existing token shards do not establish document boundaries or exclusions. Use pinned FineMath fallback if raw-document audit cannot be established.
6. Choose topology and terminal steps from measured complete-step timing. Start with the simplest qualified topology; use permitted sequential execution if partitioning costs too much integration time.

## Unresolved before scientific launch
Actual instance cutoff, approved durable checkpoint destination, released model snapshot, audited raw corpus, production driver, all accelerator/distributed/resume evidence. No deadline or backup location has been guessed. No scientific outcomes have been inspected in this session. Initial engineering timebox should begin when implementation starts; it has not been consumed by a background training job.
