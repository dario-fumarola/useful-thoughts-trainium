# trn2 training bundle — bootstrap a fresh Trainium2 instance

This zip contains everything needed to start a NEW pre-training (vanilla) or
thinking-pretraining run on a fresh trn2 instance, without pulling large
artifacts through GitHub. Built 2026-09-02 on the original box.

## Contents

- `pre-training/` — all training code (train.py, think_train.py, thinking.py,
  think_gen.py = the KV-cached fast sampler, test_think_fast.py = its CPU
  equivalence gate, monitors, oracle/infer eval scripts)
- `eval/` — benchmark scoring (bench_score.py, compare_report.py, 500-example
  jsonl benchmark sets)
- `run_*.sh` — launchers with supervisor/watchdog/monitor plumbing:
  `run_vanilla3.sh` (vanilla ladder), `run_think.sh <run>` (thinking; flags in
  `runs/<run>/config.sh`), `run_ab_fast.sh`
  (fast-vs-slow acceptance gate)
- `data/corpus/` — the tokenized training mixture (~12GB: fineweb-edu +
  finemath + dclm shards, manifest, val split, allowlist)
- `checkpoints/vanilla_ladder/step_0001900.pt` — 870M-token rung (bf16,
  model-only): the run-10c init/frozen-referee checkpoint
- `checkpoints/vanilla_ladder/step_0003040.pt` — 1169M-token rung (newest;
  the run-11 init)
- `checkpoints/think10c_from870M/step_0001400.pt` — the consolidated
  thinking checkpoint (beats its vanilla twin on held-out CE; model-only,
  no optimizer shards — for evals/inits, not exact resume)
- `TRAINIUM_SETUP.md` — how the original instance was provisioned
- `SPEEDUP_THINKING.md` — the fast-pipeline design + implementation log
- `requirements_freeze.txt` — exact package pins of the working venv

## Bootstrap on a fresh trn2

1. Provision per `TRAINIUM_SETUP.md` (Ubuntu 22.04 Neuron DLAMI or manual
   Neuron SDK install; this bundle was built against neuronx-cc 2.27.5334,
   torch-neuronx per the freeze file).
2. Unzip to **exactly** `/home/ubuntu/ema-trainium` — the run scripts
   hardcode `ROOT=/home/ubuntu/ema-trainium`:
       cd /home/ubuntu && unzip trn2_bundle_20260902.zip -d ema-trainium
3. Create the venv and install pins:
       cd ~/ema-trainium && python3 -m venv .venv
       source .venv/bin/activate
       pip install -r requirements_freeze.txt
   (transformers MUST stay at the pinned 5.15.x — think_gen.ScatterCache
   subclasses its Cache API, and the CPU gate `USE_TORCH_XLA=0 python
   pre-training/train/test_think_fast.py` must print ALL EQUIVALENCE TESTS
   PASSED before any fast-gen training.)
4. Start a run (each script refuses to start if another Neuron job owns the
   device, takes the init rung as its argument, and detaches poorly — use
   setsid):
       # vanilla continuation of the ladder:
       setsid nohup bash run_vanilla3.sh > /dev/null 2>&1 &
       # thinking, fast pipeline, from the 870M rung:
       setsid nohup bash run_think.sh <run_name> > /dev/null 2>&1 &
       # thinking, fast, m=4 tournament, from the 1169M rung:
5. Watch `runs/<run-folder>/{STATUS*,MONITOR*.md,ntp_loss.png,*.log}`.
   First step compiles several XLA graphs (minutes); a fresh box has no
   /var/tmp/neuron-compile-cache, so the very first step can take ~15-30 min.

## Gotchas carried over from the original box (hard-won)

- 124GB host RAM, NO swap: the launchers include a watchdog that kills
  neuronx-cc when MemAvailable < 4GB. Do not remove it.
- Killing a supervisor leaves its monitor-loop `sleep 300` child holding the
  flock for up to 5 min — a same-folder relaunch fails silently until it
  expires.
- Never put the literal string "train.py" in an interactive command line
  while a run is live: the device-owner guard greps for it.
- Checkpoints: `.pt` = model (bf16), `.optim_rank*.pt` = ZeRO-1 shards; a
  `--resume latest` needs both, an `--init-from` needs only the model file.
