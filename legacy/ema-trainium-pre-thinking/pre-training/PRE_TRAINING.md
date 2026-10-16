# Task: Set up a from-scratch LLM pretraining PoC on AWS Trainium

## Context

You are on an AWS EC2 Trainium instance (trn1 family, but check if we are on trn2 or later). I want to pretrain a ~0.6B parameter LLM **from scratch** as a proof of concept. The plan:

- Take **Qwen3-0.6B** from Hugging Face, keep its architecture and tokenizer, but **re-initialize all weights from scratch** (no pretrained knowledge should survive). Do not hardcode assumptions about the architecture — load the config and inspect it (hidden size, layers, attention variant, tied embeddings, etc.) and derive everything from there.
- Train on a ~3B token mixture (details below) with a standard next-token-prediction objective.
- This is a PoC: prioritize a pipeline that runs end-to-end and is easy to re-run over squeezing out maximum performance. But do follow standard pretraining best practices (proper init, LR warmup + cosine decay, gradient clipping, bf16).

## Constraints

- **Storage cap: 500 GB total on this volume.** Check free disk space before and during downloads (`df -h`). Never download a full multi-TB dataset — use per-subset sampling/streaming, and delete raw data after tokenization if space gets tight. Tokenized shards for 3B tokens are ~6 GB at 2 bytes/token, so the final footprint is small; the risk is only in careless raw downloads.
- **Hardware: AWS Trainium.** This is not CUDA. First verify what is installed (`neuron-ls`, `torch-neuronx`, `neuronx-cc`, `optimum-neuron`, NeuronX Distributed). Prefer the most standard supported path for pretraining a HF-architecture model on Trainium (e.g. optimum-neuron / NeuronX Distributed pretraining flow). If the Neuron SDK is missing or broken, stop and report what's needed rather than silently falling back to CPU.
- Check how many Neuron cores are available and size data parallelism / batch size accordingly. Be aware of Neuron's compilation step: warm up compilation with a short run before the real run, and use the persistent compile cache.

## Data mixture (~3B tokens total)

| Weight | Dataset (Hugging Face) | Subset/split | ~Tokens |
|--------|------------------------|--------------|---------|
| 70% | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | 2.1B |
| 15% | `HuggingFaceTB/finemath` | `finemath-4plus` | 450M |
| 10% | `mlfoundations/dclm-baseline-1.0` | any shard subset | 300M |
| 5%  | `HuggingFaceTB/smollm-corpus` | `python-edu` | 150M |

Notes:
- Use `datasets` in **streaming mode** (or download only the number of parquet shards needed) — count tokens as you go with the Qwen tokenizer and stop each source once its budget is hit, plus ~2% extra for a held-out eval split.
- If `python-edu` turns out to store code by reference (e.g. blob IDs requiring separate download), and fetching it is convoluted, substitute a comparable HF code-education dataset (e.g. a Stack-Edu Python subset) and note the substitution.
- Verify each dataset's current schema on the Hub before writing the pipeline; don't assume column names.
- The account is already logged in to HuggingFace. In case it is not, you can check .env, where I stored an access token. If neither works, stop and ask me.

## Pipeline to build

1. **Environment audit** — Neuron SDK, Python env, disk, memory. Report findings before proceeding.
2. **Download + tokenize** — one script that streams each source, tokenizes with the Qwen3-0.6B tokenizer, appends an EOS between documents, and writes packed uint16/uint32 binary shards (nanoGPT-style `.bin` + index, or memory-mapped arrays). Keep each source in **separate shard files** tagged by domain.
3. **Mixture dataloader** — samples documents/sequences from the four domains with the weights above (sampling, not concatenation, so every batch reflects the mix). Fixed sequence length (use the model's native context or 2048 to keep compilation cheap). Deterministic given a seed.
4. **Held-out eval** — a small per-domain validation set (~10M tokens/domain) and an eval function reporting loss **per domain**, not just aggregate.
5. **Model init** — load Qwen3-0.6B config, instantiate the model, re-initialize all parameters with standard init (e.g. normal(0, 0.02) with depth-scaled residual projections, or the init specified in the model's own config/class). Assert no pretrained weights remain (e.g. compare a checksum against the pretrained checkpoint). Print a parameter-count summary.
6. **Training loop** — bf16, AdamW, warmup + cosine decay, grad clipping at 1.0, gradient accumulation to reach a reasonable effective batch (~0.5–1M tokens/step), periodic checkpointing to disk (keep last N to respect the storage cap), loss logging (per-domain when eval runs), and clean resume-from-checkpoint.
7. **Smoke test** — before the full run, do a short run (~50–100M tokens or a fixed step count) to validate compilation, throughput, loss going down, and checkpoint/resume. Report tokens/sec and an ETA for the full 3B-token run. **Stop after the smoke test and report** — do not launch the full run without my confirmation.

## Deliverables

- `data/` scripts: `download_and_tokenize.py`, shards on disk with a manifest (token counts per domain).
- `train/` scripts: dataloader, model init, training loop, eval.
- A `README.md` documenting: environment found, how to run each stage, throughput measured, ETA for full run, disk usage.
- A short report of any deviations you had to make (dataset schema changes, Neuron limitations, substitutions).

## Style

- Ask before anything destructive or that downloads >50 GB in one go.
- Prefer plain, hackable Python scripts over heavy frameworks — I will later extend this loop with an RL objective interleaved with the LM loss, so keep the training step function modular and easy to modify.
- Prefer standard choices in terms of pre-training, i.e., a pipeline that follows the standard recipes of pre-training. We value stability for this poc.