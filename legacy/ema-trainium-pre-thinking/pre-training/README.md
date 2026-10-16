# Pretraining Qwen3-0.6B from scratch on AWS Trainium

A from-scratch pretraining PoC: Qwen3-0.6B's architecture and tokenizer, all
596M weights re-initialised, trained with next-token prediction on a ~3B token
mixture. Plain `torch-xla` + `torchrun` — no training framework — so the loss
function stays a single small function you can extend.

---

## 1. Environment found

Audited before any code was written; everything below was verified on this box.

| | |
|---|---|
| Instance | `trn1.2xlarge` — 1 Neuron device, **2 NeuronCores** |
| HBM | 32 GB per device = **16 GB per core** (see the gotcha below) |
| Host | 8 vCPU, 30 GB RAM, 484 GB disk |
| OS / Python | Ubuntu 24.04.4, Python 3.12.3 |
| Driver | `aws-neuronx-dkms` present, `neuron-ls` healthy |

Python stack (matches the known-good set in `../TRAINIUM_SETUP.md`):

```
torch 2.9.1          torch-xla 2.9.0       torch-neuronx 2.9.0.2.15.32035
neuronx-cc 2.27.5334 libneuronxla 2.2.17544  islpy 2026.1   (NOT 2026.2.x)
transformers 5.15.1  datasets 5.0.1        numpy 2.5.2
```

> **`neuron-ls` reports 32 GB, but a single core can only use 16 GB.** The
> compiler is explicit: `exceeds HBM limit of 16.00GB for Trn1`. This
> contradicts the note in `TRAINIUM_SETUP.md` saying one core can allocate the
> full 32 GB, and it is the root cause of most of the work in section 5.

---

## 2. Quick start

```bash
source ../.venv/bin/activate          # must ACTIVATE, not call .venv/bin/python
export PATH=/opt/aws/neuron/bin:$PATH

# Stage 1 - build the corpus (~35 min, writes 12 GB)
python data/download_and_tokenize.py --out data/corpus --total-tokens 3e9

# Stage 2 - inspect the model that will be trained
python train/model_init.py --verify-scratch

# Stage 3 - check the mixture sampler
cd train && python data.py --corpus ../data/corpus

# Stage 4 - train (from pre-training/train/)
torchrun --nproc_per_node=2 train.py \
  --corpus ../data/corpus --micro-bs 1 --grad-accum 128 --seq-len 2048 \
  --zero1 --loss-chunks 4 --max-steps 5722 --warmup-steps 200 \
  --lr 3e-4 --min-lr 3e-5 --eval-every 200 --ckpt-every 500

# resume after an interruption
torchrun --nproc_per_node=2 train.py ... --resume latest
```

---

## 3. Data

`data/download_and_tokenize.py` streams each source, tokenizes with the
Qwen3-0.6B tokenizer, separates documents with `<|endoftext|>` (151643 — *not*
`eos_token_id`, which is 151645 `<|im_end|>`, the chat terminator), and writes
packed **uint32** shards. uint32 is mandatory: the vocab is 151936, so uint16
would silently wrap every id above 65535.

Raw source files are deleted immediately after tokenizing, so peak disk stays
near one source shard plus the growing output.

| Domain | Source | Weight | Train tokens | Val tokens |
|---|---|---:|---:|---:|
| fineweb-edu | `HuggingFaceFW/fineweb-edu` `sample/10BT` | 73.7% | 2,210.5M | 10.0M |
| finemath | `HuggingFaceTB/finemath` `finemath-4plus` | 15.8% | 473.7M | 10.0M |
| dclm | `mlfoundations/dclm-baseline-1.0` | 10.5% | 315.8M | 10.0M |
| **total** | | | **3,000.0M** | **30.0M** |

Held-out tokens are taken from the *head* of each stream and never written to
the train shards, so the two cannot overlap.

**Measured**: 1.4–1.6M tokens/sec tokenization; whole corpus in ~35 min; **12 GB
on disk**.

The mixture dataloader (`train/data.py`) *samples* per sequence from the domain
weights rather than concatenating, so every batch reflects the mixture. Sampling
is fully determined by `(seed, rank, step, micro-step)`: reproducible, and each
data-parallel rank draws a disjoint stream. Verified empirically at
0.726/0.173/0.102 against targets of 0.737/0.158/0.105.

---

## 4. Model init

`train/model_init.py` loads the config from the Hub and derives everything from
it — nothing about the architecture is hardcoded.

```
qwen3 | hidden 1024 | 28 layers | heads 16/8 (GQA) | head_dim 128
intermediate 3072 | vocab 151936 | tied embeddings
total 596,049,920 params  (155.6M embedding + 440.5M non-embedding)
```

Init is N(0, 0.02) with the standard depth-scaled residual rule: the 56
projections writing into the residual stream (`o_proj`, `down_proj` × 28 layers)
get `0.02/sqrt(2*28) = 0.00267`, so residual variance does not grow with depth.
RMSNorm weights are ones.

`--verify-scratch` downloads the real checkpoint and compares every tensor:

```
tensors compared : 311
identical to ckpt: 0
PASS - no pretrained weights remain
```

---

## 5. Five Trainium-specific fixes

Each of these was a hard failure, not a tuning choice.

**1. `eager` attention, not SDPA.** Qwen3 RMSNorm-normalises q and k per head.
Under bf16 autocast those stay fp32 while v is cast to bf16, and SDPA rejects
the mix outright: `Expected query, key, and value to have the same dtype`.

**2. Gradient checkpointing is mandatory.** The full fwd+bwd graph peaks at
**25.7 GB** against a 16 GB limit. It also needs `use_reentrant=True,
preserve_rng_state=False`: the non-reentrant path calls
`_get_device_module("xla")` unconditionally, which resolves `getattr(torch,
"xla")` and raises. Qwen3 has `attention_dropout=0.0` and no other dropout, so
recomputation is deterministic and the RNG state is genuinely unnecessary.

**3. `torch._register_device_module("xla", torch_xla)`.** Checkpoint's backward
enters `torch.random.fork_rng()`, which resolves the device module *before* it
honours `enabled=False`. With `preserve_rng_state=False` the module is never
used — registering it only satisfies the lookup.

**4. Chunked cross-entropy for seq_len 2048.** The logits for one sequence are
`2048 × 151936 × 2 = 622 MB` in bf16 — larger than every activation in the
28-layer trunk combined, and the single allocation that overflowed the core.
`--loss-chunks 4` computes the vocab projection on one quarter of the sequence
at a time, each slice checkpointed so its logits are freed and recomputed in
backward. Verified numerically identical to the unchunked loss (diff 9.5e-07).

**5. `torch_xla.amp.syncfree.AdamW`, not `torch.optim.AdamW`.** This one is
worth the most attention. Stock AdamW pushes the learning rate through
`_get_value()`, which calls `.item()` in eager mode — turning the LR back into
a **baked graph constant**. A warmup/cosine schedule therefore changes the graph
every step and recompiles it:

| | step time | throughput |
|---|---:|---:|
| stock AdamW, changing LR | **257 s** | 64 tok/s |
| syncfree AdamW, changing LR | **3.9 s** | 4,335 tok/s |

A **66× difference**, and it presents as "training is mysteriously slow", not as
an error. This is the `TRAINIUM_SETUP.md` "everything recompiles constantly"
symptom with a non-obvious cause.

---

## 6. Training loop

`train/train.py`. bf16 autocast with fp32 master weights, AdamW
(β=0.9/0.95, wd=0.1, no decay on biases or norms), linear warmup → cosine decay
to `--min-lr`, gradient clipping at 1.0, gradient accumulation, per-domain eval,
checkpointing with pruning, and resume.

Two XLA rules shape the file:

* **`torch_xla.sync()` once per *micro*-step**, not per optimizer step. Syncing
  per optimizer step would fold all 128 accumulation micro-steps into one
  enormous graph. This way there are exactly two graphs: forward/backward, and
  the optimizer update.
* **No `.item()` in the hot loop.** Loss and grad-norm stay on device and are
  read only at logging boundaries; each read is a host sync.

**The objective is deliberately isolated** in `lm_loss(model, x, y, ...)`, and
`train_step()` does nothing but loop micro-batches over it. An RL term added
later slots in beside `lm_loss` without touching the loop.

### Checkpointing under ZeRO-1

ZeRO-1 shards Adam's `m` and `v` across ranks, so **optimizer state is not
replicated**. Model weights are written once by the master; each rank writes its
own optimizer shard (`step_NNN.optim_rankK.pt`). A master-only save would
silently restore `1/world_size` of the optimizer state. Resume refuses to run if
the world size changed, rather than corrupting the state.

Each checkpoint is ~9 GB (3.0 GB model + 3.0 GB × 2 shards); `--keep-checkpoints
2` bounds this at ~18 GB.

---

## 7. Measured results

Smoke test: 30 steps on the real 3B corpus, effective batch **524,288
tokens/step** (micro_bs 1 × grad_accum 128 × 2 ranks × 2048).

| | |
|---|---|
| Throughput | **4,335 tok/s** sustained (121 s/step, flat — no recompilation) |
| First step | ~500 s including compilation; cached in `/var/tmp/neuron-compile-cache` |
| Loss | 12.17 at init (= ln 151936, exactly random) → falling monotonically |
| Per-domain eval @ step 9 | fineweb-edu 8.70 · finemath 8.70 · dclm 8.85 |
| Corpus on disk | 12 GB |
| Checkpoint | ~9 GB each |

### ETA for the full 3B-token run

`3e9 / 4,335 tok/s ≈ 692,000 s` = **~192 hours ≈ 8 days**, i.e. **5,722 steps**
at 121 s.

**This scales out with no code change.** The model needs only data parallelism
— it fits in one core, so tensor/pipeline parallelism (the thing NeuronX
Distributed exists to provide) is not required. `--nproc_per_node=32` on a
trn1.32xlarge is a launch-flag change and should bring the run to roughly half a
day; `--nnodes` extends the same path to multi-node. `--zero1` shards optimizer
state as the world grows.

---

## 8. Deviations from the original spec

1. **The 5% `python-edu` code slice was dropped**, weights renormalised across
   the remaining three domains (70/15/10 → 73.7/15.8/10.5). `smollm-corpus`
   `python-edu` stores code only by `blob_id` with **no text column**, and
   `stack-edu` — the substitute the spec suggested — has exactly the same
   problem. Both gated pretraining-grade alternatives (`starcoderdata`,
   `the-stack-v2`) returned 403 for this account's token. An ungated fallback
   with real text does exist (`codeparrot/github-code-clean`, filtered to
   Python) if you want the domain back, but it carries no educational-quality
   score. **Chosen by you.**

2. **optimum-neuron / NeuronX Distributed were not used.** optimum-neuron pins
   `transformers~=4.57.1` and `accelerate==1.8.1`, which would downgrade the
   transformers 5.15.1 that `TRAINIUM_SETUP.md` records as verified. NxD
   installs cleanly but is a NeMo-style YAML framework built for 3D parallelism
   — heavy for 2 cores and the least hackable for the planned RL extension.
   Plain torch-xla preserves the verified stack and scales data-parallel to any
   instance size. **Chosen by you.**

3. **`gnorm` logs as `nan` when `--zero1` is on.** ZeRO-1 performs gradient
   clipping internally inside `step()`, so no separate norm is computed. The
   clipping *is* applied (`max_norm=1.0`); only the reported number is missing.
   Cosmetic.

4. **`TRAINIUM_SETUP.md` is wrong about per-core HBM.** It states a single core
   can allocate the full 32 GB. The compiler enforces 16 GB. Worth correcting,
   since it drives every memory decision above.

## 9. Files

```
data/download_and_tokenize.py   stream -> tokenize -> uint32 shards + manifest
data/corpus/manifest.json       token counts, shard lists, weights per domain
train/model_init.py             config-driven scratch init + --verify-scratch
train/data.py                   weighted mixture sampler (deterministic)
train/evaluate.py               per-domain held-out loss
train/train.py                  training loop
```
