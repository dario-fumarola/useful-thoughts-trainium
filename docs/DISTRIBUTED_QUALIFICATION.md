# Distributed optimizer and resume qualification — 2026-10-16

The real two-rank FP32 optimizer check passes on this instance. This qualifies
tiny-model update semantics and numeric resume, not Qwen memory, BF16 execution,
sampling, full-model checkpointing or production throughput.

`train/qualify_distributed.py` uses a 66-parameter `Linear(5,7) → Tanh →
Linear(7,3)` model. Odd leading dimensions exercise padded ZeRO shards. Each
update uses two different ranks, two accumulated microbatches and three examples
per microbatch, equivalent to a 12-example concatenated CPU batch. The rank-local
mean losses are divided by accumulation once; ZeRO performs world-size mean
reduction, global clipping at 0.07, AdamW and weight all-gather. No other gradient
reducer or clipper runs. All four updates clip, change every parameter, remain
finite, and leave both full replicas bitwise identical. Maximum error versus CPU
FP32 is **2.98e-8**; pre-clipping reduced gradient norms also match CPU.

The checkpoint at step 2 contains every rank's FP32 master weights, optimizer
moments, model and CPU-reference state, Python/NumPy/CPU-Torch/XLA RNG, step and
recipe. Hashes and a completion marker are published after both shard writes and
a real rendezvous. The uninterrupted process completes step 4 and exits; a new
process pair loads the checkpoint and repeats steps 3–4.

The fresh `distributed-005` run declares numeric resume tolerance **1e-7 absolute
plus 1e-6 relative** before execution. It passes that criterion. Bitwise model and
optimizer resume **fails** and is recorded as `resume_bitwise=false`; the earlier
strict failure remains in `distributed-004.resume.fixed.log`.

| Fresh-process continuation check | Maximum difference across ranks |
| --- | --- |
| All RNG states, next stochastic inputs | Exact |
| First resumed local gradient | Exact |
| Second resumed local gradient | 5.96e-8; at most 0.149 of allowed tolerance |
| Final model weights | 2.98e-8; at most 0.062 of tolerance |
| Final FP32 master shards | 2.98e-8; at most 0.062 of tolerance |
| Final Adam moments/step state | 2.33e-10; at most 0.003 of tolerance |
| CPU reference model and optimizer | Exact |

The small numeric differences are compatible with FP32 rounding when restored
state enters a different compiled graph; that explanation is an inference, not a
proved cause. No tolerance was enlarged to accept the strict bitwise failure.

## Installed-stack fixes and topology

1. Installed `torch_neuronx` torchrun initialization overwrites explicit visible
   core lists with `LOCAL_RANK`. The initial guarded torchrun attempt therefore
   aborted before model execution. Use `xmp.spawn` with explicit visible cores.
   `distributed-003.topology.json` records both worker PIDs on physical device 1
   (cores 4–7), simultaneously with the independent forward worker on device 0.
   `xmp` reports rank-visible cores 4 and 5. The assignments were verified on the
   runtime, not inferred from environment variables alone.
2. Installed `libneuronxla` can compile an optimizer graph successfully, emit
   `wrapped_neff.hlo`, and then try copying it to an unspecified `None` output
   path. `train/runtime_compat.py:configure_runtime()` supplies that missing
   temporary artifact path in the process-local Python call. It gates on exact
   library version and source hash, leaves installed files and compiler arguments
   intact, and propagates real compiler failures. A fresh empty compile cache was
   used for `distributed-005`. Unpatched errors remain in the `002`/`003` logs.
3. Ordinary PyTorch optimizer loading leaves scalar Adam `step` state on CPU.
   Syncfree AdamW needs it on XLA. Call
   `restore_syncfree_device_state(optimizer)` after ZeRO state loading. The
   uncorrected fresh-process failure is retained in `distributed-004.resume.log`.

## Reproduction and artifacts

Choose a new output and cache path for a fresh cold-cache qualification. Run the
two commands sequentially in **separate processes**:

```bash
source scripts/env.sh
PJRT_DEVICE=NEURON NEURONCORE_NUM_DEVICES=2 \
NEURON_RT_VISIBLE_CORES=4,5 NEURON_RT_ROOT_COMM_ID=127.0.0.1:64187 \
NEURON_COMPILE_CACHE_URL=/home/ubuntu/dario/local/neuron-cache-distributed-005 \
timeout --signal=TERM --kill-after=20s 600s \
python train/qualify_distributed.py --phase uninterrupted \
  --output local/qualification/distributed-005 --expected-cores 4,5

PJRT_DEVICE=NEURON NEURONCORE_NUM_DEVICES=2 \
NEURON_RT_VISIBLE_CORES=4,5 NEURON_RT_ROOT_COMM_ID=127.0.0.1:64187 \
NEURON_COMPILE_CACHE_URL=/home/ubuntu/dario/local/neuron-cache-distributed-005 \
timeout --signal=TERM --kill-after=20s 600s \
python train/qualify_distributed.py --phase resume \
  --output local/qualification/distributed-005 --expected-cores 4,5
```

Evidence is local, outside Git: `local/qualification/distributed-005/` contains
per-rank phase receipts, checkpoint manifests/shards, final states, and actual
local gradient tensors at every step. The sibling `.uninterrupted.log` and
`.resume.log` contain compiler and step outputs. `resume.optimizer_components.json`
separates master weights from moments. Both rank receipts must be read when
reporting maxima; `resume.json` contains rank 0's component details.

CPU checks after the related checkpoint guards: **121 passed**, plus toy smoke,
in `local/checks/tests.9b2Bxw/`. These verify file corruption, missing shards,
cross-arm/rank/step/recipe resume rejection, output identity, the independent
concatenated-gradient oracle, and the compiler compatibility behavior.

Production still needs its own actual Qwen C/R update, full checkpoint
save/restore and stochastic continuation tests. This result must not be used as
their replacement.

## First full Qwen attempt: one valid update, then memory failure

`local/qualification/qwen-c-001` tested actual Qwen3-1.7B on four ranks, BF16
compute, FP32 parameters/master state, four unique prefixes per update. Its
immutable executable snapshot is `qwen-c-001-source` (source hash
`699ac67bc4e5f7126fd227d1fb73b53ea0250c8561225426d07aa5a888146d75`).

The first real update completed: CE 1.5026466, pre-clipping global gradient norm
6.2664828, and 99.012% of the driver's 9,920 sampled FP32 values changed. Its
complete checkpoint passed hash verification. All 310 parameters have initialized
Adam state on every rank; each shard contains 430,143,744 FP32 master elements.
The saved model has 1,720,574,976 unique FP32 parameters and retains the embedding
tie. Independent spread sampling against released safetensors found 99.022% of
19,840 values changed. Evidence: `qwen-c-001/step1.integrity.json`.

The second optimizer graph **failed** compiler memory qualification: reported
peak **26.84 GB exceeds the 24.00 GB limit** (`NCC_EOOM002`). First-step success
does not establish steady-state memory safety: initialized moment inputs make
the later optimizer graph larger. The runtime monitor observed up to
24,315,122,964 allocated bytes before this compile-time failure. No second update
or full-model resume pass exists, and the identical failing continuation was not
relaunched. The surviving first-step checkpoint, hardlinked resume preparation,
651 monitor samples, compiler errors and source snapshot remain intact.

This BF16 run is infrastructure evidence only. Separate replay qualification
rejected BF16 and selected full FP32 for the scientific experiment. The planned
repair is a new, fully FP32, 32-rank qualification with smaller optimizer shards,
followed by actual checkpoint continuation and ten complete steady-state steps.
