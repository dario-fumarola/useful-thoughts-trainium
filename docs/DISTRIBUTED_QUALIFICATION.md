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
+ 1e-6 relative** before execution. It passes that criterion. Bitwise model and
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
