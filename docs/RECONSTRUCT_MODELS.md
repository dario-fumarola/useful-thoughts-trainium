# Reconstruct the trained model weights

The overnight model artifacts are lossless BF16 changes relative to **Qwen/Qwen3-1.7B-Base**, revision **ea980cb0a6c2ae4b936e82123acc929f1cec04c1**. They restore the exact trained tensor values; this is not quantization, a new training run or an optimizer-state archive. Scientific outcomes and regressions remain in each study's report.

Obtain that exact upstream snapshot and a model's `.npz` delta from the model release. In this repository, with PyTorch, NumPy and safetensors available, run:

```bash
USE_TORCH_XLA=0 python scripts/restore_model_delta.py \
  --base-dir /path/to/pinned-Qwen3-1.7B-Base \
  --delta /path/to/model-name.npz \
  --out /path/to/new-model-directory
```

The output is a standard Hugging Face model directory containing reconstructed BF16 weights, the original configuration and tokenizer, the upstream Apache 2.0 license, and provenance. Use the base model's completion interface. These are continued-pretraining checkpoints rather than instruction-tuned chat models.

The tool verifies the base weights, configuration, tokenizer and license against their recorded SHA-256 values, validates every reconstructed tensor, saves the model and checks every saved tensor again. The pinned model uses a tied output embedding; equal-valued payload compression does not introduce new architectural parameter sharing. The output directory must be new.

A complete native model and optimizer snapshot is still needed to resume the original Trainium training process. The restored Hugging Face directory contains model weights only. Original native snapshots remain on the owner's node.

The selected ordinary baseline has passed an actual full-model round trip and strict Hugging Face load: all 311 state tensors match the original trained model bit for bit. Codec tests also cover every possible BF16 bit pattern, tied payloads, scalar tensors, incorrect bases, corrupted payloads, changed shapes and missing keys. These verify model artifacts; they do not independently replicate training or prediction results.
