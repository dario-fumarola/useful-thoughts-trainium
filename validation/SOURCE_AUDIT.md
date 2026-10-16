# Source audit performed in this session

## Initialization

File: `pre-training/train/model_init.py` in the uploaded archive.

SHA256: `f5a3d4c30713b3ade1bb946d31c0e0cd7466c97aed3d45b5ea93dfb86f559c93`

The reviewed construction path creates a model from configuration and then initializes it from scratch:

```text
L74: def build_model(model_id=MODEL_ID, seq_len=2048, seed=1234, dtype=torch.float32,
L86:     model = AutoModelForCausalLM.from_config(
L88:     n_res = init_weights_from_scratch(model, cfg, seed=seed)
```

Changing the model identifier on this path does not load released pretrained weights. The new loader in this kit uses `from_pretrained`; it still needs actual instance-side qualification.

## Last clean-token target

The original OGeo buffer set `tgt[:T - 1] = x[1:]` and left the final clean target at zero. All five reviewed training call sites already compute the complete shifted `y` vector. The patch passes that vector into the buffer and verifies its alignment.

The actual extracted class was exercised in 12 combinations of sequence length, thought length and prediction horizon. The final clean label changed to the correct external next token; every buffer token, attention entry, position, branch target and thought mask remained unchanged. This is not a test of the complete Neuron trainer.

Exact original and patched hashes are in `source_manifest.json`; the generated diff is in `../patches/legacy-targets.patch`.

## What was not reproduced

There are no actual model checkpoints in this executable test. No claims about run 95/98 model quality, the size of NTP damage caused by this bug, or Neuron throughput follow from these tests.
