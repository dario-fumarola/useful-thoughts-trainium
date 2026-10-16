# CPU FP64 derivative reference formulas

These are exact copies of the reviewed helper and four tiny CPU fixtures used by
the [first-batch derivative diagnostic](../c_firstbatch_fp64_derivative_20261016.json).
They change process-local diagnostic callables. The production driver and installed
libraries are unchanged.

The method is **FP64 differentiable arithmetic with frozen native-CPU FP32 RoPE
constants**. A plain `model.double()` retains explicit FP32 casts in native Qwen3
RMSNorm and eager attention. This helper preserves FP64 through RMSNorm and
softmax, while keeping the native formulas, scaling, causal mask, and eval behavior.
The full runner captures native CPU FP32 RoPE cosine/sine values before casting
the model and supplies them to this helper, which returns those exact constants
cast to double. The full reference uses fixed
positions 0–511, eager attention, eval mode, and no KV cache.

The full derivative runner loaded the released Qwen3-1.7B-Base FP32 weights,
verified their exact roundtrip after conversion and the embedding/head tie, and
accumulated mean causal cross-entropy over 16 microbatches of eight TRAIN records.
Hooks verified FP64 component arithmetic; all 310 parameter gradients were finite
FP64. The result receipt binds the runner, input order, constants, and full saved
gradients. These fixtures check formula/gradient parity on FP32 inputs, FP64 dtype
preservation, and frozen RoPE behavior on a tiny model. They do not qualify an
accelerator update or substitute for the full derivative run.

The recorded environment is PyTorch **2.9.1+cu128**, git
`5811a8d7da873dd699ff6687092c225caffcf1bb`, and Transformers **5.15.1**.
The installed `modeling_qwen3.py` SHA256 is
`cbb7f2dc274c2f5592746c0dc6985ca50353efa07376f92cc922b77680a74f69`.
No package upgrade is implied. To run only the tiny fixtures in that environment:

```bash
source scripts/env.sh
USE_TORCH_XLA=0 python -m pytest -q validation/diagnostics/f64_reference_ops_tests.py
```

Exact-byte SHA256 values from the already completed review and fixture run:

| Artifact | SHA256 |
| --- | --- |
| `f64_reference_ops.py` | `9159354fc598a1adc7135fd67da1a5d7ac8b70468368192d6faeaf8f50a4088a` |
| `f64_reference_ops_tests.py` | `5059f3b8deb33b0da6c41a04fe3b4b68fe4dfa0bc0560cba2d2eff59b89b5e4e` |
| Existing `local/qualification/f64_reference_ops_tests-002.log` | `29d5f001aad8140574443506f96dc283b28de12c643ec5fcb3e2f222070576c4` |

Publishing these copies did not rerun the numerical reference. Device RoPE can
approximate cosine/sine differently, so frozen CPU constants are not a unique
physical oracle. Between-microbatch cancellation ratios do not rule out
cancellation within a microbatch, token, or matrix multiplication. Existing
pointwise failures and acceptance bounds remain unchanged.
