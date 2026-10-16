"""Instantiate Qwen3-0.6B's architecture with freshly-initialised weights.

Nothing here is hardcoded to Qwen3: the config is loaded from the Hub and every
shape/depth decision is derived from it, so pointing MODEL_ID at another causal
LM should work unchanged.
"""

import math

import torch
import torch.nn as nn

MODEL_ID = "Qwen/Qwen3-0.6B"


def build_config(model_id=MODEL_ID, max_position_embeddings=None):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    if max_position_embeddings is not None:
        # Shrinking the context to what we actually train on keeps the RoPE
        # tables small; it does not change parameter count.
        cfg.max_position_embeddings = max_position_embeddings
    cfg.use_cache = False           # never needed for training, and it would
                                    # make shapes vary across steps on XLA
    return cfg


def _residual_projection_names(cfg):
    """Projections that write *into* the residual stream.

    These get the GPT-2/LLaMA depth-scaled init: std / sqrt(2 * n_layers), so
    the residual stream's variance does not grow with depth.
    """
    return ("o_proj", "down_proj")


def init_weights_from_scratch(model, cfg, seed=1234):
    """Standard pretraining init: N(0, initializer_range), depth-scaled residuals."""
    torch.manual_seed(seed)
    std = getattr(cfg, "initializer_range", 0.02)
    n_layers = cfg.num_hidden_layers
    res_std = std / math.sqrt(2 * n_layers)
    res_names = _residual_projection_names(cfg)

    n_res = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            s = res_std if name.endswith(res_names) else std
            if s == res_std:
                n_res += 1
            nn.init.normal_(module.weight, mean=0.0, std=s)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif "RMSNorm" in type(module).__name__:
            # RMSNorm has weight only; identity init is ones.
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
        elif isinstance(module, (nn.LayerNorm,)):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # Re-tie after init so lm_head and embed_tokens remain the same storage.
    if getattr(cfg, "tie_word_embeddings", False):
        model.tie_weights()
    return n_res


def build_model(model_id=MODEL_ID, seq_len=2048, seed=1234, dtype=torch.float32,
                attn_implementation="eager"):
    """`eager` attention is deliberate, for two reasons.

    Qwen3 RMSNorm-normalises q and k per head. Under bf16 autocast those norms
    stay fp32 while v is cast to bf16, and SDPA rejects the mixed dtypes
    outright. Eager attention lets autocast cast both matmul operands itself.
    It also lowers to XLA more predictably, which is why the repo's inference
    script makes the same choice.
    """
    from transformers import AutoModelForCausalLM
    cfg = build_config(model_id, max_position_embeddings=seq_len)
    model = AutoModelForCausalLM.from_config(
        cfg, dtype=dtype, attn_implementation=attn_implementation)
    n_res = init_weights_from_scratch(model, cfg, seed=seed)
    return model, cfg, n_res


def param_summary(model, cfg):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            emb += m.weight.numel()
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    return dict(total=total, trainable=trainable, embedding=emb,
                non_embedding=total - emb, tied_embeddings=tied)


def assert_not_pretrained(model, model_id=MODEL_ID, tol=1e-6):
    """Verify no pretrained weight survived, by comparing against the real ckpt.

    Compares per-tensor mean/std fingerprints; identical shapes with identical
    statistics would mean the pretrained values leaked through.
    """
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    ref_sd = ref.state_dict()
    our_sd = model.state_dict()

    checked, identical, missing = 0, [], []
    for k, v in our_sd.items():
        if k not in ref_sd:
            missing.append(k)
            continue
        r = ref_sd[k].float()
        o = v.float()
        if r.shape != o.shape:
            continue
        checked += 1
        if torch.allclose(r, o, atol=tol, rtol=0):
            identical.append(k)
    del ref, ref_sd
    return dict(checked=checked, identical=identical, missing=missing)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--verify-scratch", action="store_true",
                    help="download the pretrained ckpt and prove no weight survived")
    a = ap.parse_args()

    model, cfg, n_res = build_model(seq_len=a.seq_len, seed=a.seed)
    s = param_summary(model, cfg)

    print("=" * 60)
    print(f"architecture      : {cfg.model_type}")
    print(f"hidden / layers   : {cfg.hidden_size} / {cfg.num_hidden_layers}")
    print(f"heads (q/kv)      : {cfg.num_attention_heads} / {cfg.num_key_value_heads}"
          f"  head_dim={cfg.head_dim}")
    print(f"intermediate      : {cfg.intermediate_size}")
    print(f"vocab             : {cfg.vocab_size}")
    print(f"ctx (training)    : {cfg.max_position_embeddings}")
    print(f"tied embeddings   : {s['tied_embeddings']}")
    print("-" * 60)
    print(f"total params      : {s['total']:,}  ({s['total']/1e6:.1f}M)")
    print(f"  embedding       : {s['embedding']:,}  ({s['embedding']/1e6:.1f}M)")
    print(f"  non-embedding   : {s['non_embedding']:,}  ({s['non_embedding']/1e6:.1f}M)")
    print(f"depth-scaled projs: {n_res} tensors at std={0.02/math.sqrt(2*cfg.num_hidden_layers):.5f}")
    print("=" * 60)

    if a.verify_scratch:
        print("verifying against pretrained checkpoint ...")
        r = assert_not_pretrained(model)
        print(f"  tensors compared : {r['checked']}")
        print(f"  identical to ckpt: {len(r['identical'])}")
        if r["identical"]:
            raise SystemExit(f"FAIL: pretrained weights survived: {r['identical'][:5]}")
        print("  PASS - no pretrained weights remain")


def untie_lm_head(model):
    """Break the embed_tokens/lm_head tie on a freshly built CPU model.

    Training runs end up with the two matrices untied and diverged (verified:
    saved checkpoints hold two distinct, differing tensors), because moving
    the model to the XLA device re-materialises the shared storage as separate
    per-parameter tensors. A tied CPU model must therefore be untied BEFORE
    load_state_dict, or the checkpoint's two matrices collapse into one
    storage and the last-loaded key silently overwrites the other (~1 nat of
    quality loss on our vanilla-350 checkpoint; this corrupted the init of
    Stage-A thinking runs 1-4 via --init-from).
    """
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
    return model
