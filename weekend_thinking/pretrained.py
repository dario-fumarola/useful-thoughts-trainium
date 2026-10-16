"""Strict LOCAL Hugging Face weight loading; untested against Qwen here.

Do not use the old build_model() for a mature-base experiment: that function
explicitly initializes random weights. This loader never does a from_config.
Requires the user-qualified Neuron/Transformers environment on the instance.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import torch


def load_base(model_dir: str | Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_dir=Path(model_dir)
    if not (model_dir/'config.json').exists():
        raise FileNotFoundError('Expected a local pretrained snapshot, including config.json.')
    model=AutoModelForCausalLM.from_pretrained(
        str(model_dir),local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
    model,info=model
    if info.get('missing_keys') or info.get('unexpected_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError(f'Non-exact checkpoint load; inspect before training: {info}')
    tok=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)
    cfg=model.config
    if getattr(cfg,'tie_word_embeddings',False):model.tie_weights()
    if len(tok)>model.get_input_embeddings().weight.shape[0]:
        raise RuntimeError('Tokenizer vocabulary exceeds embedding rows.')
    return model,tok


def sampled_fingerprint(model) -> dict:
    """A cheap integrity signature, NOT a full cryptographic weight-file hash."""
    h=hashlib.sha256();count=0
    for name,tensor in model.named_parameters():
        flat=tensor.detach().reshape(-1)
        # Floating linspace rounds the final index up for 300M-element
        # embeddings. Compute evenly spaced integer indices exactly on host.
        n=flat.numel(); k=min(32,n)
        idx=torch.tensor([i*(n-1)//max(1,k-1) for i in range(k)],
                         device=flat.device,dtype=torch.long)
        h.update(name.encode());h.update(flat[idx].float().cpu().numpy().tobytes());count+=tensor.numel()
    return {'sampled_parameter_sha256':h.hexdigest(),'parameters':count,
            'qualification':'Compare at initialization and after CPU/device transfer; also record full source-file SHA256s.'}
