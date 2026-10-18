import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.neuron_backend import NeuronCausalBackend
from train.partial_state_answer_driver import cache_prefill_check
from train.partial_state_answers import NumberLineDecoder


def test_padded_cached_prefill_matches_ordinary_causal_forward_on_real_cpu_model():
    # CPU architecture fixture only, never a released-weight or Trainium pass.
    torch.manual_seed(17)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=151650,hidden_size=16,
        intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,
        num_key_value_heads=1,head_dim=8,tie_word_embeddings=True,
        attn_implementation='eager')).eval()
    backend=NeuronCausalBackend(model,checkpoint_trunk=False)
    decoder=NumberLineDecoder(model,None)
    result,vectors=cache_prefill_check(backend,decoder,[2,3,4,5],width=8,cap=6)
    assert result['max_logp_error']<1e-3 and result['argmax_equal']
    assert vectors['ordinary'].shape==vectors['cached'].shape==(1,151650)
