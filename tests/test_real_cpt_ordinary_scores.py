"""All real ordinary-window transitions and continuation suffix must agree."""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.document_qa_span_scores import score_spans
from train.neuron_backend import NeuronCausalBackend


def test_all527_native_targets_match_independent_dense_HF_and_suffix16():
    torch.manual_seed(29)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=19,hidden_size=16,
        intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,
        num_key_value_heads=1,head_dim=8,tie_word_embeddings=True,
        pad_token_id=0,eos_token_id=0,attn_implementation='eager')).eval()
    backend=NeuronCausalBackend(model,checkpoint_trunk=False,head_chunk=32)
    x=torch.randint(1,19,(2,528));ids=x.tolist()
    rows=score_spans(backend,[[v[0]] for v in ids],[v[1:] for v in ids],
        width=528,target_slots=528,pad_id=0)
    suffix=score_spans(backend,[v[:512] for v in ids],[v[512:] for v in ids],
        width=528,target_slots=16,pad_id=0)
    with torch.no_grad():
        lp=model(x).logits[:,:-1].float().log_softmax(-1)
        gold=lp.gather(-1,x[:,1:,None])[:,:,0]
    for i,row in enumerate(rows):
        assert row['tokens']==527 and row['full_logp']==sum(row['token_logps'])
        assert torch.allclose(torch.tensor(row['token_logps']),gold[i],atol=1e-6)
        assert torch.allclose(torch.tensor(row['token_logps'][-16:]),
            torch.tensor(suffix[i]['token_logps']),atol=1e-6)
        assert 0 not in ids[i]
