import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from train.document_qa_span_scores import score_spans
from train.neuron_backend import NeuronCausalBackend


def backend():
    torch.manual_seed(21)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=19,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        tie_word_embeddings=True,pad_token_id=0,eos_token_id=0,attn_implementation='eager')).eval()
    return NeuronCausalBackend(model,checkpoint_trunk=False,head_chunk=2)


def test_variable_complete_spans_match_independent_dense_HF_predictions():
    b=backend();past=[[1,2,3],[5,6]];future=[[4,5,0],[7,0]]
    out=score_spans(b,past,future,width=12,target_slots=4,pad_id=0)
    for context,target,row in zip(past,future,out):
        with torch.no_grad():
            ids=torch.tensor([[*context,*target]])
            lp=b.model(ids).logits[0,len(context)-1:-1].float().log_softmax(-1)
            expected=lp.gather(-1,torch.tensor(target)[:,None])[:,0]
        assert torch.allclose(torch.tensor(row['token_logps']),expected,atol=1e-6)
        assert row['tokens']==len(target) and row['full_logp']==sum(row['token_logps'])


def test_target_causality_and_padding_invariance():
    b=backend();a=score_spans(b,[[1,2,3]],[[4,5,6]],width=12,target_slots=4,pad_id=0)[0]
    altered=score_spans(b,[[1,2,3]],[[4,5,8]],width=12,target_slots=4,pad_id=0)[0]
    padded=score_spans(b,[[1,2,3]],[[4,5,6]],width=16,target_slots=4,pad_id=0)[0]
    assert a['token_logps'][:2]==altered['token_logps'][:2]
    assert torch.allclose(torch.tensor(a['token_logps']),torch.tensor(padded['token_logps']),atol=1e-6)


@pytest.mark.parametrize('contexts,targets,width,slots',[
    ([[1,2]],[[3,4,5]],4,4),([[1,2]],[[3,4,5]],8,2),
    ([[1,2]],[[]],8,4),([[]],[[1]],8,4),([[1,2]],[[19]],8,4),
])
def test_incomplete_or_invalid_spans_are_explicit_errors(contexts,targets,width,slots):
    with pytest.raises(ValueError):score_spans(backend(),contexts,targets,width=width,target_slots=slots,pad_id=0)
