"""Original text keeps its causal context, label weights and dense gradients."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config,Qwen3ForCausalLM

from train import scaffold_learning as s
from train import useful_note_sft_data
from train.masked_causal_ce import masked_component_causal_ce
from train.neuron_backend import NeuronCausalBackend


def source():
    return dict(split='train',kind='calculation',ids=[i%29 for i in range(528)],
                work_header_ids=[5,6],verified_work_ids=[7,8,9],stop_id=30)


class Tokenizer:
    eos_token_id=30
    def __len__(self):return 31


def model():
    torch.manual_seed(19)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=8,
        intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,
        num_key_value_heads=1,head_dim=4,tie_word_embeddings=True,
        attn_implementation='eager',attention_dropout=0.)).eval()


def loss(m,pack,weight=1.):
    return masked_component_causal_ce(NeuronCausalBackend(m,head_chunk=64),
        pack['input_ids'],pack['target_positions'],pack['labels'],
        {'source':pack['source_mask'],'work':pack['work_mask']},
        {'source':1.,'work':weight})


def test_every_arm_preserves_all_original_targets_and_supplies_only_header(monkeypatch):
    monkeypatch.setattr(useful_note_sft_data,'EOS',30)
    row=source();counts=[]
    for arm in s.ARMS[:-1]:
        p=s.training_pack(row,Tokenizer(),arm)
        assert p['input_ids'][0,:528].tolist()==row['ids']
        assert p['source_mask'].sum()==527
        assert p['target_positions'][p['source_mask']].tolist()==list(range(1,528))
        assert p['labels'][p['source_mask']].tolist()==row['ids'][1:]
        assert not (p['source_mask']&p['work_mask']).any()
        assert p['target_positions'][p['work_mask']].tolist()==(
            [] if arm=='scaffold_raw' else [530] if arm=='scaffold_format' else [530,531,532,533])
        counts.append(int(p['work_mask'].sum()))
    assert counts==[0,1,4]


def test_work_changes_cannot_reach_original_predictions_or_their_gradients(monkeypatch):
    monkeypatch.setattr(useful_note_sft_data,'EOS',30)
    row=source();base=model();expected=copy.deepcopy(base)
    dense=F.cross_entropy(expected(torch.tensor([row['ids'][:-1]])).logits.float().transpose(1,2),
                          torch.tensor([row['ids'][1:]]));dense.backward()
    for arm in s.ARMS[:-1]:
        m=copy.deepcopy(base);p=s.training_pack(row,Tokenizer(),arm);v=loss(m,p,weight=0.)
        torch.testing.assert_close(v.loss,dense,rtol=1e-6,atol=1e-6)
        v.loss.backward()
        for (name,a),(other,b) in zip(m.named_parameters(),expected.named_parameters()):
            assert name==other
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6)
    # Even an arbitrary, longer work suffix must not change source likelihood.
    changed={**row,'verified_work_ids':[28,27,26,25,24]*10}
    a=loss(base,s.training_pack(row,Tokenizer(),'scaffold_trace')).losses['source']
    b=loss(base,s.training_pack(changed,Tokenizer(),'scaffold_trace')).losses['source']
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_source_coefficient_does_not_depend_on_trace_length(monkeypatch):
    monkeypatch.setattr(useful_note_sft_data,'EOS',30)
    m=model();row=source()
    a=loss(m,s.training_pack(row,Tokenizer(),'scaffold_trace'))
    assert a.counts['source'].item()==527 and a.counts['work'].item()==4
    torch.testing.assert_close(a.loss,a.losses['source']+a.losses['work'])
    ordinary={**row,'kind':'natural'}
    p=s.training_pack(ordinary,Tokenizer(),'scaffold_trace')
    assert p['work_mask'].sum()==0
    assert loss(m,p).losses['work'].item()==0.


def test_refuse_truncation_and_held_sources(monkeypatch):
    monkeypatch.setattr(useful_note_sft_data,'EOS',30)
    for row in [{**source(),'split':'confirm'},{**source(),'ids':[1]*527},
                {**source(),'verified_work_ids':[7]*128}]:
        with pytest.raises(ValueError):s.training_pack(row,Tokenizer(),'scaffold_trace')
