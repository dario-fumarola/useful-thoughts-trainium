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


def test_combined_source_and_work_gradients_match_independent_dense_targets(monkeypatch):
    monkeypatch.setattr(useful_note_sft_data,'EOS',30)
    row=source();m=model();reference=copy.deepcopy(m)
    pack=s.training_pack(row,Tokenizer(),'scaffold_trace');actual=loss(m,pack)
    logits=reference(pack['input_ids'][:,:-1]).logits.float()
    token_losses=F.cross_entropy(logits.transpose(1,2),pack['input_ids'][:,1:],reduction='none')
    expected=token_losses[:,:527].mean()+token_losses[:,529:533].mean()
    torch.testing.assert_close(actual.loss,expected,rtol=1e-6,atol=1e-6)
    actual.loss.backward();expected.backward()
    for (name,a),(other,b) in zip(m.named_parameters(),reference.named_parameters()):
        assert name==other
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6)


def test_direct_evaluation_binds_worker_and_requires_all_six_terminals(tmp_path):
    import json
    from train import scaffold_learning_evaluate as scoring,uniform_lm_evaluate as worker
    from train import partial_state_driver as driver
    from weekend_thinking.artifacts import sha256_file
    data=tmp_path/'data';data.write_text('{}\n');manifest=tmp_path/'manifest'
    manifest.write_text(json.dumps({'files':{'CONFIRM':{'sha256':sha256_file(data)}}}))
    output=tmp_path/'score'
    p=dict(schema='scaffold-learning-likelihood-v1',world_size=8,micro_batch=4,
        source_files=driver.source_files(),evaluator_sha256=sha256_file(scoring.__file__),
        worker_evaluator_sha256=sha256_file(worker.__file__),data_file=str(data),
        data_sha256=sha256_file(data),data_manifest_file=str(manifest),
        data_manifest_sha256=sha256_file(manifest),model_name='B0',checkpoint=None,
        cohort='CONFIRM',items=4096)
    release=dict(jobs={str(output):{}},all_terminal_models={})
    with pytest.raises(ValueError,match='All six'):scoring.validate(p,release,output)
    release['all_terminal_models']={f'scaffold_{a}-seed{n}':{'step':128}
        for a in ('raw','format','trace') for n in (11,12)}
    scoring.validate(p,release,output)
    p['worker_evaluator_sha256']='f'*64
    with pytest.raises(ValueError,match='evaluator'):scoring.validate(p,release,output)
