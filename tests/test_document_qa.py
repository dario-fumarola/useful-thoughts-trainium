import copy

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from train.document_qa_data import original_seed_ids, select_cohort, supervised_trace
from train import document_qa_packing as packing
from train.masked_causal_ce import masked_causal_ce
from train.neuron_backend import NeuronCausalBackend
from train.useful_note_sft_data import EOS


class Tokenizer:
    eos_token_id = EOS
    def encode(self, text, add_special_tokens=False): return list(text.encode())
    def decode(self, ids, **kwargs): return bytes(ids).decode()


def row():
    return dict(split='train', question=dict(item_id='qa-example', question='Where did the inventor live?',
        contexts=[dict(title='Invention', text='Ada invented the object. She lived in Lyon.')]),
        answer='Lyon', trace='Step 1: Who invented the object?\nEvidence [1]: Ada\n'
        'Step 2: Where did #1 live?\nEvidence [1]: Lyon')


def test_prefix_only_prompts_reject_unprojected_gold():
    r=row();a=packing.prompt(r['question'],'prior')
    r['answer']='poison';r['trace']='poison'
    assert packing.prompt(r['question'],'prior')==a
    with pytest.raises(ValueError):packing.prompt(r,'prior')


def test_seed_identity_is_not_generic_relation_template():
    r=dict(decomposed_instances=[dict(dataset='trex',id=10)],
           component_question_texts=['located in country'])
    assert original_seed_ids(r)=={'trex:10'}
    other=copy.deepcopy(r);other['decomposed_instances'][0]['id']=11
    assert original_seed_ids(other).isdisjoint(original_seed_ids(r))


def test_actual_directed_split_and_within_held_independence():
    def identity(i,**changes):
        return dict(item_id=str(i),question=str(i),support_text={str(i)},
                    support_titles={str(i)},visible_text={str(i)},
                    visible_titles={str(i)},seed_ids={str(i)},**changes)
    records=[identity(i) for i in range(30)]
    train,held=select_cohort(records,held_count=4,train_count=12)
    bad=identity(100);bad['visible_text'].update(held[0]['support_text'])
    forbidden=identity(101);forbidden['seed_ids'].update(held[1]['seed_ids'])
    # Put the two offenders only in the training candidate pool by finding IDs
    # ranked after all initially selected held questions.
    import hashlib
    rank=lambda r:hashlib.sha256(('document-evidence-v16-audit-only:'+r['item_id']).encode()).hexdigest()
    for record in (bad,forbidden):
        while rank(record)<max(map(rank,held)):
            record['item_id']=str(int(record['item_id'])+1)
    train2,held2=select_cohort(records+[bad,forbidden],held_count=4,train_count=12)
    assert held2==held and bad not in train2 and forbidden not in train2


@pytest.mark.parametrize('view',['direct','prior','post'])
def test_complete_targets_shared_source_masks_and_no_truncation(view):
    r=row();tok=Tokenizer();p=packing.training_pack(r,tok,view,width=1024,target_slots=512)
    source,task=p['note_close_mask'],p['future_mask'];ids=p['input_ids'][0].tolist()
    positions=p['target_positions'][task].tolist();labels=p['labels'][task].tolist()
    assert labels==[ids[i] for i in positions] and labels[-1]==EOS
    assert not (source&task).any()
    if view=='post':
        first=labels.index(EOS)
        assert tok.decode(labels[:first])=='Lyon' and labels.count(EOS)==2
        assert tok.decode(labels[first+1:-1])==packing.POST_OPEN+r['trace']
    elif view=='prior':
        assert tok.decode(labels[:-1])==r['trace']+packing.FINAL_OPEN+r['answer']
    else:assert tok.decode(labels[:-1])==r['answer']
    altered=copy.deepcopy(r);altered['answer']='Elsewhere';altered['trace']='Changed teacher'
    other=packing.training_pack(altered,tok,view,width=1024,target_slots=512)
    assert torch.equal(p['labels'][source],other['labels'][other['note_close_mask']])
    with pytest.raises(ValueError):packing.training_pack(r,tok,view,width=16,target_slots=512)


def test_every_augmented_example_gets_two_plain_and_two_augmented_views():
    assert [packing.view_for('qa_prior',e) for e in range(4)]==['direct','prior','direct','prior']
    assert [packing.view_for('qa_post',e) for e in range(4)]==['direct','post','direct','post']
    assert [packing.view_for('qa_direct',e) for e in range(4)]==['direct']*4


def test_driver_accumulation_covers_each_global_example_once():
    from train.partial_state_driver import DOCUMENT_SCHEMA,batch_at,order_for,source_files,validate
    p=dict(schema=DOCUMENT_SCHEMA,model_id='Qwen/Qwen3-1.7B-Base',world_size=8,micro_batch=1,
        global_batch=32,accumulation=4,width=3328,target_slots=512,schedule='constant',
        parameter_dtype='bf16',optimizer_dtype='fp32',head_chunk=32,items=128,
        purpose='document_qa_qualification',steps=8,save_every=2,seed=2026101921,lr=1e-5,
        view_schedule='plain_even_pass_augmented_odd_pass',
        loss='half_mean_fixed_source256_plus_half_mean_task_per_example',
        source_loss_weight=.5,task_loss_weight=.5,source_sample_labels=256,
        data_sha256='a'*64,data_manifest_sha256='b'*64,source_files=source_files())
    validate(p)
    for key,value in [('accumulation',1),('micro_batch',4),('source_sample_labels',128),
                      ('view_schedule','always_augmented'),('source_loss_weight',True)]:
        with pytest.raises(ValueError):validate({**p,key:value})
    rows=[]
    for i in range(128):
        r=row();r['question']['item_id']=str(i);r['answer']=str(i);rows.append(r)
    order=order_for(p)
    for step in (0,4):
        seen=[]
        for micro in range(4):
            for rank in range(8):
                pack=batch_at(rows,Tokenizer(),p,order,'qa_direct',step,rank,micro)
                labels=pack['labels'][pack['future_mask']].tolist()
                seen.append(int(Tokenizer().decode(labels[:-1])))
        assert seen==order[step*32:(step+1)*32] and len(set(seen))==32


def test_trace_uses_verified_canonical_support_and_backward_references():
    raw=dict(answer_text='Lyon',contexts=[dict(is_supporting=True)],
             decomposed_instances=[dict(id=1,answer_text='Ada'),dict(id=2,answer_text='Lyon')])
    canonical=dict(answer='Lyon',question_decomposition=[dict(id=1,question='Who invented it?',answer='Ada',paragraph_support_idx=0),
        dict(id=2,question='Where did #1 live?',answer='Lyon',paragraph_support_idx=0)])
    assert 'Where did #1 live?' in supervised_trace(raw,canonical)
    other=copy.deepcopy(canonical);other['question_decomposition'][0]['question']='Who is #2?'
    with pytest.raises(ValueError):supervised_trace(raw,other)


@pytest.mark.parametrize('view',['direct','prior','post'])
def test_component_loss_and_accumulated_gradients_match_dense_batch(view):
    torch.manual_seed(21)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=19,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        tie_word_embeddings=True,pad_token_id=0,eos_token_id=0,attn_implementation='eager')).eval()
    reference=copy.deepcopy(model);tok=Tokenizer()
    total=0.
    for i in range(2):
        r=row();r['question']['item_id']=str(i);r['question']['question']+='?'*i
        p=packing.training_pack(r,tok,view,width=320,target_slots=256)
        active=p['note_close_mask']|p['future_mask']
        p['input_ids']%=19;p['labels']=torch.where(active,p['labels']%19,-100)
        result=masked_causal_ce(NeuronCausalBackend(model,head_chunk=32),**p,note_weight=.5,future_weight=.5)
        (result.loss/2).backward()
        x=p['input_ids'];logits=reference(x[:,:-1]).logits.float()
        ce=F.cross_entropy(logits.transpose(1,2),x[:,1:],reduction='none')
        expected=sum(.5*ce[0,p['target_positions'][mask]-1].mean()
                     for mask in (p['note_close_mask'],p['future_mask']))
        assert torch.allclose(result.loss,expected,atol=1e-6)
        total=total+expected/2
    total.backward()
    for (name,p),(other,q) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==other and torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5),name
