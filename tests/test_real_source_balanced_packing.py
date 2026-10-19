"""Scientific mask/exposure/normalization invariants for the source-balanced arm."""
import copy
from collections import Counter

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from train import real_source_balanced_packing as packing
from train.real_trace_packing import prefix_calculation, OPEN
from train.partial_state_driver import BALANCED_SCHEMA, batch_at, order_for, source_files, validate
from train.masked_causal_ce import masked_causal_ce
from train.neuron_backend import NeuronCausalBackend
from train.useful_note_sft_data import EOS


class Tokenizer:
    eos_token_id = EOS
    def __len__(self): return 151669
    def encode(self, text, add_special_tokens=False): return [ord(c) for c in text]
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)


def example():
    tok = Tokenizer()
    prefix = tok.encode('Source context: ' * 9 + '\n(2 + 3) * 4 = ')
    text, ids = prefix_calculation(prefix, tok)
    return dict(split='train', prefix_ids=prefix, continuation16_ids=tok.encode('20. More text...'),
                full_calculation_text=text, full_calculation_ids=ids, stop_id=EOS)


def protocol(science=False):
    return dict(schema=BALANCED_SCHEMA, model_id='Qwen/Qwen3-1.7B-Base',
        items=5120 if science else 128, world_size=8, micro_batch=4, global_batch=32,
        accumulation=1, width=640, target_slots=640, schedule='constant',
        parameter_dtype='bf16', optimizer_dtype='fp32', head_chunk=32,
        purpose='real_source_balanced_study' if science else 'real_source_balanced_qualification',
        steps=320 if science else 8, seed=2026101913, save_every=160 if science else 2,
        lr=1e-5, source_files=source_files(),
        data_sha256='437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915',
        view_schedule='same_two_component_view_every_occurrence',
        loss='half_mean_source_prefix_plus_half_mean_task_per_example',
        source_loss_weight=.5, task_loss_weight=.5, checkpoint_export_cleanup=True,
        checkpoint_memory_diagnostic=True, hardware_qualification_file='pending-actual.json',
        hardware_qualification_sha256='a'*64)


@pytest.mark.parametrize('arm',packing.ARMS)
def test_identical_source_component_and_untruncated_native_task(arm):
    row=example();pack=packing.training_pack(row,Tokenizer(),arm)
    source,task=pack['note_close_mask'],pack['future_mask']
    assert not (source&task).any()
    assert pack['target_positions'][source].tolist()==list(range(1,len(row['prefix_ids'])))
    assert pack['labels'][source].tolist()==row['prefix_ids'][1:]
    assert pack['labels'][task].tolist()[-16:]==row['continuation16_ids']
    assert EOS not in pack['labels'][source|task]
    if arm=='source_cot':
        opening=range(len(row['prefix_ids']),len(row['prefix_ids'])+len(Tokenizer().encode(OPEN)))
        assert set(opening).isdisjoint(pack['target_positions'][source|task].tolist())
        assert pack['labels'][task].tolist()[:-16]==row['full_calculation_ids']+Tokenizer().encode('Answer:\n')
    else:assert task.sum()==16


def test_future_annotations_cannot_change_computed_trace_or_source_labels():
    row=example();other=copy.deepcopy(row);other['continuation16_ids']=[777]*16
    other['numeric_value']='9999999'
    a=packing.training_pack(row,Tokenizer(),'source_cot')
    b=packing.training_pack(other,Tokenizer(),'source_cot')
    assert torch.equal(a['labels'][a['note_close_mask']],b['labels'][b['note_close_mask']])
    assert a['labels'][a['future_mask']].tolist()[:-16]==b['labels'][b['future_mask']].tolist()[:-16]


@pytest.mark.parametrize('arm',packing.ARMS)
def test_actual_two_component_dense_loss_and_all_gradients_match(arm):
    torch.manual_seed(17)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=19,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        tie_word_embeddings=True,pad_token_id=0,eos_token_id=0,attn_implementation='eager')).eval()
    reference=copy.deepcopy(model)
    backend=NeuronCausalBackend(model,head_chunk=32,checkpoint_trunk=True)
    pack=packing.training_pack(example(),Tokenizer(),arm)
    # Preserve the real packing geometry while using a tiny synthetic vocabulary.
    pack['input_ids']=pack['input_ids']%19
    active=pack['note_close_mask']|pack['future_mask']
    pack['labels']=torch.where(active,pack['labels']%19,-100)
    result=masked_causal_ce(backend,**pack,note_weight=.5,future_weight=.5)
    x=pack['input_ids'];logits=reference(x[:,:-1]).logits.float()
    token_ce=F.cross_entropy(logits.transpose(1,2),x[:,1:],reduction='none')
    means=[]
    for mask in [pack['note_close_mask'],pack['future_mask']]:
        positions=pack['target_positions'][mask]-1
        means.append(token_ce[0,positions].mean())
    expected=.5*means[0]+.5*means[1]
    assert torch.allclose(result.loss,expected,atol=1e-6)
    # A length-weighted union mean is the old objective, not this intervention.
    union=token_ce[0,pack['target_positions'][active]-1].mean()
    assert abs(float(expected.detach()-union.detach()))>1e-6
    result.loss.backward();expected.backward()
    for (name,p),(other,q) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==other and p.grad is not None and q.grad is not None
        assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5),name


def test_new_schema_is_guarded_and_keeps_same_two_pass_exposure():
    for science in [False,True]:
        p=protocol(science);validate(p)
        for key,value in [('schema','real-target-sft-v15'),('source_loss_weight',0),
                          ('task_loss_weight',True),('target_slots',128),
                          ('view_schedule','alternating'),('steps',640)]:
            with pytest.raises(ValueError):validate({**p,key:value})
    p=protocol(True);order=order_for(p)
    assert len(order)==10240 and Counter(order)==Counter({i:2 for i in range(5120)})
    short=protocol();rows=[example() for _ in range(128)]
    for arm in packing.ARMS:
        b=batch_at(rows,Tokenizer(),short,order_for(short),arm,0,0)
        assert b['input_ids'].shape==(4,640) and b['labels'].shape==(4,640)
        assert b['note_close_mask'].all(0).any() and b['future_mask'].any()


@pytest.mark.parametrize('change',[dict(split='dev'),dict(stop_id=1),dict(full_calculation_ids=[9])])
def test_no_evaluation_or_unverified_teacher_can_enter_training(change):
    row=example();row.update(change)
    with pytest.raises(ValueError):packing.training_pack(row,Tokenizer(),'source_cot')
