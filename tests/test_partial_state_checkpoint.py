import copy

import pytest

from train.partial_state_checkpoint import terminal_scope, verify
from train.partial_state_driver import source_files
from weekend_thinking.artifacts import canonical_hash


def inputs():
    p=dict(schema='partial-state-sft-v6',model_id='Qwen/Qwen3-1.7B-Base',
        items=128,world_size=4,micro_batch=8,global_batch=32,accumulation=1,
        width=128,target_slots=64,schedule='constant',parameter_dtype='bf16',
        optimizer_dtype='fp32',head_chunk=32,purpose='TRAIN_overfit_only',
        steps=128,seed=17,save_every=32,lr=1e-5,source_files=source_files())
    r=dict(purpose=p['purpose'],arm='reader',pipeline_schema=p['schema'],
           spec_hash=canonical_hash(p),planned_steps=128)
    c=dict(purpose=p['purpose'],status='complete',actual_steps=128,planned_steps=128)
    return p,r,c


def test_only_completed_nonfinal_terminal_scope_is_accepted():
    terminal_scope(*inputs())
    for which,key,value in [(0,'purpose','qualification'),(0,'steps',6),
                            (1,'arm','unbound'),(2,'actual_steps',64),
                            (2,'status','stopped_at_boundary')]:
        values=list(copy.deepcopy(inputs()));values[which][key]=value
        with pytest.raises(ValueError):
            terminal_scope(*values)


def test_generic_or_qualification_checkpoint_cannot_be_presented_as_reader():
    for value in [{},dict(checkpoint_scope='development_model_only_step64'),
                  dict(checkpoint_scope='partial-state-TRAIN128-terminal128',developmental_only=False)]:
        with pytest.raises(ValueError):
            verify(value)
