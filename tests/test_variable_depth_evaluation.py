import json
from types import SimpleNamespace

import pytest

from train.partial_state_checkpoint import identity, terminal_scope
from train.partial_state_driver import source_files
from weekend_thinking.artifacts import canonical_hash


def test_only_complete_pilot_writer_is_a_new_frozen_model():
    p=dict(schema='variable-depth-writer-v11', model_id='Qwen/Qwen3-1.7B-Base',
        world_size=4,micro_batch=8,global_batch=32,accumulation=1,width=192,
        target_slots=128,schedule='constant',parameter_dtype='bf16',optimizer_dtype='fp32',
        head_chunk=32,purpose='variable_depth_pilot',curriculum='variable',items=640,
        steps=160,save_every=80,lr=1e-5,seed=2026101911,source_files=source_files(),
        data_sha256='1b8a328479cd05718753b89e2f99e0f696e2cbb62079d505cc372fbb927111d3')
    run=dict(purpose=p['purpose'],arm='writer',pipeline_schema=p['schema'],
             spec_hash=canonical_hash(p),planned_steps=160)
    done=dict(purpose=p['purpose'],status='complete',actual_steps=160,planned_steps=160)
    terminal_scope(p,run,done)
    with pytest.raises(ValueError):
        terminal_scope(p,{**run,'arm':'reader'},done)
    with pytest.raises(ValueError):
        terminal_scope(p,run,{**done,'actual_steps':80})


def test_old_identity_executes_original_validator_in_its_own_source_tree(tmp_path,monkeypatch):
    producer=tmp_path/'frozen';(producer/'train').mkdir(parents=True)
    (producer/'train/partial_state_checkpoint.py').write_text('# original immutable validator\n')
    expected={'checkpoint_scope':'partial-state-full-terminal-v7','checkpoint_sha256':'actual'}
    def run(command,**kwargs):
        assert kwargs['cwd']==producer
        assert kwargs['env']['PYTHONPATH'].split(':')[0]==str(producer)
        assert kwargs['check'] is True and kwargs['timeout']==600
        assert command[-3:]==['model','protocol','closure']
        assert not kwargs.get('shell',False)
        return SimpleNamespace(stdout=json.dumps(expected))
    monkeypatch.setattr('subprocess.run',run)
    assert identity('model','protocol','closure',producer)=={
        **expected,'producer_source_root':str(producer)}


def test_missing_original_validator_does_not_fall_back_to_current_source(tmp_path):
    with pytest.raises(ValueError,match='Original producer'):
        identity('model','protocol','closure',tmp_path)
