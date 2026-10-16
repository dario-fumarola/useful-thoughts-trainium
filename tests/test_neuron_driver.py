"""CPU checkpoint integrity and lineage guards; no accelerator claim."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from train.neuron_driver import (save_checkpoint, stream_save, validate_resume_run,
                                 validate_resume_state, verify_checkpoint)


class FakeXM:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.barriers = []

    def rendezvous(self, tag):
        self.barriers.append(tag)
        if tag.endswith('-written'):
            # The publication marker must be absent until every write finished.
            assert not (self.checkpoint / 'COMPLETE.json').exists()

    def get_rng_state(self, device):
        return 123


def checkpoint_fixture(tmp_path, world=1):
    model = torch.nn.Linear(3, 2)
    model.config = SimpleNamespace(tie_word_embeddings=False)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    xm = FakeXM(tmp_path / 'step-0000001')
    state = dict(spec_hash='spec', order_hash='order', counters=dict(
        source_records=2, source_tokens=12, sampled_actions=0),
        arm='vanilla', seed=7, global_batch=2, accumulation=2,
        parameter_dtype='fp32', compute_dtype='bf16')
    directory = save_checkpoint(tmp_path, model, optimizer, 1, state, 0, world, xm, 'cpu')
    return directory, xm


def test_checkpoint_publication_and_file_integrity(tmp_path):
    directory, xm = checkpoint_fixture(tmp_path)
    manifest = verify_checkpoint(directory, 'spec', 1)
    assert manifest['step'] == 1
    assert xm.barriers == ['checkpoint-1-start', 'checkpoint-1-written', 'checkpoint-1-complete']
    shard = torch.load(directory / 'rank-00000.pt', weights_only=True)
    assert shard['rank'] == 0 and shard['rng']['device'] == 123
    with pytest.raises(ValueError, match='spec/topology'):
        verify_checkpoint(directory, 'other', 1)
    with pytest.raises(ValueError, match='spec/topology'):
        verify_checkpoint(directory, 'spec', 2)
    with (directory / 'rank-00000.pt').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        verify_checkpoint(directory, 'spec', 1)


def test_missing_shard_cannot_publish_complete(tmp_path):
    with pytest.raises(FileNotFoundError):
        checkpoint_fixture(tmp_path, world=2)
    assert not (tmp_path / 'step-0000001' / 'COMPLETE.json').exists()


def test_stream_save_cannot_overwrite_checkpoint(tmp_path):
    path = tmp_path / 'state.pt'
    stream_save(path, {'value': torch.tensor([1.0])})
    with pytest.raises(FileExistsError):
        stream_save(path, {'value': torch.tensor([2.0])})
    assert torch.load(path, weights_only=True)['value'].item() == 1.0


@pytest.fixture
def resume_parts():
    expected = dict(run_id='run', spec_hash='spec', world=2, rank=0,
                    order_hash='order', arm='thought', seed=7, batch=8,
                    accumulation=4, precision='bf16')
    manifest = dict(run_id='run', spec_hash='spec', world_size=2, step=3)
    weights = dict(run_id='run', spec_hash='spec', step=3, model={})
    shard = dict(spec_hash='spec', world_size=2, rank=0, step=3,
                 order_hash='order', arm='thought', seed=7, global_batch=8,
                 accumulation=4, parameter_dtype='fp32', compute_dtype='bf16',
                 optimizer=dict(base_state={0: {'step': torch.tensor(3.)}},
                                sharded_master_weights={0: torch.ones(2)}),
                 counters=dict(source_records=24, source_tokens=100, sampled_actions=40))
    return manifest, weights, shard, expected


def test_resume_accepts_matching_full_identity(resume_parts):
    manifest, weights, shard, expected = resume_parts
    validate_resume_state(manifest, weights, shard, **expected)


@pytest.mark.parametrize('field,value', [('arm', 'vanilla'), ('seed', 8),
    ('rank', 1), ('step', 2), ('world_size', 4), ('global_batch', 16),
    ('accumulation', 8), ('order_hash', 'other'), ('compute_dtype', 'fp32')])
def test_resume_rejects_wrong_arm_rank_step_or_recipe(resume_parts, field, value):
    manifest, weights, shard, expected = resume_parts
    shard[field] = value
    with pytest.raises(ValueError, match=f'identity mismatch: {field}'):
        validate_resume_state(manifest, weights, shard, **expected)


def test_resume_rejects_mixed_model_step_or_missing_masters(resume_parts):
    manifest, weights, shard, expected = resume_parts
    weights['step'] = 2
    with pytest.raises(ValueError, match='model identity mismatch: step'):
        validate_resume_state(manifest, weights, shard, **expected)
    weights['step'] = 3
    del shard['optimizer']['sharded_master_weights']
    with pytest.raises(ValueError, match='FP32 masters'):
        validate_resume_state(manifest, weights, shard, **expected)


def test_resume_rejects_wrong_output_run_identity(tmp_path):
    run = dict(run_id=tmp_path.name, arm='thought', seed=7, spec_hash='spec')
    (tmp_path / 'run.json').write_text(json.dumps(run))
    validate_resume_run(tmp_path, arm='thought', seed=7, spec_hash='spec')
    for field, value in [('arm', 'vanilla'), ('seed', 8), ('run_id', 'other')]:
        changed = copy.deepcopy(run)
        changed[field] = value
        (tmp_path / 'run.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=f'output identity mismatch: {field}'):
            validate_resume_run(tmp_path, arm='thought', seed=7, spec_hash='spec')
