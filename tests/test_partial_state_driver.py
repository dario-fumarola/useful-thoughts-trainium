import copy
from collections import Counter, defaultdict

import pytest

from train.partial_state_driver import checkpoint_identity, order_for, role_for, source_files, validate
from weekend_thinking.artifacts import canonical_hash


def protocol():
    return dict(schema='partial-state-sft-v6', model_id='Qwen/Qwen3-1.7B-Base',
        items=128, world_size=4, micro_batch=8, global_batch=32, accumulation=1,
        width=128, target_slots=64, schedule='constant', parameter_dtype='bf16',
        optimizer_dtype='fp32', head_chunk=32, purpose='qualification', steps=16,
        seed=17, save_every=3, lr=1e-5, source_files=source_files(), data_sha256='data')


def test_exposure_order_and_per_expression_reader_cross_are_exact():
    p = protocol(); order = order_for(p)
    assert len(order) == 512 and order == order_for(p)
    byitem = defaultdict(Counter)
    for pos, item in enumerate(order):
        byitem[item][role_for('reader', pos, p['items'], item)] += 1
    assert len(byitem) == 128
    assert all(c == Counter(reader_teacher=2, reader_empty=1, direct=1) for c in byitem.values())
    for begin in range(0, 512, 128):
        assert sorted(order[begin:begin+128]) == list(range(128))
    assert all(role_for('writer', pos, 128, item) == 'writer' for pos, item in enumerate(order))


@pytest.mark.parametrize('key,value', [('world_size',32), ('micro_batch',1), ('purpose','science'),
    ('steps',129), ('save_every',0), ('lr',float('nan')), ('parameter_dtype','fp32')])
def test_unqualified_scope_or_shape_is_rejected(key, value):
    p = protocol(); validate(p); p[key] = value
    with pytest.raises(ValueError):
        validate(p)


def test_checkpoint_identity_requires_same_arm_cursor_and_full_model_lineage():
    p = protocol(); state = dict(spec_hash=canonical_hash(p), arm='writer', seed=17,
        order_hash='order', protocol_sha256='protocol', pipeline_schema='partial-state-sft-v6',
        data_sha256='data', counters=dict(expressions=96, supervised_tokens=1234))
    manifest = dict(run_id='run', world_size=4, step=3, spec_hash=state['spec_hash'], metadata=state)
    weights = dict(step=3, run_id='run', spec_hash=state['spec_hash'])
    saved = dict(state, rank=1, world_size=4, step=3)
    assert checkpoint_identity(manifest,weights,saved,p,'writer','run',1,'order','protocol') == 3
    for key, value in [('arm','reader'), ('step',2), ('order_hash','other'), ('rank',0),
                       ('counters',dict(expressions=64,supervised_tokens=1234))]:
        bad = copy.deepcopy(saved); bad[key] = value
        with pytest.raises(ValueError):
            checkpoint_identity(manifest,weights,bad,p,'writer','run',1,'order','protocol')
