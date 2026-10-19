from collections import Counter

import pytest

from train.partial_state_data import execute, teacher_partial
from train.partial_state_driver import source_files, validate
from train.variable_depth_data import generate, shapes, training_shapes


def test_curricula_have_exact_depth_counts_and_no_expression_overlap():
    data = generate(2026101911)
    fixed, varied = data['TRAIN_fixed'], data['TRAIN_variable']
    assert Counter(len(r['partial_steps']) for r in fixed) == {2:640}
    assert Counter(len(r['partial_steps']) for r in varied) == {i:128 for i in range(1,6)}
    seen = set()
    for panel, rows in data.items():
        for r in rows:
            assert r['expression'] not in seen
            seen.add(r['expression'])
            answer, steps = execute(r['expression_tree'])
            assert r['answer'] == str(answer) and r['partial_steps'] == steps[:-1]
            assert r['partial_note'] == teacher_partial(r['expression_tree'])
            assert r['panel'] == ('TRAIN' if panel.startswith('TRAIN') else panel)
    assert len(data['DEV_depth6']) == 256
    assert all(len(r['partial_steps']) == 6 for r in data['DEV_depth6'])


def test_determinism_and_prefix_exclusion_do_not_change_sizes():
    first = generate(17)
    assert generate(17) == first
    excluded = [r['expression'] for rows in first.values() for r in rows]
    second = generate(17, excluded)
    assert not set(excluded) & {r['expression'] for rows in second.values() for r in rows}
    assert {k:len(v) for k,v in second.items()} == {k:len(v) for k,v in first.items()}


def test_tree_catalogue_covers_bounded_ordered_structures():
    assert [len(shapes(i)) for i in range(8)] == [1,1,2,5,14,42,132,429]
    assert all(s in shapes(n+1) for n in range(1,6) for s in training_shapes(n))


def test_intervention_cannot_launch_a_reader_or_an_unbounded_recipe():
    p=dict(schema='variable-depth-writer-v11', model_id='Qwen/Qwen3-1.7B-Base',
        world_size=4,micro_batch=8,global_batch=32,accumulation=1,width=192,
        target_slots=128,schedule='constant',parameter_dtype='bf16',optimizer_dtype='fp32',
        head_chunk=32,purpose='variable_depth_pilot',curriculum='variable',items=640,
        steps=160,save_every=80,lr=1e-5,seed=2026101911,source_files=source_files(),
        data_sha256='1b8a328479cd05718753b89e2f99e0f696e2cbb62079d505cc372fbb927111d3')
    validate(p)
    for key,value in [('steps',320),('items',5120),('curriculum','other'),('data_sha256','other'),
                      ('width',128),('purpose','TRAIN5120_partial_state_study')]:
        with pytest.raises(ValueError):
            validate({**p,key:value})
