"""Only the bounded, TRAIN-only two-forward qualification can execute."""
from collections import Counter

import pytest

from train import partial_state_driver as driver
from test_real_cpt_scaffold import Tokenizer, example


def protocol():
    return dict(schema=driver.SCAFFOLD_SCHEMA, model_id='Qwen/Qwen3-1.7B-Base',
        experiment_id='real-cpt-scaffold-v29', purpose='real_cpt_scaffold_qualification',
        items=128, steps=8, save_every=4, seed=2026102029, lr=1e-5,
        world_size=8, micro_batch=4, global_batch=32, accumulation=1,
        auxiliary_micro_batch=1, width=640, target_slots=640, head_chunk=32,
        schedule='constant', parameter_dtype='bf16', optimizer_dtype='fp32',
        view_schedule='clean_plus_raw_format_computation_cycle',
        loss='clean_LM_plus_fraction_weighted_auxiliary_continuation_work',
        source_loss_weight=1., continuation_loss_weight=.5, work_loss_weight=1.,
        auxiliary_batch_fraction=.25, parent_data_sha256=driver.SCAFFOLD_PARENT_SHA,
        data_manifest_sha256=driver.SCAFFOLD_MANIFEST_SHA, data_sha256='a'*64,
        checkpoint_export_cleanup=True, checkpoint_memory_diagnostic=True,
        source_files=driver.source_files())


@pytest.mark.parametrize('change', [dict(purpose='real_cpt_scaffold_study'),
    dict(steps=1280), dict(items=20480), dict(world_size=4),
    dict(auxiliary_micro_batch=4), dict(auxiliary_batch_fraction=1),
    dict(continuation_loss_weight=True), dict(work_loss_weight=.1),
    dict(parent_data_sha256='b'*64), dict(data_manifest_sha256='b'*64),
    dict(checkpoint_export_cleanup=False)])
def test_no_science_or_changed_loss_shape_release(change):
    p = protocol(); driver.validate(p)
    with pytest.raises(ValueError): driver.validate({**p, **change})


def test_actual_two_pass_order_and_rank_batches_cover_all_three_conditions():
    p = protocol(); driver.validate(p)
    rows = [example(document=f'c{i}') for i in range(32)]
    rows += [example('ordinary', document=f'o{i}') for i in range(96)]
    order = driver.order_for(p, rows)
    assert Counter(order) == Counter({i:2 for i in range(128)})
    for step in range(8):
        for rank in range(8):
            passes = driver.scaffold_passes_at(rows, Tokenizer(), p, order,
                driver.SCAFFOLD_QUALIFICATION_ARM, step, rank)
            (name, source, sw), (aux_name, aux, aw) = passes
            assert name == 'clean' and aux_name == 'auxiliary'
            assert source['input_ids'].shape == (4,640)
            assert aux['input_ids'].shape == (1,640)
            assert sw == {'source':1.} and aw == {'work':.25,'continuation':.125}
            assert int(aux['component_masks']['continuation'].sum()) == 16
            work = int(aux['component_masks']['work'].sum())
            if step % 3 == 0: assert work == 0
            elif step % 3 == 1: assert work == len(Tokenizer().encode('Answer:\n'))
            else: assert work > len(Tokenizer().encode('Answer:\n'))
    with pytest.raises(ValueError):
        driver.batch_at(rows, Tokenizer(), p, order, driver.SCAFFOLD_QUALIFICATION_ARM, 0, 0)
    with pytest.raises(ValueError):
        driver.scaffold_passes_at(rows, Tokenizer(), p, order, 'computation', 0, 0)
