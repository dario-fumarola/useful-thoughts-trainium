"""The adapter permits only eight all-view updates, never a learning campaign."""
import pytest
import torch

from train import partial_state_driver as driver


def protocol():
    return dict(schema=driver.PROGRAM_SCHEMA, experiment_id='controlled-computation-v28',
        purpose='permutation_program_qualification', model_id='Qwen/Qwen3-1.7B-Base',
        world_size=4, micro_batch=8, global_batch=32, accumulation=1,
        width=256, target_slots=64, head_chunk=32, schedule='constant',
        parameter_dtype='bf16', optimizer_dtype='fp32', items=128,
        steps=8, save_every=4, seed=2026102028, lr=1e-5,
        view_schedule='qualification_all_four_views_each_update',
        loss='separate_mean_answer_work_per_example', answer_loss_weight=1., work_loss_weight=1.,
        data_sha256='a'*64, data_manifest_sha256='b'*64,
        source_files=driver.source_files())


def test_only_bounded_mixed_qualification_is_valid():
    p = protocol(); driver.validate(p)
    for key, value in [('purpose', 'permutation_program_study'), ('steps', 4096),
                       ('world_size', 8), ('save_every', 2), ('items', 32768),
                       ('lr', 1e-4), ('work_loss_weight', True),
                       ('view_schedule', 'trace_only')]:
        changed = dict(p, **{key: value})
        with pytest.raises(ValueError):
            driver.validate(changed)


def test_each_update_spans_same32_items_and_all_four_views(monkeypatch):
    p = protocol(); order = driver.order_for(p)
    assert len(order) == 256
    assert sorted(order[:128]) == sorted(order[128:]) == list(range(128))
    rows = [dict(item_id=str(i)) for i in range(128)]; calls = []

    def spy(row, tok, view, **shape):
        calls.append((row['item_id'], view, shape))
        # Dispatch/ordering only. Real pack semantics are checked separately
        # against the actual tokenizer and all32,768 TRAIN questions.
        return dict(input_ids=torch.tensor([[1, 2, 0]]),
            target_positions=torch.tensor([[1]]), labels=torch.tensor([[2]]),
            answer_mask=torch.tensor([[True]]), work_mask=torch.tensor([[False]]))

    monkeypatch.setattr(driver.permutation_program_packing, 'training_pack', spy)
    for step in range(8):
        begin = len(calls)
        for rank in range(4):
            batch = driver.batch_at(rows, None, p, order, driver.PROGRAM_QUALIFICATION_ARM, step, rank)
            assert batch['input_ids'].shape == (8, 3)
        current = calls[begin:]
        assert [v[0] for v in current] == [str(i) for i in order[step*32:(step+1)*32]]
        assert all(sum(v[1] == mode for v in current) == 8 for mode in ('direct', 'empty', 'trace', 'pause'))
        assert all(v[2] == dict(width=256, target_slots=64) for v in current)
    with pytest.raises(ValueError):
        driver.batch_at(rows, None, p, order, 'trace', 0, 0)


def study_protocol():
    from train.permutation_program_retention import POLICY
    p=protocol();p.update(purpose=driver.PROGRAM_STUDY_PURPOSE,items=32768,steps=4096,save_every=1024,
        program_arm='trace',view_schedule='interleaved_fixed_item_plus_epoch_parity',
        checkpoint_retention=POLICY,data_sha256=driver.PROGRAM_DATA_SHA,
        data_manifest_sha256=driver.PROGRAM_MANIFEST_SHA)
    return p


def test_science_protocol_has_fixed_exposures_and_cannot_self_release(tmp_path):
    from types import SimpleNamespace
    p=study_protocol();driver.validate(p)
    for key,value in [('steps',2048),('seed',3),('lr',1e-6),('program_arm','writer'),
                      ('checkpoint_retention','keep_whatever')]:
        with pytest.raises(ValueError):driver.validate(dict(p,**{key:value}))
    args=SimpleNamespace(execution_release=None,arm='trace',output=tmp_path)
    with pytest.raises(ValueError):driver.validate_program_execution(p,args)


def test_science_views_match_all_four_question_exposures():
    from collections import Counter
    p=study_protocol();order=driver.order_for(p)
    assert Counter(order)==Counter({i:4 for i in range(32768)})
    for arm in ('direct','trace','pause'):
        seen={i:Counter() for i in range(32768)}
        for pos,index in enumerate(order):
            seen[index][driver.permutation_program_packing.view_for(arm,pos//32768,index)]+=1
        augmented='empty' if arm=='direct' else arm
        assert all(v==Counter(direct=2,**{augmented:2}) for v in seen.values())
