import copy
from collections import Counter, defaultdict

import pytest

from train import real_trace_packing as packing
from train.partial_state_driver import REAL_SCHEMA, order_for, source_files, validate
from train.useful_note_sft_data import EOS


class Tokenizer:
    eos_token_id = EOS

    def __len__(self):
        return 151669

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


def example():
    tok = Tokenizer()
    prefix = tok.encode('Source context: ' * 9 + '\n(2 + 3) * 4 = ')
    trace, ids = packing.prefix_calculation(prefix, tok)
    return dict(split='train', prefix_ids=prefix, continuation16_ids=tok.encode('20. More text...'),
                full_calculation_text=trace, full_calculation_ids=ids, stop_id=EOS)


def labels(pack):
    mask = pack['future_mask'][0]
    return pack['target_positions'][0, mask].tolist(), pack['labels'][0, mask].tolist()


def test_raw_native_source_is_preserved_without_artificial_eos():
    row = example(); pack = packing.training_pack(row, Tokenizer(), 'raw')
    positions, target = labels(pack)
    original = row['prefix_ids'] + row['continuation16_ids']
    assert positions == list(range(1, len(original)))
    assert target == original[1:]
    assert pack['input_ids'][0, :len(original)].tolist() == original
    assert EOS not in target
    assert not pack['note_close_mask'].any()
    assert (pack['labels'][~pack['future_mask']] == -100).all()


@pytest.mark.parametrize('view', ['prior', 'post'])
def test_augmented_native_ids_and_conditioning_mask(view):
    row = example(); tok = Tokenizer()
    pack = packing.training_pack(row, tok, view)
    positions, target = labels(pack)
    past, future = row['prefix_ids'], row['continuation16_ids']
    opening = tok.encode(packing.OPEN); trace = row['full_calculation_ids']
    if view == 'prior':
        ids = past + opening + trace + tok.encode(packing.ANSWER) + future
        gap = range(len(past), len(past) + len(opening))
        assert EOS not in target
    else:
        ids = past + future + opening + trace + [EOS]
        gap = range(len(past) + 16, len(past) + 16 + len(opening))
        assert target[-1] == EOS and positions[-1] == len(ids)-1
    assert pack['input_ids'][0, :len(ids)].tolist() == ids
    assert set(positions) == set(range(1, len(ids))) - set(gap)
    assert target == [ids[p] for p in positions]
    assert not pack['note_close_mask'].any()  # ONE ordinary token mean


def test_teacher_labels_do_not_depend_on_future_or_stored_answer():
    row = example(); tok = Tokenizer()
    before = packing.prefix_calculation(row['prefix_ids'], tok)
    row['continuation16_ids'] = tok.encode('99. More text...')
    row['numeric_value'] = '999999'
    assert packing.prefix_calculation(row['prefix_ids'], tok) == before
    pack = packing.training_pack(row, tok, 'prior')
    _, target = labels(pack)
    assert target[-16:] == row['continuation16_ids']


@pytest.mark.parametrize('change', [dict(split='dev'), dict(stop_id=1),
    dict(continuation16_ids=[1]), dict(full_calculation_text='The answer is20.')])
def test_invalid_or_nontraining_rows_are_rejected(change):
    row = example(); row.update(change)
    with pytest.raises(ValueError):
        packing.training_pack(row, Tokenizer(), 'prior')


def test_overlength_is_rejected_without_truncation():
    with pytest.raises(ValueError):
        packing.training_pack(example(), Tokenizer(), 'post', width=128)


def test_real_arm_schedule_gives_every_document_both_views():
    p = dict(steps=320, global_batch=32, items=5120, seed=2026101910)
    order = order_for(p)
    for arm in packing.ARMS:
        views = defaultdict(Counter)
        for pos, index in enumerate(order):
            views[index][packing.view_for(arm, pos, p['items'], index)] += 1
        expected = Counter(raw=2) if arm == 'real_raw' else Counter(raw=1, **{arm[5:]:1})
        assert len(views) == 5120 and all(v == expected for v in views.values())


def test_new_training_scope_is_only_eight_update_mechanics():
    p = dict(schema=REAL_SCHEMA, model_id='Qwen/Qwen3-1.7B-Base',
        items=128, world_size=8, micro_batch=4, global_batch=32, accumulation=1,
        width=640, target_slots=640, schedule='constant', parameter_dtype='bf16',
        optimizer_dtype='fp32', head_chunk=32, purpose='real_trace_qualification',
        steps=8, seed=2026101910, save_every=2, lr=1e-5, source_files=source_files(),
        data_sha256='437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915',
        view_schedule='one_plain_one_augmented_per_two_passes',
        loss='single_mean_all_active_tokens_per_example', checkpoint_export_cleanup=True,
        checkpoint_memory_diagnostic=True)
    validate(p)
    for key, value in [('purpose','science'), ('steps',320), ('world_size',4),
                       ('loss','sum_component_means'), ('data_sha256','other')]:
        bad = copy.deepcopy(p); bad[key] = value
        with pytest.raises(ValueError):
            validate(bad)


def test_real_study_cannot_bypass_declared_seeds_or_hardware_receipt():
    p = dict(schema=REAL_SCHEMA,model_id='Qwen/Qwen3-1.7B-Base',items=5120,
        world_size=8,micro_batch=4,global_batch=32,accumulation=1,width=640,target_slots=640,
        schedule='constant',parameter_dtype='bf16',optimizer_dtype='fp32',head_chunk=32,
        purpose='real_trace_cpt_study',steps=320,seed=2026101913,save_every=160,lr=1e-5,
        source_files=source_files(),data_sha256='437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915',
        view_schedule='one_plain_one_augmented_per_two_passes',
        loss='single_mean_all_active_tokens_per_example',hardware_qualification_file='pending-actual-audit.json',
        hardware_qualification_sha256='a'*64)
    validate(p)  # Main separately verifies the actual receipt before allocation.
    for key,value in [('seed',17),('hardware_qualification_file',None),
                      ('hardware_qualification_sha256','invented'),('steps',640)]:
        with pytest.raises(ValueError):validate({**p,key:value})
