import copy
from collections import Counter

import pytest

from train import real_target_packing as packing
from train.partial_state_driver import TARGET_SCHEMA, batch_at, order_for, source_files, validate
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
    text, ids = packing.prefix_calculation(prefix, tok)
    return dict(split='train', prefix_ids=prefix, continuation16_ids=tok.encode('20. More text...'),
                full_calculation_text=text, full_calculation_ids=ids, stop_id=EOS)


@pytest.mark.parametrize('arm', packing.ARMS)
def test_prefix_condition_is_present_but_has_no_supervised_position(arm):
    row, tok = example(), Tokenizer()
    pack = packing.training_pack(row, tok, arm)
    original = row['prefix_ids']
    assert pack['input_ids'][0, :len(original)].tolist() == original
    mask = pack['future_mask'][0]
    positions = pack['target_positions'][0, mask].tolist()
    labels = pack['labels'][0, mask].tolist()
    assert min(positions) >= len(original)
    assert not pack['note_close_mask'].any()
    assert not set(positions).intersection(range(len(original)))
    assert labels[-16:] == row['continuation16_ids']
    assert EOS not in labels
    if arm == 'target_direct':
        assert positions == list(range(len(original), len(original)+16))
        assert labels == row['continuation16_ids']
    else:
        opening, closing = tok.encode(packing.OPEN), tok.encode(packing.ANSWER)
        assert positions[0] == len(original)+len(opening)
        assert labels == row['full_calculation_ids'] + closing + row['continuation16_ids']
        assert set(range(len(original), len(original)+len(opening))).isdisjoint(positions)
        assert '5 * 4 = 20' in row['full_calculation_text']  # Root calculation is taught.


def test_teacher_computation_is_prefix_only_and_future_is_kept_native():
    row = example()
    changed = copy.deepcopy(row)
    changed['continuation16_ids'] = [777] * 16
    changed['numeric_value'] = '1234567'
    a = packing.training_pack(row, Tokenizer(), 'target_cot')
    b = packing.training_pack(changed, Tokenizer(), 'target_cot')
    aa = a['labels'][a['future_mask']].tolist()
    bb = b['labels'][b['future_mask']].tolist()
    assert aa[:-16] == bb[:-16]
    assert bb[-16:] == [777] * 16


def test_vocabulary_validation_does_not_rebuild_the_vocab_for_each_token():
    class CountingTokenizer(Tokenizer):
        calls = 0

        def __len__(self):
            self.calls += 1
            return super().__len__()

    tok = CountingTokenizer()
    packing.training_pack(example(), tok, 'target_cot')
    assert tok.calls == 1


@pytest.mark.parametrize('change', [dict(split='dev'), dict(stop_id=1),
    dict(continuation16_ids=[1]), dict(full_calculation_text='unverified answer'),
    dict(full_calculation_ids=[9]), dict(prefix_ids=[1] * 513),
    dict(continuation16_ids=[EOS] * 16)])
def test_invalid_inputs_never_silently_enter_cot_optimization(change):
    row = example()
    row.update(change)
    with pytest.raises(ValueError):
        packing.training_pack(row, Tokenizer(), 'target_cot')


def test_shape_overflow_is_rejected_without_trimming_targets():
    with pytest.raises(ValueError):
        packing.training_pack(example(), Tokenizer(), 'target_cot', target_slots=16)
    with pytest.raises(ValueError):
        packing.training_pack(example(), Tokenizer(), 'target_direct', width=128)


def protocol(science=False):
    return dict(schema=TARGET_SCHEMA, model_id='Qwen/Qwen3-1.7B-Base',
        items=5120 if science else 128, world_size=8, micro_batch=4, global_batch=32,
        accumulation=1, width=640, target_slots=128, schedule='constant',
        parameter_dtype='bf16', optimizer_dtype='fp32', head_chunk=32,
        purpose='real_target_sft_study' if science else 'real_target_sft_qualification',
        steps=320 if science else 8, seed=2026101913, save_every=160 if science else 2,
        lr=1e-5, source_files=source_files(),
        data_sha256='437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915',
        view_schedule='same_target_view_every_occurrence',
        loss='single_mean_target_tokens_prefix_unscored', checkpoint_export_cleanup=True,
        checkpoint_memory_diagnostic=True, hardware_qualification_file='pending-actual.json',
        hardware_qualification_sha256='a'*64)


def test_new_objective_requires_its_own_identity_and_actual_qualification():
    for science in [False, True]:
        p = protocol(science)
        validate(p)
        for key, value in [('schema','real-trace-cpt-v10'), ('target_slots',640),
                           ('loss','single_mean_all_active_tokens_per_example'),
                           ('view_schedule','one_plain_one_augmented_per_two_passes'),
                           ('steps',640), ('data_sha256','unknown')]:
            with pytest.raises(ValueError):
                validate({**p, key:value})
    for key, value in [('seed',17), ('hardware_qualification_file',None),
                       ('hardware_qualification_sha256','not-an-audit')]:
        with pytest.raises(ValueError):
            validate({**protocol(True), key:value})


def test_exact_same_source_exposures_and_rank_partition_for_both_arms():
    p = protocol(True)
    order = order_for(p)
    assert len(order) == 10240
    assert Counter(order) == Counter({i:2 for i in range(5120)})
    short = protocol()
    rows = [example() for _ in range(128)]
    for arm in packing.ARMS:
        batch = batch_at(rows, Tokenizer(), short, order_for(short), arm, 0, 0)
        assert batch['input_ids'].shape == (4, 640)
        assert batch['labels'].shape == (4, 128)
        assert batch['future_mask'].sum() > 0
