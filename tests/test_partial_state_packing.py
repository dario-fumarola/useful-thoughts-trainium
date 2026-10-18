import pytest
import torch

from train.partial_state_data import row, writer_prompt
from train.partial_state_packing import COMPONENTS, batch_packs, training_pack
from train.useful_note_sft_data import EOS


class BytesTokenizer:
    eos_token_id = EOS

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(text.encode())


def example():
    return row(('*', ('+', 37, 18), ('-', 92, 57)), 'TRAIN')


def test_exact_causal_masks_eos_and_unscored_prompt_in_all_components():
    tok = BytesTokenizer(); r = example()
    for component in COMPONENTS:
        p = training_pack(r, tok, component, width=256, target_slots=128)
        active = p['note_close_mask'] | p['future_mask']
        positions = p['target_positions'][active]
        assert torch.equal(p['labels'][active], p['input_ids'][0, positions])
        assert p['labels'][active][-1] == EOS
        assert p['input_ids'][0, positions[-1] + 1] == EOS  # Same pad ID is inactive.
        assert not active[0, len(positions):].any()
        if component == 'writer':
            assert p['note_close_mask'].sum() == len(r['partial_note'].encode()) + 1
            assert not p['future_mask'].any()
            prompt = writer_prompt(r['expression_tree']).encode()
            assert p['input_ids'][0, :positions[0]].tolist() == list(prompt)
        else:
            assert not p['note_close_mask'].any()
        if component == 'cot':
            target = (r['partial_note'] + 'Answer:\n' + r['answer'] + '\n').encode()
            assert p['labels'][active].tolist() == list(target) + [EOS]


def test_reader_teacher_is_conditioning_only_and_batch_keeps_every_example():
    tok = BytesTokenizer(); r = example()
    p = training_pack(r, tok, 'reader_teacher', width=256, target_slots=128)
    active = p['future_mask']
    assert p['labels'][active].tolist() == list((r['answer'] + '\n').encode()) + [EOS]
    out = batch_packs([p, p])
    assert out['input_ids'].shape == (2, 256)
    assert out['future_mask'].sum() == 2 * p['future_mask'].sum()
    r['panel'] = 'DEV_structure'
    with pytest.raises(ValueError):
        training_pack(r, tok, 'writer')


def test_full_cot_baseline_can_state_final_operation_with_ordinary_completion_loss():
    r=example();tok=BytesTokenizer()
    p=training_pack(r,tok,'cot_full',width=256,target_slots=128)
    text=bytes(p['labels'][p['future_mask']][:-1].tolist()).decode()
    assert text=='37 + 18 = 55\n92 - 57 = 35\n55 * 35 = 1925\nAnswer:\n1925\n'
    assert not p['note_close_mask'].any()
    # The separate writer still receives intermediate-only supervision.
    w=training_pack(r,tok,'writer',width=256,target_slots=128)
    assert '55 * 35' not in bytes(w['labels'][w['note_close_mask']][:-1].tolist()).decode()


def test_overwidth_and_tampered_partial_are_rejected_without_truncation():
    tok = BytesTokenizer(); r = example()
    with pytest.raises(ValueError):
        training_pack(r, tok, 'cot', width=32)
    r['partial_note'] += '55 * 35 = 1925\n'
    with pytest.raises(ValueError):
        training_pack(r, tok, 'writer')
    r = example(); r['answer'] = '1926'
    with pytest.raises(ValueError):
        training_pack(r, tok, 'reader_teacher')
