"""Answer coefficients and available-input compute controls survive packing."""
import pytest
import torch

from train.permutation_program_data import family, question_from_dict
from train.permutation_program_packing import (EOS, FINAL, PAUSES_PER_STEP,
    inference_context, pause_id, training_pack, view_for)


class Characters:
    eos_token_id = EOS

    def encode(self, text, add_special_tokens=False):
        return [ord(v) for v in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(v) for v in ids)

    def get_vocab(self):
        return {'<|fim_pad|>': 151662}

    def convert_tokens_to_ids(self, text):
        return self.get_vocab().get(text)

    def convert_ids_to_tokens(self, value):
        return next((k for k, v in self.get_vocab().items() if v == value), None)


def test_answer_tokens_and_coefficients_are_identical_across_views():
    row = family('TRAIN', 5, 19)[0]; tok = Characters()
    all_answers = []
    for view in ('direct', 'empty', 'trace', 'pause'):
        packed = training_pack(row, tok, view, width=1024, target_slots=256)
        active = packed['answer_mask'] | packed['work_mask']
        positions = packed['target_positions'][active]
        assert not (packed['answer_mask'] & packed['work_mask']).any()
        assert torch.equal(packed['labels'][active], packed['input_ids'][0, positions])
        assert packed['labels'][packed['answer_mask']].tolist() == tok.encode(FINAL + row['answer']) + [EOS]
        all_answers.append(packed['labels'][packed['answer_mask']].tolist())
        assert packed['work_mask'].any().item() == (view == 'trace')
        assert all(p >= 1 for p in positions.tolist())
    assert all(v == all_answers[0] for v in all_answers)


def test_pauses_are_known_inputs_never_prediction_targets_or_stops():
    tok = Characters(); row = family('TRAIN', 4, 29)[0]
    q = question_from_dict(row['question'])
    pack = training_pack(row, tok, 'pause', width=1024, target_slots=256)
    pause_positions = (pack['input_ids'][0] == pause_id(tok)).nonzero()[:, 0]
    assert len(pause_positions) == PAUSES_PER_STEP * len(q.program)
    selected = set(pack['target_positions'][pack['answer_mask'] | pack['work_mask']].tolist())
    assert selected.isdisjoint(pause_positions.tolist())
    assert pack['labels'][pack['answer_mask']][-1] == EOS != pause_id(tok)
    assert inference_context(q, tok, 'pause').count(pause_id(tok)) == len(pause_positions)
    # Inference gets no answer/teacher row; true labels cannot select the budget.
    row['answer'] = 'POISON'
    assert inference_context(q, tok, 'pause').count(pause_id(tok)) == 6 * len(q.program)
    with pytest.raises(ValueError):
        training_pack(row, tok, 'pause', width=1024, target_slots=256)


def test_every_question_gets_both_training_views_and_held_labels_are_rejected():
    for arm in ('direct', 'trace', 'pause'):
        for i in range(12):
            views = [view_for(arm, epoch, i) for epoch in range(4)]
            assert views.count('direct') == 2
            assert views.count('empty' if arm == 'direct' else arm) == 2
    held = family('DEV_depth', 0, 17)[0]
    with pytest.raises(ValueError):
        training_pack(held, Characters(), 'trace', width=1024, target_slots=256)


def test_missing_pause_embedding_or_small_packs_fail_explicitly():
    class NoPause(Characters):
        def get_vocab(self):
            return {}
    row = family('TRAIN', 3, 7)[0]
    with pytest.raises(ValueError):
        training_pack(row, NoPause(), 'pause', width=1024, target_slots=256)
    with pytest.raises(ValueError):
        training_pack(row, Characters(), 'trace', width=16, target_slots=2)
