import math

import pytest

from train.document_evidence_plan_likelihood import (
    calibration_check, context_text, physical_batch, persist_calibration, score_original_batch,
    scoring_example, selected_ids,
)
from train.useful_note_sft_data import EOS


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))


def question():
    return dict(item_id='q', question='Which place?', contexts=[
        dict(title='One', text='A fact.'), dict(title='Two', text='Another fact.')])


def test_native_generated_answer_is_never_replayed_into_likelihood_context():
    attempt = dict(status='ok', raw_text='Evidence: [2] [1]\nFinal answer:\nWRONG CLAIM\n')
    indices, flag = selected_ids(attempt, 2)
    assert indices == [2, 1] and flag['fallback'] is False
    record = scoring_example(question(), 'Paris', 'selected_id_replay', CharacterTokenizer(), indices)
    context = ''.join(map(chr, record['context_ids']))
    assert context.endswith('Evidence: [2] [1]\nFinal answer:\n')
    assert 'WRONG CLAIM' not in context and 'Paris' not in context
    assert record['target_ids'] == [*map(ord, 'Paris'), EOS]


@pytest.mark.parametrize('attempt,reason', [
    (dict(status='error', raw_text='Evidence: [1]\nFinal answer:\nParis'), 'generation_error'),
    (dict(status='ok', raw_text='Evidence:\nFinal answer:\nParis'), 'plan_empty'),
    (dict(status='ok', raw_text='Evidence: [1] [1]\nFinal answer:\nParis'), 'plan_unsupported'),
    (dict(status='ok', raw_text='Evidence: [1]\nA factual claim.\nFinal answer:\nParis'), 'plan_unsupported'),
])
def test_failed_or_invalid_original_selection_is_explicit_empty_not_dropped(attempt, reason):
    indices, flag = selected_ids(attempt, 2)
    assert indices == [] and flag == dict(fallback=True, reason=reason, **(
        {} if reason == 'generation_error' else {'parser_reason': flag['parser_reason']}))
    assert context_text(question(), 'selected_id_replay', indices) == context_text(question(), 'empty')


def test_donor_pointers_are_not_clipped_or_sorted_to_fit_the_recipient():
    with pytest.raises(ValueError):
        context_text(question(), 'other_id_replay', [3])
    with pytest.raises(ValueError):
        context_text(question(), 'other_id_replay', [1, 1])
    assert context_text(question(), 'other_id_replay', [2, 1]).endswith(
        'Evidence: [2] [1]\nFinal answer:\n')


def test_physical_padding_never_creates_an_independent_scoring_observation():
    batch = [dict(context_ids=[1], target_ids=[2]), dict(context_ids=[3], target_ids=[4])]
    seen = []

    def scorer(backend, contexts, targets, *, width):
        seen.append((contexts, targets, width))
        return [dict(full_logp=-target[0]) for target in targets]

    values, padding = score_original_batch(None, batch, 3584, scorer)
    assert values == [dict(full_logp=-2), dict(full_logp=-4)]
    assert padding == 2
    assert seen == [([[1], [3], [3], [3]], [[2], [4], [4], [4]], 3584)]
    with pytest.raises(ValueError):
        physical_batch([])
    with pytest.raises(ValueError):
        physical_batch(batch*3)


def reference(tokens=16, shift=0.0):
    return [dict(tokens=tokens, token_logps=[-1.0+shift]*tokens,
                 full_logp=tokens*(-1.0+shift)) for _ in range(4)]


def test_calibration_requires_both_token_and_mean_span_tolerances():
    assert calibration_check(reference(shift=.00001), reference())['passed']
    result = calibration_check(reference(shift=.0002), reference())
    assert result['max_token_error'] < .001
    assert result['absolute_mean_span_error'] > .001
    assert not result['passed']
    assert not calibration_check(reference(tokens=15), reference())['passed']
    assert not calibration_check(reference(shift=math.nan), reference())['passed']


def test_complete_native_answer_span_cannot_be_shortened_to_fit():
    with pytest.raises(ValueError):
        scoring_example(question(), 'x'*32, 'direct', CharacterTokenizer())


def test_two_calibration_panels_persist_without_clobbering_original_witnesses(tmp_path):
    import json

    checks = dict(QA=dict(passed=True, max_token_error=.00001))
    persist_calibration(tmp_path, 'QA', checks)
    original = (tmp_path/'own-model-TRAIN-calibration-QA.json').read_bytes()
    checks['ordinary'] = dict(passed=True, absolute_mean_span_error=.00002)
    persist_calibration(tmp_path, 'ordinary', checks)
    assert json.loads((tmp_path/'own-model-TRAIN-calibration.json').read_text()) == checks
    assert (tmp_path/'own-model-TRAIN-calibration-QA.json').read_bytes() == original
    assert json.loads((tmp_path/'own-model-TRAIN-calibration-ordinary.json').read_text()) == checks['ordinary']
    with pytest.raises(FileExistsError):
        persist_calibration(tmp_path, 'QA', checks)


def test_failed_second_panel_is_saved_before_the_caller_can_raise(tmp_path):
    import json

    checks = dict(QA=dict(passed=True))
    persist_calibration(tmp_path, 'QA', checks)
    checks['ordinary'] = dict(passed=False, reason='parity_bound_failed')
    persist_calibration(tmp_path, 'ordinary', checks)
    assert json.loads((tmp_path/'own-model-TRAIN-calibration.json').read_text()) == checks
    assert json.loads((tmp_path/'own-model-TRAIN-calibration-ordinary.json').read_text()) == checks['ordinary']
