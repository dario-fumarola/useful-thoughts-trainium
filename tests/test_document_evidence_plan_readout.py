import pytest

from train.document_evidence_plan_readout import (
    DRAWS, SEED, paired_contrast, pilot_decisions, score_condition,
)


def inputs():
    qs = [dict(item_id=str(i), question='A question', contexts=[
        dict(title='A title', text='Some text')]) for i in range(2)]
    gold = {q['item_id']: dict(answer='Paris', answer_aliases=['City of Paris'], support_indices=[1]) for q in qs}
    return qs, gold


def attempt(key, text, **extra):
    result = dict(item_id=key, mode='selection', parser_mode='prior', status='ok',
        text=text, raw_text=text, stop_type='token:151643', actions=[1, 151643])
    result.update(extra)
    return result


def test_answer_correctness_and_pointer_validity_are_scored_independently():
    qs, gold = inputs()
    rows, summary = score_condition([
        attempt('0', 'Evidence: [0]\nFinal answer:\nParis'),
        attempt('1', 'Evidence: [1]\nFinal answer:\nLondon')], qs, gold, mode='selection')
    assert summary['correct'] == 1 and summary['items'] == 2
    assert rows[0]['canonical_em'] == 1 and not rows[0]['valid_nonempty_selection']
    assert rows[1]['canonical_em'] == 0 and rows[1]['valid_nonempty_selection']


def test_failed_execution_and_unterminated_cap_remain_in_original_denominator():
    qs, gold = inputs()
    rows, summary = score_condition([
        attempt('0', 'Evidence: [1]\nFinal answer:\nParis', status='error', stop_type='error'),
        attempt('1', 'Evidence: [1]\nFinal answer:\nParis', stop_type='length_cap')], qs, gold, mode='selection')
    assert summary['items'] == 2 and summary['correct'] == 0
    assert summary['execution_errors'] == summary['capped'] == 1
    assert summary['parsed_statuses'] == dict(execution_error=1, unsupported=1)
    assert rows[0]['selection']['status'] == 'execution_error'


def test_same_anchored_parser_has_no_bare_answer_recovery_or_alias_substitution():
    qs, gold = inputs()
    rows, summary = score_condition([
        attempt('0', 'Paris', mode='direct'),
        attempt('1', 'Final answer:\nCity of Paris', mode='direct')], qs, gold, mode='direct')
    assert summary['correct'] == 0 and summary['attributed_alias_em'] == .5
    assert rows[0]['parsed']['status'] == 'unsupported'
    assert 'selection' not in rows[1]


def test_missing_or_repeated_original_questions_prevent_partial_readout():
    qs, gold = inputs()
    row = attempt('0', 'Evidence: [1]\nFinal answer:\nParis')
    with pytest.raises(ValueError):
        score_condition([row], qs, gold, mode='selection')
    with pytest.raises(ValueError):
        score_condition([row, row], qs, gold, mode='selection')


def result(point, lower, upper=.2):
    return dict(items=255, difference_treatment_minus_control=point, ci95=[lower, upper],
                bootstrap_draws=DRAWS, bootstrap_seed=SEED)


def test_net_recipe_gain_does_not_establish_inference_assistance():
    decision = pilot_decisions(result(.04, .001), result(.02, -.01), result(.05, .01), 243, 255)
    assert decision['net_recipe_primary_pass'] and decision['selection_integrity_pass']
    assert decision['eligible_for_same_protocol_replication']
    assert not decision['inference_assistance_pass']
    assert not decision['two_seed_success_established']


def test_original_accuracy_and_nonempty_selection_gates_are_not_relaxed():
    assert not pilot_decisions(result(.029, .01), result(.1, .01), result(.1, .01), 255, 255)['net_recipe_primary_pass']
    assert not pilot_decisions(result(.04, 0), result(.1, .01), result(.1, .01), 255, 255)['net_recipe_primary_pass']
    assert not pilot_decisions(result(.04, .01), result(.1, .01), result(.1, .01), 242, 255)['selection_integrity_pass']
    assert pilot_decisions(result(.04, .01), result(.1, .01), result(.1, .01), 243, 255)['inference_assistance_pass']
    with pytest.raises(ValueError):
        pilot_decisions(result(.04, .01), result(.1, .01), result(.1, .01), 255, 256)


def test_paired_bootstrap_uses_original_ids_and_is_order_independent():
    control = {str(i): int(i < 100) for i in range(255)}
    treatment = {str(i): int(i < 120) for i in reversed(range(255))}
    a = paired_contrast(control, treatment)
    b = paired_contrast(dict(reversed(list(control.items()))), treatment)
    assert a == b and a['items'] == 255
    assert a['difference_treatment_minus_control'] == pytest.approx(20/255)
    assert a['ci95'][0] > 0 and a['bootstrap_draws'] == DRAWS
    with pytest.raises(ValueError):
        paired_contrast(control, {**treatment, 'padding': 1})
