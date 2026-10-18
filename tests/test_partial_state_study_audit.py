"""Prevent partial-review and comparator omissions in the matched study audit."""
import json

import pytest

from scripts.audit_partial_state_study import ARMS, PANELS, SEEDS, require_complete


def registry(tmp_path):
    jobs = {}
    for seed in SEEDS:
        jobs[seed] = {}
        for arm in ARMS:
            control = tmp_path/seed/arm
            control.mkdir(parents=True)
            (control/'completion.json').write_text(json.dumps(
                {'status': 'COMPLETE_ALL_FOUR_FIXED_DEV_PANELS'}))
            jobs[seed][arm] = {'control': str(control),
                               'outputs': {p: str(control/p) for p in PANELS}}
    base = tmp_path/'B0'
    base.mkdir()
    (base/'completion.json').write_text('{"status":"COMPLETE_ALL_FOUR_FIXED_DEV_PANELS"}')
    return dict(schema='partial-state-v7-matched5120-evaluation-registry-v1',
                seeds=list(SEEDS), panels=list(PANELS), action_budget=96, jobs=jobs,
                untouched_reference=dict(control=str(base), outputs={p: str(base/p) for p in PANELS},
                                         modes=['direct', 'cot', 'empty']))


def test_no_outcomes_required_to_check_entire_question_closure(tmp_path):
    value = registry(tmp_path)
    # There are deliberately no gold, answer, model or note files in this fixture.
    require_complete(value)
    last = value['jobs'][SEEDS[-1]]['cot_full']
    (tmp_path/SEEDS[-1]/'cot_full'/'completion.json').write_text(
        '{"status":"running"}')
    with pytest.raises(ValueError, match='Every original controller'):
        require_complete(value)


@pytest.mark.parametrize('removed', ['cot', 'cot_full', 'direct', 'reader', 'writer'])
def test_no_competitor_or_split_component_can_be_dropped(tmp_path, removed):
    value = registry(tmp_path)
    del value['jobs'][SEEDS[0]][removed]
    with pytest.raises(ValueError, match='Both ordinary CoT'):
        require_complete(value)


def test_old_inference_budget_cannot_be_silently_reused(tmp_path):
    value = registry(tmp_path)
    value['action_budget'] = 80
    with pytest.raises(ValueError, match='96-action'):
        require_complete(value)


def test_untouched_reference_must_close_before_any_outcome_review(tmp_path):
    value = registry(tmp_path)
    (tmp_path/'B0'/'completion.json').write_text('{"status":"running"}')
    with pytest.raises(ValueError, match='untouched reference must close'):
        require_complete(value)


def test_untouched_reference_cannot_be_omitted(tmp_path):
    value = registry(tmp_path)
    del value['untouched_reference']
    with pytest.raises(ValueError, match='untouched reference is required'):
        require_complete(value)
