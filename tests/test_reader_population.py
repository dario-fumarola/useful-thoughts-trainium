import pytest

from scripts.census_reader_population import position_summary, aggregate


def record(length=1024, accepted=(600,), rejected=(700,)):
    return dict(document_id='same-document', split='train', bounded_native_tokens=length,
        status='accepted' if accepted else 'no_valid_bounded_raw_equation',
        selected_absolute_cut=accepted[0] if accepted else None,
        accepted_candidate_count=len(accepted),
        candidate_audit=[dict(absolute_cut=c, status='accepted') for c in accepted]+
            [dict(absolute_cut=c, status='source_numeric_mismatch') for c in rejected])


def test_prediction_positions_and_document_presence_are_different_denominators():
    row = position_summary(record())
    assert row['positions'] == 1024-16-512+1
    assert row['verified_boundaries'] == 1 and row['pre_label_boundaries'] == 2
    summary = aggregate([row])
    assert summary['document_contains_verified_prevalence'] == 1
    assert summary['position_weighted_verified_prevalence'] == 1/497


def test_hash_position_is_unchanged_by_future_labels():
    a = position_summary(record(accepted=(600,), rejected=(700,)))
    b = position_summary(record(accepted=(700,), rejected=(600,)))
    assert a['chosen_cut'] == b['chosen_cut']
    assert 512 <= a['chosen_cut'] <= 1008


def test_short_sources_and_early_equations_do_not_enter_full_prefix_population():
    a = position_summary(record(500, accepted=(200,), rejected=()))
    b = position_summary(record(528, accepted=(512,), rejected=()))
    c = position_summary(record(1024, accepted=(200,), rejected=()))
    assert a['positions'] == 0 and a['chosen_cut'] is None
    assert b['positions'] == 1 and b['chosen_cut'] == 512 and b['chosen_verified'] == 1
    assert c['verified_boundaries'] == 0


def test_document_and_position_weighted_expectations_retain_their_estimands():
    a = position_summary(record(528, accepted=(512,), rejected=()))
    b = position_summary(record(627, accepted=(), rejected=()))
    summary = aggregate([a, b])
    assert summary['document_weighted_expected_verified_prevalence'] == .5
    assert summary['position_weighted_verified_prevalence'] == 1/101


def test_curator_accounting_and_duplicate_candidates_cannot_be_silently_ignored():
    bad = record(); bad['accepted_candidate_count'] = 9
    with pytest.raises(ValueError): position_summary(bad)
    with pytest.raises(ValueError): position_summary(record(accepted=(600, 600)))
    bad = record(); bad['candidate_audit'][0]['status'] = 'new_undefined_status'
    with pytest.raises(ValueError): position_summary(bad)


def test_bounds_exclude_foreign_splits_and_unobserved_tail():
    bad = record(); bad['split'] = 'test'
    with pytest.raises(ValueError): position_summary(bad)
    with pytest.raises(ValueError): position_summary(record(accepted=(1010,)))
    assert position_summary(record(8208, accepted=(8192,), rejected=()))['positions'] == 7681


def test_natural_confirmation_selection_ignores_candidate_outcomes():
    from scripts.prepare_reader_confirmation import select_natural
    rows = [dict(document_id=f'doc-{i}', used=False, positions=500,
                 chosen_cut=512+i, chosen_verified=i%2) for i in range(20)]
    a = select_natural(iter(rows), 8)
    changed = [dict(r, chosen_verified=1-r['chosen_verified'], future_score=-999,
                    accepted_candidate_count=100) for r in reversed(rows)]
    assert a == select_natural(iter(changed), 8)
    assert len({r['document_id'] for r in a}) == 8
    rows[0]['used'] = True
    rows[1]['positions'] = 0
    assert not {'doc-0', 'doc-1'} & {r['document_id'] for r in select_natural(rows, 8)}


def test_natural_confirmation_never_fills_a_shortfall_with_used_sources():
    from scripts.prepare_reader_confirmation import select_natural
    rows = [dict(document_id='old', used=True, positions=100, chosen_cut=512)]
    with pytest.raises(ValueError): select_natural(rows, 1)
