"""Paired secondary accuracy must retain failures and strict source identities."""
from copy import deepcopy

import pytest

from train.real_cpt_scalar_results import outcomes
from train.scalar_completion import parse_initial_scalar_v2


def fixture():
    sources = {str(i): dict(document_id=f'doc-{i}', numeric_value='42') for i in range(5)}
    rows = [dict(item_id=key, document_id=value['document_id'], mode=mode,
                 text='42\n', stop_type='eos', status='ok', source_note_status='ok',
                 parsed=parse_initial_scalar_v2('42\n', 'eos'))
            for key, value in sources.items() for mode in ('raw', 'empty', 'own')]
    return sources, rows


def test_execution_and_note_errors_remain_paired_zero_outcomes():
    sources, rows = fixture()
    own = [row for row in rows if row['mode'] == 'own']
    own[0]['status'] = 'error'
    own[1]['source_note_status'] = 'error'
    own[2].update(text='unknown', parsed=parse_initial_scalar_v2('unknown', 'eos'))
    own[3].update(text='41\n', parsed=parse_initial_scalar_v2('41\n', 'eos'))
    out = outcomes(rows, sources)
    assert len(out['own']) == len(out['raw']) == 5
    assert sum(row['score'] for row in out['own']) == 1
    assert sum(row['score'] for row in out['raw']) == 5
    assert [row['item_id'] for row in out['own']] == [row['item_id'] for row in out['raw']]


def test_missing_duplicate_and_cross_document_observations_are_rejected():
    sources, rows = fixture()
    with pytest.raises(ValueError, match='all three modes'):
        outcomes(rows[:-1], sources)
    with pytest.raises(ValueError, match='duplicate'):
        outcomes(rows + [rows[0]], sources)
    bad = deepcopy(rows); bad[0]['document_id'] = 'different'
    with pytest.raises(ValueError, match='pairing changed'):
        outcomes(bad, sources)


def test_stored_parser_change_cannot_create_an_extra_correct_answer():
    sources, rows = fixture()
    rows[0]['parsed'] = dict(status='valid', value='41')
    with pytest.raises(ValueError, match='Frozen parser'):
        outcomes(rows, sources)
