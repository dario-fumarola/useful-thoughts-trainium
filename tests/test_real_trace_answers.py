"""Behavioral checks for scoring-only gold and conservative end-to-end accuracy."""
import json
from pathlib import Path

import pytest

from train import real_trace_answers as answers
from weekend_thinking.artifacts import sha256_file


@pytest.mark.parametrize('text,stop,expected,correct,category', [
    ('0.50\nA derivation follows.', 'length_cap', '1/2', True, 'valid'),
    ('12', 'token:151643', '12', True, 'valid'),
    ('13\n', 'length_cap', '12', False, 'valid'),
    ('12', 'length_cap', '12', False, 'incomplete'),
    ('The answer is 12', 'token:151643', '12', False, 'format_error'),
    ('6*2', 'token:151643', '12', False, 'unsupported'),
    ('1,000', 'token:151643', '1000', False, 'unsupported'),
])
def test_complete_scalar_only_without_search_repair_or_computation(text, stop, expected, correct, category):
    result = answers.classify(dict(text=text, stop_type=stop, status='ok', source_note_status='ok'), expected)
    assert result['correct'] is correct and result['category'] == category


@pytest.mark.parametrize('status,note_status', [('error', 'ok'), ('ok', 'error'), ('error', 'error')])
def test_a_correct_partial_error_is_preserved_but_cannot_count_as_system_success(status, note_status):
    result = answers.classify(dict(text='12', stop_type='token:151643', status=status,
                                  source_note_status=note_status), '12')
    assert result['parsed']['value'] == '12'
    assert not result['correct'] and result['category'] == 'execution_error'


def test_generation_loader_does_not_require_or_open_gold(tmp_path, monkeypatch):
    monkeypatch.setattr(answers, 'sources', lambda: {})
    values = [dict(item_id=str(i), document_id=f'doc{i}',
                   contexts={mode: [42, 43] for mode in answers.MODES},
                   note_status=dict(own='ok', other='ok')) for i in range(256)]
    (tmp_path/'contexts.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
    (tmp_path/'native-contexts.json').write_text('{}')
    (tmp_path/'native-reference.pt').write_bytes(b'fixture; actual model validation belongs to hardware')
    origin = tmp_path/'origin'
    origin.mkdir()
    (origin/'protocol.json').write_text(json.dumps(dict(model=dict(id='fixture'))))
    (origin/'notes-bundle.json').write_text('{}')
    protocol = dict(schema=answers.SCHEMA, items=256, shards=4, modes=list(answers.MODES),
                    max_actions=32, prefill_width=672, stop_ids=[151643], source_files={},
                    ordered_item_ids=[v['item_id'] for v in values], gold_sha256='not-present-and-never-opened',
                    reused_origin=str(origin), model=dict(id='fixture'),
                    origin_protocol_sha256=sha256_file(origin/'protocol.json'),
                    original_notes_sha256=sha256_file(origin/'notes-bundle.json'),
                    files={n: sha256_file(tmp_path/n) for n in
                           ('contexts.jsonl', 'native-contexts.json', 'native-reference.pt')})
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    _, actual = answers.load(tmp_path)
    assert actual == values
    protocol['files']['gold.jsonl'] = 'forbidden'
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='decoding identity'):
        answers.load(tmp_path)
