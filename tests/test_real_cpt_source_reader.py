"""Reserved payloads must remain opaque, even when their JSON is invalid."""
import json

import pytest

from scripts.check_real_cpt_scaffold_data import native_selection


def test_reserved_test_payload_is_never_parsed(tmp_path):
    good = dict(id='selected', document_id='train-doc', split='train', prefix=[1], future=[2])
    p = tmp_path/'records.jsonl'
    p.write_bytes(json.dumps(good, separators=(',',':')).encode() + b'\n'
        + b'{"id":"reserved","document_id":"test-doc","split":"test","prefix":INVALID JSON PAYLOAD\n'
        + b'{"id":"unused","document_id":"dev-doc","split":"dev","prefix":ALSO UNPARSED\n')
    selected, counts = native_selection(p, {'selected'})
    assert selected == {'selected': good}
    assert counts == {'headers:train':1,'payloads:train':1,'headers:test':1,'headers:dev':1}


def test_unknown_or_missing_source_headers_fail_closed(tmp_path):
    p = tmp_path/'records.jsonl'
    p.write_text('{"id":"one","split":"unknown","prefix":[]}\n')
    with pytest.raises(ValueError): native_selection(p, set())
    p.write_text('{"id":"one","document_id":"d","split":"dev","prefix":[]}\n')
    with pytest.raises(ValueError): native_selection(p, {'missing'})
