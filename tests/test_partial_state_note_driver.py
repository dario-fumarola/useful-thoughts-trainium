import json

import pytest

from train.partial_state_note_driver import SCHEMA, LEGACY_SCHEMA, load_bundle, note_controls
from weekend_thinking.artifacts import sha256_file


def test_intact_other_controls_preserve_failed_and_successful_empty_notes():
    questions=[dict(item_id='a'),dict(item_id='b'),dict(item_id='c')]
    notes={'a':dict(item_id='a',status='error',content_ids=[]),
           'b':dict(item_id='b',status='ok',content_ids=[]),
           'c':dict(item_id='c',status='ok',content_ids=[2,3])}
    controls=note_controls(questions,notes)
    assert [v['own']['status'] for v in controls]==['error','ok','ok']
    assert sorted(v['other']['item_id'] for v in controls)==['a','b','c']
    assert sum(v['other']['status']=='error' for v in controls)==1
    assert all(v['item_id']!=v['other_item_id'] for v in controls)


@pytest.mark.parametrize('schema',[SCHEMA,LEGACY_SCHEMA])
def test_bundle_cannot_replace_an_actual_writer_note_after_generation(tmp_path,schema):
    questions=[dict(item_id='a'),dict(item_id='b')]
    notes={'a':dict(item_id='a',status='ok',content_ids=[2,3]),
           'b':dict(item_id='b',status='error',content_ids=[4])}
    raw=tmp_path/'notes.jsonl';raw.write_text(''.join(json.dumps(v)+'\n' for v in notes.values()))
    protocol=tmp_path/'protocol.json';writer=dict(id='writer-TRAIN128-step128')
    protocol.write_text(json.dumps(dict(schema=schema,writer=writer)))
    bundle=dict(schema=schema,status='complete_all_attempts',ordered_item_ids=['a','b'],
        writer=writer,source_pins={str(p):sha256_file(p) for p in [raw,protocol]},
        controls=note_controls(questions,notes))
    file=tmp_path/'notes-bundle.json';file.write_text(json.dumps(bundle))
    assert load_bundle(file,['a','b'])['controls']==bundle['controls']
    bundle['controls'][0]['own']['content_ids']=[9,9]
    file.write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='actual original writer'):
        load_bundle(file,['a','b'])
