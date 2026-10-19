import json
from pathlib import Path

import pytest

from train.real_trace_scope import SCHEMAS, panel_scope, protocol_items, binding_statuses
from train import real_trace_answers as answers
from weekend_thinking.artifacts import sha256_file


@pytest.mark.parametrize('kind', ['notes','scalar'])
def test_new_cohort_requires_new_identity_without_changing_legacy(kind):
    old, new = SCHEMAS[kind]
    assert panel_scope('DEV_AST256', kind) == (old, 256)
    assert panel_scope('DEV_legacy256', kind) == (old, 256)
    assert panel_scope('DEV_fresh384', kind) == (new, 384)
    assert protocol_items(dict(schema=new, items=384, panel='DEV_fresh384'), kind) == 384
    for schema, count, panel in [(old,384,'DEV_fresh384'), (new,256,'DEV_fresh384'),
                                  (new,384,'DEV_AST256'), (new,512,'DEV_fresh384')]:
        with pytest.raises(ValueError):
            protocol_items(dict(schema=schema, items=count, panel=panel), kind)
    with pytest.raises(ValueError):
        panel_scope('TEST', kind)


def test_new_bindings_are_not_silently_accepted_by_old_panel():
    value = 'FROZEN_AUDITED_REAL_TARGET_TERMINAL320'
    assert value in binding_statuses('DEV_fresh384')
    assert value not in binding_statuses('DEV_AST256')


def test_all_384_native_attempts_per_mode_keep_parser_and_note_failures(tmp_path, monkeypatch):
    """CPU collector fixture;no model/native-hardware qualification is implied."""
    import transformers
    class Tokenizer:
        def get_vocab(self):
            return {str(i):i for i in range(128)} | {'eos':151643}

        def decode(self, ids, **kwargs):
            return ''.join(chr(i) for i in ids)

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a,**k:Tokenizer())
    monkeypatch.setattr(answers, 'sources', lambda:{})
    contexts = [dict(item_id=str(i), document_id=f'doc{i}',
                     contexts={m:[42,43] for m in answers.MODES},
                     note_status=dict(own='error' if i == 0 else 'ok',other='ok')) for i in range(384)]
    (tmp_path/'contexts.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in contexts))
    (tmp_path/'native-contexts.json').write_text('{}')
    (tmp_path/'native-reference.pt').write_bytes(b'CPU collector fixture only')
    origin = tmp_path/'origin'
    origin.mkdir()
    (origin/'protocol.json').write_text(json.dumps(dict(model=dict(id='fixture'))))
    (origin/'notes-bundle.json').write_text('{}')
    gold = tmp_path/'scoring-only-gold.jsonl'
    gold.write_text(''.join(json.dumps(dict(item_id=str(i),numeric_value='12'))+'\n' for i in range(384)))
    protocol = dict(schema=SCHEMAS['scalar'][1],items=384,shards=4,panel='DEV_fresh384',
        modes=list(answers.MODES),max_actions=32,prefill_width=672,stop_ids=[151643],source_files={},
        ordered_item_ids=[v['item_id'] for v in contexts],gold_sha256=sha256_file(gold),
        reused_origin=str(origin),model=dict(id='fixture'),model_dir=str(tmp_path),
        origin_protocol_sha256=sha256_file(origin/'protocol.json'),
        original_notes_sha256=sha256_file(origin/'notes-bundle.json'),
        files={n:sha256_file(tmp_path/n) for n in ['contexts.jsonl','native-contexts.json','native-reference.pt']},limits=[])
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'generation-closure.json').write_text(json.dumps(dict(status='ALL_FOUR_GENERATION_SESSIONS_AND_DEVICE_PHYSICALLY_CLOSED')))
    for shard in range(4):
        d = tmp_path/'answers'/f'shard-{shard}'
        d.mkdir(parents=True)
        values = []
        for i in range(shard,384,4):
            text = '12' if i < 128 else '13' if i < 256 else '12*1'
            for mode in answers.MODES:
                content = [ord(c) for c in text]
                values.append(dict(item_id=str(i),document_id=f'doc{i}',mode=mode,prompt_ids=[42,43],
                    text=text,content_ids=content,actions=content+[151643],n_actions=len(content)+1,
                    stop_type='token:151643',status='ok',wall_seconds=0.,
                    source_note_status=contexts[i]['note_status'].get(mode,'ok')))
        assert len(values) == 480
        file = d/'answers.jsonl'
        file.write_text(''.join(json.dumps(v)+'\n' for v in values))
        (d/'run.json').write_text(json.dumps(dict(status='complete_all_attempts',rows=480,
            protocol_sha256=sha256_file(tmp_path/'protocol.json'),answers_sha256=sha256_file(file))))
        (d/'native.json').write_text(json.dumps(dict(passed=True)))
        (d/'cache-prefill.json').write_text(json.dumps(dict(max_logp_error=0.)))
    result = answers.score(tmp_path,gold)
    # score() writes an immutable summary; inspect it even if it returns None.
    saved = json.loads((tmp_path/'summary.json').read_text())
    assert saved['rows'] == 1920
    for mode in answers.MODES:
        cell = saved['modes'][mode]
        assert cell['documents'] == 384
        assert cell['correct'] == (127 if mode == 'own' else 128)
        assert cell['categories']['unsupported'] == 128
        assert sum(cell['categories'].values()) == 384
    assert saved['modes']['own']['categories']['execution_error'] == 1
