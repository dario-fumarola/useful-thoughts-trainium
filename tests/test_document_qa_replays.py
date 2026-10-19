import json
from pathlib import Path
import pytest

from train.document_qa_evaluate import prepare_replays,sources
from weekend_thinking.artifacts import sha256_file


class Tok:
    def encode(self,text,add_special_tokens=False):return list(text.encode())


def fixture(tmp_path):
    original=tmp_path/'original';original.mkdir()
    qfile=tmp_path/'questions.jsonl'
    qfile.write_text(''.join(json.dumps(dict(item_id=str(i),question='Which place?',contexts=[dict(title='Place',text='Visible input.')]))+'\n' for i in range(512)))
    for n in ('native-reference.pt','native-contexts.json'):(original/n).write_bytes(b'fixture-not-a-model')
    p=dict(source_files=sources(),purpose='FIXED_HELD_DOCUMENT_QA',modes=['prior'],items=512,shards=4,
        question_file=str(qfile),question_file_sha256=sha256_file(qfile),ordered_item_ids=list(map(str,range(512))),model_dir='fixture',
        files={n:sha256_file(original/n) for n in ('native-reference.pt','native-contexts.json')})
    (original/'protocol.json').write_text(json.dumps(p))
    for shard in range(4):
        d=original/'answers'/f'shard-{shard}';d.mkdir(parents=True)
        attempts=[dict(item_id=str(i),mode='prior',text=f'Trace{i}\nFinal answer: Place\n',status='ok' if i else 'error',stop_type='qa_answer_line' if i else 'error') for i in range(shard,512,4)]
        (d/'attempts.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in attempts))
        (d/'run.json').write_text(json.dumps(dict(status='complete_all_attempts',protocol_sha256=sha256_file(original/'protocol.json'),attempts_sha256=sha256_file(d/'attempts.jsonl'))))
    return original


def test_full_cohort_cyclic_donor_controls_preserve_failed_trace(tmp_path,monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:Tok())
    original=fixture(tmp_path);out=tmp_path/'replays';p=prepare_replays(original,out)
    own=[json.loads(v) for v in (out/'prompts-own_replay.jsonl').read_text().splitlines()]
    other=[json.loads(v) for v in (out/'prompts-other_replay.jsonl').read_text().splitlines()]
    assert len(own)==len(other)==512
    assert own[0]['donor_status']=='error' and own[0]['trace_text']=='Trace0'
    assert [v['donor_item_id'] for v in other]==list(map(str,[*range(1,512),0]))
    assert all(v['donor_item_id']!=v['item_id'] for v in other)
    assert other[-1]['donor_status']=='error' and other[-1]['trace_text']=='Trace0'
    assert p['modes']==['own_replay','other_replay']


def test_missing_original_shard_blocks_controls(tmp_path,monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:Tok())
    original=fixture(tmp_path);(original/'answers/shard-3/run.json').unlink()
    with pytest.raises(FileNotFoundError):prepare_replays(original,tmp_path/'out')
