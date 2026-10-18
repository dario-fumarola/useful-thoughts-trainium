"""Boundaries specific to the question-only calculation diagnostic."""
import json

import pytest

from train import calculation_pilot as pilot
from weekend_thinking.artifacts import atomic_json, sha256_file


def test_partial_cot_preserves_complete_question_and_both_empty_separators():
    prefix=[11,12,13]
    controls=dict(filler=[4,4],shuffled=[7,6],other=[8,9])
    values=pilot.make_contexts(prefix,[6,7],[20,21],[20,22],controls,controls)
    assert set(values)==set(pilot.MODES)
    assert values['raw']==prefix and values['empty']==prefix+[198,198]
    assert values['generated']==prefix+[198,6,7,198]
    assert values['verified']==prefix+[198,20,21,198]
    assert values['corrupted']==prefix+[198,20,22,198]
    assert all(v[:len(prefix)]==prefix for v in values.values())
    values['raw'][0]=999
    assert prefix==[11,12,13] and values['empty']==[11,12,13,198,198]


@pytest.fixture
def closed(tmp_path):
    ids=[f'gsm8k:train:{i:05d}' for i in range(32)]
    p=dict(item_ids=ids)
    atomic_json(tmp_path/'protocol.json',p)
    for s in range(4):
        directory=tmp_path/'answers'/f'shard-{s}';directory.mkdir(parents=True)
        values=[dict(item_id=k,mode=m,status='error',generated_text='',n_generated_tokens=0)
                for k in ids[s::4] for m in pilot.MODES]
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
        atomic_json(directory/'run.json',dict(status='complete',rows=80,shard=s,
            protocol_sha256=sha256_file(tmp_path/'protocol.json'),
            predictions_sha256=sha256_file(directory/'predictions.jsonl'),model_signature={'weights':'actual-base'}))
    return tmp_path,p


def test_complete_answer_generation_can_include_failures_without_gold_access(closed):
    output,p=closed
    assert len(pilot.closed_answers(output,p))==320
    assert not (output/'gold.json').exists()


@pytest.mark.parametrize('damage',['partial','missing','duplicate','identity','hash'])
def test_gold_barrier_rejects_incomplete_resealed_or_mismatched_answer_shards(closed,damage):
    output,p=closed; d=output/'answers'/'shard-2';r=pilot.read(d/'run.json')
    if damage=='partial':r['status']='running'
    elif damage=='identity':r['model_signature']={'weights':'different'}
    elif damage=='hash':r['predictions_sha256']='0'*64
    else:
        values=pilot.rows(d/'predictions.jsonl')
        if damage=='missing':values.pop()
        else:values[-1]=values[0]
        (d/'predictions.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
        r['predictions_sha256']=sha256_file(d/'predictions.jsonl')
    atomic_json(d/'run.json',r,overwrite=True)
    with pytest.raises(ValueError):pilot.closed_answers(output,p)


def test_project_gold_never_opens_source_before_generation_closure(closed,monkeypatch):
    output,p=closed
    r=pilot.read(output/'answers/shard-3/run.json');r['status']='running'
    atomic_json(output/'answers/shard-3/run.json',r,overwrite=True)
    monkeypatch.setattr(pilot,'load',lambda path:(p,{}))
    # An absent source proves no source stat/read/hash occurs before the gate.
    with pytest.raises(ValueError,match='before gold access'):
        pilot.project_gold(output,output/'forbidden-source.parquet')
    assert not (output/'gold.json').exists()


def test_answer_failures_count_even_when_partial_text_contains_right_number():
    gold=[dict(item_id='a',answer='#### 12'),dict(item_id='b',answer='#### 12')]
    predictions=[dict(item_id='a',status='truncated',generated_text='#### 12'),
                 dict(item_id='b',status='error',generated_text='#### 12')]
    rows,summary=pilot.score_predictions(gold,predictions)
    assert summary['items']==2 and summary['accuracy']==0
    assert [r['score'] for r in rows]==[0,0]


def test_numeric_and_greedy_metrics_remain_separate():
    assert ('corrupted','verified') in pilot.CONTRASTS
    assert ('other','generated') in pilot.CONTRASTS
    assert ('raw','generated') in pilot.CONTRASTS
    assert ('verified_other','verified') in pilot.CONTRASTS
    assert pilot.numeric_answer('untrusted reasoning\n#### 1,200\n')=='1200'
    assert pilot.numeric_answer('1200') is None
