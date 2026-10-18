"""Policy averaging, causal controls and closure in the short-note diagnostic."""
import json

import pytest

from train import self_note_pilot as pilot
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file


def question_fixture():
    questions=[dict(item_id=f"gsm8k:train:{i:05d}",document_id=f"gsm8k:train:{i:05d}",
        split="train",task="gsm8k",question=f"Question{i}") for i in range(32)]
    notes={(q["item_id"],s):dict(content_ids=[10+i,20+s]) for i,q in enumerate(questions) for s in range(4)}
    prefixes={q["item_id"]:[100+i,200] for i,q in enumerate(questions)}
    return questions,notes,prefixes


def test_all_samples_have_same_question_prefix_and_empty_preserves_separators():
    qs,notes,prefixes=question_fixture();cells=pilot.build_cells(qs,notes,prefixes)
    assert len(cells)==416 and len({c['item_id'] for c in cells})==416
    assert {c['mode'] for c in cells}==set(pilot.MODES)-{'direct256'}
    assert all(c['answer_cap']==192 for c in cells)
    for c in cells:
        prefix=prefixes[c['question_id']]
        assert c['context_ids'][:len(prefix)]==prefix
        assert c['document_id']==c['question_id']
        if c['mode']=='empty192':assert c['context_ids']==prefix+[198,198]
        elif c['mode']=='own':assert c['context_ids']==prefix+[198]+notes[(c['question_id'],c['sample_index'])]['content_ids']+[198]
        elif c['mode']=='other':
            donor=notes[(c['donor_question_id'],c['sample_index'])]['content_ids']
            assert c['context_ids']==prefix+[198]+donor+[198]
    assert notes[(qs[0]['item_id'],0)]['content_ids']==[10,20]


def test_controls_preserve_length_empty_paths_and_deterministic_donors():
    a=pilot.controls([1,2,3,4,5],[7,8],'q',2)
    assert a==pilot.controls([1,2,3,4,5],[7,8],'q',2)
    assert sorted(a['shuffled'])==[1,2,3,4,5] and a['other']==[7,8,7,8,7]
    assert pilot.controls([],[],'q',2)==dict(own=[],other=[],shuffled=[])
    assert pilot.controls([1,2],[],'q',2)['other']==[220,220]


def metric_fixture():
    ids=['q1','q2'];values=[]
    for q in ids:
        for mode in pilot.MODES:
            scores=[0,1,0,1] if mode in pilot.MODES[:3] else [1]
            for sample,value in enumerate(scores):
                values.append(dict(question_id=q,mode=mode,value=value,sample=sample))
    return ids,values


def test_average_is_single_sample_policy_estimate_not_oracle_or_128_independent_items():
    ids,values=metric_fixture();actual=pilot.aggregate_questions(values,ids,'value')
    assert len(actual['own'])==2 and [r['score'] for r in actual['own']]==[.5,.5]
    assert [r['score'] for r in actual['direct256']]==[1,1]
    paired=pilot.paired_cluster_bootstrap(actual['direct256'],actual['own'],repeats=100)
    assert paired['items']==paired['documents']==2 and paired['difference_b_minus_a']==-.5


@pytest.mark.parametrize('damage',['missing','duplicate','nonfinite'])
def test_all_four_samples_are_required_before_question_aggregation(damage):
    ids,values=metric_fixture()
    if damage=='missing':values.pop(0)
    elif damage=='duplicate':values.append(values[0])
    else:values[0]['value']=float('nan')
    with pytest.raises(ValueError):pilot.aggregate_questions(values,ids,'value')


@pytest.fixture
def closed(tmp_path):
    qs,notes,prefixes=question_fixture();cells=pilot.build_cells(qs,notes,prefixes)
    p=dict(item_ids=[q['item_id'] for q in qs]);x=dict(questions=qs,cells=cells)
    atomic_json(tmp_path/'protocol.json',p)
    for shard in range(4):
        d=tmp_path/'answers'/f'shard-{shard}';d.mkdir(parents=True)
        values=[{**{k:c[k] for k in ('item_id','document_id','question_id','sample_index','mode')},
            'context_sha256':canonical_hash(c['context_ids']),'status':'error','generated_text':''}
            for c in pilot.selected_cells(p,x,shard)]
        (d/'predictions.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
        atomic_json(d/'run.json',dict(status='complete',rows=104,
            protocol_sha256=sha256_file(tmp_path/'protocol.json'),
            rows_sha256=sha256_file(d/'predictions.jsonl'),model_signature={'B0':'exact'}))
    return tmp_path,p,x


def test_execution_failures_close_without_dropping_rows_or_reading_gold(closed):
    output,p,x=closed
    actual,signature=pilot.completed_rows(output,p,x,'answers')
    assert len(actual)==416 and signature=={'B0':'exact'}
    assert not (output/'gold.json').exists()


@pytest.mark.parametrize('damage',['missing','duplicate','context','cluster','identity','running'])
def test_closure_rejects_incomplete_or_resealed_pairing_damage(closed,damage):
    output,p,x=closed;d=output/'answers/shard-1';r=pilot.read(d/'run.json')
    values=pilot.rows(d/'predictions.jsonl')
    if damage=='missing':values.pop()
    elif damage=='duplicate':values[-1]=values[0]
    elif damage=='context':values[0]['context_sha256']='wrong'
    elif damage=='cluster':values[0]['document_id']='wrong'
    elif damage=='identity':r['model_signature']={'B0':'different'}
    else:r['status']='running'
    (d/'predictions.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
    r['rows_sha256']=sha256_file(d/'predictions.jsonl');atomic_json(d/'run.json',r,overwrite=True)
    with pytest.raises(ValueError):pilot.completed_rows(output,p,x,'answers')


def test_gold_projection_cannot_parse_even_existing_known_gold_before_answer_closure(closed,monkeypatch):
    output,p,x=closed
    r=pilot.read(output/'answers/shard-0/run.json');r['status']='running'
    atomic_json(output/'answers/shard-0/run.json',r,overwrite=True)
    monkeypatch.setattr(pilot,'load',lambda output:(p,x))
    with pytest.raises(ValueError,match='must close'):pilot.project_gold(output)
    assert not (output/'gold.json').exists()


@pytest.mark.parametrize('damage',['changed_gold','disagreeing_prior_receipt'])
def test_closed_answers_do_not_allow_gold_that_disagrees_with_pinned_prior_likelihood(closed,monkeypatch,damage):
    output,p,x=closed;previous=output/'previous';previous.mkdir()
    source=previous/'gold.json';source.write_text('{"not":"the sealed gold"}')
    expected='a'*64
    for shard in range(4):
        d=previous/'likelihood'/f'shard-{shard}';d.mkdir(parents=True)
        atomic_json(d/'run.json',dict(gold_sha256='b'*64 if damage=='disagreeing_prior_receipt' and shard==2 else expected))
    p.update(previous_pilot=str(previous),gold_source=str(source),gold_source_sha256=expected)
    monkeypatch.setattr(pilot,'load',lambda output:(p,x))
    with pytest.raises(ValueError,match='all4 pinned'):pilot.project_gold(output)
    assert not (output/'gold.json').exists()


def test_frozen_parser_counts_truncated_and_execution_error_paths_as_zero():
    gold=[dict(item_id='a',answer='#### 5'),dict(item_id='b',answer='#### 5')]
    pred=[dict(item_id='a',status='truncated',generated_text='#### 5'),
          dict(item_id='b',status='error',generated_text='#### 5')]
    scored,_=pilot.score_predictions(gold,pred)
    assert [r['score'] for r in scored]==[0,0]
