"""Closed teacher barrier, same-condition controls, paired utility and failure retention."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from train import teacher_prepend_probe as probe
from weekend_thinking.artifacts import canonical_hash


def fixture():
    ids = [f'train:{i}' for i in range(16)]
    docs = [f'doc:{i}' for i in range(16)]
    p = dict(schema=probe.teacher.SCHEMA, source_files=probe.teacher.sources(),
        selection=dict(split='train',count=16,item_ids=ids,document_ids=docs,item_ids_sha256=canonical_hash(ids)),
        conditions=list(probe.CONDITIONS),future_exposure=dict(prefix_only=False,hindsight=True),
        total_notes=32,examples_per_condition=16,batch_size=4,group_size=1,cap=256,temperature=.7,
        compute='fp32',baseline_atol=.002,replay_atol=.001,native_vocab={'equal':True,'entries':4},
        format=probe.teacher.teacher_format(23).__dict__,
        prompt_width=532,batch_seeds=[probe.teacher.batch_seed(ids[i:i+4],i//4) for i in range(0,16,4)])
    p['parent_selection'] = dict(p['selection'],count=128,
        item_ids=ids+[f'train:{i}' for i in range(16,128)],
        document_ids=docs+[f'doc:{i}' for i in range(16,128)])
    p['parent_selection']['item_ids_sha256']=canonical_hash(p['parent_selection']['item_ids'])
    rows = [dict(item_id=k,document_id=d,split='train',prefix=[10]*512,future=[11]*16) for k,d in zip(ids,docs)]
    inputs = dict(rows=rows,template={'token_segments':[[20],[21],[22,23]]},prompts={})
    notes = {}; report = dict(status='complete',notes=32,
        forward_checks={c:dict(atol=.002,max_error=.0004) for c in probe.CONDITIONS},
        replay_checks={c:dict(atol=.001,max_error=.00001) for c in probe.CONDITIONS})
    for c in probe.CONDITIONS:
        inputs['prompts'][c] = [probe.teacher.build_prompt(r['prefix'],r['future'] if c=='hindsight' else None,
                                                        inputs['template']['token_segments']) for r in rows]
        notes[c] = [dict(item_id=k,document_id=d,split='train',condition=c,future_exposed=c=='hindsight',
            batch=i//4,batch_row=i%4,batch_seed=p['batch_seeds'][i//4],
            prompt_sha256=canonical_hash(inputs['prompts'][c][i]),
            actions=[10,151645],action_mask=[True]*2+[False]*254,behavior_logps=[-1.,-2.],
            content_ids=[10],text='note',words=1,instruction_word_limit_met=True,stop_type='token:151645',
            decode_iterations=2,sampling_batch_wall_seconds=3.,replay_batch_wall_seconds=.1 if i<4 else 0.,
            first_batch_replay_max_error=.00001 if i<4 else None) for i,(k,d) in enumerate(zip(ids,docs))]
    return p,inputs,report,notes,{10,11,151643,151645}


def test_exact_complete_bundle_accepts_empty_and_both_teacher_stop_ids():
    args = fixture(); p,inputs,report,notes,defined = args
    for condition in probe.CONDITIONS:
        n=notes[condition][0]
        n.update(actions=[151643],action_mask=[True]+[False]*255,behavior_logps=[-1.],content_ids=[],
                 text='',words=0,instruction_word_limit_met=True,stop_type='token:151643')
    probe.validate_notes(*args)
    assert inputs['prompts']['prefix_only'][0] != inputs['prompts']['hindsight'][0]


@pytest.mark.parametrize('mutation,match',[
    (lambda p,i,r,n: r.update(status='running'),'All32'),
    (lambda p,i,r,n: n['hindsight'].pop(),'all16'),
    (lambda p,i,r,n: r['forward_checks']['hindsight'].update(max_error=.0021),'gate failed'),
    (lambda p,i,r,n: r['replay_checks'].pop('prefix_only'),'All32'),
    (lambda p,i,r,n: n['prefix_only'][0]['behavior_logps'].__setitem__(0,float('nan')),'support'),
    (lambda p,i,r,n: n['prefix_only'][0]['actions'].__setitem__(0,999),'support'),
    (lambda p,i,r,n: n['prefix_only'][0]['action_mask'].__setitem__(3,True),'support'),
    (lambda p,i,r,n: n['prefix_only'][0].update(future_exposed=True),'support'),
    (lambda p,i,r,n: i['prompts']['prefix_only'].__setitem__(0,i['prompts']['hindsight'][0]),'support'),
    (lambda p,i,r,n: n['prefix_only'][0].update(content_ids=[151645]),'stop/content'),
    (lambda p,i,r,n: n['prefix_only'][0].update(stop_type='length_cap'),'stop/content'),
    (lambda p,i,r,n: n['prefix_only'][1].update(sampling_batch_wall_seconds=4.),'shared B4'),
    (lambda p,i,r,n: p['parent_selection']['item_ids'].__setitem__(0,'changed'),'first16'),
])
def test_partial_failed_or_changed_teacher_is_rejected(mutation,match):
    p,inputs,report,notes,defined = fixture()
    mutation(p,inputs,report,notes)
    with pytest.raises(ValueError,match=match):probe.validate_notes(p,inputs,report,notes,defined)


def test_controls_use_next_global_item_within_condition_and_exact_recipient_lengths():
    p,_,_,notes,_ = fixture(); ids=p['selection']['item_ids']
    for c,offset in [('prefix_only',0),('hindsight',100)]:
        for i,note in enumerate(notes[c]):note['content_ids']=[offset+i]*(i%3)
    controls=probe.build_controls(ids,notes,[70,71],17)
    assert controls == probe.build_controls(ids,notes,[70,71],17)
    indexed={(r['item_id'],r['condition']):r for r in controls}
    for c in probe.CONDITIONS:
        for i,key in enumerate(ids):
            row=indexed[key,c];content=notes[c][i]['content_ids']
            donor=notes[c][(i+1)%16]['content_ids'] or [70,71]
            assert row['donor_item_id']==ids[(i+1)%16]
            assert row['content_ids']['own']==content and row['content_ids']['empty']==[]
            assert all(len(row['content_ids'][m])==len(content) for m in ('own','filler','shuffled','other'))
            assert sorted(row['content_ids']['shuffled'])==sorted(content)
            assert row['content_ids']['other']==(donor*256)[:len(content)]
    prefix=list(range(512));rows=probe.contexts(prefix,ids[1],indexed)
    assert len(rows)==11 and rows[0]==('raw','raw',prefix)
    for c,m,context in rows[1:]:
        assert context==indexed[ids[1],c]['content_ids'][m]+[271]+prefix


def test_copying_counts_token_ngrams_and_longest_contiguous_span():
    future=list(range(16))
    result=probe.copying([99]+future[2:11]+[98]+future[:4],future)
    assert result['longest_contiguous_copied_span']==9
    assert result['future_4gram_note_occurrences']==7 and result['future_8gram_note_occurrences']==2
    assert result['has_future_4gram'] and result['has_future_8gram']
    assert probe.copying([],future)['longest_contiguous_copied_span']==0


def score_fixture():
    old,inputs,_,notes,_ = fixture()
    p=dict(selection=old['selection'],experiment_id=probe.SCHEMA)
    rows=[]
    for key,doc in zip(p['selection']['item_ids'],p['selection']['document_ids']):
        for condition,mode,context in [('raw','raw',[])]+[(c,m,[]) for c in probe.CONDITIONS for m in probe.MODES]:
            value=-3. if condition=='raw' else -5.
            if mode=='own':value=-4. if condition=='prefix_only' else -2.
            rows.append(dict(item_id=key,document_id=doc,split='train',condition=condition,mode=mode,
                             status='ok',token_log_likelihoods=[value]+[0.]*15))
    return p,inputs,notes,rows


def test_reducer_retains_absolute_arms_paired_conditions_and_all_document_denominators():
    p,inputs,notes,rows=score_fixture();result=probe.reduce_rows(p,rows,inputs,notes)
    contrasts=result['paired_document_contrasts']
    assert result['status']=='complete' and result['rows']==176 and len(result['per_document'])==32
    assert result['absolute_mean_log_likelihoods']['raw/K16']==-3.
    assert contrasts['prefix_only/own_minus_empty/K16']['difference_b_minus_a']==1.
    assert contrasts['hindsight/own_minus_raw/K16']['difference_b_minus_a']==1.
    delta=contrasts['hindsight_minus_prefix_only/own_minus_empty/K16']
    assert delta['difference_b_minus_a']==2. and delta['documents']==delta['items']==16
    assert delta['bootstrap_draws']==5000 and delta['bootstrap_seed']==17
    rows[3].update(status='error',error='fixture failure');rows[3].pop('token_log_likelihoods')
    result=probe.reduce_rows(p,rows,inputs,notes)
    assert result['status']=='incomplete' and result['failures']==1 and len(result['per_document'])==32
    assert result['paired_document_contrasts']=={} and result['absolute_mean_log_likelihoods']=={}
    with pytest.raises(ValueError,match='exact all16'):probe.reduce_rows(p,rows[:-1],inputs,notes)
    with pytest.raises(ValueError,match='exact all16'):probe.reduce_rows(p,rows+[rows[0]],inputs,notes)


def test_closed_barrier_precedes_reader_model(tmp_path,monkeypatch):
    def missing(_):raise ValueError('Teacher not complete')
    monkeypatch.setattr(probe,'load_inputs',missing)
    monkeypatch.setattr(probe,'setup_model',lambda *a,**k:pytest.fail('Reader before teacher closure'))
    with pytest.raises(ValueError,match='Teacher not complete'):probe.score(tmp_path,'cpu')
    assert not (tmp_path/'scoring').exists()


def test_running_teacher_rejected_before_any_saved_future_or_notes_are_read(tmp_path):
    (tmp_path/'generation').mkdir()
    (tmp_path/'generation/run.json').write_text(json.dumps(dict(status='running',notes=32)))
    with pytest.raises(ValueError,match='fully closed teacher'):
        probe.closed_teacher(tmp_path,tmp_path/'absent-control')
    (tmp_path/'generation/run.json').write_text(json.dumps(dict(status='complete',notes=32)))
    with pytest.raises(ValueError,match='fully closed teacher'):
        probe.closed_teacher(tmp_path,tmp_path/'absent-control')


@pytest.mark.parametrize('values',[[float('nan')]*16,[-1.]*15])
def test_score_attempts_all_rows_and_retains_nonfinite_or_short_failure(tmp_path,monkeypatch,values):
    old,inputs,_,notes,_=fixture(); ids=old['selection']['item_ids']
    p=dict(model_dir='unopened-B0',model={'fixture':'B0'})
    controls={(r['item_id'],r['condition']):r for r in probe.build_controls(ids,notes,[70],17)}
    refs={key:[-1.]*16 for key in ids};(tmp_path/'protocol.json').write_text('{}')
    monkeypatch.setattr(probe,'load_inputs',lambda _:(p,inputs,notes,{},controls,refs))
    monkeypatch.setattr(probe,'setup_model',lambda *a,**k:(None,None,None,p['model'],None))
    monkeypatch.setattr(probe,'score_continuations',lambda *a,**k:[SimpleNamespace(token_log_likelihoods=values)])
    with pytest.raises(RuntimeError,match='failed rows retained'):probe.score(tmp_path,'cpu')
    d=tmp_path/'scoring';rows=[json.loads(s) for s in (d/'scores.jsonl').read_text().splitlines()]
    report=json.loads((d/'run.json').read_text())
    assert len(rows)==176 and all(r['status']=='error' for r in rows)
    assert report['status']=='failed' and report['failures']==176 and report['rows']==176
    assert all('sixteen finite' in r['error'] for r in rows)
