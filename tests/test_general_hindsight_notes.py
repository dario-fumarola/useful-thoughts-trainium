"""Causal serialization, genuine-path retention and full-document denominators."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train import general_hindsight_notes as e
from weekend_thinking.artifacts import canonical_hash


class Tok:
    def decode(self, ids, **kwargs):
        return ''.join({1:'a',2:'\nNext passage',3:'b',4:' '}.get(i,str(i)) for i in ids)


def fixture():
    source=dict(item_id='a',document_id='doc-a',split='train',prefix_ids=[10,11],natural_future16_ids=[3]*16,reanchor_ids=[11])
    p=dict(teacher_parts=dict(initial=[20],middle=[21],final=[22]),vocab_size=151936,banned_ids=[9],parts=dict(old_open=[30],old_close=[31]))
    return source,p


def note(source,p,actions,status='ok'):
    raw=actions[:-1] if actions and actions[-1]==151643 else actions
    extraction=e.truncate_content(raw,Tok())
    return dict(item_id=source['item_id'],document_id=source['document_id'],status=status,actions=actions,raw_content_ids=raw,
        raw_text=e.decode(Tok(),raw),prefix_sha256=canonical_hash(source['prefix_ids']),future_sha256=canonical_hash(source['natural_future16_ids']),
        teacher_prompt_sha256=canonical_hash(e.teacher_prompt(source,p['teacher_parts'])),n_actions=len(actions),wall_seconds=1.,
        extraction=extraction,content_ids=extraction['content_ids'],content_text=extraction['content_text'],
        stop_type='token:151643' if actions and actions[-1]==151643 else 'length_cap')


def test_teacher_prompt_is_explicit_hindsight_and_native_prefixes_are_never_reencoded():
    r,p=fixture();prefix=copy.deepcopy(r['prefix_ids'])
    assert e.teacher_prompt(r,p['teacher_parts'])==[20,10,11,21]+[3]*16+[22]
    changed=dict(r,natural_future16_ids=[4]*16)
    assert e.teacher_prompt(changed,p['teacher_parts'])==[20,10,11,21]+[4]*16+[22]
    assert r['prefix_ids']==prefix


def test_selection_rejects_threshold_drift_reordering_and_document_replacement(monkeypatch):
    monkeypatch.setattr(e,'N',2)
    rows=[dict(item_id=k,document_id='d'+k,split='train',prefix=[1]*512,future=[3]*16) for k in ('a','b')]
    s=dict(item_ids=['a','b'],document_ids=['da','db'],split='train',count=2,
        item_ids_sha256=canonical_hash(['a','b']),document_ids_sha256=canonical_hash(['da','db']),
        origins=[dict(raw_K1_surprisal_nats=1.7),dict(raw_K1_surprisal_nats=2.)],
        threshold_surprisal_nats=1.7,threshold_comparison='>=',likelihood_ranking=False,
        held256_document_overlap=0,priorDEV572_document_overlap=0)
    e.validate_selection(s,rows)
    for key,value in [('threshold_surprisal_nats',1.6),('held256_document_overlap',1),('likelihood_ranking',True)]:
        with pytest.raises(ValueError):e.validate_selection(dict(s,**{key:value}),rows)
    with pytest.raises(ValueError):e.validate_selection(s,list(reversed(rows)))
    with pytest.raises(ValueError):e.validate_selection(s,[dict(rows[0],document_id='db'),rows[1]])
    low=copy.deepcopy(s);low['origins'][0]['raw_K1_surprisal_nats']=1.69999
    with pytest.raises(ValueError):e.validate_selection(low,rows)


def test_merged_newline_token_is_retained_and_all_raw_actions_remain():
    r,p=fixture();v=note(r,p,[4,1,2,3,151643]);e.validate_note(v,r,p,Tok())
    assert v['actions']==[4,1,2,3,151643] and v['raw_content_ids']==[4,1,2,3]
    assert v['content_ids']==[4,1,2] and v['content_text']==' a\nNext passage'


@pytest.mark.parametrize('actions',[[151643],[1]*64])
def test_empty_eos_and_full_cap_are_retained_successes(actions):
    r,p=fixture();e.validate_note(note(r,p,actions),r,p,Tok())


def test_success_cannot_end_early_without_eos_or_change_native_prompt():
    r,p=fixture();v=note(r,p,[1,3])
    with pytest.raises(ValueError,match='EOS or cap64'):e.validate_note(v,r,p,Tok())
    v=note(r,p,[1,151643]);v['teacher_prompt_sha256']='changed'
    with pytest.raises(ValueError):e.validate_note(v,r,p,Tok())


@pytest.mark.parametrize('bad',[[9,151643],[151643,1,151643],[True,151643]])
def test_undefined_support_interior_eos_and_bool_tokens_are_rejected(bad):
    r,p=fixture()
    with pytest.raises(ValueError):e.validate_note(note(r,p,bad),r,p,Tok())


def test_partial_error_is_kept_but_is_not_relabelled_as_success():
    r,p=fixture();v=note(r,p,[1,3],status='error');e.validate_note(v,r,p,Tok())
    assert v['raw_content_ids']==v['actions']==[1,3]


def test_global_intact_donor_includes_empty_and_unmatched_lengths():
    r,p=fixture();notes={'a':dict(content_ids=[1,2,3]),'b':dict(content_ids=[]),'c':dict(content_ids=[1])}
    ids=['a','b','c'];donors=e.donors(ids)
    assert donors==dict(a='b',b='c',c='a')
    c=dict(other_content_ids={key:notes[d]['content_ids'] for key,d in donors.items()})
    x=e.contexts(r,p,notes,c)
    assert x['raw']==[10,11] and x['teacher']==[10,11,30,1,2,3,31,11]
    assert x['empty']==x['other']==[10,11,30,31,11]
    assert notes['a']['content_ids']==[1,2,3]


def matrix():
    ids=['a','b','c'];docs=['da','db','dc'];values=[]
    for key,doc in zip(ids,docs):
        for path in e.PATHS:
            for mode,value in [('raw',-4.),('empty',-3.),('teacher',-2.),('other',-5.)]:
                values.append(dict(item_id=key,document_id=doc,path=path,mode=mode,status='ok',natural16_ll=value))
    return ids,docs,values


def test_paired_direction_and_strict_per_document_acceptance_use_document_denominator():
    ids,docs,v=matrix();v[2]['natural16_ll']=-3.  # Exact tie with empty is NOT accepted.
    r=e.reduce_rows(v,ids,docs)['I3'];c=r['contrasts']['teacher_minus_empty']
    assert c['strict_positive_documents']==2 and c['denominator']==3 and c['unavailable_documents']==0
    assert c['difference_b_minus_a']==pytest.approx(2/3) and c['ci95'][0]>=0
    assert r['contrasts']['teacher_minus_other']['difference_b_minus_a']>0


def test_one_error_blocks_complete_mean_and_ci_but_keeps_all_documents():
    ids,docs,v=matrix();v[2]=dict(v[2],status='error');del v[2]['natural16_ll']
    r=e.reduce_rows(v,ids,docs)['I3'];c=r['contrasts']['teacher_minus_empty']
    assert r['absolute']['teacher']['mean'] is None
    assert not c['available'] and c['strict_positive_documents']==2 and c['unavailable_documents']==1 and c['denominator']==3


def test_missing_duplicate_and_wrong_document_cells_never_form_a_filtered_summary():
    ids,docs,v=matrix()
    for bad in [v[:-1],v+[v[0]], [dict(v[0],document_id='other')]+v[1:]]:
        with pytest.raises(ValueError):e.reduce_rows(bad,ids,docs)


@pytest.mark.parametrize('lp',[[0.]*15,[0.]*15+[float('nan')],[0.]*15+[.1]])
def test_short_nonfinite_positive_likelihoods_are_explicit_errors(lp):
    with pytest.raises(ValueError):e.likelihood(lp)


def test_generation_exception_preserves_genuine_actions_before_next_attempt(tmp_path,monkeypatch):
    from train import evaluate
    r,p=fixture();p.update(deadline_utc='2099-01-01T00:00:00+00:00',writer_prefill_width=64)
    (tmp_path/'protocol.json').write_text('{}')
    monkeypatch.setattr(e,'load',lambda path:(p,[r]))
    backend=SimpleNamespace(model=None,banned_ids=[],sync=lambda:None)
    monkeypatch.setattr(e.shared,'load_model',lambda *args:(backend,Tok(),e.B0))
    monkeypatch.setattr(e.shared,'base_gate',lambda *args:dict(fixture=True))
    partial=evaluate.Decoded([1,2,3],[1,2,3],'a\nNext passageb','a\nNext passageb','error')
    class FailingDecoder:
        def __init__(self,*args,**kwargs):pass
        def decode(self,prompt,**kwargs):
            assert kwargs['answer_line_stop'] is False and kwargs['max_new_tokens']==64 and kwargs['stop_ids']==(151643,)
            raise evaluate.GenerationFailure(RuntimeError('actual device failure'),partial)
    monkeypatch.setattr(evaluate,'GreedyDecoder',FailingDecoder)
    e.generate(tmp_path,0,p['deadline_utc'],'cpu')
    v=e.shared.rows(tmp_path/'notes/shard-0/notes.jsonl')[0]
    assert v['status']=='error' and v['actions']==[1,2,3] and v['raw_content_ids']==[1,2,3] and v['content_ids']==[1,2]
    assert 'actual device failure' in v['error']


def test_physical_closure_is_required_even_with_all_note_files(tmp_path):
    p=dict(control_dir=str(tmp_path));(tmp_path/'protocol.json').write_text('{}')
    closure=dict(status='COMPLETE_PHYSICALLY_CLOSED',all4_exit0=True,all12_owned_PID_start_identities_absent=True,
        all_sessions_absent=True,assigned_devices_empty=True,owned_group_survivors=[],protocol_sha256=e.sha256_file(tmp_path/'protocol.json'))
    file=tmp_path/'generate.closure.json';file.write_text(json.dumps(closure));assert e.require_closure(p,tmp_path,'generate')
    closure['assigned_devices_empty']=False;file.write_text(json.dumps(closure))
    with pytest.raises(ValueError,match='physical closure'):e.require_closure(p,tmp_path,'generate')


def test_actual_score_path_retains_both_nonfinite_mixer_errors(tmp_path,monkeypatch):
    from train import interface_mixer_probe as mixer
    r,p=fixture();p.update(deadline_utc='2099-01-01T00:00:00+00:00')
    (tmp_path/'protocol.json').write_text('{}');(tmp_path/'controls.json').write_text('{}')
    notes={'a':dict(content_ids=[1],status='ok')};control=dict(other_donors={'a':'a'},other_content_ids={'a':[1]})
    monkeypatch.setattr(e,'load',lambda path:(p,[r]));monkeypatch.setattr(e,'reader_inputs',lambda *args:(notes,{},e.B0,control))
    backend=SimpleNamespace(device='cpu',sync=lambda:None)
    monkeypatch.setattr(e.shared,'load_model',lambda *args:(backend,None,e.B0));monkeypatch.setattr(e.shared,'base_gate',lambda *args:{})
    monkeypatch.setattr(mixer,'full_logits',lambda *args:torch.zeros(1,16,5))
    monkeypatch.setattr(mixer,'token_logps',lambda *args:torch.zeros(1,16))
    monkeypatch.setattr(mixer,'mixed_token_logps',lambda *args:(torch.full((1,16),float('nan')),None,torch.zeros(1,16)))
    e.score(tmp_path,0,p['deadline_utc'],'cpu');v=e.shared.rows(tmp_path/'scores/shard-0/scores.jsonl')
    assert len(v)==8 and all(x['status']=='ok' for x in v[:2]) and all(x['status']=='error' for x in v[2:])
    assert e.shared.read(tmp_path/'scores/shard-0/run.json')['rows']==8
