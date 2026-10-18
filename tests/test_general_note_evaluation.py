"""GENERAL coverage, causal barriers, intact controls and failure denominators."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from train import general_note_evaluation as e
from weekend_thinking.artifacts import canonical_hash,sha256_file


def rows():
    return [dict(item_id=k,document_id='doc-'+k,split='dev',prefix_ids=[3+i]*512,
        reanchor_ids=[3+i]*24,natural_future16_ids=[1+i]*16) for i,k in enumerate(('a','b'))]


def protocol():
    return dict(writer=e.B0,deadline_utc='2099-01-01T00:00:00+00:00',writer_prefill_width=544,
        parts=dict(old_open=[151667],old_close=[151668,198,220]),vocab_size=151936,banned_ids=[])


def controls():
    notes={'a':dict(status='ok',content_ids=[7]*128),'b':dict(status='ok',content_ids=[])}
    control=e.shared.other_controls(dict(ordered_item_ids=['a','b']),notes)
    return notes,control


def test_future_free_projection_is_exact_native512_and_last24(monkeypatch):
    monkeypatch.setattr(e,'N',2);r=rows();prefixes=[e.shared.prefix_row(v) for v in r]
    e.validate_prefixes(prefixes,['a','b'],['doc-a','doc-b'])
    for field,value in [('natural_future16_ids',[1]*16),('numeric_token_mask',[True])]:
        bad=copy.deepcopy(prefixes);bad[0][field]=value
        with pytest.raises(ValueError):e.validate_prefixes(bad,['a','b'],['doc-a','doc-b'])
    bad=copy.deepcopy(prefixes);bad[0]['prefix_ids'].pop()
    with pytest.raises(ValueError):e.validate_prefixes(bad,['a','b'],['doc-a','doc-b'])


def test_contexts_use_intact_global_donors_and_exact_full_target_copy():
    a,b=rows();p=protocol();notes,c=controls();byid={'a':a,'b':b}
    x=e.contexts(a,p,notes,c,byid)
    suffix=p['parts']['old_close']+a['reanchor_ids'];base=a['prefix_ids']+p['parts']['old_open']
    assert x['raw']==a['prefix_ids']
    assert x['own']==base+[7]*128+suffix
    assert x['empty']==x['other']==base+suffix
    assert x['target_copy']==base+a['natural_future16_ids']+suffix
    assert x['corrupt_copy']==base+b['natural_future16_ids']+suffix
    y=e.contexts(b,p,notes,c,byid)
    assert y['other']==b['prefix_ids']+p['parts']['old_open']+[7]*128+p['parts']['old_close']+b['reanchor_ids']


@pytest.mark.parametrize('bad',[[0.]*15,[0.]*15+[float('nan')],[0.]*15+[float('inf')],[0.]*15+[.01]])
def test_full16_vector_is_required_without_scalar_or_numeric_reduction(bad):
    with pytest.raises(ValueError):e.likelihood(bad)


def test_reader_future_file_requires_pinned_native_targets_and_no_arithmetic_fields(tmp_path):
    rr=rows();file=tmp_path/'reader.json';file.write_text(json.dumps(dict(rows=rr)))
    p=dict(protocol(),reader_inputs_file=str(file),reader_inputs_sha256=sha256_file(file))
    prefixes=[e.shared.prefix_row(r) for r in rr]
    assert e.reader_rows(p,prefixes)==rr
    rr[0]['numeric_token_mask']=[True];file.write_text(json.dumps(dict(rows=rr)))
    with pytest.raises(ValueError):e.reader_rows(p,prefixes)
    p['reader_inputs_sha256']=sha256_file(file)
    with pytest.raises(ValueError):e.reader_rows(p,prefixes)


def matrix():
    ids=['a','b','c'];docs=['da','db','dc'];values=[]
    for key,doc in zip(ids,docs):
        for path in e.PATHS:
            for mode,ll in [('raw',-10.),('empty',-10.5),('own',-8.),('other',-9.),('target_copy',-1.),('corrupt_copy',-12.)]:
                values.append(dict(item_id=key,document_id=doc,path=path,mode=mode,status='ok',natural16_ll=ll))
    return ids,docs,values


def test_contrast_direction_headroom_tax_and_primary_document_gates():
    ids,docs,v=matrix();g=e.reduce_rows(v,ids,docs)['I3']
    assert g['contrasts']['own_minus_raw']['difference_b_minus_a']==2
    assert g['contrasts']['own_minus_other']['difference_b_minus_a']==1
    assert g['contrasts']['target_copy_minus_raw']['difference_b_minus_a']==9
    assert g['contrasts']['target_copy_minus_own']['difference_b_minus_a']==7
    assert g['gates']==dict(own_over_raw=True,own_over_other=True,useful_content=True,target_copy_headroom=True,empty_tax_qualified=True)
    for row in v:
        if row['mode']=='other':row['natural16_ll']=-8.
    assert not e.reduce_rows(v,ids,docs)['I3']['gates']['useful_content']


def test_all_documents_count_once_and_one_error_blocks_complete_pair():
    ids,docs,v=matrix()
    for row in v:
        if row['mode']=='own':row['natural16_ll']=-10.+[1.,2.,6.][ids.index(row['item_id'])]
    assert e.reduce_rows(v,ids,docs)['I3']['contrasts']['own_minus_raw']['difference_b_minus_a']==3
    bad=next(row for row in v if row['item_id']=='a' and row['path']=='I3' and row['mode']=='own')
    bad['status']='error';bad.pop('natural16_ll')
    g=e.reduce_rows(v,ids,docs)['I3']
    assert g['absolute']['own']==dict(items=3,errors=1,mean=None)
    assert not g['contrasts']['own_minus_raw']['available'] and not g['gates']['useful_content']
    assert g['absolute']['raw']['mean']==-10.


@pytest.mark.parametrize('defect',['missing','duplicate','document','nonfinite'])
def test_partial_or_mixed_publication_cannot_form_summary(defect):
    ids,docs,v=matrix()
    if defect=='missing':v.pop()
    elif defect=='duplicate':v.append(dict(v[0]))
    elif defect=='document':v[0]['document_id']='other'
    else:v[0]['natural16_ll']=float('nan')
    with pytest.raises(ValueError):e.reduce_rows(v,ids,docs)


@pytest.mark.parametrize('ids,docs',[([],[]),(['a','a'],['da','db'])])
def test_empty_or_duplicated_panel_never_creates_a_bootstrap(ids,docs):
    with pytest.raises(ValueError):e.reduce_rows([],ids,docs)


def test_actual_score_refuses_before_future_file_or_model_when_generation_not_closed(tmp_path,monkeypatch):
    p=protocol();p['control_dir']=str(tmp_path/'missing-control')
    monkeypatch.setattr(e,'load',lambda path:(p,[]))
    monkeypatch.setattr(e,'reader_rows',lambda *args:pytest.fail('No future read before physical note closure'))
    monkeypatch.setattr(e.shared,'load_model',lambda *args:pytest.fail('No model before note barrier'))
    with pytest.raises(FileNotFoundError):e.score(tmp_path,0,'B0',p['deadline_utc'],'cpu')


def test_B0_self_is_not_duplicated():
    with pytest.raises(ValueError):e.reader_identity(protocol(),'self')
    p=protocol();p['writer']=dict(id='W',checkpoint='own',checkpoint_sha256='ownsha')
    assert e.reader_identity(p,'self')==p['writer'] and e.reader_identity(p,'B0')==e.B0


def test_actual_generation_preserves_partial_actions_and_total128_stop_contract(tmp_path,monkeypatch):
    from train import evaluate
    p=protocol();r=rows()[0];(tmp_path/'protocol.json').write_text('{}')
    monkeypatch.setattr(e,'load',lambda path:(p,[e.shared.prefix_row(r)]))
    backend=SimpleNamespace(model=None,banned_ids=[],sync=lambda:None)
    monkeypatch.setattr(e.shared,'load_model',lambda *args:(backend,None,e.B0))
    monkeypatch.setattr(e.shared,'base_gate',lambda *args:{})
    class Decoder:
        def __init__(self,*args,**kwargs):pass
        def decode(self,prompt,**kwargs):
            assert prompt==r['prefix_ids']+[151667] and kwargs['max_new_tokens']==128
            assert kwargs['stop_ids']==(151668,151643) and not kwargs['answer_line_stop']
            partial=evaluate.Decoded([7,8],[7,8],'partial','partial','error')
            raise evaluate.GenerationFailure(RuntimeError('genuine device failure'),partial)
    monkeypatch.setattr(evaluate,'GreedyDecoder',Decoder)
    e.generate(tmp_path,0,p['deadline_utc'],'cpu')
    row=e.shared.rows(tmp_path/'notes/shard-0/notes.jsonl')[0]
    assert row['status']=='error' and row['actions']==row['content_ids']==[7,8]
    assert row['n_actions']==row['n_content_tokens']==2


def test_actual_full_logit_score_retains_raw_empty_and_copy_when_own_or_donor_failed(tmp_path,monkeypatch):
    from train import interface_mixer_probe as mixer
    p=protocol();rr=rows();notes,c=controls();notes['a']['status']='error'
    monkeypatch.setattr(e,'SHARDS',1);(tmp_path/'protocol.json').write_text('{}');(tmp_path/'controls.json').write_text('{}')
    monkeypatch.setattr(e,'load',lambda path:(p,[e.shared.prefix_row(r) for r in rr]))
    monkeypatch.setattr(e,'notes_and_controls',lambda *args:(notes,{},e.B0,c))
    monkeypatch.setattr(e,'reader_rows',lambda *args:rr)
    backend=SimpleNamespace(device='cpu',sync=lambda:None)
    monkeypatch.setattr(e.shared,'load_model',lambda *args:(backend,None,e.B0))
    monkeypatch.setattr(e.shared,'base_gate',lambda *args:{})
    monkeypatch.setattr(mixer,'full_logits',lambda *args:torch.tensor([[0.,1.,2.,3.]]).expand(16,-1))
    e.score(tmp_path,0,'B0',p['deadline_utc'],'cpu')
    values=e.shared.rows(tmp_path/'scores/B0/shard-0/scores.jsonl')
    assert len(values)==24 and sum(v['status']=='error' for v in values)==4
    assert all(v['status']=='ok' for v in values if v['mode'] in ('raw','empty','target_copy','corrupt_copy'))
    for v in values:
        if v['status']=='ok':assert v['natural16_ll']==sum(v['token_logps'])
    costs=e.shared.saved_costs({k:dict(v,actions=[],wall_seconds=1.) for k,v in notes.items()},values)
    assert costs['scoring_shared_timing_groups']==12 and costs['raw_full_logits_seconds_sum']>0
