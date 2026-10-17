"""Writer-only causal packing and prospective strict prepend TRAIN label rule."""
import copy
from dataclasses import asdict
import json

import pytest
import torch

from train import positive_writer_data as data, warmstart_data
from weekend_thinking.artifacts import canonical_hash, sha256_file
from weekend_thinking.core import Format


def fmt():
    return Format((2,),(3,4),3,0,header_ids=(17,18),additional_stop_ids=(0,))


def example(length=2):
    return dict(item_id='train:0',document_id='doc:0',split='train',training_layout='writer_only',
        prefix=[10,11]*256,future=[7001+i for i in range(16)],content_ids=[7]*length)


@pytest.mark.parametrize('length',[0,2,256])
def test_writer_only_input_targets_causal_positions_and_zero_future_mask(length):
    row=example(length);before=copy.deepcopy(row);f=fmt()
    packed=warmstart_data.pack_example(row,f,'warmstart')
    ids,pos,labels=(packed[k][0] for k in ('input_ids','target_positions','labels'))
    n,m=packed['note_close_mask'][0],packed['future_mask'][0]
    prompt=list(f.header_ids)+row['prefix']+list(f.open_ids)
    rendered=prompt+row['content_ids']+list(f.close_ids)
    assert ids.shape==(832,) and pos.shape==labels.shape==n.shape==m.shape==(258,)
    assert ids[:len(rendered)].tolist()==rendered and ids[len(rendered):].eq(f.pad_id).all()
    assert int(n.sum())==length+2 and not m.any()
    assert pos[n].tolist()==list(range(len(prompt),len(rendered)))
    assert labels[n].tolist()==row['content_ids']+list(f.close_ids)
    assert labels[~n].eq(-100).all() and pos[~n].eq(0).all()
    assert torch.equal(ids[pos[n]],labels[n]) and int(pos[n][0])-1==len(prompt)-1
    assert not set(ids.tolist())&set(row['future']) and row==before
    # Changing true continuations, including EOS==pad, changes no writer tensor.
    row['future']=[f.pad_id]*16
    other=warmstart_data.pack_example(row,f,'warmstart')
    assert all(torch.equal(packed[k],other[k]) for k in packed)


def test_writer_layout_raw_control_and_legacy_joint_are_unchanged():
    row=example();row['future']=[0]+[12]*15
    raw=warmstart_data.pack_example(row,fmt(),'raw_control')
    assert raw['input_ids'].shape==(1,544) and raw['target_positions'].shape==(1,16)
    assert raw['input_ids'][0,:528].tolist()==row['prefix']+row['future']
    assert raw['labels'].tolist()==[row['future']] and raw['future_mask'].all() and not raw['note_close_mask'].any()
    legacy=dict(row);legacy.pop('training_layout')
    explicit=dict(legacy,training_layout='joint_reanchor32')
    a=warmstart_data.pack_example(legacy,fmt(),'warmstart');b=warmstart_data.pack_example(explicit,fmt(),'warmstart')
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert a['input_ids'].shape==(1,864) and a['target_positions'].shape==(1,274) and int(a['future_mask'].sum())==16


def test_writer_only_rejects_overcap_and_wrong_canonical_close_or_layout():
    with pytest.raises(ValueError,match='at most256'):
        warmstart_data.pack_example(example(257),fmt(),'warmstart')
    with pytest.raises(ValueError,match='two-token'):
        warmstart_data.pack_example(example(),Format((2,),(3,),3,0),'warmstart')
    row=example();row['training_layout']='unknown'
    with pytest.raises(ValueError,match='layout'):
        warmstart_data.pack_example(row,fmt(),'warmstart')


def fixture(ids=('train:0','train:1')):
    p=dict(selection=dict(item_ids=list(ids),document_ids=['doc:'+key for key in ids],split='train'))
    scores=[];notes={}
    for key in ids:
        def add(interface,mode,i,value):
            scores.append(dict(item_id=key,document_id='doc:'+key,split='train',interface=interface,mode=mode,candidate_index=i,status='ok',token_log_likelihoods=[value]+[0.]*15))
        for mode,value in [('raw',-20.),('repeat32',-19.),('repeat64',-18.)]:add('raw',mode,None,value)
        add('prepend_plain','empty',None,-20.)
        for i in range(16):
            notes[key,i]=dict(content_ids=[7])
            for mode in ('own','filler','shuffled','other'):
                add('prepend_plain',mode,i,(-5. if i==2 else -6.) if mode=='own' else -30.)
    return p,scores,notes


def row(scores,mode='own',i=2,key='train:0'):
    return next(r for r in scores if r['item_id']==key and r['mode']==mode and r['candidate_index']==i)


def test_highest_sum_k16_first_then_strict_raw_and_same_index_filters():
    p,scores,notes=fixture()
    row(scores,i=2)['token_log_likelihoods']=[-.1]+[-1.]*15
    row(scores,i=5)['token_log_likelihoods']=[-5.]+[0.]*15
    # An unrelated candidate's high control must not replace same-index control.
    row(scores,mode='other',i=0)['token_log_likelihoods']=[100.]+[0.]*15
    selected,audit=data.choose_candidates(p,list(reversed(scores)),notes)
    assert selected==[('train:0',5),('train:1',2)]
    assert audit[0]['selected'] and audit[0]['strict_comparator_pass']['raw']
    assert audit[0]['contrasts']['own_minus_raw']==15.


@pytest.mark.parametrize('mode',['raw','empty','filler','shuffled','other'])
def test_every_strict_comparator_tie_rejects_without_fallback(mode):
    p,scores,notes=fixture(('train:0',))
    row(scores,mode=mode,i=None if mode in {'raw','empty'} else 2)['token_log_likelihoods']=[-5.]+[0.]*15
    selected,audit=data.choose_candidates(p,scores,notes)
    assert selected==[] and audit[0]['source_candidate_index']==2 and not audit[0]['selected']
    assert f'not_strictly_above_{mode}' in audit[0]['rejection_reasons']


def test_empty_first_tied_winner_never_falls_back_to_nonempty():
    p,scores,notes=fixture(('train:0',))
    row(scores,i=1)['token_log_likelihoods']=[-5.]+[0.]*15;notes['train:0',1]['content_ids']=[]
    selected,audit=data.choose_candidates(p,scores,notes)
    assert selected==[] and audit[0]['source_candidate_index']==1 and not audit[0]['nonempty']


@pytest.mark.parametrize('bad',['dev','duplicate','missing','failed_unselected','nan_unselected','document'])
def test_selection_keeps_all_score_denominators_and_identity(bad):
    p,scores,notes=fixture()
    if bad=='dev':scores[0]['split']='dev'
    elif bad=='duplicate':scores.append(copy.deepcopy(scores[0]))
    elif bad=='missing':scores.pop()
    elif bad=='failed_unselected':row(scores,mode='filler',i=0)['status']='error'
    elif bad=='nan_unselected':row(scores,mode='other',i=0)['token_log_likelihoods'][0]=float('nan')
    else:scores[0]['document_id']='wrong'
    with pytest.raises(ValueError):data.choose_candidates(p,scores,notes)


def test_preparation_keeps_all128_audits_and_discloses_writer_only_labels(tmp_path,monkeypatch):
    ids=[f'train:{i}' for i in range(128)];p,scores,notes=fixture(ids)
    p.update(hashes={k:'a'*64 for k in ['corpus_protocol','corpus_summary','corpus_barrier','controls']},model_manifest_sha256='b'*64,thought_format=asdict(fmt()))
    p['selection']['records_file']='unopened-records.jsonl'
    records=[]
    for key in ids:
        r=dict(example(),item_id=key,document_id='doc:'+key);records.append(r)
        notes[key,2].update(prefix_sha256=canonical_hash(r['prefix']),document_id='doc:'+key,actions=[7,0],behavior_logps=[-1.,-2.],action_mask=[True,True]+[False]*254,stop_type='token:0',decode_iterations=32,group=0,group_seed=123)
    directory=tmp_path/'prepend';directory.mkdir();(directory/'protocol.json').write_text('{}');(directory/'summary.json').write_text('{}')
    monkeypatch.setattr(data,'completed_scores',lambda _: (p,fmt(),notes,scores,{'scoring':[]},{'generation':[],'model':{'fixture':'B0'}}))
    monkeypatch.setattr(data,'load_selected',lambda *a,**k:records)
    output=tmp_path/'labels';data.prepare(output,directory)
    audit=json.loads((output/'selection-audit.json').read_text());manifest=json.loads((output/'dataset.json').read_text())
    rows=[json.loads(s) for s in (output/'examples.jsonl').read_text().splitlines()]
    assert audit['items']==audit['selected']==len(audit['rows'])==len(rows)==128
    assert manifest['schema']==data.SCHEMA and manifest['training_layout']=='writer_only' and manifest['utility_reader']=='prepend_plain'
    assert manifest['writer_future_supervision_tokens']==0 and manifest['canonical_close_tokens_are_synthetic']
    assert manifest['examples_sha256']==sha256_file(output/'examples.jsonl') and manifest['audit_sha256']==sha256_file(output/'selection-audit.json')
    assert manifest['item_ids_sha256']==canonical_hash(ids) and manifest['source_sha256']==manifest['selector_source_sha256']
    assert all(r['training_layout']=='writer_only' and r['sampled_actions']==[7,0] and r['sampled_stop_type']=='token:0' and not r['future_tokens_used_for_writer_ce'] for r in rows)
    assert rows[0]['canonical_close_ids']==[3,4] and rows[0]['future']==records[0]['future']


def test_incomplete_score_source_refuses_future_loading(tmp_path,monkeypatch):
    directory=tmp_path/'prepend';directory.mkdir();p,_,notes=fixture()
    (directory/'summary.json').write_text(json.dumps(dict(schema=data.prepend.SCHEMA,split='train',status='running')))
    monkeypatch.setattr(data.prepend,'read_protocol',lambda _: (p,fmt()))
    monkeypatch.setattr(data.prepend,'inputs',lambda _: ({},notes,{},{}))
    monkeypatch.setattr(data,'load_selected',lambda *a,**k:pytest.fail('Incomplete source read canonical futures'))
    with pytest.raises(ValueError,match='completed prepend'):
        data.prepare(tmp_path/'out',directory)
    assert not (tmp_path/'out').exists()
