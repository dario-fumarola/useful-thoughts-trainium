"""CPU context/reducer/global-barrier contracts for closed-corpus prepend scoring."""
import json
from types import SimpleNamespace

import pytest

from train import candidate_prepend as probe
from weekend_thinking.artifacts import canonical_hash
from weekend_thinking.core import Format


def protocol(ids=None):
    ids=ids or [f'train:{i}' for i in range(128)]
    return dict(schema=probe.SCHEMA,experiment_id=probe.SCHEMA,
        selection=dict(item_ids=ids,document_ids=['doc:'+k for k in ids],split='train',count=len(ids),records_file='unopened-records.jsonl'),
        interfaces={'prepend_plain':{'separator_ids':[271]}},separator_ids=[271],horizons=[1,16],
        strata={k:'TRAIN_selected' for k in ids},limitations=['TRAIN future-selected diagnostic'],
        hashes={'corpus_barrier':'a'*64},expected_shard_score_rows=1088,model_dir='unopened-model')


def test_prepend_keeps_entire_prefix_last_and_global_controls_exact():
    prefix=list(range(1000,1512));p=protocol()
    controls={i:dict(content_ids=dict(own=[10+i,20+i],filler=[30,30],shuffled=[20+i,10+i],other=[40+i,50+i])) for i in range(16)}
    rows=probe.contexts(prefix,controls,p)
    assert len(rows)==68 and rows[0]==('raw','raw',None,prefix)
    assert rows[1][-1]==prefix+prefix[-32:] and rows[2][-1]==prefix+prefix[-64:]
    assert rows[3]==('prepend_plain','empty',None,[271]+prefix)
    content=[r for r in rows if r[2] is not None]
    assert len(content)==64 and all(r[-1][-512:]==prefix for r in content)
    for interface,mode,index,context in content:
        assert interface=='prepend_plain' and context==controls[index]['content_ids'][mode]+[271]+prefix
    assert controls[0]['content_ids']['own']==[10,20]


def test_global_closed_corpus_barrier_precedes_models_and_new_futures(tmp_path,monkeypatch):
    p=protocol();fmt=Format((2,),(3,),5,0)
    monkeypatch.setattr(probe,'read_protocol',lambda _: (p,fmt))
    def incomplete(_):raise ValueError('All2048 notes/global controls must close')
    monkeypatch.setattr(probe,'inputs',incomplete)
    monkeypatch.setattr(probe,'setup_model',lambda *a,**k:pytest.fail('Model before complete corpus barrier'))
    monkeypatch.setattr(probe,'load_selected',lambda *a,**k:pytest.fail('Future before complete corpus barrier'))
    with pytest.raises(ValueError,match='All2048'):
        probe.score(tmp_path,0,'cpu')
    assert not (tmp_path/'scoring').exists()


def test_raw_references_compare_repeats_to_identical_contexts_not_scanner(tmp_path):
    p=protocol(['train:0','train:1']);p['corpus_dir']=str(tmp_path)
    scanner=tmp_path/'scanner.jsonl';scanner.write_text(''.join(json.dumps(dict(item_id=k,status='ok',token_log_likelihoods=[-1.]*16))+'\n' for k in p['selection']['item_ids']))
    old=dict(paths={'raw_references':str(scanner)})
    repeat=[]
    for shard in range(8):
        d=tmp_path/'scoring'/f'shard-{shard:02d}';d.mkdir(parents=True)
        if shard==0:
            repeat=[dict(item_id=k,interface='raw',mode=m,status='ok',token_log_likelihoods=[value]*16) for k in p['selection']['item_ids'] for m,value in [('repeat32',-2.),('repeat64',-3.)]]
        (d/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in repeat) if shard==0 else '')
    refs=probe.raw_references(p,old)
    assert len(refs)==6 and refs['train:0','raw']==[-1.]*16 and refs['train:0','repeat32']==[-2.]*16 and refs['train:0','repeat64']==[-3.]*16
    repeat[0]['token_log_likelihoods'][0]=float('nan')
    (tmp_path/'scoring/shard-00/scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in repeat))
    with pytest.raises(ValueError,match='Invalid/duplicate'):
        probe.raw_references(p,old)


def test_reducer_preserves_train_prepend_identity_means_oracle_and_failures():
    p=protocol(['train:0','train:1']);rows=[]
    for key in p['selection']['item_ids']:
        def add(interface,mode,index,value):
            rows.append(dict(item_id=key,document_id='doc:'+key,split='train',interface=interface,mode=mode,candidate_index=index,status='ok',token_log_likelihoods=[value]+[0.]*15,n_note_content_tokens=1,note_stop_type='length_cap'))
        for mode in ('raw','repeat32','repeat64'):add('raw',mode,None,-1.)
        add('prepend_plain','empty',None,-2.)
        for i in range(16):
            for mode in ('own','filler','shuffled','other'):
                add('prepend_plain',mode,i,(-1. if i==3 else -3.) if mode=='own' else -2.5)
    result=probe.reduce_rows(p,rows)
    assert result['schema']==probe.SCHEMA and result['experiment_id']==probe.SCHEMA and result['split']=='train'
    assert all(r['stratum']=='TRAIN_selected' and 'quartile' not in r for r in result['per_document'])
    group=result['groups']['all'];assert group['prepend_plain/K1/oracle_max_own_minus_empty']['mean_complete_documents_only']==1.
    assert group['prepend_plain/K1/mean16_own_minus_empty']['mean_complete_documents_only']==-.875
    assert group['prepend_plain/K1/max16_own_minus_max16_other']['mean_complete_documents_only']==1.5
    failed=next(r for r in rows if r['item_id']=='train:0' and r['mode']=='own');failed['status']='error';failed.pop('token_log_likelihoods')
    result=probe.reduce_rows(p,rows);entry=result['groups']['all']['prepend_plain/K1/oracle_max_own_minus_empty']
    assert result['failures']==1 and entry['expected_documents']==2 and entry['complete_documents']==1 and 'document_bootstrap' not in entry


@pytest.mark.parametrize('values',[[float('nan')]*16,[-1.]*15])
def test_invalid_return_is_saved_as_error_row(tmp_path,monkeypatch,values):
    p=protocol();fmt=Format((2,),(3,),5,0);(tmp_path/'protocol.json').write_text('{}')
    prefix=list(range(512));key=p['selection']['item_ids'][0]
    notes={(key,i):dict(prefix_sha256=canonical_hash(prefix)) for i in range(16)}
    controls={(key,i):dict(content_ids={m:[10] for m in ('own','filler','shuffled','other')}) for i in range(16)}
    monkeypatch.setattr(probe,'read_protocol',lambda _: (p,fmt))
    monkeypatch.setattr(probe,'inputs',lambda _: ({},notes,controls,{'model':{'fixture':'B0'}}))
    monkeypatch.setattr(probe,'raw_references',lambda *a: {(key,m):[-1.]*16 for m in ('raw','repeat32','repeat64')})
    monkeypatch.setattr(probe,'setup_model',lambda *a: (None,None,None,{'fixture':'B0'},None))
    monkeypatch.setattr(probe,'load_selected',lambda *a,**k: [dict(item_id=key,document_id='doc:'+key,prefix=prefix,future=[10]*16)])
    monkeypatch.setattr(probe,'score_continuations',lambda *a,**k: [SimpleNamespace(token_log_likelihoods=values)])
    with pytest.raises(RuntimeError,match='Failed actual reader row retained'):
        probe.score(tmp_path,0,'cpu')
    d=tmp_path/'scoring/shard-00';rows=[json.loads(x) for x in (d/'scores.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['status']=='error' and 'sixteen finite' in rows[0]['error']
    assert json.loads((d/'run.json').read_text())['status']=='failed'
