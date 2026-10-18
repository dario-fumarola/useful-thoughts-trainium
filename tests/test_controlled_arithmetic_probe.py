"""Synthetic context, causal likelihood, coverage and retained-failure checks."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from train import controlled_arithmetic_probe as e
from weekend_thinking.artifacts import sha256_file


def protocol():
    return dict(writer=e.base.B0, deadline_utc='2099-01-01T00:00:00+00:00', writer_prefill_width=32,
        control_dir='missing', parts=dict(old_open=[151667],old_close=[151668,198,220]),
        vocab_size=151936,banned_ids=[])


def rows():
    return [dict(item_id=key,document_id='doc-'+key,split='synthetic_dev',prefix_ids=[1,2+i],
        reanchor_ids=[2+i],target_ids=[1,2,3],numeric_only_mask=[False,True,False],numeric_only_supported=True,
        supplied_note_ids=dict(correct=[6,7],wrong=[6,8],partial_correct=[9],partial_wrong=[10]))
        for i,key in enumerate(('a','b'))]


def prefix(row):
    return e.dataset.prefix_projection(row)


def notes_controls():
    notes=dict(a=dict(status='ok',content_ids=[11]*128),b=dict(status='ok',content_ids=[]))
    return notes,e.shared.other_controls(dict(ordered_item_ids=['a','b']),notes)


def test_only_exact_future_free_projection_enters_writer(monkeypatch):
    monkeypatch.setattr(e,'N',2);pp=[prefix(r) for r in rows()]
    e.validate_prefixes(pp,['a','b'],['doc-a','doc-b'])
    for field,value in [('target_ids',[123]),('value',123),('supplied_note_ids',{})]:
        bad=copy.deepcopy(pp);bad[0][field]=value
        with pytest.raises(ValueError):e.validate_prefixes(bad,['a','b'],['doc-a','doc-b'])
    bad=copy.deepcopy(pp);bad[0]['split']='train'
    with pytest.raises(ValueError):e.validate_prefixes(bad,['a','b'],['doc-a','doc-b'])


def test_contexts_keep_all_native_notes_and_intact_empty_donor():
    row=rows()[0];p=protocol();notes,c=notes_controls();ctx=e.contexts(row,p,notes,c)
    before=row['prefix_ids']+p['parts']['old_open'];after=p['parts']['old_close']+row['reanchor_ids']
    assert ctx['raw']==row['prefix_ids']
    assert ctx['empty']==ctx['other']==before+after
    assert ctx['own']==before+[11]*128+after
    for mode,key in e.NOTE_KEYS.items():assert ctx[mode]==before+row['supplied_note_ids'][key]+after
    assert set(ctx)==set(e.MODES)
    original=copy.deepcopy(ctx)
    row['target_ids']=[99]*3;row['value']=999
    assert e.contexts(row,p,notes,c)==original
    assert c['other_donors']=={'a':'b','b':'a'}


def test_contexts_refuse_overwidth_instead_of_truncating():
    row=rows()[0];notes,c=notes_controls();notes['a']['content_ids']=[11]*1024
    with pytest.raises(ValueError):e.contexts(row,protocol(),notes,c)


@pytest.mark.parametrize('lp', [[-1,-2],[-1,-2,float('nan')],[-1,-2,float('inf')],[-1,-2,.1]])
def test_all_fullspan_tokens_required(lp):
    with pytest.raises(ValueError):e.metrics(lp,rows()[0])


def test_numeric_is_descriptive_and_does_not_hide_boundary_or_sign_cost():
    row=rows()[0]
    assert e.metrics([-2,-3,-4],row)==dict(fullspan_ll=-9,numeric_only_ll=-3)
    row['numeric_only_mask']=[False]*3;row['numeric_only_supported']=False
    assert e.metrics([-2,-3,-4],row)==dict(fullspan_ll=-9,numeric_only_ll=None)


def test_real_full_logits_causal_shift_and_actual_logit_mixing():
    from train.interface_mixer_probe import full_logits,token_logps,mixed_token_logps
    class Backend:
        device='cpu'
        model=SimpleNamespace(lm_head=lambda h: h)
        @staticmethod
        def hidden(ids):
            # A tiny deterministic causal trunk, with no model imports/weights.
            return torch.nn.functional.one_hot(ids.remainder(5),5).float().cumsum(1)
        @staticmethod
        def sync():pass
    backend=Backend();past=[1,2];target=[3,4,1]
    raw=full_logits(backend,past,target)[:3]
    changed=full_logits(backend,past,[0,4,1])[:3]
    assert torch.equal(raw[0],changed[0]) and not torch.equal(raw[1],changed[1])
    assert torch.equal(raw[0],backend.hidden(torch.tensor([past]))[0,-1])
    note=full_logits(backend,past+[4],target)[:3];y=torch.tensor([target])
    mixed,_,end=mixed_token_logps(raw,note,y,[.375],token_logps(raw,y))
    expected=token_logps(raw*.625+note*.375,y)
    assert torch.allclose(mixed,expected) and torch.equal(end,token_logps(note,y))
    # Mixing target-token logps is not the chosen interpolation.
    assert not torch.allclose(mixed,.625*token_logps(raw,y)+.375*token_logps(note,y))


def matrix():
    ids=['a','b','c'];docs=['da','db','dc'];values=[]
    means=dict(raw=-10.,empty=-10.5,full_correct=-1.,full_wrong=-9.,partial_correct=-7.,partial_wrong=-11.,own=-8.,other=-9.)
    for key,doc in zip(ids,docs):
        for path in e.PATHS:
            for mode,value in means.items():values.append(dict(item_id=key,document_id=doc,path=path,mode=mode,status='ok',fullspan_ll=value,numeric_only_ll=value/2))
    return ids,docs,values


def test_contrast_directions_headroom_and_all_document_denominators():
    ids,docs,v=matrix();g=e.reduce_rows(v,ids,docs)['I3']['fullspan_ll']
    assert g['contrasts']['full_correct_minus_raw']['difference_b_minus_a']==9
    assert g['contrasts']['full_correct_minus_full_wrong']['difference_b_minus_a']==8
    assert g['contrasts']['partial_correct_minus_partial_wrong']['difference_b_minus_a']==4
    assert g['contrasts']['own_minus_other']['difference_b_minus_a']==1
    assert g['raw_mean_maximum_possible_gain']==10
    assert g['absolute']['own']['items']==3


@pytest.mark.parametrize('defect',['missing','duplicate','document','nonfinite'])
def test_partial_or_mixed_rows_refuse_summary(defect):
    ids,docs,v=matrix()
    if defect=='missing':v.pop()
    elif defect=='duplicate':v.append(copy.deepcopy(v[0]))
    elif defect=='document':v[0]['document_id']='wrong'
    else:v[0]['fullspan_ll']=float('nan')
    with pytest.raises(ValueError):e.reduce_rows(v,ids,docs)


def test_error_and_unsupported_never_become_complete_case_or_zero_imputation():
    ids,docs,v=matrix()
    bad=next(r for r in v if r['item_id']=='a' and r['path']=='I3' and r['mode']=='own')
    bad['status']='error';bad.pop('fullspan_ll');bad.pop('numeric_only_ll')
    g=e.reduce_rows(v,ids,docs)['I3']
    assert g['fullspan_ll']['absolute']['own']==dict(items=3,errors=1,unsupported=0,mean=None)
    assert not g['fullspan_ll']['contrasts']['own_minus_raw']['available']
    assert g['fullspan_ll']['absolute']['raw']['mean']==-10
    v=matrix()[2];v[0]['numeric_only_ll']=None
    g=e.reduce_rows(v,ids,docs)['I3']
    assert g['numeric_only_ll']['absolute']['raw']==dict(items=3,errors=0,unsupported=1,mean=None)
    assert g['fullspan_ll']['absolute']['raw']['mean']==-10


def test_actual_score_refuses_before_future_or_model_if_generation_open(tmp_path,monkeypatch):
    p=protocol();p['control_dir']=str(tmp_path/'missing')
    monkeypatch.setattr(e,'load',lambda output:(p,[]))
    monkeypatch.setattr(e,'reader_rows',lambda *a:pytest.fail('Future before closure'))
    monkeypatch.setattr(e.shared,'load_model',lambda *a:pytest.fail('Model before closure'))
    with pytest.raises(FileNotFoundError):e.score(tmp_path,0,p['deadline_utc'],'cpu')


def test_actual_generation_retains_genuine_partial_and_total128_cap(tmp_path,monkeypatch):
    from train import evaluate
    p=protocol();row=rows()[0];(tmp_path/'protocol.json').write_text('{}')
    monkeypatch.setattr(e,'load',lambda output:(p,[prefix(row)]))
    backend=SimpleNamespace(model=None,banned_ids=[],sync=lambda:None)
    monkeypatch.setattr(e.shared,'load_model',lambda *a:(backend,None,e.base.B0))
    monkeypatch.setattr(e.shared,'base_gate',lambda *a:{})
    class Decoder:
        def __init__(self,*args,**kwargs):pass
        def decode(self,prompt,**kwargs):
            assert prompt==row['prefix_ids']+[151667] and kwargs['max_new_tokens']==128
            assert kwargs['stop_ids']==(151668,151643) and not kwargs['answer_line_stop']
            raise evaluate.GenerationFailure(RuntimeError('failed after actual actions'),evaluate.Decoded([4,5],[4,5],'partial','partial','error'))
    monkeypatch.setattr(evaluate,'GreedyDecoder',Decoder)
    e.generate(tmp_path,0,p['deadline_utc'],'cpu')
    saved=e.shared.rows(tmp_path/'notes/shard-0/notes.jsonl')[0]
    assert saved['status']=='error' and saved['actions']==saved['content_ids']==[4,5]


@pytest.mark.parametrize('model_id',['B0','W'])
def test_actual_score_keeps_raw_empty_supplied_notes_when_own_or_donor_failed(tmp_path,monkeypatch,model_id):
    from train import interface_mixer_probe as mixer
    p=protocol();rr=rows();notes,c=notes_controls();notes['a']['status']='error'
    rr[1].update(target_ids=[1,2,3,4,5],numeric_only_mask=[False,True,True,True,False])
    if model_id!='B0':p['writer']=dict(id=model_id,checkpoint='terminal/model.pt')
    monkeypatch.setattr(e,'SHARDS',1);(tmp_path/'protocol.json').write_text('{}');(tmp_path/'controls.json').write_text('{}')
    monkeypatch.setattr(e,'load',lambda path:(p,[prefix(r) for r in rr]))
    monkeypatch.setattr(e,'controls',lambda *args:(notes,c));monkeypatch.setattr(e,'reader_rows',lambda *args:rr)
    backend=SimpleNamespace(device='cpu',sync=lambda:None)
    def load_model(actual_p,identity,device):
        assert identity==p['writer']
        return backend,None,identity
    monkeypatch.setattr(e.shared,'load_model',load_model)
    monkeypatch.setattr(e.shared,'base_gate',lambda *args:{})
    def logits(backend,past,target):
        return torch.arange(16)[:,None]*torch.arange(12)[None]*.01+torch.sin(torch.arange(12)+(sum(past)%17)*.1)
    native_token_lp=mixer.token_logps;native_mixed=mixer.mixed_token_logps;shapes=[]
    # The tiny vocabulary represents unused EOS with ID0; actual target IDs
    # stay untouched. This avoids allocating a released-size head in a fixture.
    def tiny_token_lp(logits,targets):
        return native_token_lp(logits,torch.where(targets==151643,0,targets))
    def tracked_mixed(raw,note,targets,weights,raw_lp):
        shapes.append((raw.shape,note.shape,targets.shape,raw_lp.shape,targets.tolist()))
        return native_mixed(raw,note,targets,weights,raw_lp)
    monkeypatch.setattr(mixer,'full_logits',logits)
    monkeypatch.setattr(mixer,'token_logps',tiny_token_lp)
    monkeypatch.setattr(mixer,'mixed_token_logps',tracked_mixed)
    e.score(tmp_path,0,p['deadline_utc'],'cpu')
    vv=e.shared.rows(tmp_path/'scores/shard-0/scores.jsonl')
    assert len(vv)==32 and sum(v['status']=='error' for v in vv)==4
    assert all(v['status']=='ok' for v in vv if v['mode'] not in ('own','other'))
    byid={row['item_id']:row for row in rr}
    assert shapes and all(raw==note==(16,12) and target==lp==(1,16) for raw,note,target,lp,_ in shapes)
    expected_targets=[row['target_ids']+[151643]*(16-len(row['target_ids'])) for row in rr]
    assert all(target[0] in expected_targets for *_,target in shapes)
    for v in vv:
        if v['status']!='ok':continue
        row=byid[v['item_id']];n=len(row['target_ids']);ctx=e.contexts(row,p,notes,c)
        raw=logits(backend,row['prefix_ids'],row['target_ids'])[:n]
        note=logits(backend,ctx[v['mode']],row['target_ids'])[:n];y=torch.tensor([row['target_ids']])
        expected=(native_token_lp(raw,y) if v['mode']=='raw' else native_token_lp(note,y) if v['path']=='I3'
            else native_mixed(raw,note,y,[.375],native_token_lp(raw,y))[0])
        assert len(v['token_logps'])==n and v['fullspan_ll']==sum(v['token_logps'])
        assert torch.equal(torch.tensor(v['token_logps']),expected[0])
    costs=e.shared.saved_costs({k:dict(v,actions=[],wall_seconds=1.) for k,v in notes.items()},vv)
    assert costs['scoring_shared_timing_groups']==16 and costs['raw_full_logits_seconds_sum']>0


def test_trained_writer_uses_existing_terminal_validator_without_alias_or_matrix(monkeypatch):
    actual=dict(id='W',checkpoint='model.pt',checkpoint_sha256='a'*64,provenance={'p':'sha'},checkpoint_files={'model.pt':'sha'},training_source_files={'s':'sha'},run_id='real',step=320,spec_hash='b'*64)
    seen=[]
    monkeypatch.setattr(e.shared,'trained_identity',lambda *args:seen.append(args) or actual)
    assert e.writer_identity()==e.base.B0
    assert e.validate_writer(e.writer_identity('model.pt','modelsha','completesha','training.json'))==actual
    assert seen==[('model.pt','modelsha','completesha','training.json')]
    with pytest.raises(ValueError):e.writer_identity('model.pt')
    with pytest.raises(ValueError):e.validate_writer(dict(actual,id='vanilla'))


def test_reader_bytes_must_match_original_hash_and_same_future_free_projection(tmp_path):
    rr=rows();file=tmp_path/'reader.json';file.write_text(json.dumps(dict(rows=rr)))
    p=dict(reader_inputs_file=str(file),reader_inputs_sha256=sha256_file(file));pp=[prefix(r) for r in rr]
    assert e.reader_rows(p,pp)==rr
    rr[0]['prefix_ids']=[5];file.write_text(json.dumps(dict(rows=rr)))
    with pytest.raises(ValueError):e.reader_rows(p,pp)
    p['reader_inputs_sha256']=sha256_file(file)
    with pytest.raises(ValueError):e.reader_rows(p,pp)


def test_CPU_prepare_accepts_real_native_metadata_shape_without_model_or_future_projection(tmp_path,monkeypatch):
    from transformers import AutoTokenizer
    from train import interface_oracle_probe as oracle
    from weekend_thinking.artifacts import canonical_hash
    class Tokenizer:
        @staticmethod
        def encode(text,**kwargs):return list(text.encode())
        @staticmethod
        def get_vocab():return {str(i):i for i in list(range(256))+[151643,151667,151668]}
    tok=Tokenizer();data=tmp_path/'data';data.mkdir();native=tmp_path/'native';native.mkdir()
    for name in ('protocol.json','inputs.json'):(native/name).write_text('{}')
    (native/'cpu-reference').mkdir()
    for name in ('reference.json','next-logps.pt'):(native/'cpu-reference'/name).write_text('{}')
    rr=[]
    for i in range(256):
        r=dict(item_id=str(i),document_id='doc-'+str(i),split='synthetic_dev',prefix_text='P=',reanchor_text='=',target_text=' 123\n',
            supplied_notes=dict(correct='good',wrong='bad',partial_correct='part',partial_wrong='wrongpart'),
            numeric_only_mask=[False,True,True,True,False],numeric_only_supported=True)
        for key in ('prefix','reanchor','target'):r[key+'_ids']=tok.encode(r[key+'_text'])
        r['supplied_note_ids']={key:tok.encode(value) for key,value in r['supplied_notes'].items()};rr.append(r)
    (data/'reader-inputs.json').write_text(json.dumps(dict(rows=rr)))
    (data/'prefixes.jsonl').write_text(''.join(json.dumps(prefix(r))+'\n' for r in rr))
    manifest=dict(schema=e.dataset.SCHEMA,status='FROZEN_BEFORE_ANY_MODEL_OUTCOME',items=256,families={k:64 for k in e.dataset.FAMILIES},
        source_files={'train/controlled_arithmetic_data.py':sha256_file(e.dataset.__file__)},input_hashes={},
        prefixes_sha256=sha256_file(data/'prefixes.jsonl'),reader_inputs_sha256=sha256_file(data/'reader-inputs.json'),
        ordered_item_ids=[r['item_id'] for r in rr],model_dir=str(tmp_path/'model'),tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()),limitations=[])
    (data/'manifest.json').write_text(json.dumps(manifest));monkeypatch.setattr(e,'DATA_MANIFEST_SHA256',sha256_file(data/'manifest.json'))
    # Exact consumed legacy API: manifest SHA is inside assets, not a top-level key.
    old=dict(model_dir=manifest['model_dir'],assets=dict(manifest_sha256='asset-sha'),vocab_size=151936)
    monkeypatch.setattr(oracle,'load',lambda path:(old,dict(parts=protocol()['parts'])))
    monkeypatch.setattr(oracle,'reference',lambda *args:(dict(signature='released'),None))
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *args,**kwargs:tok)
    output=tmp_path/'prepared'
    p=e.prepare(output,data,native,protocol()['deadline_utc'],tmp_path/'control')
    assert p['model_manifest_sha256']=='asset-sha' and p['vocab_size']==151936
    assert p['writer']==e.base.B0 and p['writer_prefill_width']==32
    assert len(e.shared.rows(output/'prefixes.jsonl'))==256
    assert all(set(r)=={'item_id','document_id','split','prefix_ids','reanchor_ids'} for r in e.shared.rows(output/'prefixes.jsonl'))
    assert not (output/'notes').exists() and not (output/'scores').exists()
