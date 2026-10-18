"""Frozen sixteen-way partition, native-reference reuse and fixed96 refusal."""
import copy,json
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from train import useful_note_gsm8k as g
from train import useful_note_gsm8k_equivalence as e
from train.evaluate import Decoded

class Tok:
    eos_token_id=151643
    def encode(self,text,**kwargs):return list(text.encode())
    def decode(self,ids,**kwargs):
        if list(ids)==[198,151667]:return '\n<think>'
        if list(ids)==[151668,198]:return '</think>\n'
        return bytes(ids).decode()

def fixture(tmp,monkeypatch):
    qs=[dict(item_id=f'gsm8k:train:{i:05}',document_id=f'd{i}',task='gsm8k',split='train',question='x'*(1000 if i==511 else 5)) for i in range(512)]
    tok=Tok();fmtdata=dict(format=dict(header_ids=[],open_ids=[198,151667],close_ids=[151668,198],pad_id=151643,stop_id=151668,additional_stop_ids=[151643]))
    prompt=dict(instruction='Show work and finish #### <number>.',demonstrations=[])
    for name,v in [('prompt.json',prompt),('format.json',fmtdata),('protocol.json',{})]:(tmp/name).write_text(json.dumps(v))
    model=dict(id='B0',checkpoint=None,checkpoint_sha256=None);fmt=g.validate_format(tok,fmtdata)
    p=dict(model=model,shards=16,deadline_utc='future',ordered_item_ids=[q['item_id'] for q in qs],purpose='synthetic',source_files={},input_hashes={},limitations=[],**g.measure_widths(qs,prompt,fmt,tok))
    backend=SimpleNamespace(model=SimpleNamespace(parameters=lambda:iter([torch.zeros(1)])),sync=lambda:None,banned_ids=[])
    monkeypatch.setattr(g,'load',lambda *a:(p,qs));monkeypatch.setattr(g.shared,'check',lambda *a:None)
    monkeypatch.setattr(g.shared,'load_model',lambda *a:(backend,tok,model));gates=[]
    monkeypatch.setattr(g.shared,'base_gate',lambda b,p,index,*a:gates.append(index) or dict(max_error=0.))
    decoders=[]
    class Decoder:
        def __init__(self,*a,**k):self.tokenizer=tok;self.calls=[];decoders.append(self)
        def decode(self,prompt,**kw):
            self.calls.append(kw)
            return Decoded([35]*8,[35]*8,'#### 1\n','#### 1\n','answer_line') if kw['answer_line_stop'] else Decoded([42,151668],[42],'*','*','token:151668')
    monkeypatch.setattr(g.evaluator,'GreedyDecoder',Decoder)
    return p,qs,model,gates,decoders

@pytest.mark.parametrize('value',[4,16])
def test_explicit_frozen_shard_choices_and_default4(value):
    assert g.validate_shards(value)==value
    assert signature(g.prepare).parameters['shards'].default==4

@pytest.mark.parametrize('value',[0,2,8,32,True,4.,None])
def test_no_implicit_quota_or_shard_fallback(value):
    with pytest.raises(ValueError):g.validate_shards(value)


def test_partition_preserves512_unique_and_first32_two_per_worker():
    qs=[dict(item_id=str(i)) for i in range(512)]
    for count in (4,16):
        pieces=[g.evaluator.shard_questions(qs,i,count) for i in range(count)]
        assert {q['item_id'] for piece in pieces for q in piece}=={str(i) for i in range(512)}
        assert sum(map(len,pieces))==512 and all(len(v)==512//count for v in pieces)
    assert all(len([q for q in g.evaluator.shard_questions(qs,i,16) if int(q['item_id'])<32])==2 for i in range(16))


def test_real_policy16_shards_all1536_and_four_native_refs(tmp_path,monkeypatch):
    p,qs,model,gates,decoders=fixture(tmp_path,monkeypatch)
    for i in range(16):g.generate(tmp_path,i,'future','cpu')
    values,reports,_,actual=g.complete_predictions(tmp_path,p,qs)
    assert len(values)==1536 and len({(v['item_id'],v['mode']) for v in values})==1536
    assert actual==model and gates==list(range(4))*4
    assert [r['native_reference_index'] for r in reports]==gates
    assert all(r['rows']==96 for r in reports)
    for decoder in decoders:
        assert {v['prefill_width'] for v in decoder.calls if v['answer_line_stop']}=={p['answer_prefill_width']}
    path=tmp_path/'generation/shard-15/predictions.jsonl';vv=g.shared.rows(path);path.write_text(''.join(json.dumps(v)+'\n' for v in vv[:-1]))
    r=g.shared.read(tmp_path/'generation/shard-15/run.json');r['predictions_sha256']=g.sha256_file(path)
    (tmp_path/'generation/shard-15/run.json').write_text(json.dumps(r))
    with pytest.raises(ValueError):g.complete_predictions(tmp_path,p,qs)


def test_qualifier_reuses_real_generate_item_and_measures_all512(tmp_path,monkeypatch):
    p,qs,model,gates,decoders=fixture(tmp_path,monkeypatch);out=tmp_path/'qual';out.mkdir();(out/'protocol.json').write_text(json.dumps(dict(schema=e.SCHEMA)))
    q=dict(deadline_utc='future',reference_model=model,full_preparation=str(tmp_path),ordered_item_ids=p['ordered_item_ids'][:32],common_widths={k:p[k] for k in ('question_prefill_width','answer_prefill_width','question_prompt_lengths','reader_suffix_lengths')})
    monkeypatch.setattr(e,'load',lambda *a:(q,p,qs));sizes=[];original=g.measure_widths
    monkeypatch.setattr(g,'measure_widths',lambda questions,*a:sizes.append(len(questions)) or original(questions,*a))
    read=Path.read_text
    def guard(path,*a,**k):
        if path.name=='gold.jsonl':raise AssertionError('Equivalence opened gold')
        return read(path,*a,**k)
    monkeypatch.setattr(Path,'read_text',guard);e.generate(out,15,'future','cpu')
    values=g.shared.rows(out/'generation/shard-15/predictions.jsonl');r=g.shared.read(out/'generation/shard-15/run.json')
    assert sizes==[512] and gates==[3] and r['rows']==6 and r['status']=='complete_equivalence_attempts'
    assert {v['item_id'] for v in values}=={p['ordered_item_ids'][15],p['ordered_item_ids'][31]}
    assert all(kw['prefill_width']==p['answer_prefill_width'] for kw in decoders[0].calls if kw['answer_line_stop'])
    assert p['question_prefill_width']>original(qs[:32],g.shared.read(tmp_path/'prompt.json'),g.validate_format(Tok(),g.shared.read(tmp_path/'format.json')),Tok())['question_prefill_width']
    assert e.SCHEMA!=g.SCHEMA


def native_rows():
    ids=[str(i) for i in range(32)];vv=[dict(item_id=i,mode=m,status='ok',answer_actions=[3],thought_actions=[]) for i in ids for m in g.MODES]
    return ids,vv,{e.key(v):e.projection(v) for v in vv}


def test_native_comparison_is_exact_actions_not_accuracy():
    ids,vv,refs=native_rows();assert e.reduce_rows(vv,refs,ids)==[]
    vv[0]['answer_actions']=[4];d=e.reduce_rows(vv,refs,ids)
    assert d==[dict(item_id='0',mode='direct',differing_fields=['answer_actions'])]

@pytest.mark.parametrize('defect',['missing','duplicate','replacement_duplicate','reference_missing'])
def test_fixed96_cannot_certify_partial_or_duplicate_paths(defect):
    ids,vv,refs=native_rows()
    if defect=='missing':vv.pop()
    elif defect=='duplicate':vv.append(copy.deepcopy(vv[0]))
    elif defect=='replacement_duplicate':vv[-1]=copy.deepcopy(vv[0])
    else:refs.pop(next(iter(refs)))
    with pytest.raises(ValueError):e.reduce_rows(vv,refs,ids)
