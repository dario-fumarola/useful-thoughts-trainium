"""Question-only grouped decoder reuse, exact coverage and failure denominators."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train import useful_note_gsm8k as e
from train.evaluate import Decoded, GenerationFailure
from weekend_thinking.artifacts import canonical_hash, sha256_file


class Tokenizer:
    eos_token_id=151643

    def encode(self,text,**kwargs):return list(text.encode())

    def decode(self,ids,**kwargs):
        if list(ids)==[198,151667]:return '\n<think>'
        if list(ids)==[151668,198]:return '</think>\n'
        return bytes(ids).decode()


def format_data():
    return dict(format=dict(header_ids=[],open_ids=[198,151667],close_ids=[151668,198],
        pad_id=151643,stop_id=151668,additional_stop_ids=[151643]))


def questions(n=4):
    return [dict(item_id=f'gsm8k:train:{i:05}',document_id=f'd{i}',task='gsm8k',split='train',question=f'How many is{i}+1?') for i in range(n)]


def trained(key='W'):
    return dict(id=key,checkpoint='/tmp/science/step-0320/model.pt',checkpoint_sha256='a'*64,
        provenance={'/tmp/run.json':'b'*64},checkpoint_files={'/tmp/science/model.pt':'a'*64},
        training_source_files={'train/driver.py':'c'*64},run_id='actual-science',step=320,spec_hash='d'*64)


@pytest.mark.parametrize('key',['vanilla','thought','C','B0-as-W'])
def test_legacy_arm_aliases_cannot_forge_new_model_identity(key):
    with pytest.raises(ValueError,match='actual B0/W/Cprime/T'):e.validate_identity(dict(trained(),id=key))


def test_B0_cannot_hide_checkpoint_and_trained_requires_full_lineage(monkeypatch):
    assert e.resolve_identity('B0')['checkpoint'] is None
    for key in ('W','Cprime','T'):assert e.validate_identity(trained(key))['id']==key
    with pytest.raises(ValueError,match='disguise'):e.resolve_identity('B0',checkpoint=Path('/tmp/model.pt'))
    with pytest.raises(ValueError,match='unchanged'):e.validate_identity(dict(id='B0',loaded_checkpoint={'run_id':'trained'}))
    for missing in ('checkpoint_files','training_source_files','spec_hash','step'):
        identity=trained();identity.pop(missing)
        with pytest.raises(ValueError,match='completed science'):e.validate_identity(identity)
    monkeypatch.setattr(e.shared,'trained_identity',lambda *args:trained('W'))
    with pytest.raises(ValueError,match='differs'):
        e.resolve_identity('Cprime','/tmp/model.pt','a'*64,'b'*64,'/tmp/protocol.json')


def test_noheader_native_format_and_common_three_mode_width():
    tok=Tokenizer();fmt=e.validate_format(tok,format_data());qs=questions()
    prompt=dict(instruction='Show calculations, finish ####.',demonstrations=[])
    widths=e.measure_widths(qs,prompt,fmt,tok)
    for q in qs:
        prefix=tok.encode(e.evaluator.question_prompt(q['question'],prompt))
        suffix=tok.encode(e.evaluator.reader_suffix(q['question']))
        own=prefix+list(fmt.open_ids)+[42]*128+list(fmt.close_ids)+suffix
        assert max(len(prefix),len(own))<=widths['answer_prefill_width']
        assert len(prefix)<=widths['question_prefill_width']
    bad=format_data();bad['format']['header_ids']=[1]
    with pytest.raises(ValueError,match='no-header'):e.validate_format(tok,bad)
    bad=format_data();bad['format']['additional_stop_ids']=[]
    with pytest.raises(ValueError,match='first-close'):e.validate_format(tok,bad)


class Decoder:
    def __init__(self,tok,failure=False):self.tokenizer=tok;self.calls=[];self.failure=failure

    def decode(self,prompt,**kwargs):
        self.calls.append((prompt,kwargs))
        if self.failure:
            self.failure=False
            raise GenerationFailure(RuntimeError('fixture failure'),Decoded([65],[65],'A','A','error'))
        if not kwargs['answer_line_stop']:
            return Decoded([42,151668],[42],'*','*','token:151668')
        return Decoded([35]*8,[35]*8,'#### 1\n','#### 1\n','answer_line')


def pipeline(tmp_path,monkeypatch,failure=False):
    monkeypatch.setattr(e,'N',4);monkeypatch.setattr(e,'SHARDS',2)
    monkeypatch.setattr(e.shared,'check',lambda _:None)
    qs=questions();tok=Tokenizer();prompt=dict(instruction='Calculate; finish #### <number>.',demonstrations=[])
    (tmp_path/'prompt.json').write_text(json.dumps(prompt));(tmp_path/'format.json').write_text(json.dumps(format_data()))
    (tmp_path/'protocol.json').write_text('{}')
    identity=dict(id='B0',checkpoint=None,checkpoint_sha256=None)
    p=dict(model=identity,shards=2,deadline_utc='future',ordered_item_ids=[q['item_id'] for q in qs],purpose='TRAIN development',
        **e.measure_widths(qs,prompt,e.validate_format(tok,format_data()),tok),source_files={},input_hashes={},limitations=[])
    monkeypatch.setattr(e,'load',lambda output:(p,qs))
    backend=SimpleNamespace(model=SimpleNamespace(parameters=lambda:iter([torch.zeros(1)])),
                            sync=lambda:None,banned_ids=())
    monkeypatch.setattr(e.shared,'load_model',lambda *args:(backend,tok,identity))
    events=[]
    monkeypatch.setattr(e.shared,'base_gate',lambda *args:events.append('gate') or dict(scope='fixture'))
    decoders=[]
    def make(*args,**kwargs):
        assert events[-1]=='gate'
        decoder=Decoder(tok,failure=failure);decoders.append(decoder);return decoder
    monkeypatch.setattr(e.evaluator,'GreedyDecoder',make)
    return p,qs,decoders


def test_real_generate_item_reuse_runs_all_modes_per_model_job_without_gold(tmp_path,monkeypatch):
    p,qs,decoders=pipeline(tmp_path,monkeypatch)
    original=Path.read_text
    def guard(path,*args,**kwargs):
        assert path.name!='gold.jsonl','Generation opened gold'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guard)
    for shard in range(2):e.generate(tmp_path,shard,'future',device='cpu')
    values,reports,_,identity=e.complete_predictions(tmp_path,p,qs)
    assert len(values)==12 and len(decoders)==2 and identity['id']=='B0'
    for decoder in decoders:
        assert len(decoder.calls)==8 #2direct+2empty answers+2thought+2thought answers
        answers=[kw for _,kw in decoder.calls if kw['answer_line_stop']]
        assert len(answers)==6 and {kw['prefill_width'] for kw in answers}=={p['answer_prefill_width']}
        thoughts=[kw for _,kw in decoder.calls if not kw['answer_line_stop']]
        assert len(thoughts)==2 and all(kw['max_new_tokens']==128 for kw in thoughts)
    assert {(v['item_id'],v['mode']) for v in values}=={(q['item_id'],m) for q in qs for m in e.MODES}


def test_actual_generation_failure_retains_partial_path_and_all_denominators(tmp_path,monkeypatch):
    p,qs,_=pipeline(tmp_path,monkeypatch,failure=True)
    for shard in range(2):e.generate(tmp_path,shard,'future',device='cpu')
    values,_,_,_=e.complete_predictions(tmp_path,p,qs)
    failures=[r for r in values if r['status']=='error']
    assert len(failures)==2 and all(r['answer_actions']==[65] and r['raw_generated_text']=='A' for r in failures)
    gold=[dict(q,answer='#### 1') for q in qs]
    absolute,contrasts,_=e.reduce_predictions(gold,values,p['ordered_item_ids'])
    assert absolute['direct']['items']==4 and absolute['direct']['correct']==2
    assert absolute['direct']['statuses']['error']==2 and absolute['direct']['unsupported']==2
    assert absolute['one_thought']['correct']==4
    assert contrasts['thought_minus_direct']['difference_b_minus_a']==.5
    assert contrasts['thought_minus_empty_format']['ci95']==[0,0]


@pytest.mark.parametrize('defect',['missing','duplicate','wrong_mode','wrong_model','wrong_doc','bad_status','wrong_count'])
def test_complete_loader_rejects_semantic_gaps_even_with_refreshed_file_hash(tmp_path,monkeypatch,defect):
    p,qs,_=pipeline(tmp_path,monkeypatch)
    for shard in range(2):e.generate(tmp_path,shard,'future',device='cpu')
    d=tmp_path/'generation/shard-0';rows=[json.loads(s) for s in (d/'predictions.jsonl').read_text().splitlines()]
    if defect=='missing':rows.pop()
    elif defect=='duplicate':rows[-1]=rows[0]
    elif defect=='wrong_mode':rows[0]['mode']='one_thought'
    elif defect=='wrong_model':rows[0]['model_id']='vanilla'
    elif defect=='wrong_doc':rows[0]['document_id']='not-original'
    elif defect=='bad_status':rows[0]['status']='recovered_correct'
    else:rows[0]['n_answer_tokens']+=1
    (d/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    run=json.loads((d/'run.json').read_text());run['predictions_sha256']=sha256_file(d/'predictions.jsonl')
    (d/'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError):e.complete_predictions(tmp_path,p,qs)


def test_summary_cannot_open_gold_before_last_shard_closes(tmp_path,monkeypatch):
    p,_,_=pipeline(tmp_path,monkeypatch);p['gold_file']=str(tmp_path/'gold.jsonl');p['gold_sha256']='unknown'
    e.generate(tmp_path,0,'future',device='cpu')
    original=Path.read_text
    def guard(path,*args,**kwargs):
        assert path.name!='gold.jsonl','Gold opened before full path barrier'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guard)
    with pytest.raises((FileNotFoundError,ValueError)):e.summarize(tmp_path)
    assert not (tmp_path/'summary.json').exists()


def test_strict_parser_retains_caps_and_wrong_format_not_recovered_answers(tmp_path,monkeypatch):
    p,qs,_=pipeline(tmp_path,monkeypatch)
    for shard in range(2):e.generate(tmp_path,shard,'future',device='cpu')
    values,_,_,_=e.complete_predictions(tmp_path,p,qs)
    for row in values:
        if row['mode']=='direct':row.update(status='truncated',generated_text='#### 1\n',answer_stop_type='length_cap')
        if row['mode']=='empty_format':row.update(status='format_error',generated_text='The number is1.')
    absolute,contrasts,_=e.reduce_predictions([dict(q,answer='#### 1') for q in qs],values,p['ordered_item_ids'])
    assert absolute['direct']['accuracy']==absolute['empty_format']['accuracy']==0
    assert absolute['one_thought']['accuracy']==1
    assert absolute['direct']['unsupported']==absolute['direct']['answer_caps']==4
    assert absolute['empty_format']['statuses']['format_error']==4
    assert all(c['ci95']==[1,1] and c['items']==4 for c in contrasts.values())
    with pytest.raises(ValueError,match='distinct attempted'):
        e.reduce_predictions([dict(q,answer='#### 1') for q in qs],values[:-1],p['ordered_item_ids'])


def test_summary_is_complete_immutable_scored_output(tmp_path,monkeypatch):
    p,qs,_=pipeline(tmp_path,monkeypatch)
    for shard in range(2):e.generate(tmp_path,shard,'future',device='cpu')
    gold=tmp_path/'gold.jsonl';gold.write_text(''.join(json.dumps(dict(q,answer='#### 1'))+'\n' for q in qs))
    p.update(gold_file=str(gold),gold_sha256=sha256_file(gold))
    result=e.summarize(tmp_path)
    assert result['paths']==12 and result['items']==4 and result['model']['id']=='B0'
    assert all(v['correct']==4 and v['items']==4 for v in result['absolute'].values())
    assert len((tmp_path/'scores.jsonl').read_text().splitlines())==12
    with pytest.raises(FileExistsError):e.summarize(tmp_path)
