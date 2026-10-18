"""Prefix-only policy, whole-path retention, extraction and full-vocab score seams."""
import copy,json
from dataclasses import asdict
from inspect import signature
from types import SimpleNamespace
import pytest
import torch
from train import general_prefix_helper_probe as e
from train.evaluate import Decoded,GenerationFailure
from weekend_thinking.artifacts import canonical_hash

class Tok:
    def decode(self,ids,**kwargs):return ''.join({7:'alpha',8:'\nbeta',9:'\n',6:'x'}.get(v,'z') for v in ids)
    def encode(self,*args,**kwargs):raise AssertionError('Extraction must never retokenize')

def row(i=0):return dict(item_id=f'i{i}',document_id=f'd{i}',split='dev',prefix_ids=[3+i]*512,reanchor_ids=[3+i]*24,natural_future16_ids=[1+i]*16)
def p():return dict(writer=e.base.B0,ordered_item_ids=['i0','i1'],deadline_utc='2099-01-01T00:00:00+00:00',vocab_size=151936,banned_ids=[],
    writer_parts=dict(plain_open=[151667],fewshot_initial=[11,12],fewshot_final=[13]),writer_prefill_widths=dict(plain=544,fewshot=544),parts=dict(old_open=[151667],old_close=[151668,198]),control_dir='/unused')
def note(q,policy='plain',actions=None,status='ok'):
    a=[7,8,151668] if actions is None else actions;content=a[:-1] if a and a[-1] in (151668,151643) else a
    return dict(item_id=q['item_id'],document_id=q['document_id'],policy=policy,status=status,actions=a,content_ids=content,
        n_actions=len(a),n_content_tokens=len(content),prefix_sha256=canonical_hash(q['prefix_ids']),
        writer_prompt_sha256=canonical_hash(e.writer_prompt(q['prefix_ids'],p()['writer_parts'])[policy]),
        raw_text=Tok().decode(content),stop_type=f'token:{a[-1]}' if a and a[-1] in (151668,151643) else 'length_cap',wall_seconds=1.,extraction=e.truncate_content(content,Tok()))
def notes():return {f'{policy}/i{i}':note(row(i),policy,[7,8,151668] if i==0 else [151643]) for policy in e.POLICIES for i in range(2)}

def test_prompt_api_has_only_native_prefix_and_fixed_parts():
    assert list(signature(e.writer_prompt).parameters)==['prefix_ids','parts']
    parts=p()['writer_parts'];v=e.writer_prompt([3,4],parts)
    assert v==dict(plain=[3,4,151667],fewshot=[11,12,3,4,13])
    assert parts==p()['writer_parts']

@pytest.mark.parametrize('actions',[[151643],[151668],[7]*63+[151668],[7]*64])
def test_total64_action_contract_includes_native_stop(actions):
    v=note(row(),actions=actions);e.validate_note(v,row(),p(),Tok())
    assert v['content_ids']==(actions[:-1] if actions[-1] in (151643,151668) else actions)

@pytest.mark.parametrize('change',[lambda v:v.update(actions=[7]*65),lambda v:v.update(content_ids=[8]),lambda v:v.update(prefix_sha256='bad'),lambda v:v.update(writer_prompt_sha256='bad')])
def test_native_note_changes_rejected(change):
    v=note(row());change(v)
    with pytest.raises(ValueError):e.validate_note(v,row(),p(),Tok())


def test_firstline_keeps_merged_token_suffix_and_never_reencodes():
    v=note(row(),actions=[7,8,6,151668]);assert v['extraction']['content_ids']==[7,8]
    assert v['extraction']['content_text']=='alpha\nbeta'
    assert v['content_ids']==[7,8,6]


def test_global_donors_remain_intact_by_policy_and_extraction():
    pp=p();nn=notes();variants=e.note_variants(pp,nn)
    for key in e.VARIANTS:
        c=e.shared.other_controls(pp,variants[key])
        assert c['other_donors']=={'i0':'i1','i1':'i0'}
        assert c['other_content_ids']['i0']==[]
        assert c['other_content_ids']['i1']==variants[key]['i0']['content_ids']
    assert nn['plain/i0']['content_ids']==[7,8]


def test_controls_refuse_before_any_notes_or_future_access(monkeypatch,tmp_path):
    called=[]
    monkeypatch.setattr(e.base,'closure',lambda *a:(_ for _ in ()).throw(ValueError('not physically closed')))
    monkeypatch.setattr(e,'complete_notes',lambda *a:called.append('notes'))
    with pytest.raises(ValueError):e.controls(tmp_path,p(),[row()])
    assert called==[]


def test_generate_preserves_genuine_partial_on_decode_error(monkeypatch,tmp_path):
    from train import evaluate
    pp=p();q=row();(tmp_path/'protocol.json').write_text('{}')
    monkeypatch.setattr(e,'load',lambda *a:(pp,[q]));monkeypatch.setattr(e.shared,'load_model',lambda *a:(SimpleNamespace(model=None,sync=lambda:None,banned_ids=[]),Tok(),e.base.B0))
    monkeypatch.setattr(e.shared,'base_gate',lambda *a:dict(max_error=0.))
    class Decoder:
        def __init__(self,*a,**k):self.n=0
        def decode(self,prompt,**kwargs):
            assert kwargs['max_new_tokens']==64 and kwargs['stop_ids']==(151668,151643)
            self.n+=1
            if self.n==1:return Decoded([7]*64,[7]*64,'alpha'*64,'alpha'*64,'length_cap')
            raise GenerationFailure(ValueError('synthetic'),Decoded([6],[6],'x','x','error'))
    monkeypatch.setattr(evaluate,'GreedyDecoder',Decoder)
    e.generate(tmp_path,0,pp['deadline_utc'],'cpu')
    vv=e.shared.rows(tmp_path/'notes/shard-0/notes.jsonl')
    assert len(vv)==2 and vv[0]['n_actions']==64
    assert vv[1]['status']=='error' and vv[1]['actions']==[6] and vv[1]['raw_text']=='x'
    assert e.shared.read(tmp_path/'notes/shard-0/run.json')['rows']==2


def scores():
    return [dict(item_id=f'i{i}',document_id=f'd{i}',variant=variant,path=path,mode=mode,status='ok',natural16_ll={'raw':-10.,'empty':-9.,'own':-8.,'other':-11.,'target_copy':-1.,'corrupt_copy':-12.}[mode])
        for i in range(2) for variant in e.VARIANTS for path in e.base.PATHS for mode in e.base.MODES]

def test_primary_is_fewshot_firstline_mixed_not_any_secondary():
    vv=scores()
    for v in vv:
        if v['variant']=='fewshot/firstline' and v['path']=='I6w0.375' and v['mode']=='own':v['natural16_ll']=-13.
    r=e.reduce_rows(vv,['i0','i1'],['d0','d1'])
    assert r['primary']['variant']=='fewshot/firstline' and not r['primary']['passed']
    assert r['variants']['plain/full']['I3']['gates']['useful_content']
    assert r['variants']['plain/full']['I3']['contrasts']['target_copy_minus_raw']['difference_b_minus_a']==9.

@pytest.mark.parametrize('mutate',[lambda v:v.pop(),lambda v:v.append(copy.deepcopy(v[0]))])
def test_missing_duplicate_or_partial_variant_never_complete(mutate):
    vv=scores();mutate(vv)
    with pytest.raises(ValueError):e.reduce_rows(vv,['i0','i1'],['d0','d1'])


def test_error_stays_in_denominator_and_blocks_primary():
    vv=scores()
    next(v for v in vv if v['variant']=='fewshot/firstline' and v['path']=='I6w0.375' and v['mode']=='own').update(status='error')
    r=e.reduce_rows(vv,['i0','i1'],['d0','d1'])
    assert not r['primary']['passed'];a=r['variants']['fewshot/firstline']['I6w0.375']['absolute']['own']
    assert a==dict(items=2,errors=1,mean=None)


def test_score_real_full_logit_mixer_and_shared_costs(monkeypatch,tmp_path):
    import train.interface_mixer_probe as mixer
    pp=p();rr=[row(),row(1)];nn=notes();vv=e.note_variants(pp,nn);ctrl=dict(variants={k:e.shared.other_controls(pp,n) for k,n in vv.items()})
    (tmp_path/'protocol.json').write_text('{}');(tmp_path/'controls.json').write_text('{}')
    monkeypatch.setattr(e,'load',lambda *a:(pp,rr));monkeypatch.setattr(e,'controls',lambda *a:(nn,vv,ctrl));monkeypatch.setattr(e.base,'reader_rows',lambda *a:rr)
    backend=SimpleNamespace(device='cpu',sync=lambda:None);monkeypatch.setattr(e.shared,'load_model',lambda *a:(backend,None,e.base.B0));monkeypatch.setattr(e.shared,'base_gate',lambda *a:dict(max_error=0.))
    calls=[]
    def full(_backend,past,target):
        calls.append(tuple(past));return torch.arange(16*4,dtype=torch.float32).reshape(16,4)/100+len(past)/100
    monkeypatch.setattr(mixer,'full_logits',full)
    e.score(tmp_path,0,pp['deadline_utc'],'cpu');values=e.shared.rows(tmp_path/'scores/shard-0/scores.jsonl')
    assert len(values)==48 and all(v['status']=='ok' for v in values)
    assert len(calls)==len(set(calls))  # true full-V context work reused, not LP interpolation
    cost=e.shared.saved_costs(nn,values)
    assert cost['raw_full_logits_seconds_sum']>0 and cost['total_generated_actions']==8
    assert all(v['shared_raw_forward_seconds']>0 for v in values if v['mode']=='raw')


def test_prepare_uses_exact_fixed_demos_but_never_actual_future(monkeypatch,tmp_path):
    from transformers import AutoTokenizer
    base_dir=tmp_path/'base';base_dir.mkdir();(base_dir/'protocol.json').write_text('{}');(base_dir/'prefixes.jsonl').write_text('prefixes')
    future=tmp_path/'DO_NOT_OPEN.json';future.write_text('actual future secret')
    decl=tmp_path/'declaration.json';demo='\n\nPassage:\nSynthetic past\nObserved continuation:\nSynthetic future\nHelper note: Fixed helper'
    original='OLD FUTURE-AWARE INSTRUCTION'+demo*3+'\n\nPassage:\n';decl.write_text(json.dumps(dict(teacher=dict(fixed_initial_text=original))))
    monkeypatch.setattr(e,'DEMO_SHA',e.sha256_file(decl));old={k:None for k in ('assets','model_dir','model_manifest_sha256','base_signature','native_reference','native_inputs')}
    old.update(parts=p()['parts'],vocab_size=151936,banned_ids=[],ordered_item_ids=[f'i{i}' for i in range(256)],ordered_document_ids=[f'd{i}' for i in range(256)],reader_inputs_file=str(future),reader_inputs_sha256='future_hash',input_hashes={})
    prefixes=[e.shared.prefix_row(row(i)) for i in range(256)];monkeypatch.setattr(e.base,'load',lambda *a:(old,prefixes))
    encoded=[]
    class Encoder:
        def encode(self,text,**kw):encoded.append(text);return [len(text)]
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:Encoder())
    actual_read=e.Path.read_text
    def guarded_read(path,*a,**k):
        if path==future:raise AssertionError('Future opened during prepare')
        return actual_read(path,*a,**k)
    monkeypatch.setattr(e.Path,'read_text',guarded_read)
    result=e.prepare(tmp_path/'new',base_dir,decl,'2099-01-01T00:00:00+00:00',tmp_path/'control')
    assert result['schema']==e.SCHEMA and result['max_actions']==64
    assert encoded[0].startswith(e.INSTRUCTION) and 'OLD FUTURE-AWARE INSTRUCTION' not in encoded[0]
    assert encoded[0].count('Observed continuation:')==3 and encoded[0].endswith('Passage:\n')
    assert encoded[1]=='\nHelper note:' and 'actual future secret' not in ''.join(encoded)
    assert result['primary'].startswith('fewshot/firstline fixedI6')


def test_raw_generation_cost_is_not_discounted_by_four_extractions():
    nn=notes();values=[]
    for variant in e.VARIANTS:
        for path in e.base.PATHS:values.append(dict(timing_group='one-actual-raw',wall_seconds=2.,shared_raw_forward_seconds=2.,shared_by_paths=list(e.base.PATHS)))
    v=e.shared.saved_costs(nn,values)
    assert v['total_generated_actions']==8 and v['note_generation_seconds_sum']==4.
    assert v['scoring_cell_seconds_sum']==2. and v['raw_full_logits_seconds_sum']==2.
