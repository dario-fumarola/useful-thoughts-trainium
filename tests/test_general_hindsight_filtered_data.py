"""Selection, causal native projections, closed coverage and retained failures."""
import copy
import json
from pathlib import Path

import pytest
import torch

from train import general_hindsight_filtered_data as d
from train import general_hindsight_filtered_batch as b
from weekend_thinking.artifacts import canonical_hash,sha256_file


class Tok:
    all_special_ids=[151643,151644]
    eos_token_id=151643
    pieces={1:'x',2:'y',3:'z',4:'helper',5:' \t',6:'\n',7:'next',99:'x'*16,100:' y'*16}
    def get_vocab(self):return {str(i):i for i in list(range(21))+[99,100]}
    def decode(self,ids,**kwargs):return ''.join(self.pieces.get(i,str(i)) for i in ids)
    def encode(self,*args,**kwargs):raise AssertionError('Native labels must not be re-encoded')


def tiny(monkeypatch):
    for name,value in [('RAW_N',4),('TEACHER_N',2),('LABEL_N',1),('SHARDS',2)]:monkeypatch.setattr(d,name,value)


def test_metadata_order_has_no_target_score_or_input_iteration_dependence():
    a=[dict(item_id=k,document_id='d'+k,split='train') for k in ('a','b','c')]
    first=[r['item_id'] for r in d.pool_order(a)]
    altered=[dict(r,future='not inspected',gain=999-i) for i,r in enumerate(reversed(a))]
    assert [r['item_id'] for r in d.pool_order(altered)]==first
    with pytest.raises(ValueError):d.pool_order(a+[dict(a[0],item_id='another')])
    with pytest.raises(ValueError):d.pool_order([dict(a[0],split='dev')])


def raw_matrix():
    order=[dict(pool_index=i,item_id=str(i),document_id='d'+str(i)) for i in range(4)]
    scores=[dict(r,status='ok',token_logps=[-v]+[-.1]*15,raw_K1_surprisal_nats=v) for r,v in zip(order,[1.7,.5,2.,100.])]
    return scores,order


def test_first_qualifiers_not_best_or_last_qualifiers_are_selected(monkeypatch):
    tiny(monkeypatch);scores,order=raw_matrix();chosen,audit=d.choose_teachers(list(reversed(scores)),order)
    assert chosen==[0,2] and audit['qualifiers']==3
    scores[-1]['token_logps'][0]=-999;scores[-1]['raw_K1_surprisal_nats']=999
    assert d.choose_teachers(scores,order)[0]==[0,2]


def test_unknown_earlier_raw_is_never_silently_ineligible(monkeypatch):
    tiny(monkeypatch);scores,order=raw_matrix();scores[0]=dict(order[0],status='error')
    assert d.choose_teachers(scores,order)[1]['status']=='FAILED_UNKNOWN_EARLIER_ELIGIBILITY'
    scores,order=raw_matrix();scores[-1]=dict(order[-1],status='error')
    assert d.choose_teachers(scores,order)[0]==[0,2] # Error AFTER the fixed final selected index.


def test_shortfall_missing_duplicate_and_nonfinite_raw_fail_without_replacement(monkeypatch):
    tiny(monkeypatch);scores,order=raw_matrix()
    for x in scores:x.update(token_logps=[-.2]*16,raw_K1_surprisal_nats=.2)
    assert d.choose_teachers(scores,order)[1]['status']=='FAILED_FIXED_TEACHER_QUOTA'
    with pytest.raises(ValueError):d.choose_teachers(scores[:-1],order)
    with pytest.raises(ValueError):d.choose_teachers(scores+[scores[0]],order)
    scores[0]['token_logps'][1]=float('nan')
    with pytest.raises(ValueError):d.choose_teachers(scores,order)


def test_raw_K1_cannot_drift_from_saved_first_token(monkeypatch):
    tiny(monkeypatch);scores,order=raw_matrix();scores[0]['raw_K1_surprisal_nats']=3.
    with pytest.raises(ValueError,match='K1'):d.choose_teachers(scores,order)


def test_native_projection_rejects_truncation_undefined_and_changed_source():
    r=dict(split='train',prefix_ids=[1]*512,natural_future16_ids=[2]*16,reanchor_ids=[1]*24,source528_sha256=canonical_hash([1]*512+[2]*16))
    d.validate_native(r,{1,2})
    for changed in [dict(r,natural_future16_ids=[2]*15),dict(r,source528_sha256='wrong'),dict(r,prefix_ids=[3]+[1]*511),dict(r,split='test')]:
        with pytest.raises(ValueError):d.validate_native(changed,{1,2})


def test_raw_CPU_preparation_never_decodes_TEST_or_DEV_payloads(tmp_path,monkeypatch):
    from transformers import AutoTokenizer
    tiny(monkeypatch);tok=Tok();monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    monkeypatch.setattr(d.shared,'load_model',lambda *a,**k:(_ for _ in ()).throw(AssertionError('CPU must not load model')))
    source=tmp_path/'records';membership=tmp_path/'membership';design=tmp_path/'design';design.write_text('{}')
    originals=[]
    with source.open('w') as stream:
        stream.write('{"id":"t","document_id":"dt","split":"test","prefix":FORBIDDEN_NON_JSON_future\n')
        stream.write('{"id":"v","document_id":"dv","split":"dev","prefix":FORBIDDEN_NON_JSON_future\n')
        for i in range(4):
            r=dict(id=str(i),document_id='d'+str(i),split='train',prefix=[i]+[1]*511,future=[2]*16);originals.append(r);stream.write(json.dumps(r)+'\n')
    membership.write_text('{"split":"test","opaque":FORBIDDEN_NON_JSON}\n'+''.join(json.dumps(dict(record_id=r['id'],document_id=r['document_id'],split='train'))+'\n' for r in originals))
    ordered=d.pool_order([dict(item_id=r['id'],document_id=r['document_id'],split='train') for r in originals])
    design.write_text(json.dumps(dict(raw_pool=dict(documents=4,salt=d.SALT,ordered_item_ids_sha256=canonical_hash([r['item_id'] for r in ordered]),ordered_document_ids_sha256=canonical_hash([r['document_id'] for r in ordered])))))
    sm=tmp_path/'source-manifest';sm.write_text(json.dumps(dict(output_sha256=sha256_file(source),document_membership_sha256=sha256_file(membership))))
    old=tmp_path/'pilot';old.mkdir()
    fields=dict(model_dir='unused',assets={},model_manifest_sha256='unused',base_signature={},native_reference='unused',native_inputs={},writer=d.pilot.B0,vocab_size=151936,banned_ids=[],teacher_parts={},parts={},writer_prefill_width=800,teacher_policy={})
    (old/'protocol.json').write_text(json.dumps(fields))
    monkeypatch.setattr(d,'SOURCE_SHA',sha256_file(source));monkeypatch.setattr(d,'DESIGN_SHA',sha256_file(design));monkeypatch.setattr(d,'PILOT_PROTOCOL_SHA',sha256_file(old/'protocol.json'))
    out=tmp_path/'out';before=sha256_file(source);m=d.prepare_raw(out,source,membership,sm,design,old,'2099-01-01T00:00:00+00:00')
    assert m['raw_items']==4 and sha256_file(source)==before and m['source_counts']==dict(train=4,dev=1,test=1)
    native=[r for i in range(2) for r in d.shared.rows(d.shard_file(out,'raw',i))]
    assert {r['item_id'] for r in native}=={'0','1','2','3'} and all(r['split']=='train' for r in native)
    for r in native:assert r['prefix_ids']==originals[int(r['item_id'])]['prefix'] and r['natural_future16_ids']==[2]*16
    assert [r['pool_index'] for r in d.shared.rows(d.shard_file(out,'raw',0))]==[0,2]


@pytest.mark.parametrize('content,reason',[([5,6],'empty_stripped_text'),([151644],'native_special_or_thought_marker'),([151667],'native_special_or_thought_marker'),([151668],'native_special_or_thought_marker')])
def test_plain_text_gates_include_whitespace_and_non_special_thought_markers(content,reason):
    out=d.label_eligibility(dict(status='ok',content_ids=content),[2]*16,Tok(),{151643,151644,151667,151668})
    assert not out['eligible'] and reason in out['reasons']


def test_full16_copy_checked_as_tokens_and_retokenized_whitespace_only_text():
    t=Tok();forbid={151643,151668}
    exact=d.label_eligibility(dict(status='ok',content_ids=[1]*16),[1]*16,t,forbid)
    assert exact['full16_token_copy'] and exact['full16_character_copy']
    other=d.label_eligibility(dict(status='ok',content_ids=[99]),[1]*16,t,forbid)
    assert not other['full16_token_copy'] and other['full16_character_copy'] and not other['eligible']
    whitespace=d.label_eligibility(dict(status='ok',content_ids=[100]),[5,2]*8,t,forbid)
    assert whitespace['full16_character_copy']
    assert d.label_eligibility(dict(status='ok',content_ids=[1]*8),[1]*16,t,forbid)['eligible'] # No new shorter-overlap gate.


def test_no_case_or_punctuation_normalization_in_literal_copy_gate():
    class CaseTok(Tok):pieces={1:'A'*16,2:'a'*16,3:'a.'*16}
    assert d.label_eligibility(dict(status='ok',content_ids=[2]),[1]*16,CaseTok(),set())['eligible']
    assert d.label_eligibility(dict(status='ok',content_ids=[3]),[2]*16,CaseTok(),set())['eligible']


def test_global_donor_projection_preserves_error_empty_and_full_population(monkeypatch):
    monkeypatch.setattr(d,'TEACHER_N',3)
    rows={i:dict(item_id=str(i),document_id='d'+str(i)) for i in range(3)}
    notes={0:dict(content_ids=[4]*5,status='ok'),1:dict(content_ids=[],status='ok'),2:dict(content_ids=[3],status='error')}
    out=d.global_reader_projection(rows,notes)
    assert out[0]['other_note'] is notes[1] and out[1]['other_note'] is notes[2] and out[2]['other_note'] is notes[0]
    assert len(out)==3 and out[0]['other_note']['content_ids']==[]
    with pytest.raises(ValueError):d.global_reader_projection(rows,{0:notes[0],1:notes[1]})


def native_row():
    return dict(item_id='a',document_id='da',split='train',pool_index=0,teacher_index=0,prefix_ids=[1]*512,natural_future16_ids=[2]*16,
        reanchor_ids=[1]*24,original_source_ids=[1]*512+[2]*16,teacher_note_ids=[4],teacher_note_text='helper')


def test_native_writer_has_no_future_input_and_reader_labels_exact_original16():
    r=native_row();parts=dict(old_open=[198,151667],old_close=[151668,198]);tok=Tok()
    w=d.pack_native(r,tok,'writer',parts=parts,width=640,target_slots=65)
    changed=dict(r,natural_future16_ids=[3]*16,original_source_ids=[1]*512+[3]*16)
    w2=d.pack_native(changed,tok,'writer',parts=parts,width=640,target_slots=65)
    assert torch.equal(w['input_ids'],w2['input_ids']) and w['future_mask'].sum()==0 and w['note_close_mask'].sum()==2
    assert w['labels'][0,:2].tolist()==[4,151668]
    rr=d.pack_native(r,tok,'reader',parts=parts,width=640,target_slots=16)
    context=r['prefix_ids']+parts['old_open']+[4]+parts['old_close']+r['reanchor_ids']
    assert rr['input_ids'][0,:len(context)+16].tolist()==context+[2]*16
    assert rr['labels'][0].tolist()==[2]*16 and rr['target_positions'][0].tolist()==list(range(len(context),len(context)+16))
    assert rr['note_close_mask'].sum()==0 and rr['future_mask'].sum()==16


def test_ordinary_control_labels_all527_identical_source_tokens():
    r=native_row();v=d.pack_native(r,Tok(),'source',parts={},width=544,target_slots=527)
    assert v['input_ids'][0,:528].tolist()==r['original_source_ids'] and v['labels'][0].tolist()==r['original_source_ids'][1:]
    assert v['future_mask'].sum()==527


def test_native_adapter_never_truncates_overwidth_or_forges_symbolic_labels():
    r=native_row();parts=dict(old_open=[198,151667],old_close=[151668,198])
    with pytest.raises(ValueError):d.pack_native(r,Tok(),'reader',parts=parts,width=520,target_slots=16)
    with pytest.raises(ValueError):d.pack_native(dict(r,teacher_note_ids=[151668]),Tok(),'writer',parts=parts,width=640,target_slots=65)


def test_teacher_generation_failure_retains_actual_actions_and_firstline_prefix():
    from train.evaluate import Decoded,GenerationFailure
    row=native_row();p=dict(teacher_parts=dict(initial=[10],middle=[11],final=[12]),writer_prefill_width=800)
    class Failed:
        def decode(self,*args,**kwargs):
            assert kwargs['stop_ids']==(151643,) and kwargs['max_new_tokens']==64 and not kwargs['answer_line_stop']
            raise GenerationFailure(RuntimeError('device error'),Decoded([4,6,7],[4,6,7],'helper\nnext','helper\nnext','error'))
    v=b.generate_one(Failed(),row,p,Tok())
    assert v['status']=='error' and v['actions']==[4,6,7] and v['raw_content_ids']==[4,6,7] and v['content_ids']==[4,6]


def decision_fixture():
    records=[];values=[]
    for i in range(2):
        r=dict(native_row(),teacher_index=i,pool_index=i,item_id=str(i),document_id='d'+str(i),teacher_note=dict(status='ok',content_ids=[4],content_text='helper',raw_content_ids=[4]))
        records.append(r)
        for path in d.pilot.PATHS:
            for mode,ll in [('raw',-1.),('empty',-3.),('teacher',-2.),('other',-1.5)]:
                values.append(dict(teacher_index=i,path=path,mode=mode,status='ok',natural16_ll=ll))
    p=dict(forbidden_content_ids=[151643,151644,151667,151668],teacher_policy=dict(synthetic_illustrative_examples=[dict(note='helper')]))
    return values,records,p


def test_demo_copy_and_losing_to_raw_other_are_descriptive_not_new_semantic_gates():
    values,records,p=decision_fixture();out=b.decision_rows(values,records,p,Tok())
    assert all(v['accepted'] and v['demo_note_exact_whitespace_normalized_match'] for v in out)
    assert all(not v['paths']['I3']['all_three'] and not v['paths']['I3']['strict_teacher_positive']['raw'] for v in out)


def test_teacher_error_ties_and_missing_cells_never_become_filtered_acceptances():
    values,records,p=decision_fixture();values[2]['status']='error';values[10]['natural16_ll']=-3.
    out=b.decision_rows(values,records,p,Tok())
    assert not out[0]['accepted'] and not out[1]['accepted']
    with pytest.raises(ValueError):b.decision_rows(values[:-1],records,p,Tok())
    with pytest.raises(ValueError):b.decision_rows(values+[values[0]],records,p,Tok())


def test_closed_barrier_demands_all48_identities_and_actual_device_release(tmp_path):
    (tmp_path/'protocol.json').write_text('{}');p=dict(control_dir=str(tmp_path))
    c=dict(status='COMPLETE_PHYSICALLY_CLOSED',protocol_sha256=sha256_file(tmp_path/'protocol.json'),all16_exit0=True,
        all48_owned_PID_start_identities_absent=True,all_sessions_absent=True,assigned_devices_empty=True,owned_group_survivors=[])
    f=tmp_path/'raw.closure.json';f.write_text(json.dumps(c));assert d.require_closure(tmp_path,p,'raw')
    c['all48_owned_PID_start_identities_absent']=False;f.write_text(json.dumps(c))
    with pytest.raises(ValueError):d.require_closure(tmp_path,p,'raw')
