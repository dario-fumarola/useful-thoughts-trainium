"""Real-source scalar closure, causal selection, explicit answer-copy payloads."""
from fractions import Fraction
import json
from pathlib import Path

import pytest

from train import computable_equations as probe
from weekend_thinking.artifacts import canonical_hash, sha256_file


class Characters:
    def encode(self,text,*,add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]
    def decode(self,ids,**kw):return ''.join(chr(i) for i in ids)
    def get_vocab(self):return {chr(i):i for i in range(128)}


class Merged(Characters):
    def encode(self,text,*,add_special_tokens=False):
        out=[]
        while text:
            if text.startswith('24;'):out.append(1000);text=text[3:]
            else:out.append(ord(text[0]));text=text[1:]
        return out
    def decode(self,ids,**kw):return ''.join('24;' if i==1000 else chr(i) for i in ids)


def record(tok=None,expression='6*4',rhs='24; natural continuation.',split='train',doc='one'):
    tok=tok or Characters();past=' Context: '+expression+'='
    prefix=tok.encode('p'*(512-len(tok.encode(past)))+past)
    future=tok.encode(rhs+' additional source context')[:16]
    assert len(prefix)==512 and len(future)==16
    return dict(id='id:'+doc,document_id='doc:'+doc,split=split,prefix=prefix,future=future)


def selected(r,tok):
    choice,_=probe.choose_boundary(r['prefix'],r['document_id'],r['id'],tok)
    assert choice is not None
    return choice


@pytest.mark.parametrize('expression,rhs,value',[
    ('6*4','24; continuation.',24),('12+13','25; continuation.',25),
    ('12-5','7; continuation.',7),('2^3','8; continuation.',8),
    ('2.5*3','7.5; continuation.',Fraction(15,2)),
    ('(-2)*3','-6; continuation.',-6),('1-1','0; continuation.',0),
])
def test_one_binary_operation_is_sufficient_and_copy_is_full_target(expression,rhs,value):
    tok=Characters();r=record(tok,expression,rhs);choice=selected(r,tok)
    row,status=probe.curate(r,choice,tok,Path('source.jsonl'),'source-sha')
    assert status['status']=='accepted' and Fraction(row['numeric_value'])==value
    assert row['correct_note_ids']==row['copy_note_ids']==row['target_ids']
    assert row['correct_note_text']==tok.decode(row['target_ids'])
    assert row['numeric_surface_text']==row['verified_numeric_rhs']['literal_text']
    assert 'NOT a proper intermediate' in row['supervision']
    assert row['target_ids']==row['natural_future16_ids'][:len(row['target_ids'])]
    assert row['original_source_ids']==r['prefix']+r['future']
    assert row['reanchor_text']==choice['expression']+'='
    assert tok.decode(row['reanchor_ids'])==row['reanchor_text']


def test_zero_operation_and_known_visible_whole_are_prefix_ineligible():
    tok=Characters();r=record(tok,'24','24; continuation.')
    choice,log=probe.choose_boundary(r['prefix'],r['document_id'],r['id'],tok)
    assert choice is None and log[-1]['reason']=='no_binary_operation'
    past='Earlier Rs.24; Context: 6*4='
    r['prefix']=tok.encode('p'*(512-len(past))+past)
    choice,log=probe.choose_boundary(r['prefix'],r['document_id'],r['id'],tok)
    assert choice is None and log[-1]['reason']=='ambiguous_decimal_boundary_copy_span'


@pytest.mark.parametrize('rhs',['24 × 2; remainder.','24 (2); remainder.','24 million people.',
    '24,000; remainder.','24e2; remainder.','24 1/2; remainder.','24!; remainder.',
    '24.5+2; remainder.'])
def test_full_numeral_closing_boundary_never_accepts_a_numeric_fragment(rhs):
    tok=Characters();r=record(tok,rhs=rhs)
    row,status=probe.curate(r,selected(r,tok),tok,'source','sha')
    assert row is None and status['status']=='unsupported_source_label'


@pytest.mark.parametrize('rhs,certificate,kind',[
    (' 24; later prose.',';','sentence'),('24. Next paragraph.','.','sentence'),
    ('24\nNext paragraph.','\n','line'),('24 = 6*4; later.','=','equality'),
    ('24 dollars paid.','d','word'),
])
def test_target_includes_actual_native_closing_certificate(rhs,certificate,kind):
    tok=Characters();r=record(tok,rhs=rhs);target=probe.source_target(selected(r,tok),r['prefix']+r['future'],tok)
    assert target['closing_boundary_char']==certificate and target['completion_kind']==kind
    assert target['serialized_text'].endswith(certificate)
    assert target['token_ids'][target['closing_boundary_token_start']:]==target['closing_boundary_ids']
    assert sum(target['numeric_token_mask'])==2
    assert target['numeric_token_mask'][-1] is False
    assert target['closing_boundary_source_char']==512+target['closing_boundary_char_offset']


def test_merged_number_and_closure_stays_one_original_token_and_unequal_corruption_is_retained():
    tok=Merged();r=record(tok);row,status=probe.curate(r,selected(r,tok),tok,'source','sha')
    assert status['status']=='accepted'
    assert row['target_ids']==[1000] and row['numeric_token_mask']==[True]
    assert row['closing_boundary_ids']==[1000]
    assert row['correct_note_text']=='24;'
    assert row['corrupt_note_text'].endswith(';') and row['corrupt_note_text']!='24;'
    assert row['equal_note_token_length'] is False  # No filtering to rescue length matching.


def test_corrupt_payload_changes_only_exact_numeral_and_keeps_leading_and_boundary_bytes():
    tok=Characters();r=record(tok,rhs=' 24; continuation.')
    row,status=probe.curate(r,selected(r,tok),tok,'source','sha')
    assert status['status']=='accepted'
    t=row['verified_numeric_rhs'];start,end=t['numeral_char_start'],t['numeral_char_end']
    wrong=row['corrupt_note_text'];correct=row['correct_note_text']
    assert wrong[:start]==correct[:start] and wrong[end:]==correct[end:]
    assert len(row['corrupt_note_ids'])==len(row['target_ids']) and row['equal_note_token_length']
    assert Fraction(row['corrupt_numeric_value'])!=Fraction(row['numeric_value'])
    assert row['corrupt_numeric_surface_text']==wrong[start:end]


def test_unsupported_source_is_distinct_from_complete_exact_numeric_mismatch():
    tok=Characters();r=record(tok,rhs='25; continuation.')
    row,status=probe.curate(r,selected(r,tok),tok,'source','sha')
    assert row is None and status['status']=='source_numeric_mismatch'
    r['future']=tok.encode('24              ')
    row,status=probe.curate(r,selected(r,tok),tok,'source','sha')
    assert row is None and status['status']=='unsupported_source_label'


def test_complete_target_never_truncated_to_fit16():
    tok=Characters();r=record(tok);choice=selected(r,tok)
    full=r['prefix']+tok.encode(' '*15+'24; further source')
    with pytest.raises(probe.safe.Unsupported,match='not_within16'):
        probe.source_target(choice,full,tok)


def test_first_selected_cut_is_unchanged_when_every_later_token_changes():
    tok=Characters();past='p'*180+' Context: 6*4='
    first=tok.encode(past);one=first+tok.encode('24; later prose. '*40)
    two=first+[999999]*300
    chosen,_=probe.choose_boundary(one[:512],'doc','item',tok)
    same,_=probe.choose_boundary(two[:512],'doc','item',tok)
    assert chosen==same and chosen['cut']==len(first)


def test_wrong_selected_rhs_does_not_fall_through_to_later_valid_equality():
    tok=Characters();text='p'*180+' Context: 6*4=25; Later context: 7*4=28; '
    prefix=tok.encode((text+'x'*512)[:512]);r=dict(id='item',document_id='doc',split='train',prefix=prefix,future=tok.encode('x'*16))
    choice=selected(r,tok);assert choice['expression']=='6*4'
    row,status=probe.curate(r,choice,tok,'source','sha')
    assert row is None and status['status']=='source_numeric_mismatch'


def test_test_arrays_are_never_json_parsed_and_pool_order_is_deterministic(tmp_path,monkeypatch):
    tok=Characters();train=record(tok,doc='train');dev=record(tok,split='dev',doc='dev')
    source=tmp_path/'source.jsonl'
    source.write_text(json.dumps(train)+'\n'+
        '{"id":"test-id","document_id":"test-doc","split":"test","prefix":NOT_JSON,"future":NOT_JSON}\n'+json.dumps(dev)+'\n')
    monkeypatch.setattr(probe,'SOURCE_SHA',sha256_file(source))
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained',lambda *a,**kw:tok)
    monkeypatch.setattr(probe,'TRAIN_ITEMS',1);monkeypatch.setattr(probe,'DEV_ITEMS',1)
    out=tmp_path/'output';m=probe.prepare(source,out,tmp_path/'tokenizer','2099-01-01T00:00:00Z')
    rows=[json.loads(s) for s in (out/'accepted.jsonl').read_text().splitlines()]
    assert len(rows)==m['documents']==2 and m['ready_for_requested_pools']
    assert set(r['split'] for r in rows)=={'train','dev'}
    assert all(r['source_file']==str(source.resolve()) for r in rows)
    assert probe.select_pools(rows)==probe.select_pools(rows[::-1])
    assert m['accepted_sha256']==sha256_file(out/'accepted.jsonl')
    assert m['selected_dev_item_ids']==m['phase1_dev_item_ids']
    assert not m['held_post_sft_dev_item_ids']  # No fallback to fabricate another reserve.


def test_quota_shortfall_is_explicit_and_never_shrinks_request():
    rows=[dict(item_id='one',document_id='doc-one',split='dev')]
    pools=probe.select_pools(rows)
    assert pools['train']['requested']==5120 and pools['train']['available']==0
    assert pools['dev']['requested']==512 and pools['dev']['available']==1
    assert len(pools['dev']['selected_item_ids'])==1


def test_expired_preparation_fails_before_any_source_or_tokenizer_read(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'sha256_file',lambda *a:pytest.fail('Source touched after deadline'))
    with pytest.raises(TimeoutError):probe.prepare(tmp_path/'missing',tmp_path/'out',tmp_path/'tok','2000-01-01T00:00:00Z')
    assert not (tmp_path/'out').exists()
