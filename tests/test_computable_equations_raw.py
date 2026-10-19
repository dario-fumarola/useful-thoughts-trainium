"""Bounded raw-source identity, native UTF8 cuts, split and numeric safety."""
import json
from pathlib import Path
import unicodedata

import pytest

from train import computable_equations_raw as probe


class Characters:
    is_fast=True
    truncation_side='right'
    def __init__(self):self.calls=[]
    def encode(self,text,*,add_special_tokens=False):return [ord(c) for c in text]
    def decode(self,ids,**kw):return ''.join(chr(i) for i in ids)
    def get_vocab(self):return {chr(i):i for i in range(128)}
    def __call__(self,text,**kw):
        self.calls.append((text,kw));ids=self.encode(text)[:kw['max_length']]
        return dict(input_ids=ids,offset_mapping=[(i,i+1) for i in range(len(ids))])


def fixture(body=' Context: 6*4=24; natural continuation. ',split='train',start=140):
    text='p'*start+body+' z'*600
    raw=dict(text=text,url='https://example.org/source')
    meta=dict(source_line=7,document_id='doc-one',split=split,url=raw['url'],
        content_sha256=probe.text_digest(text),record_id='old-one',token_count=len(text))
    return meta,json.dumps(raw,ensure_ascii=False).encode(),text


def test_native_raw_chunk_prefix_target_and_absolute_offsets_are_exact():
    meta,line,text=fixture();tok=Characters();row,audit,counts=probe.curate_document(meta,line,tok,'raw.jsonl')
    assert row is not None and counts['accepted']==1 and audit['accepted_candidate_count']==1
    a,b=row['source_chunk_char_start'],row['source_chunk_char_end'];end=row['prefix_source_char_end']
    assert row['original_source_text']==text[a:b] and row['prefix_text']==text[a:end]
    assert row['absolute_source_cut']==row['source_chunk_token_start']+len(row['prefix_ids'])
    assert len(row['original_source_ids'])==528 and row['correct_note_ids']==row['target_ids']
    assert row['source_sha256']==probe.RAW_SHA and row['raw_source_line']==7
    assert row['raw_rhs_validation']['status']=='valid' and row['raw_rhs_validation']['value']=='24'
    assert row['audited_original_record_id']=='old-one' and row['source_url']==meta['url']


def test_later_raw_window_uses_exact_last512_past_and_original16_future():
    meta,line,text=fixture(start=1000);tok=Characters();row,audit,_=probe.curate_document(meta,line,tok,'raw')
    assert row is not None and len(row['prefix_ids'])==512
    assert row['source_chunk_token_start']==row['absolute_source_cut']-512
    assert row['natural_future16_ids']==tok.encode(text)[row['absolute_source_cut']:row['absolute_source_cut']+16]
    assert row['source_chunk_token_end']-row['source_chunk_token_start']==528


def test_full_answer_copy_and_numeric_only_corruption_do_not_change_source():
    meta,line,text=fixture();row,_,_=probe.curate_document(meta,line,Characters(),'raw')
    assert row['correct_note_text']=='24;' and row['numeric_surface_text']=='24'
    assert row['corrupt_note_text'][-1]==';' and row['corrupt_numeric_value']!='24'
    assert row['original_source_text']==text[:528]
    assert 'NOT a proper intermediate' in row['supervision']


def test_raw_unicode_is_preserved_without_operator_normalization():
    meta,line,text=fixture(' Context: 6×4 = 24; natural continuation. ')
    row,_,_=probe.curate_document(meta,line,Characters(),'raw')
    assert row is not None and row['reanchor_text']=='6×4 ='
    assert '×' in row['prefix_text'] and row['original_source_text']==text[:528]


def test_tokenizer_normalization_is_rejected_instead_of_rewriting_raw_bytes():
    class Normalizes(Characters):
        def encode(self,text,*,add_special_tokens=False):return super().encode(unicodedata.normalize('NFC',text))
    meta,line,_=fixture(' café e\u0301 Context: 6*4=24; natural continuation. ')
    row,audit,_=probe.curate_document(meta,line,Normalizes(),'raw')
    assert row is None
    assert any(x.get('reason')=='raw_UTF8_native_window_not_lossless' for x in audit['candidate_audit'])


def test_test_metadata_fails_before_any_raw_json_decode():
    with pytest.raises(ValueError,match='Never decode raw TEST'):
        probe.curate_document(dict(split='test'),b'not JSON',Characters(),'raw')


@pytest.mark.parametrize('change',[dict(url='wrong'),dict(content_sha256='wrong')])
def test_raw_line_identity_mismatch_fails_without_skipping_document(change):
    meta,line,_=fixture();meta.update(change)
    with pytest.raises(ValueError,match='audited membership'):
        probe.curate_document(meta,line,Characters(),'raw')


def test_unsupported_first_source_equation_does_not_hide_later_supervised_valid_one():
    meta,line,_=fixture(' Context: 6*4=25; Later context: 7*4=28; ')
    row,audit,counts=probe.curate_document(meta,line,Characters(),'raw')
    assert row is not None and row['numeric_value']=='28'
    assert counts==dict(source_numeric_mismatch=1,accepted=1)
    assert 'SUPERVISED bounded raw' in row['boundary']['selection_policy']


def test_all_source_valid_candidates_are_counted_but_one_fixed_hash_is_selected():
    meta,line,_=fixture(' Context: 6*4=24; Later context: 7*4=28; ')
    tok=Characters();row,audit,counts=probe.curate_document(meta,line,tok,'raw')
    valid=[e for e in audit['candidate_audit'] if e['status']=='accepted']
    assert counts['accepted']==2 and len(valid)==audit['accepted_candidate_count']==2
    assert row['supervised_choice_hash']==min(e['choice_hash'] for e in valid)
    assert row==probe.curate_document(meta,line,tok,'raw')[0]


def test_full_raw_identity_hashes_are_computed_once_per_document(monkeypatch):
    meta,line,text=fixture(' Context: 6*4=24; Later context: 7*4=28; ')
    original=probe.hashlib.sha256;calls={'line':0,'text':0};text_bytes=text.encode()
    def track(data=b''):
        if data==line:calls['line']+=1
        if data==text_bytes:calls['text']+=1
        return original(data)
    monkeypatch.setattr(probe.hashlib,'sha256',track)
    row,_,counts=probe.curate_document(meta,line,Characters(),'raw')
    assert row is not None and counts['accepted']==2
    assert calls==dict(line=1,text=1)


def test_character_and_native_token_limits_are_applied_before_search(monkeypatch):
    monkeypatch.setattr(probe,'MAX_CHARACTERS',600);monkeypatch.setattr(probe,'MAX_TOKENS',550)
    meta,line,_=fixture(start=700);tok=Characters();row,audit,_=probe.curate_document(meta,line,tok,'raw')
    assert row is None and len(tok.calls[0][0])==600 and tok.calls[0][1]['max_length']==550
    assert audit['tokenized_characters']==600 and audit['bounded_native_tokens']==550


def test_source_window_requires_real528_tokens_and_causal_offsets():
    tok=Characters();ids=tok.encode('x'*527);offsets=[(i,i+1) for i in range(527)]
    with pytest.raises(probe.base.safe.Unsupported,match='incomplete'):
        probe.source_window(ids,offsets,'x'*527,128,tok)
    ids=tok.encode('x'*528);offsets=[(i,i+1) for i in range(528)];offsets[-1]=(527,999)
    with pytest.raises(probe.base.safe.Unsupported,match='offsets'):
        probe.source_window(ids,offsets,'x'*528,128,tok)


def test_binary_raw_iterator_never_decodes_unlisted_or_test_lines(tmp_path):
    path=tmp_path/'raw.jsonl';meta,line,_=fixture();meta['source_line']=2
    path.write_bytes(b'\xffINVALID_TEST_BYTES\n'+line+b'\n'+b'\xffUNLISTED_BYTES\n')
    tasks=list(probe.raw_tasks(path,{2:meta}))
    assert tasks==[(meta,line+b'\n',str(path.resolve()))]


def test_membership_keeps_original_split_and_rejects_duplicates(tmp_path):
    path=tmp_path/'membership';path.write_text('\n'.join(json.dumps(r) for r in [
        dict(source_line=1,document_id='a',split='train'),dict(source_line=2,document_id='b',split='dev'),
        dict(source_line=3,document_id='c',split='test')])+'\n')
    assert set(probe.membership(path))=={1,2}
    with path.open('a') as f:f.write(json.dumps(dict(source_line=4,document_id='a',split='dev'))+'\n')
    with pytest.raises(ValueError,match='Duplicate'):probe.membership(path)


def test_candidate_pool_order_is_new_fixed_hash_and_does_not_replace_prior_training():
    rows=[dict(item_id='i'+str(i),document_id='d'+str(i),split='dev') for i in range(5)]
    pool=probe.select_pools(rows)
    assert pool==probe.select_pools(rows[::-1])
    assert pool['dev']['requested']==512 and pool['dev']['available']==5
    assert pool['train']['requested']==5120 and pool['train']['available']==0
    assert probe.SALT!=probe.supervised.SALT


def test_expired_raw_pass_reads_no_source_or_membership(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'sha256_file',lambda *a:pytest.fail('Expired source read'))
    with pytest.raises(TimeoutError):probe.prepare(tmp_path/'raw',tmp_path/'members',tmp_path/'out',tmp_path/'tok','2000-01-01T00:00:00Z')
    assert not (tmp_path/'out').exists()


def test_compound_minimum_rejects_one_operation_before_rhs_validation(monkeypatch):
    meta,line,_=fixture()
    monkeypatch.setattr(probe.base,'curate',lambda *a:pytest.fail('One-op RHS must not be inspected'))
    row,audit,_=probe.curate_document(meta,line,Characters(),'raw',minimum_binary_ops=2)
    assert row is None
    assert any(v.get('reason')=='below_declared_prefix_operation_minimum' for v in audit['candidate_audit'])


def test_compound_minimum_keeps_exact_later_computation_and_raw_source():
    meta,line,text=fixture(' Context: 6*4=24; Later context: (6*4)+1=25; ')
    row,_,counts=probe.curate_document(meta,line,Characters(),'raw',minimum_binary_ops=2)
    assert row['boundary']['binary_ops']==2 and row['numeric_value']=='25'
    assert counts['accepted']==1
    assert row['original_source_text']==text[:528] and row['raw_rhs_validation']['value']=='25'


def test_compound_minimum_preserves_rejection_of_wrong_source_answer():
    meta,line,_=fixture(' Context: (6*4)+1=26; ')
    row,_,counts=probe.curate_document(meta,line,Characters(),'raw',minimum_binary_ops=2)
    assert row is None and counts['source_numeric_mismatch']==1


@pytest.mark.parametrize('minimum',[0,7,True,1.5])
def test_invalid_operation_minimum_rejected_before_raw_decode(minimum):
    with pytest.raises(ValueError,match='minimum binary operation'):
        probe.curate_document(dict(split='train'),b'not JSON',Characters(),'raw',minimum_binary_ops=minimum)
