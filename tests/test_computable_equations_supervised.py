"""Future-aware equation selection stays explicit and document-disjoint."""
import json

import pytest

from train import computable_equations_supervised as probe


class Characters:
    def encode(self,text,*,add_special_tokens=False):return [ord(c) for c in text]
    def decode(self,ids,**kw):return ''.join(chr(i) for i in ids)
    def get_vocab(self):return {chr(i):i for i in range(128)}


def record(text,doc='one',split='train'):
    tok=Characters();prefix=tok.encode(('p'*140+' Context: '+text+' z'*512)[:512])
    row=dict(id='item:'+doc,document_id='doc:'+doc,split=split,prefix=prefix,future=tok.encode('z'*16))
    header=dict(item_id=row['id'],document_id=row['document_id'],split=split,prefix=prefix)
    return header,json.dumps(row)


def test_later_valid_source_equation_is_supervised_curation_not_stopping_time_fallback():
    header,line=record('6*4=25; Later context: 7*4=28; ')
    row,audit,counts=probe.curate_document(header,line,Characters(),'source')
    assert row is not None and row['boundary']['expression']=='7*4'
    assert row['numeric_value']=='28' and row['correct_note_text']=='28;'
    assert counts==dict(source_numeric_mismatch=1,accepted=1)
    assert audit['accepted_candidate_count']==1 and audit['status']=='accepted'
    assert 'SUPERVISED' in row['boundary']['selection_policy']
    assert 'supervised-equation-v2' in row['item_id']


def test_all_accepted_equations_are_audited_then_one_fixed_hash_wins(monkeypatch):
    header,line=record('6*4=24; Later context: 7*4=28; ')
    tok=Characters();seen=[];original=probe.base.curate
    def track(record,selected,*args):
        seen.append(selected['expression']);return original(record,selected,*args)
    monkeypatch.setattr(probe.base,'curate',track)
    row,audit,counts=probe.curate_document(header,line,tok,'source')
    assert seen==['6*4','7*4'] and counts['accepted']==2
    valid=[e for e in audit['candidate_audit'] if e['status']=='accepted']
    assert len(valid)==2 and row['supervised_choice_hash']==min(e['choice_hash'] for e in valid)
    assert audit['accepted_candidate_count']==2
    again=probe.curate_document(header,line,tok,'source')[0]
    assert row==again


def test_source_validation_future_changes_may_change_supervised_membership():
    one,line=record('6*4=24; Later context: 7*4=28; ')
    _,audit,_=probe.curate_document(one,line,Characters(),'source')
    two,line=record('6*4=25; Later context: 7*4=28; ')
    row,new_audit,_=probe.curate_document(two,line,Characters(),'source')
    assert audit['accepted_candidate_count']==2 and new_audit['accepted_candidate_count']==1
    assert row['boundary']['expression']=='7*4'  # Explicitly future-aware, never called an online anchor rule.


@pytest.mark.parametrize('source', ['6*4=24 × 2; ', '6*4=24 million people; ', '6*4=25; '])
def test_no_valid_equation_is_retained_with_complete_rejection_counts(source):
    header,line=record(source);row,audit,counts=probe.curate_document(header,line,Characters(),'source')
    assert row is None and audit['status']=='no_valid_source_equation'
    assert sum(counts.values())==1 and audit['accepted_candidate_count']==0


def test_test_header_rejected_before_line_decode():
    with pytest.raises(ValueError,match='Never parse TEST'):
        probe.curate_document(dict(split='test'),'not valid JSON',Characters(),'source')


def test_known_currency_copy_still_rejected_before_rhs():
    header,line=record('Earlier Rs.24; Context: 6*4=24; ')
    row,audit,counts=probe.curate_document(header,line,Characters(),'source')
    assert row is None and not counts
    assert audit['candidate_audit'][-1]['reason']=='ambiguous_decimal_boundary_copy_span'


def test_exact_source_lhs_and_payload_bytes_are_not_normalized():
    header,line=record('6×4 = 24; ')
    row,audit,_=probe.curate_document(header,line,Characters(),'source')
    assert row['reanchor_text']=='6×4 =' and row['correct_note_text']==' 24;'
    assert row['correct_note_ids']==row['target_ids']
    assert row['corrupt_note_text'][0]==' ' and row['corrupt_note_text'][-1]==';'
    assert row['original_source_ids']==json.loads(line)['prefix']+json.loads(line)['future']


def test_pool_order_is_fixed_and_quotas_are_not_shrunk():
    rows=[dict(item_id='i'+str(i),document_id='d'+str(i),split='dev') for i in range(3)]
    pool=probe.select_pools(rows)
    assert pool==probe.select_pools(rows[::-1])
    assert pool['dev']['available']==3 and pool['dev']['requested']==512
    assert pool['train']['requested']==5120 and pool['train']['available']==0
    assert probe.SCHEMA!=probe.base.SCHEMA and probe.SALT!=probe.base.SALT
