"""Exact arithmetic, past-only choice, lossless source cuts and no RHS fallback."""
from fractions import Fraction
import json
from pathlib import Path

import pytest

import train.calculation_boundaries as probe
from weekend_thinking.artifacts import canonical_hash, sha256_file


class Characters:
    def encode(self,text,*,add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]
    def decode(self,ids,**kw):return ''.join(chr(i) for i in ids)
    def get_vocab(self):return {chr(i):i for i in range(128)}


class Merged(Characters):
    def __init__(self,pieces):self.pieces=pieces
    def encode(self,text,*,add_special_tokens=False):
        values=[]
        while text:
            piece=next((s for s in sorted(self.pieces,key=len,reverse=True) if text.startswith(s)),None)
            if piece:values.append(self.pieces[piece]);text=text[len(piece):]
            else:values.append(ord(text[0]));text=text[1:]
        return values
    def decode(self,ids,**kw):
        inv={v:k for k,v in self.pieces.items()}
        return ''.join(inv[i] if i in inv else chr(i) for i in ids)


def record(tok=None,expr='2+3*4',rhs='14',doc='one',split='train'):
    tok=tok or Characters();text='Context: '+expr+'='
    prefix=[ord('a')]*(512-len(tok.encode(text)))+tok.encode(text)
    future=tok.encode(' '+rhs+'; natural continuation.')[:16]
    assert len(prefix)==512 and len(future)==16
    return dict(id='item:'+doc,document_id='doc:'+doc,split=split,prefix=prefix,future=future)


@pytest.mark.parametrize('text,value', [
    ('2+3*4',14),('-.5 + 1.25*2',2),('3*(1/2+1/4)',Fraction(9,4)),
    ('(2+3)^2',25),('2**-3+1',Fraction(9,8)),('\u22122+3\u00d74',10),
    ('6\u00f72+1',4),('(-2+3)*-4',-4),
])
def test_exact_decimals_fractions_signs_and_bounded_powers(text,value):
    p=probe.parse_numeric(text);assert p.value==value and p.binary_ops>=2


@pytest.mark.parametrize('text', [
    '__import__("os").system("x")','x+2*3','True+2*3','[1,2][0]+3','2//3+4',
    '1e2+3*4','2 1/2+3*4','1.2.3+4*5','2+3*','2/0+3','2**1000000000+3',
    '9**9**9','999999999999**8+1','1+'*70+'1',
])
def test_unsupported_or_unbounded_never_executes_untrusted_code(text):
    with pytest.raises(probe.Unsupported):probe.parse_numeric(text)


def test_first_simplifying_proper_subexpression_skips_fraction_identity():
    p=probe.parse_numeric('1/3*2/3')
    assert p.intermediate_expression=='1/3*2' and p.intermediate_value==Fraction(2,3)
    p=probe.parse_numeric('(1/2)*4*4')
    assert p.intermediate_expression=='(1/2)*4' and p.intermediate_value==2
    assert probe.parse_numeric('1/3+2/3').intermediate_value is None
    assert probe.parse_numeric('(2+3)*1').intermediate_value is None  # Would disclose whole5.
    assert probe.parse_numeric('1.0/3.0+2/3').intermediate_value is None


@pytest.mark.parametrize('past', [
    'foo2+3*4 =','x +2*3 =','x 2+3*4 =','sin(2)+3*4 =','sin (2)+3*4 =','log (2)+3*4 =',
    '\u03c0 2*3+4 =','\u03c02*3+4 =','\\frac{1}{2} 3*4+1 =',
    'values] 3*4+1 =','1e3+4*5 =','2% +3*4 =',
])
def test_maximal_lhs_does_not_accept_numeric_suffix_of_symbolic_expression(past):
    with pytest.raises(probe.Unsupported):probe.maximal_lhs(past)


def test_maximal_numeric_region_is_not_retried_as_shorter_valid_expression():
    expr=probe.maximal_lhs('Compute: 2 1/2+3*4 =')
    assert expr=='2 1/2+3*4'
    with pytest.raises(probe.Unsupported):probe.parse_numeric(expr)
    assert probe.maximal_lhs('Compute 2+3*4 =')=='2+3*4'
    assert probe.parse_numeric(probe.maximal_lhs('Compute: -2+3*4 =')).value==10


@pytest.mark.parametrize('past,value', [
    ('Earlier result14. Now2+3*4 =',14),('Earlier0.5. Now3*(1/2+1/4)=',Fraction(1,2)),
    ('Earlier1 / 2. Now calculation=',Fraction(1,2)),('Earlier\u22123. Now calculation=',-3),
    ('Earlier50%. Now calculation=',Fraction(1,2)),('Earlier1e2. Now calculation=',100),
    ('Earlier1,000. Now calculation=',1000),
])
def test_exact_visible_whole_values_and_equivalent_literal_spans_are_rejected(past,value):
    assert probe.whole_value_visible(past,Fraction(value))=='whole_value_already_visible'


def test_copy_check_never_accepts_numeric_prefix_and_marks_ambiguous_fraction_spans():
    assert probe.whole_value_visible('Earlier2020; now2+3*4=',20) is None
    assert probe.whole_value_visible('Earlier2 1/2; now calculation=',99)=='ambiguous_mixed_number_copy_span'
    assert probe.whole_value_visible('Earlier\\frac{1}{2}; now calculation=',99)=='ambiguous_fraction_copy_span'


@pytest.mark.parametrize('past,value',[
    ('Earlier Rs.132600; now calculation=',132600),
    ('Earlier Rs.38000; now calculation=',38000),
    ('Earlier Rs .132600; now calculation=',132600),
    ('Earlier Rs .1,32,600; now calculation=',132600),
    ('Earlier Re .38000; now calculation=',38000),
    ('Earlier INR .132600; now calculation=',132600),
    ('Option C.25.2 million; now calculation=',Fraction(126,5)),
    ('Earlier e.g.132600; now calculation=',132600),
    ('Earlier amount...132600; now calculation=',132600),
    ('Earlier17.02.2022; now calculation=',2022),
    ('Earlier .5.6; now calculation=',999),
])
def test_contextual_dot_cannot_hide_a_visible_integer_or_decimal(past,value):
    assert probe.whole_value_visible(past,Fraction(value))=='ambiguous_decimal_boundary_copy_span'


def test_unambiguous_leading_decimal_is_still_parsed_without_inventing_integer_copy():
    assert probe.whole_value_visible('Earlier .5; now calculation=',Fraction(1,2))=='whole_value_already_visible'
    assert probe.whole_value_visible('Earlier -.5; now calculation=',Fraction(-1,2))=='whole_value_already_visible'
    assert probe.whole_value_visible('Earlier .132600; now calculation=',132600) is None
    assert probe.whole_value_visible('Earlier Rs. 132600; now calculation=',132600)=='whole_value_already_visible'


@pytest.mark.parametrize('text,value',[
    ('1,000',1000),('1,234,567.5',Fraction(2469135,2)),
    ('1,32,600',132600),('12,34,567',1234567),('80,00,000',8000000),
    ('-1,32,600',-132600),('1,32,600/2',66300),('1,32,600%',1326),
    ('1,32,600e-2',1326),
    ('0,100',100),('1,000',1),  # Also retain possible list components.
])
def test_maximal_western_and_indian_groups_are_checked_as_whole_values(text,value):
    assert probe.whole_value_visible('Earlier '+text+'; now calculation=',Fraction(value))=='whole_value_already_visible'


@pytest.mark.parametrize('text',['4,5','1,1','1,4','1,2345','1,23,45','1234,567','1,2,3,4','1,234,56.7',
    '1,,234','1,234,,567','1,234.5,6'])
def test_malformed_or_list_like_comma_run_is_ambiguous_not_split_or_concatenated(text):
    assert probe.whole_value_visible('Earlier '+text+'; now calculation=',999)=='ambiguous_grouped_numeric_copy_span'


def test_bad_currency_boundary_is_past_ineligible_before_rhs_curation():
    tok=Characters();r=record(tok,expr='130000+2000+300*2',rhs='132600')
    text='Earlier Rs.132600; now: 130000+2000+300*2='
    r['prefix']=[ord('a')]*(512-len(text))+tok.encode(text)
    selected,audit=probe.choose_boundary(r['prefix'],r['document_id'],r['id'],tok)
    assert selected is None and audit[-1]['reason']=='ambiguous_decimal_boundary_copy_span'
    assert probe.SCHEMA.endswith('-v2') and probe.SALT=='finemath-arithmetic-boundaries-v1'


@pytest.mark.parametrize('text',['1e1000000000','9'*81])
def test_copy_check_bounds_unsafe_literals_before_fraction_construction(monkeypatch,text):
    monkeypatch.setattr(probe,'Fraction',lambda _:pytest.fail('Unsafe Fraction construction reached'))
    assert probe.whole_value_visible(text,1)=='ambiguous_or_unbounded_numeric_copy_span'


def test_original_token_cut_proper_note_corruption_and_full_source_control():
    tok=Characters();r=record(tok);selected,audit=probe.choose_boundary(r['prefix'],r['document_id'],r['id'],tok)
    assert selected and len([v for v in audit if v['status']=='eligible'])==1
    assert selected['cut']==512 and selected['prefix_text'].endswith('2+3*4=')
    assert selected['note']['correct_text']=='Compute 3*4 = 12.'
    assert selected['note']['corrupted_text']=='Compute 3*4 = 13.'
    row,status=probe.curate_selected(r,selected,tok,Path('/source'),'a'*64)
    assert status['status']=='accepted'
    assert row['ordinary_source_ids']==r['prefix']+r['future'] and len(row['ordinary_source_ids'])==528
    assert row['prefix_ids']==row['ordinary_source_ids'][:512]
    assert row['prefix_text']+row['natural_continuation_text']==row['ordinary_source_text']
    assert row['numeric_target_ids']==row['natural_future16_ids'][:3]==tok.encode(' 14')
    assert row['numeric_token_mask']==[False,True,True]
    assert row['numeric_token_mask']==row['verified_numeric_rhs']['numeric_token_mask']
    assert len(row['note_ids'])==len(row['corrupted_note_ids']) and row['note_ids']!=row['corrupted_note_ids']
    assert 'FULL ordinary source' in row['ordinary_control'] and 'not future16-only' in row['ordinary_control']


@pytest.mark.parametrize('tail',['14 \u00d7 2; rest','14 \u00f7 2; rest','14 \u2212 2; rest','14 (2+1); rest',
    '14e2; rest','14 1/2; rest','14,000; rest','14^2; rest','14 / 2 + 3; rest','14%; rest',
    '14 [2]; rest','14 times 2; rest','14x; rest','14 + 3; rest','14 \u03c0; rest','14\u221a2; rest','14\u2074; rest'])
def test_numeric_prefix_of_larger_or_unsupported_rhs_is_not_a_valid_label(tail):
    tok=Characters();selected=dict(cut=1,prefix_text='=')
    with pytest.raises(probe.Unsupported):probe.source_rhs(selected,tok.encode('='+tail),tok)


@pytest.mark.parametrize('literal,value',[('-.5',Fraction(-1,2)),('1.5 / +.5',3),('\u22121/2',Fraction(-1,2))])
def test_complete_source_rhs_exact_fraction_not_binary_float(literal,value):
    tok=Characters();rhs=probe.source_rhs(dict(cut=1,prefix_text='='),tok.encode('= '+literal+'; rest.'),tok)
    assert Fraction(rhs['value'])==value
    assert rhs['token_ids']==tok.encode(' '+literal)


def test_rhs_needs_complete_lookahead_and_original_token_alignment():
    tok=Characters();selected=dict(cut=1,prefix_text='=')
    with pytest.raises(probe.Unsupported,match='lookahead'):probe.source_rhs(selected,tok.encode('= 14'),tok)
    with pytest.raises(probe.Unsupported):probe.source_rhs(selected,tok.encode('= 14.'),tok)
    assert probe.source_rhs(selected,tok.encode('= 14. Sentence'),tok)['value']=='14'
    merged=Merged({'14;':1000})
    with pytest.raises(probe.Unsupported,match='aligned'):probe.source_rhs(selected,merged.encode('= 14; rest'),merged)


def test_merged_equals_rhs_token_is_never_a_causal_cut():
    tok=Merged({'=1':1000});r=record(tok,expr='2+3*4',rhs='14')
    # Put the RHS inside the source prefix: the '=1' token contains future bytes.
    text='Context: 2+3*4=14; rest'
    prefix=[ord('a')]*(512-len(tok.encode(text)))+tok.encode(text)
    selected,_=probe.choose_boundary(prefix,r['document_id'],r['id'],tok)
    assert selected is None


def test_selected_rhs_failure_never_switches_to_another_boundary():
    tok=Characters();text='Context: 2+3*4= 99; Calculation: 5+6*7= 47; end.'
    prefix=[ord('a')]*(512-len(text))+tok.encode(text);r=record(tok);r['prefix']=prefix
    selected,audit=probe.choose_boundary(prefix,r['document_id'],r['id'],tok)
    assert len([v for v in audit if v['status']=='eligible'])==1 and selected['expression']=='2+3*4'
    row,status=probe.curate_selected(r,selected,tok,Path('/source'),'a'*64)
    assert row is None and status['status']=='source_numeric_mismatch'


def test_search_start_and_first_boundary_are_unchanged_when_entire_suffix_changes():
    tok=Characters();r=record(tok)
    text='Context: 2+3*4= 14; Calculation: 5+6*7= 47; end.'
    prefix=[ord('a')]*(512-len(text))+tok.encode(text)
    selected,audit=probe.choose_boundary(prefix,r['document_id'],r['id'],tok)
    start,h=probe.search_start(r['document_id'],r['id'])
    assert 128<=start<=192 and start==128+int(h,16)%65 and selected['search_start']==start
    # If any changed suffix token is decoded, this fixture fails. The chosen
    # past is identical even though the old selected RHS/later cuts disappear.
    class NoSuffix(Characters):
        def decode(self,ids,**kw):
            assert 999999 not in ids,'Selected future was decoded during eligibility'
            return super().decode(ids,**kw)
    changed=prefix[:selected['cut']]+[999999]*(512-selected['cut'])
    other,other_audit=probe.choose_boundary(changed,r['document_id'],r['id'],NoSuffix())
    assert other==selected and other_audit==audit


def test_unsupported_source_is_distinct_from_a_complete_false_numeric_equality():
    tok=Characters();r=record(tok);selected,_=probe.choose_boundary(r['prefix'],r['document_id'],r['id'],tok)
    r['future']=tok.encode(' 15; remainder!!')[:16]
    assert probe.curate_selected(r,selected,tok,'/source','x')[1]['status']=='source_numeric_mismatch'
    r['future']=tok.encode(' 14 \u00d7 2; remainder')[:16]
    assert probe.curate_selected(r,selected,tok,'/source','x')[1]['status']=='unsupported_source_label'


def test_prefix_stage_never_parses_test_or_unselected_future_arrays(tmp_path):
    source=tmp_path/'source.jsonl'
    source.write_text('{"id":"test","document_id":"test","split":"test","prefix":INVALID,"future":INVALID}\n'+
        json.dumps(dict(id='train',document_id='train',split='train',prefix=[97]*512))[:-1]+',"future":INVALID}\n')
    values=list(probe.prefix_records(source))
    assert len(values)==1 and values[0][0]['item_id']=='train' and values[0][0]['prefix']==[97]*512


def test_prepare_fixture_selection_is_disjoint_hash_ordered_and_future_opening_follows_choice(tmp_path,monkeypatch):
    import transformers
    tok=Characters();values=[record(tok,doc='train'+str(i),split='train') for i in range(3)]+[record(tok,doc='dev',split='dev')]
    source=tmp_path/'source.jsonl';source.write_text(''.join(json.dumps(v)+'\n' for v in values))
    monkeypatch.setattr(probe,'SOURCE_SHA',sha256_file(source));monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:tok)
    chosen=[];original_choose=probe.choose_boundary;original_loads=json.loads
    def choose(*a,**kw):
        result=original_choose(*a,**kw);chosen.append(a[2]);return result
    def loads(text,*a,**kw):
        if '"future"' in text:
            v=original_loads(text);assert v['id'] in chosen
        return original_loads(text,*a,**kw)
    monkeypatch.setattr(probe,'choose_boundary',choose);monkeypatch.setattr(probe.json,'loads',loads)
    monkeypatch.chdir(tmp_path)
    out=tmp_path/'prepared';m=probe.prepare(Path('source.jsonl'),out,tmp_path/'tokenizer-not-loaded')
    assert m['counts']=={'accepted':4} and m['ready_for_feasibility'] is False
    accepted=[original_loads(s) for s in (out/'accepted.jsonl').read_text().splitlines()]
    assert all(r['source_file']==m['source_file']==str(source.resolve()) for r in accepted)
    for split in ('train','dev'):
        expected=sorted((r for r in accepted if r['split']==split),key=lambda r:canonical_hash(dict(salt=probe.SALT,split=split,document_id=r['document_id'],item_id=r['item_id'])))
        assert m['selection'][split]['selected_item_ids']==[r['item_id'] for r in expected]
    assert not(set(m['selection']['train']['selected_item_ids']) & set(m['selection']['dev']['selected_item_ids']))
    assert m['accepted_sha256']==sha256_file(out/'accepted.jsonl')


def test_duplicate_document_across_train_dev_fails_instead_of_leaking(tmp_path,monkeypatch):
    import transformers
    tok=Characters();values=[record(tok,doc='same',split=s) for s in ('train','dev')]
    source=tmp_path/'source.jsonl';source.write_text(''.join(json.dumps(v)+'\n' for v in values))
    monkeypatch.setattr(probe,'SOURCE_SHA',sha256_file(source));monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:tok)
    with pytest.raises(ValueError,match='disjointness'):probe.prepare(source,tmp_path/'out',tmp_path/'fake-tokenizer')


@pytest.mark.parametrize('value,text',[
    (Fraction(1,2),'0.5'),(Fraction(-1,8),'-0.125'),(Fraction(16,5),'3.2'),
    (Fraction(32148009,62500),'514.368144'),(Fraction(1,3),'1/3'),(Fraction(-2,7),'-2/7'),
    (Fraction(1,2**24),'0.000000059604644775390625'),(Fraction(1,2**25),'1/33554432'),
    (Fraction(1,5**24),'0.000000000000000016777216'),(Fraction(1000), '1000'),
])
def test_exact_decimal_display_is_finite_bounded_and_never_rounded(value,text):
    assert probe.render_value(value)==text and Fraction(text)==value


def test_decimal_display_and_equal_length_corruption_share_renderer_without_changing_identity_filter():
    parsed=probe.parse_numeric('(5-1.8)*3');pair=probe.note_pair(parsed,Characters())
    assert pair['correct_text']=='Compute 5-1.8 = 3.2.'
    assert Fraction(pair['correct_value_fraction'])==Fraction(pair['correct_value'])==Fraction(16,5)
    assert len(pair['correct_ids'])==len(pair['corrupted_ids'])
    assert probe.parse_numeric('1/3+2/3').intermediate_value is None


def test_deadline_keeps_partial_rows_and_audit_without_a_ready_manifest(tmp_path,monkeypatch):
    import transformers
    tok=Characters();values=[record(tok,doc=str(i)) for i in range(2)]
    source=tmp_path/'source.jsonl';source.write_text(''.join(json.dumps(v)+'\n' for v in values))
    monkeypatch.setattr(probe,'SOURCE_SHA',sha256_file(source));monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:tok)
    calls=[]
    def deadline(value):
        calls.append(value)
        if len(calls)==5:raise TimeoutError('Prospective fixture deadline')
    monkeypatch.setattr(probe,'check_deadline',deadline)
    out=tmp_path/'partial'
    with pytest.raises(TimeoutError):probe.prepare(source,out,tmp_path/'fake-tokenizer','2099-01-01T00:00:00+00:00')
    assert not(out/'manifest.json').exists()
    failure=json.loads((out/'preparation.failed.json').read_text())
    assert failure['status']=='partial_not_ready' and failure['counts']=={'accepted':1}
    assert len((out/'accepted.jsonl').read_text().splitlines())==len((out/'audit.jsonl').read_text().splitlines())==1
    assert failure['partial_hashes']=={name:sha256_file(out/name) for name in ('accepted.jsonl','audit.jsonl')}
