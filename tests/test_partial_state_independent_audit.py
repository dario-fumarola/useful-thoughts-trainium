from decimal import Decimal
import pytest

from scripts.audit_partial_state_panels import exact_tree, independent_score, parse_independent


@pytest.mark.parametrize('text,value', [('1925\n','1925'),('Answer:\n−1,925.5\n','-1925.5'),('  .25e+2.\n','25'),('+0\n','0')])
def test_independent_audit_accepts_the_declared_complete_number_forms(text,value):
    assert parse_independent(text)==Decimal(value)


@pytest.mark.parametrize('text',['12,34\n','3.1.4\n','4','Not sure\nAnswer:\n4\n','NaN\n'])
def test_independent_audit_rejects_invalid_or_incomplete_first_answers(text):
    assert parse_independent(text) is None


def test_independent_CoT_parser_never_rescues_later_answer_and_keeps_caps_failed():
    assert parse_independent('37 + 18 = 55\nAnswer:\n1925\n',cot=True)==Decimal(1925)
    assert parse_independent('Answer:\nunknown\nAnswer:\n1925\n',cot=True) is None
    assert parse_independent('4',eos=True)==Decimal(4)
    row=dict(mode='direct',status='ok',stop_type='length_cap',text='4\n')
    assert independent_score(row,4)==dict(score=0,status='truncated')
    assert exact_tree(('*',('+',37,18),('-',92,57)))==1925
