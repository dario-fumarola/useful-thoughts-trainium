import pytest

from train.expression_identity import expression_key


@pytest.mark.parametrize('a,b',[
    ('1 + (2*3)','1+(2 * 3)'),('1.0+2','1+2.00'),
    ('-1+2','−1+2'),('1*2+3','1×2+3'),('2^3+1','2**3+1'),
    ('(+1)+2','1+2')])
def test_formatting_and_exact_constant_aliases_do_not_create_false_novelty(a,b):
    assert expression_key(a)==expression_key(b)


@pytest.mark.parametrize('a,b',[
    ('(1+2)*3','1+2*3'),('2+3','3+2'),('7-1','2*3'),
    ('0.1+2','0.10000000000000001+2')])
def test_structure_order_and_exact_rational_constants_remain_distinct(a,b):
    assert expression_key(a)!=expression_key(b)


def test_unbounded_or_nonarithmetic_source_is_rejected():
    with pytest.raises(ValueError):expression_key('f(1)+2')
