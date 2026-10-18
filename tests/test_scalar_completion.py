"""Pure prospective syntax tests, with no old probe outcomes/gold or model."""
from fractions import Fraction

import pytest

from train.scalar_completion import MAX_TEXT, PARSER_VERSION, parse_initial_scalar_v2 as parse

EOS='token:151643'
CAP='length_cap'


@pytest.mark.parametrize('text,value',[
    ('12','12'),(' +0012','12'),('-12','-12'),('.5','1/2'),('-.5','-1/2'),
    ('+0.50','1/2'),('8.3/2','83/20'),('4.15','83/20'),(' 2 / -4','-1/2'),
    ('-0.50/+.25','-2'),('12 /3','4'),('1.','1'),('1/2.','1/2'),('1. \t','1'),('12;','12'),
    ('120 sties.','120'),('384. For composite numbers','384'),('1, next prose','1'),('1, I continue','1')])
def test_eos_completes_one_maximal_exact_rational(text,value):
    result=parse(text,EOS)
    assert result['parser_version']==PARSER_VERSION and result['status']=='valid'
    assert Fraction(result['value'])==Fraction(value)


@pytest.mark.parametrize('text',['12','12 ','12\t','12.','12. ','12;','12/3','+.5','+','-.',''])
def test_cap_needs_a_real_local_completion(text):
    assert parse(text,CAP)['status']=='incomplete'


@pytest.mark.parametrize('suffix',['I will explain.','Answer: this follows.','Next we continue.','x is a variable.'])
@pytest.mark.parametrize('boundary',['\n','\n\n','\r\n','. ','.\n','; '])
def test_terminated_numeric_line_or_sentence_ignores_later_prose_word_choice(boundary,suffix):
    assert parse('4.15'+boundary+suffix,CAP)['value']=='83/20'


@pytest.mark.parametrize('text',['12\n','12 \t\n','12\r\n','12.\n','12;\n','12/3\n'])
def test_line_delimiter_can_precede_cap_without_later_prose(text):
    assert parse(text,CAP)['status']=='valid'


@pytest.mark.parametrize('text',[
    '12=6*2','12 = 6/0','12\n=what follows is not evaluated','12 = ',
    '12 = __import__("os").system("never executed")'])
def test_equality_certifies_only_the_initial_emitted_value(text):
    result=parse(text,CAP)
    assert result['status']=='valid' and result['value']=='12' and result['completion']=='equality'


@pytest.mark.parametrize('tail',[
    '+3','-3','−3','*3','/3/2','^3','×3','÷3','%','‰','²','1/2','.5','e2','E-2','e','e+','E-','e²','e₂',
    'plus 3','minus 3','times 3','modulo 3','squared','cubed','percent','factorial',
    'and 1/2','or 13','to 15','x 3','x3','x.5','X+3','x²','x₂','\\times 3','* Explanation','< 13',': 3'])
@pytest.mark.parametrize('boundary',[' ','\n','\n\n','. '])
def test_explicit_arithmetic_or_numeric_continuations_are_rejected_even_after_line(tail,boundary):
    assert parse('12'+boundary+tail,EOS)['status']=='unsupported'


@pytest.mark.parametrize('boundary',['\n','\n\n','. '])
def test_slash_fraction_cannot_wrap_across_a_completed_line_or_sentence(boundary):
    assert parse('12'+boundary+'/3',EOS)['status']=='unsupported'


@pytest.mark.parametrize('text',[
    '1+2','1-2','1/2/3','1e2','1E-2','1abc','1_abc','1.2.3','1.. prose','1... prose',
    '1,000','1, 000','1,','1, ','1 000','1 1/2','1/0','1 /','1/\n2','1 : 2',
    '1!','1?','1*','1**','**1**','*1*','1.Next','1 . 2','1;2','1==2','1=>2','1 = =2',
    '１','−1','½','(1)','The answer is 1','\\frac{1}{2}','0x12'])
def test_ambiguous_grouping_markdown_expression_and_noninitial_formats_never_pass(text):
    assert parse(text,EOS)['status']!='valid'
    assert parse(text,CAP)['status']!='valid'


@pytest.mark.parametrize('text',['1\nI continue','1\nAnswer: continue','1. I continue','1. Answer: continue'])
def test_eos_after_prose_does_not_change_an_already_certified_local_value(text):
    assert parse(text,CAP)['value']==parse(text,EOS)['value']=='1'


@pytest.mark.parametrize('word',['hundred','thousands','million','billions','trillion','dozen','lakhs','crore','bn'])
@pytest.mark.parametrize('boundary',[' ','\n',', '])
def test_uninterpreted_scale_words_cannot_be_mislabeled_as_a_wrong_unscaled_value(word,boundary):
    assert parse('4.749574'+boundary+word,EOS)['status']=='unsupported'


@pytest.mark.parametrize('tail',[' i',' x',' e',' E-2',' e+3',' + 2',' squared'])
def test_inline_word_certificate_does_not_accept_variable_or_arithmetic_ambiguity(tail):
    assert parse('120'+tail,EOS)['status']=='unsupported'


def test_bounds_no_search_and_unknown_runtime_stop():
    for text in ('1'*65,'1'+' '*128+'/2','1'+'x'*MAX_TEXT):
        assert parse(text,EOS)['status']=='unsupported'
    assert parse('Not a number\n12',EOS)['status']=='format_error'
    assert parse('12','error')['status']=='error'
    assert parse(None,EOS)['status']=='unsupported'
    assert parse('12','eos')['value']=='12'
