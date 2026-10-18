"""Exact arithmetic traces, precedence and deliberate future-aware labels."""
from fractions import Fraction

import pytest

from train import symbolic_note_labels as labels


def test_postorder_substitutes_computed_children_and_respects_precedence():
    value, steps = labels.postorder_steps('12 * 24 + 7 * 3')
    assert value == 309
    assert steps == ['12 * 24 = 288', '7 * 3 = 21', '288 + 21 = 309']


def test_nonterminating_intermediates_remain_exact():
    value, steps = labels.postorder_steps('(1/3)+(2/3)')
    assert value == 1
    assert steps == ['1 / 3 = 1/3', '2 / 3 = 2/3', '(1/3) + (2/3) = 1']
    assert '0.333' not in str(steps)


def test_decimal_literals_are_not_binary_floats_and_unicode_is_validated():
    value, steps = labels.postorder_steps('0.1 × 0.2 + .03')
    assert value == Fraction(1, 20)
    assert steps == ['0.1 * 0.2 = 0.02', '0.02 + 0.03 = 0.05']


@pytest.mark.parametrize('expression,value,steps', [
    ('-(2+3)*4', -20, ['2 + 3 = 5', '(-5) * 4 = -20']),
    ('2^3^2', 512, ['3 ^ 2 = 9', '2 ^ 9 = 512']),
    ('2^(-3)+1', Fraction(9, 8), ['2 ^ (-3) = 0.125', '0.125 + 1 = 1.125']),
])
def test_unary_and_power_semantics(expression, value, steps):
    if expression == '2^3^2':
        # Its outer exponent9 exceeds the existing safety bound8.
        with pytest.raises(ValueError, match='power_bound'):
            labels.postorder_steps(expression)
    else:
        assert labels.postorder_steps(expression) == (value, steps)


def test_source_answer_surface_is_preserved_after_verified_steps():
    row = dict(boundary=dict(expression='(2/3)+(2/3)'), numeric_value='4/3',
               numeric_surface_text='4 / 3', teacher_note_kind=labels.POSTORDER)
    note = labels.teacher_note(row)
    assert note == 'Compute: 2 / 3 = 2/3. 2 / 3 = 2/3. (2/3) + (2/3) = 4/3. Result: 4 / 3.'
    row['numeric_surface_text'] = '5 / 3'
    with pytest.raises(ValueError, match='disagree'):
        labels.teacher_note(row)


@pytest.mark.parametrize('expression', ['__import__("os")', '1/0', '2**9', 'sqrt(4)', '1e3+2'])
def test_unsafe_or_unsupported_expressions_are_rejected_before_evaluation(expression):
    with pytest.raises(ValueError):
        labels.postorder_steps(expression)
