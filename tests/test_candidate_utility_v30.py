from fractions import Fraction

from train.candidate_utility_v30 import correctness, equation_trace, item_seed


TEACHER = '2 * 3 = 6\n6 + 4 = 10\n'


def test_canonical_requires_complete_ordered_computation():
    assert correctness('(2 * 3)=6.0\n(6+4)=10\n', TEACHER, '10')['category'] == 'verified_canonical'
    assert correctness('6 + 4 = 10\n', TEACHER, '10')['category'] == 'unsupported_or_alternative'
    assert correctness('2*3=6\n', TEACHER, '10')['category'] == 'arithmetic_error'


def test_root_match_does_not_hide_bad_intermediate():
    result = correctness('2*3=5\n6+4=10\n', TEACHER, '10')
    assert result['category'] == 'arithmetic_error' and result['root_correct']
    assert not result['all_equations_true']


def test_valid_alternative_is_not_called_arithmetic_error():
    assert correctness('4+6=10\n', TEACHER, '10')['category'] == 'unsupported_or_alternative'
    assert correctness('We should multiply first.\n', TEACHER, '10')['parsed'] is False


def test_no_executable_python_or_unbounded_power():
    for bad in ("__import__('os').system('false')=0", '2**1000000=0', '1/0=0'):
        assert correctness(bad, TEACHER, '10')['category'] == 'unsupported_or_alternative'


def test_exact_rational_and_stable_item_rng():
    assert equation_trace('1/3=1/3\n')[0][1:] == (Fraction(1,3),Fraction(1,3))
    assert item_seed('fixed-item') == item_seed('fixed-item')
    assert 1 <= item_seed('fixed-item') < 2147483647
