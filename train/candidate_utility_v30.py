"""Frozen-cohort, exact-arithmetic helpers for the bounded V30 diagnostic."""
from __future__ import annotations

from fractions import Fraction
import hashlib

from train.calculation_boundaries import parse_numeric
from train.expression_identity import expression_key


def cohort(pool):
    if len(pool) != 390 or len({r['document_id'] for r in pool}) != 390:
        raise ValueError('Original390 unique-document source pool required')
    ordered = sorted(pool, key=lambda r: hashlib.sha256(
        ('v30-fixed-cohort:' + r['item_id']).encode()).hexdigest())
    return ordered[:8], ordered[8:264]


def item_seed(item_id):
    return int.from_bytes(hashlib.sha256(('v30-sampling:'+item_id).encode()).digest()[:8],
                          'big') % 2147483646 + 1


def equation_trace(text):
    """Conservative bounded execution; no eval or language-model judge."""
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if len(result) >= 16 or line.count('=') != 1:
            raise ValueError('Unsupported equation layout')
        left, right = line.split('=')
        a, b = parse_numeric(left), parse_numeric(right)
        # A scalar RHS can be an exact rational, not a second program.
        if expression_key(right)[0] not in ('number', 'Div', 'neg'):
            raise ValueError('Unsupported computed scalar')
        result.append((expression_key(left), a.value, b.value))
    return result


def correctness(text, teacher_text, root_value):
    teacher = equation_trace(teacher_text)
    assert teacher and all(a == b for _, a, b in teacher)
    root = Fraction(root_value)
    assert teacher[-1][2] == root
    try:
        actual = equation_trace(text)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return dict(category='unsupported_or_alternative', parsed=False,
                    equations=None, root_correct=None, all_equations_true=None)
    if not actual:
        return dict(category='unsupported_or_alternative', parsed=True,
                    equations=0, root_correct=False, all_equations_true=None)
    truth = all(a == b for _, a, b in actual)
    root_correct = actual[-1][2] == root
    canonical = len(actual) == len(teacher) and all(
        x[0] == y[0] and x[2] == y[2] for x, y in zip(actual, teacher))
    category = ('verified_canonical' if canonical and truth and root_correct else
                'arithmetic_error' if not truth or not root_correct else
                'unsupported_or_alternative')
    return dict(category=category, parsed=True, equations=len(actual),
                root_correct=root_correct, all_equations_true=truth)
