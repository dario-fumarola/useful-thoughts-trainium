"""Exact supervised calculation traces; these are not generated policy notes.

The source expression is validated by the existing bounded arithmetic parser.
Every binary operation is then evaluated in left-to-right AST postorder using
Fractions. Source continuation text chooses only the final answer's surface.
"""
from __future__ import annotations

import ast
from fractions import Fraction

from train.calculation_boundaries import parse_numeric, render_value
from train.scalar_completion import parse_initial_scalar_v2

POSTORDER = 'exact-postorder-v1'
FULL = 'full-calculation-v1'


def verified_expression(row):
    parsed = parse_numeric(row['boundary']['expression'])
    surface = row['numeric_surface_text']
    answer = parse_initial_scalar_v2(surface, 'eos')
    if (parsed.value != Fraction(row['numeric_value'])
            or answer['status'] != 'valid'
            or Fraction(answer['value']) != parsed.value):
        raise ValueError('Externally verified equation and answer disagree')
    return parsed, surface


def postorder_steps(expression):
    """Return validated value and substitution steps, with exact arithmetic.

    Do not evaluate arbitrary Python, use floating point, or round intermediate
    fractions. Repeated subexpressions remain repeated computations, not data
    augmentation. Unary signs are applied before their parent's operation.
    """
    parsed = parse_numeric(expression)
    text = parsed.expression
    tree = ast.parse(text, mode='eval')
    steps = []

    def operand(value):
        rendered = render_value(value)
        return f'({rendered})' if value < 0 or '/' in rendered else rendered

    def walk(node):
        if isinstance(node, ast.Constant):
            return Fraction(ast.get_source_segment(text, node))
        if isinstance(node, ast.UnaryOp):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        left, right = walk(node.left), walk(node.right)
        if isinstance(node.op, ast.Add):
            value, symbol = left + right, '+'
        elif isinstance(node.op, ast.Sub):
            value, symbol = left - right, '-'
        elif isinstance(node.op, ast.Mult):
            value, symbol = left * right, '*'
        elif isinstance(node.op, ast.Div):
            value, symbol = left / right, '/'
        elif isinstance(node.op, ast.Pow):
            value, symbol = left ** int(right), '^'
        else:  # parse_numeric must reject any unsupported operator first.
            raise ValueError('Unsupported validated operation')
        steps.append(f'{operand(left)} {symbol} {operand(right)} = {render_value(value)}')
        return value

    value = walk(tree.body)
    if value != parsed.value or len(steps) != parsed.binary_ops:
        raise ValueError('Postorder evaluation disagrees with bounded parser')
    return value, steps


def teacher_note(row):
    parsed, surface = verified_expression(row)
    kind = row.get('teacher_note_kind', FULL)
    if kind == FULL:
        return f'Compute {row["boundary"]["expression"]} = {surface}.'
    if kind != POSTORDER:
        raise ValueError('An explicit supported teacher-note identity is required')
    value, steps = postorder_steps(parsed.expression)
    if value != parsed.value or not steps:
        raise ValueError('A postorder trace requires an actual binary calculation')
    return 'Compute: ' + '. '.join(steps) + f'. Result: {surface}.'
