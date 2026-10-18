"""Scientific data invariants: exact partials, root exclusion and split identity."""
import ast
import operator

import pytest

from train.partial_state_data import (PANELS, STRUCTURE_SHAPES, TRAIN_SHAPES,
    direct_prompt, evaluate, execute, expression, generate, reader_prompt,
    teacher_partial, writer_prompt)


def independent_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    return {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}[type(node.op)](
        independent_eval(node.left), independent_eval(node.right))


def shape(tree):
    return 0 if type(tree) is int else (shape(tree[1]), shape(tree[2]))


def leaves(tree):
    return [tree] if type(tree) is int else leaves(tree[1]) + leaves(tree[2])


def test_exact_partial_omits_root():
    tree = ('*', ('+', 37, 18), ('-', 92, 57))
    assert teacher_partial(tree) == '37 + 18 = 55\n92 - 57 = 35\n'
    assert execute(tree)[-1][-1] == (55, '*', 35, 1925)
    assert evaluate(tree) == 1925
    assert '1925' not in writer_prompt(tree) + teacher_partial(tree)
    assert reader_prompt(tree, '') == writer_prompt(tree) + 'Answer:\n'
    assert direct_prompt(tree).endswith('Answer:\n')


def test_shapes_are_disjoint_and_all_dev_axes_are_separate():
    assert set(TRAIN_SHAPES).isdisjoint(STRUCTURE_SHAPES)
    data = generate()
    seen = set()
    for panel, rows in data.items():
        n, shapes, operand_range = PANELS[panel]
        assert len(rows) == n
        for r in rows:
            tree = r['expression_tree']
            assert shape(tree) in shapes
            assert all(operand_range[0] <= v <= operand_range[1] for v in leaves(tree))
            assert r['item_id'] not in seen
            seen.add(r['item_id'])
            assert int(r['answer']) == independent_eval(ast.parse(expression(tree), mode='eval').body)
            assert len(r['partial_steps']) == len(leaves(tree)) - 2
            assert r['partial_steps'] == execute(tree)[1][:-1]
            for a, op, b, value in r['partial_steps']:
                assert value == {'+': operator.add, '-': operator.sub, '*': operator.mul}[op](a, b)
            # Changing all future/supervision fields leaves writer context unchanged.
            original = writer_prompt(tree)
            r.update(answer='999999999', partial_note='unrelated', partial_steps=[])
            assert writer_prompt(r['expression_tree']) == original


def test_deterministic_not_model_selected():
    panels = {'TRAIN': (128, TRAIN_SHAPES, (2, 99))}
    assert generate(17, panels) == generate(17, panels)
    assert generate(17, panels) != generate(18, panels)
    with pytest.raises(ValueError):
        teacher_partial(('+', 2, 3))
