"""Exact, prefix-computable partial arithmetic; no root operation in notes.

This is a diagnostic dataset, not a general-language training substitute.
Writer prompts accept expressions alone. Teacher notes are execution traces of
every non-root internal node, in postorder; final answers live separately.
"""
import argparse
import hashlib
import json
import operator
import random
from pathlib import Path

OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul}
# Unlabelled ordered trees. Operators and leaves are filled independently.
TRAIN_SHAPES = (((0, 0), (0, 0)), (((0, 0), 0), 0), (0, (0, (0, 0))))
STRUCTURE_SHAPES = (((0, (0, 0)), 0), (0, ((0, 0), 0)))
COMPOSITION_SHAPES = (((0, 0), (0, (0, 0))), (((0, 0), (0, 0)), 0))
PANELS = {
    'TRAIN': (5120, TRAIN_SHAPES, (2, 99)),
    'DEV_interpolation': (512, TRAIN_SHAPES, (2, 99)),
    'DEV_structure': (512, STRUCTURE_SHAPES, (2, 99)),
    'DEV_range': (512, TRAIN_SHAPES, (100, 199)),
    'DEV_composition': (512, COMPOSITION_SHAPES, (2, 99)),
}


def evaluate(tree):
    if type(tree) is int:
        return tree
    op, left, right = tree
    return OPS[op](evaluate(left), evaluate(right))


def expression(tree):
    if type(tree) is int:
        return str(tree)
    op, left, right = tree
    return f'({expression(left)} {op} {expression(right)})'


def execute(tree):
    """Return (value, postorder steps), including root for independent auditing."""
    if type(tree) is int:
        return tree, []
    op, left, right = tree
    a, steps_a = execute(left)
    b, steps_b = execute(right)
    value = OPS[op](a, b)
    return value, steps_a + steps_b + [(a, op, b, value)]


def teacher_partial(tree):
    _, steps = execute(tree)
    if len(steps) < 2:
        raise ValueError('Partial-state diagnostic requires at least two operations')
    return ''.join(f'{a} {op} {b} = {v}\n' for a, op, b, v in steps[:-1])


def writer_prompt(tree):
    """No target, teacher note, answer or full training row is accepted here."""
    return f'Problem: {expression(tree)}\nWork:\n'


def reader_prompt(tree, note):
    return writer_prompt(tree) + note + 'Answer:\n'


def direct_prompt(tree):
    return f'Problem: {expression(tree)}\nAnswer:\n'


def fill(shape, rng, operand_range):
    if shape == 0:
        return rng.randint(*operand_range)
    return (rng.choice(tuple(OPS)), fill(shape[0], rng, operand_range),
            fill(shape[1], rng, operand_range))


def row(tree, panel):
    text = expression(tree)
    answer, steps = execute(tree)
    return dict(schema='partial-state-arithmetic-v6', panel=panel,
                item_id=hashlib.sha256(text.encode()).hexdigest(),
                expression_tree=tree, expression=text,
                partial_note=teacher_partial(tree), partial_steps=steps[:-1],
                answer=str(answer), root_step=steps[-1],
                answer_is_intermediate_value=any(s[-1] == answer for s in steps[:-1]))


def generate(seed=2026101806, panels=None):
    """Fixed sizes/order, no model scoring or answer-based acceptance filter."""
    rng = random.Random(seed)
    seen = set()
    result = {}
    for panel, (n, shapes, operands) in (panels or PANELS).items():
        rows = []
        while len(rows) < n:
            # Cycle shapes to balance them to within one item, independent of answers.
            tree = fill(shapes[len(rows) % len(shapes)], rng, operands)
            text = expression(tree)
            if text in seen:
                continue
            seen.add(text)
            rows.append(row(tree, panel))
        result[panel] = rows
    return result


def save(output, seed=2026101806):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = generate(seed)
    files = {}
    for panel, rows in data.items():
        path = output / (panel + '.jsonl')
        with path.open('x') as stream:
            for value in rows:
                stream.write(json.dumps(value, separators=(',', ':')) + '\n')
        files[path.name] = dict(rows=len(rows), bytes=path.stat().st_size,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               incidental_answer_in_partial=sum(r['answer_is_intermediate_value'] for r in rows))
    # TRAIN subset only; no development outcome chooses mechanics examples.
    subset = output / 'overfit128.ids.json'
    subset.write_text(json.dumps([r['item_id'] for r in data['TRAIN'][:128]], indent=2))
    manifest = dict(schema='partial-state-arithmetic-v6', seed=seed,
                    generated_from='exact prefix expression only; no model filtering',
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    files=files, overfit128_sha256=hashlib.sha256(subset.read_bytes()).hexdigest(),
                    limits=['Synthetic computation diagnostic; no general-language claim.',
                            'Root step excluded structurally, not by banning numerical coincidences.',
                            'DEV panels are separate; never pool their generalization claims.',
                            'This prepares data only; no model or learning qualification.'])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=2026101806)
    args = parser.parse_args()
    print(json.dumps(save(args.output, args.seed), indent=2))
