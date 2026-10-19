"""A bounded curriculum intervention for the previously fixed-depth writer.

Both curricula start from released B0 and use the existing writer prompt and
exact nonroot targets. This changes training depth/structure coverage, not just
an EOS label. No model score or answer value selects an expression.
"""
from functools import lru_cache
import random

from train.partial_state_data import TRAIN_SHAPES, expression, fill, row


@lru_cache(None)
def shapes(operations):
    if operations == 0:
        return (0,)
    if type(operations) is not int or not 0 <= operations <= 7:
        raise ValueError('Bounded nonnegative operation count required')
    return tuple((a, b) for left in range(operations)
                 for a in shapes(left) for b in shapes(operations-1-left))


def left_chain(n):
    return 0 if n == 0 else (left_chain(n-1), 0)


def right_chain(n):
    return 0 if n == 0 else (0, right_chain(n-1))


def balanced(n):
    if n == 0:
        return 0
    left = (n-1)//2
    return balanced(left), balanced(n-1-left)


def training_shapes(nonroot_steps):
    if nonroot_steps == 2:
        return TRAIN_SHAPES
    return tuple(dict.fromkeys((left_chain(nonroot_steps+1),
                                right_chain(nonroot_steps+1), balanced(nonroot_steps+1))))


def generate(seed, excluded=()):
    """640 expressions/arm; variable arm has128 at each depth1..5.

    Fixed arm has640 at depth2. All operands2..99, operators uniform+/-/*.
    Each arm has its own deterministic stream and is expression-disjoint from
    the other arm, new DEV panels, and the complete previous dataset.
    """
    seen = set(excluded)
    result = {}
    for number, arm in enumerate(('fixed', 'variable')):
        rng = random.Random(seed + number)
        values = []
        while len(values) < 640:
            depth = 2 if arm == 'fixed' else len(values) % 5 + 1
            options = training_shapes(depth)
            shape = options[(len(values)//5 if arm == 'variable' else len(values)) % len(options)]
            tree = fill(shape, rng, (2, 99)); text = expression(tree)
            if text in seen:
                continue
            seen.add(text)
            values.append(row(tree, 'TRAIN'))
        result['TRAIN_' + arm] = values
    for number, depth in enumerate((2, 6)):
        rng = random.Random(seed + 10 + number); values = []
        options = training_shapes(depth) if depth == 2 else shapes(depth+1)
        while len(values) < 256:
            tree = fill(options[len(values) % len(options)], rng, (2, 99))
            text = expression(tree)
            if text in seen:
                continue
            seen.add(text)
            values.append(row(tree, 'DEV_depth' + str(depth)))
        result['DEV_depth' + str(depth)] = values
    return result
