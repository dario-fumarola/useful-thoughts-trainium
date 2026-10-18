"""Prefix-only arithmetic projections, intact controls and state diagnostics.

Gold answers belong to the offline scorer. None of these generation APIs accepts
a full supervised row. Notes are never repaired, ranked or selected by utility.
"""
from collections import defaultdict
import re

from train.partial_state_data import OPS, direct_prompt, execute, expression, teacher_partial, writer_prompt

QUESTION_FIELDS = {'item_id', 'panel', 'expression_tree'}
EQUATION = re.compile(r'\s*([-+]?\d+)\s*([+*\-])\s*([-+]?\d+)\s*=\s*([-+]?\d+)\s*')
EOS = 151643


def question_projection(row):
    return {key: row[key] for key in sorted(QUESTION_FIELDS)}


def tree_from_question(question):
    if set(question) != QUESTION_FIELDS:
        raise ValueError('Only an explicit question projection may enter generation')
    tree = question['expression_tree']
    # Validate the expression through the exact interpreter, without its answer.
    expression(tree)
    return tree


def prompt_tokens(question, tokenizer, *, cot=False):
    tree = tree_from_question(question)
    text = writer_prompt(tree) if cot else direct_prompt(tree)
    return tokenizer.encode(text, add_special_tokens=False)


def reader_tokens(question, note_ids, tokenizer):
    """Preserve actual native note IDs; add only the declared answer boundary.

    The canonical note ends in newline. If a student ends earlier, a newline
    separator supplies the same Answer boundary, without changing note content.
    """
    prefix = prompt_tokens(question, tokenizer, cot=True)
    if any(type(i) is not int or i < 0 or i == EOS for i in note_ids):
        raise ValueError('Content IDs exclude the sampled stop; never repair a note')
    text = tokenizer.decode(note_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    suffix = ('\n' if text and not text.endswith('\n') else '') + 'Answer:\n'
    return prefix + list(note_ids) + tokenizer.encode(suffix, add_special_tokens=False)


def donor_permutation(lengths, keys):
    """Intact derangement, matching native length wherever a peer exists.

    Singleton length bins rotate together. With one singleton, swap its donor
    with the closest-length bin's first member. Always retain the population
    of note lengths and disclose the individual mismatches.
    """
    if len(lengths) != len(keys) or len(keys) < 2 or len(set(keys)) != len(keys):
        raise ValueError('At least two unique ordered items required')
    groups = defaultdict(list)
    for i, length in enumerate(lengths):
        if type(length) is not int or length < 0:
            raise ValueError('Actual nonnegative native lengths required')
        groups[length].append(i)
    result = [None] * len(keys)
    singles = []
    for group in groups.values():
        group.sort(key=lambda i: keys[i])
        if len(group) == 1:
            singles.extend(group)
        else:
            for i, j in zip(group, group[1:] + group[:1]):
                result[i] = j
    singles.sort(key=lambda i: keys[i])
    if len(singles) > 1:
        for i, j in zip(singles, singles[1:] + singles[:1]):
            result[i] = j
    elif singles:
        i = singles[0]
        j = min((j for j in range(len(keys)) if j != i),
                key=lambda j: (abs(lengths[i] - lengths[j]), keys[j]))
        result[i], result[j] = result[j], i
    if sorted(result) != list(range(len(keys))) or any(i == j for i, j in enumerate(result)):
        raise RuntimeError('Controls must preserve every intact donor exactly once')
    return result


def teacher_controls(questions, tokenizer):
    """All supplied notes are computed from the expression, never its future."""
    trees = [tree_from_question(q) for q in questions]
    texts = [teacher_partial(tree) for tree in trees]
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    notes = [encode(text) for text in texts]
    donors = donor_permutation([len(n) for n in notes], [q['item_id'] for q in questions])
    result = []
    for i, (question, tree) in enumerate(zip(questions, trees)):
        steps = execute(tree)[1][:-1]
        wrong = [tuple(s) for s in steps]
        a, op, b, value = wrong[-1]
        wrong[-1] = (a, op, b, value + 1)
        corrupt = ''.join(f'{a} {op} {b} = {v}\n' for a, op, b, v in wrong)
        j = donors[i]
        result.append(dict(item_id=question['item_id'], notes={
            'empty': [], 'teacher': notes[i], 'corrupt': encode(corrupt),
            'other_teacher': notes[j], 'restatement': encode(expression(tree) + '\n')},
            other_teacher_item_id=questions[j]['item_id'],
            other_teacher_length_matched=len(notes[i]) == len(notes[j])))
    return result


def score_partial_state(text, tree, *, stop_type, status='ok'):
    """Canonical non-root state accuracy, distinct from complete answer accuracy.

    Different strategies may be useful without matching this canonical trace.
    Equation validity alone is insufficient: states must be the prefix's exact
    non-root postorder operations. No numerical coincidence is filtered out.
    """
    lines = text.strip().splitlines()
    matches = [EQUATION.fullmatch(line) for line in lines]
    supported = bool(lines) and all(match is not None for match in matches)
    statements = [(int(m[1]), m[2], int(m[3]), int(m[4])) for m in matches] if supported else []
    _, steps = execute(tree)
    canonical = supported and statements == steps[:-1]
    return dict(canonical_nonroot_exact=canonical,
                canonical_and_natural_stop=canonical and status == 'ok' and stop_type == f'token:{EOS}',
                equations_supported=supported,
                all_stated_equations_correct=supported and all(OPS[op](a, b) == v for a, op, b, v in statements),
                root_operation_transmitted=supported and steps[-1] in statements,
                expected_nonroot_statements=len(steps)-1,
                observed_statements=len(statements), stop_type=stop_type, status=status)
