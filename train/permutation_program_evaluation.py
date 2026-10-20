"""Prefix-only evaluation contexts, including intact partial-state controls.

Generation consumes projected questions/prompt IDs. Exact partial states are an
offline intervention derived from the available map/program, never future labels.
"""
from collections import defaultdict
import hashlib

from train.permutation_program_data import Question, question_from_dict, render
from train.permutation_program_packing import WORK, FINAL, inference_context

MODES = ('direct', 'empty', 'trace', 'pause', 'partial', 'corrupt', 'donor')


def available_item(row):
    if type(row) is not dict or set(row) != {'item_id', 'family_id', 'question'}:
        raise ValueError('Only the questions-only file, never labelled rows')
    if not all(isinstance(row[k], str) and row[k] for k in ('item_id', 'family_id')):
        raise ValueError('Explicit item and independent-family identities required')
    return question_from_dict(row['question'])


def partial_states(question):
    """Independent execution of every step except the final root operation."""
    if type(question) is not Question:
        raise ValueError('Only an available-input Question')
    node = question.start; lines = []
    for op in question.program[:-1]:
        nxt = getattr(question, op)[node]
        lines.append(f'{op}: {"ABCDEFGHIJKLMNOP"[node]} -> {"ABCDEFGHIJKLMNOP"[nxt]}\n')
        node = nxt
    return ''.join(lines)


def corrupt_partial(question):
    text = partial_states(question)
    if not text:
        return text
    lines = text.splitlines(keepends=True)
    # Deterministic single wrong destination; no utility filtering or answer use.
    old = lines[-1][-2]
    new = 'ABCDEFGHIJKLMNOP'[('ABCDEFGHIJKLMNOP'.index(old)+1) % 16]
    lines[-1] = lines[-1][:-2] + new + '\n'
    return ''.join(lines)


def controls(items):
    """Derange whole donor states at fixed depth, always outside the same family.

    Donor states are not repaired or selected for recipient utility. A foreign
    state may coincidentally mention the recipient's final node. This is not
    filtered away. Depth-one controls all have empty partial states by design.
    """
    if len({v['item_id'] for v in items}) != len(items):
        raise ValueError('Unique original questions required')
    groups = defaultdict(lambda: defaultdict(list)); notes = {}; questions = {}
    for item in items:
        q = available_item(item); questions[item['item_id']] = q
        groups[len(q.program)][item['family_id']].append(item)
        notes[item['item_id']] = partial_states(q)
    donors = {}
    for depth, families in groups.items():
        ordered = sorted(families, key=lambda f: hashlib.sha256(f.encode()).hexdigest())
        if len(ordered) < 2:
            raise ValueError('At least two independent families per depth needed for donor control')
        for index, family in enumerate(ordered):
            other = ordered[(index+1) % len(ordered)]
            source = sorted(families[family], key=lambda v: v['item_id'])
            donor = sorted(families[other], key=lambda v: v['item_id'])
            if len(source) != len(donor):
                raise ValueError('Balanced paired variants required; never repeat/drop donor notes')
            for recipient, origin in zip(source, donor):
                donors[recipient['item_id']] = origin
    return {item['item_id']: dict(partial=notes[item['item_id']],
        corrupt=corrupt_partial(questions[item['item_id']]),
        donor=notes[donors[item['item_id']]['item_id']],
        donor_item_id=donors[item['item_id']]['item_id'],
        donor_family_id=donors[item['item_id']]['family_id']) for item in items}


def prompt_ids(item, tokenizer, mode, intervention=None):
    q = available_item(item)
    if mode in ('direct', 'empty', 'trace', 'pause'):
        return inference_context(q, tokenizer, mode)
    if mode not in ('partial', 'corrupt', 'donor') or intervention is None:
        raise ValueError('Declared prefix-derived partial-state intervention required')
    text = intervention[mode]
    return tokenizer.encode(render(q)+WORK+text+FINAL, add_special_tokens=False)
