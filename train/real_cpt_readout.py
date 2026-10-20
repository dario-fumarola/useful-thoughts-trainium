"""Frozen V29 contexts and inventories, without model loading or generation.

Writer callers receive only the question projection. Continuations and numeric
answers enter the scorer only. Oracle traces include the root answer and are
plumbing controls, not evidence that a deployed writer can compute that answer.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import hashlib

from train.calculation_boundaries import render_value
from train.real_cpt_scaffold import verified_work
from train.real_trace_contexts import question_projection, reader_context
from train.real_trace_packing import encode

MATH_MODES = ('raw', 'empty', 'own', 'other', 'oracle', 'corrupt')
SCALAR_MODES = ('raw', 'empty', 'own')


@lru_cache(maxsize=4)
def defined_ids(tokenizer):
    return frozenset(tokenizer.get_vocab().values())


def donors(item_ids):
    """A fixed whole-note derangement, independent of notes and outcomes."""
    if len(item_ids) < 2 or len(set(item_ids)) != len(item_ids):
        raise ValueError('At least two unique original identities required')
    ordered = sorted(item_ids, key=lambda k: hashlib.sha256(('v29-other:'+k).encode()).hexdigest())
    return {k: ordered[(i+1) % len(ordered)] for i, k in enumerate(ordered)}


def equal_length_batches(questions):
    """Only actual equal-length rows: B8, B4, then original serial residuals."""
    groups = defaultdict(list)
    for i, q in enumerate(questions):
        groups[len(q['prefix_ids'])].append(i)
    for length in sorted(groups):
        indices = groups[length]
        cursor = 0
        for size in (8, 4, 1):
            while len(indices)-cursor >= size:
                yield indices[cursor:cursor+size]
                cursor += size
        assert cursor == len(indices)


def corrupt_work(question, tokenizer):
    """Change exactly the last computed root value by +1, from prefix alone."""
    text, _ = verified_work(question['prefix_ids'], tokenizer)
    lines = text.splitlines(keepends=True)
    left, right = lines[-1].rstrip('\n').rsplit(' = ', 1)
    lines[-1] = left+' = '+render_value(Fraction(right)+1)+'\n'
    result = ''.join(lines)
    return result, encode(tokenizer, result)


def controls(question, tokenizer, own, other):
    """All native contexts use the same forced complete Answer boundary."""
    q = question_projection(question)
    _, oracle = verified_work(q['prefix_ids'], tokenizer)
    _, corrupt = corrupt_work(q, tokenizer)
    contents = dict(raw=None, empty=[], own=own['content_ids'],
                    other=other['content_ids'], oracle=oracle, corrupt=corrupt)
    contexts = {mode: reader_context(q, tokenizer, note) for mode, note in contents.items()}
    if any(len(ctx)+16 > 672 for ctx in contexts.values()):
        raise ValueError('Complete native context exceeds fixed width; never truncate')
    return contexts


def validate_note(value, question, tokenizer, *, stop_ids, cap=128):
    """Check the intact original attempt; execution failures stay observations."""
    from train.real_trace_contexts import writer_prompt
    q = question_projection(question)
    if (value['item_id'] != q['item_id'] or value['document_id'] != q['document_id']
            or value['prompt_ids'] != writer_prompt(q, tokenizer)
            or value['status'] not in ('ok', 'error')):
        raise ValueError('Original same-question causal attempt required')
    actions = value['actions']
    defined = defined_ids(tokenizer)
    if (len(actions) > cap or any(type(t) is not int or t not in defined for t in actions)
            or any(t in stop_ids for t in actions[:-1])):
        raise ValueError('Invalid native action sequence')
    stopped = bool(actions and actions[-1] in stop_ids)
    content = actions[:-1] if stopped else actions
    if value['content_ids'] != content:
        raise ValueError('Content differs from exact original actions')
    text = tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if value['text'] != text or value['raw_text'] != text:
        raise ValueError('Text differs from intact original native content')
    if value['status'] == 'ok':
        expected = f'token:{actions[-1]}' if stopped else 'length_cap'
        if value['stop_type'] != expected or not stopped and len(actions) != cap:
            raise ValueError('Unreported natural stop or cap')
    return value


def numeric_masks(row, tokenizer):
    """Recover native source masks, including merged signs and whitespace.

    Standalone tokenization of a numeral is not its tokenization in the actual
    source. Reuse the original boundary curator on the exact native sequence;
    never locate a re-encoded string or split a token containing two components.
    """
    from train.computable_equations import source_target
    target = row['continuation16_ids']
    past = row['prefix_ids']
    selected = dict(cut=len(past),prefix_text=tokenizer.decode(past,
        skip_special_tokens=False,clean_up_tokenization_spaces=False))
    proof = source_target(selected,past+target,tokenizer)
    count = proof['token_count']
    if (not count or count > 16 or proof['literal_text'] != row['numeric_surface_text']
            or proof['closing_boundary_ids'] != row['closing_boundary_ids']
            or proof['value'] != str(Fraction(row['numeric_value']))
            or proof['token_ids'] != target[:count]):
        raise ValueError('Complete original numeral and boundary must be inside target16')
    numeric = proof['numeric_token_mask']+[False]*(16-count)
    return dict(numeral=numeric,numeral_and_boundary=[i<count for i in range(16)],
        boundary_and_whitespace=[i<count and not numeric[i] for i in range(16)],
        surrounding=[i>=count for i in range(16)])
