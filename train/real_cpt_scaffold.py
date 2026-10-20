"""V29 CPU preparation: identical clean LM plus a separate computation branch.

There is no launcher in this module. Every arm trains on the complete unmodified
source window, including the actual continuation. Calculation examples also
have a direct, empty-format, or verified-work view. Continuation and work losses
are separate means; adding a trace cannot dilute the continuation coefficient.
The extra views change supervision and compute, not the information in a source
prefix. This is supervised continued pretraining, not autonomous thinking.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random

import torch

from train.real_trace_packing import ANSWER, OPEN, encode, prefix_calculation
from train.useful_note_sft_data import EOS, packed

SCHEMA = 'real-cpt-scaffold-v29-cpu-v1'
ARMS = ('raw', 'format', 'computation')
CONTINUATION_WEIGHT = 0.5
WORK_WEIGHT = 0.1
ORDINARY_PER_CALCULATION = 3
WIDTH = 640
TARGET_SLOTS = 640


@lru_cache(maxsize=4)
def native_metadata(tokenizer):
    """Read immutable tokenizer metadata once, not once per source token.

    The pinned Qwen tokenizer's get_vocab_size is expensive in this environment.
    This API assumes the caller never adds/removes tokens after construction;
    the data manifest records the actual local tokenizer files.
    """
    return len(tokenizer), frozenset(tokenizer.all_special_ids)


@dataclass
class Views:
    clean: dict
    auxiliary: dict | None


def _component_pack(ids, components, *, width, target_slots):
    positions = sorted(p for group in components.values() for p in group)
    if len(set(positions)) != len(positions):
        raise ValueError('A target belongs to exactly one component')
    result = packed(ids, positions, note_positions=(), width=width,
                    target_slots=target_slots)
    positions_tensor = result['target_positions']
    result.pop('note_close_mask')
    result.pop('future_mask')
    groups = {name: set(group) for name, group in components.items()}
    result['component_masks'] = {
        name: torch.tensor([[p in group for p in positions]
                            + [False] * (target_slots - len(positions))])
        for name, group in groups.items()
    }
    assert all(m.shape == positions_tensor.shape and m.dtype == torch.bool
               for m in result['component_masks'].values())
    return result


def validate_source(row, tokenizer, *, training):
    allowed = ('train',) if training else ('dev', 'followup_dev')
    if row.get('split') not in allowed or row.get('kind') not in ('ordinary', 'calculation'):
        raise ValueError('Explicit source kind and matching TRAIN/DEV scope required')
    past, future = list(row['prefix_ids']), list(row['continuation16_ids'])
    vocabulary_size, special = native_metadata(tokenizer)
    if (tokenizer.eos_token_id != EOS or row.get('stop_id') != EOS
            or not 128 <= len(past) <= 512 or len(future) != 16
            or (row['kind'] == 'ordinary' and len(past) != 512)
            or any(type(t) is not int or not 0 <= t < vocabulary_size
                   or t in special for t in past + future)):
        raise ValueError('Original native, nonspecial source prefix and continuation16 required')
    return past, future


def verified_work(prefix_ids, tokenizer):
    """The API deliberately has no continuation/answer/annotation argument."""
    text, ids = prefix_calculation(prefix_ids, tokenizer)
    _, special = native_metadata(tokenizer)
    if not 1 <= len(ids) < 128 or any(t in special for t in ids):
        raise ValueError('Trace plus its first learned stop must fit128 native actions')
    return text, ids


def training_views(row, tokenizer, arm, *, width=WIDTH, target_slots=TARGET_SLOTS):
    if arm not in ARMS:
        raise ValueError('Explicit V29 arm required')
    past, future = validate_source(row, tokenizer, training=True)
    original = past + future
    clean = _component_pack(original, {'source': list(range(1, len(original)))},
                            width=width, target_slots=target_slots)
    if row['kind'] == 'ordinary':
        return Views(clean, None)
    work_text, work = verified_work(past, tokenizer)
    if (work_text != row.get('full_calculation_text')
            or work != row.get('full_calculation_ids')):
        raise ValueError('Stored work must equal the exact prefix-only computation')
    opening, answer = encode(tokenizer, OPEN), encode(tokenizer, ANSWER)
    _, special = native_metadata(tokenizer)
    if not opening or not answer or any(t in special for t in opening + answer):
        raise ValueError('Native supplied opening and learned answer boundary required')
    if arm == 'raw':
        context, work_positions = past, []
    else:
        given = past + opening
        context = given + (work if arm == 'computation' else []) + answer
        work_positions = list(range(len(given), len(context)))
    ids = context + future
    auxiliary = _component_pack(ids, {
        'work': work_positions,
        'continuation': list(range(len(context), len(ids))),
    }, width=width, target_slots=target_slots)
    return Views(clean, auxiliary)


def grouped_order(rows, *, seed, passes):
    """One calculation and three ordinary documents per four-item group.

    Every document occurs once/pass, without replacement. Every batch whose size
    is a multiple of four consequently has exactly25% auxiliary examples. Same
    seed/order for every arm. No outcome, expression length, or loss enters order.
    """
    if type(seed) is not int or type(passes) is not int or passes < 1:
        raise ValueError('Integer data-order seed and positive passes required')
    groups = {kind: [i for i, r in enumerate(rows) if r.get('kind') == kind]
              for kind in ('calculation', 'ordinary')}
    if (any(r.get('split') != 'train' for r in rows)
            or len(groups['ordinary']) != ORDINARY_PER_CALCULATION * len(groups['calculation'])
            or not groups['calculation']
            or sum(map(len, groups.values())) != len(rows)
            or len({r['document_id'] for r in rows}) != len(rows)):
        raise ValueError('Unique TRAIN documents in the frozen1:3 mixture required')
    result = []
    for epoch in range(passes):
        rng = random.Random((seed << 32) + epoch)
        calculation, ordinary = list(groups['calculation']), list(groups['ordinary'])
        rng.shuffle(calculation)
        rng.shuffle(ordinary)
        for n, index in enumerate(calculation):
            group = [index] + ordinary[3*n:3*n+3]
            rng.shuffle(group)
            result.extend(group)
    return result


def auxiliary_weights(*, calculation_examples, source_examples):
    """Convert auxiliary-batch means to the declared per-source-document loss.

    A separate auxiliary batch contains only calculation documents. Multiplying
    by its fraction prevents accidentally increasing auxiliary weights fourfold.
    Both backward passes precede one optimizer step and one gradient reduction.
    """
    if (type(calculation_examples) is not int or type(source_examples) is not int
            or not 0 < calculation_examples <= source_examples):
        raise ValueError('Actual nonempty batch counts required')
    fraction = calculation_examples / source_examples
    return {'work': WORK_WEIGHT * fraction,
            'continuation': CONTINUATION_WEIGHT * fraction}
