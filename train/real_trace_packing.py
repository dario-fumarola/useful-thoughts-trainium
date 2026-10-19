"""Native real-document CPT, with an optional verified full calculation.

This is a new supervised training experiment, not the frozen partial-state
study or an exact reproduction of TPT. All active tokens share ONE CE mean.
The inserted opening marker is a given condition, never a target following
ordinary text. Plain source fragments receive no artificial document EOS.
"""
from train.useful_note_sft_data import EOS, packed

ARMS = ('real_raw', 'real_prior', 'real_post')
OPEN = '\nCalculation:\n'
ANSWER = 'Answer:\n'


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def prefix_calculation(prefix_ids, tokenizer):
    """Exact prefix-only supervision; no continuation or gold-answer argument."""
    from train.calculation_boundaries import maximal_lhs
    from train.symbolic_note_labels import postorder_steps
    text = tokenizer.decode(prefix_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    value, steps = postorder_steps(maximal_lhs(text))
    if len(steps) < 2:
        raise ValueError('Require a compound expression available in the prefix')
    trace = '\n'.join(steps) + '\n'
    return trace, encode(tokenizer, trace)


def view_for(arm, position, items, item_index):
    """Exactly one plain and one augmented occurrence in every two passes."""
    if arm not in ARMS or items < 1 or position < 0 or not 0 <= item_index < items:
        raise ValueError('Declared real-data arm and actual item cursor required')
    return ('raw' if arm == 'real_raw' or (item_index + position // items) % 2
            else arm.removeprefix('real_'))


def training_pack(row, tokenizer, view, *, width=640, target_slots=640):
    """Preserve original native prefix/future IDs; never retokenize the splice.

    prior: prefix + GIVEN open + full trace + answer marker + true future16.
    post:  prefix + true future16 + GIVEN open + full trace + native EOS.
    raw:   prefix + true future16, without an invented end-of-document label.
    Prefix tokens after the first are supervised in every view. Trace/answer
    markers belong to ordinary CE, not a separately normalized auxiliary loss.
    """
    if row.get('split') != 'train' or view not in ('raw', 'prior', 'post'):
        raise ValueError('Only declared TRAIN views may enter optimization')
    past, future = list(row['prefix_ids']), list(row['continuation16_ids'])
    vocab_size = len(tokenizer)
    if (tokenizer.eos_token_id != EOS or row['stop_id'] != EOS
            or not 128 <= len(past) <= 512 or len(future) != 16
            or any(type(t) is not int or not 0 <= t < vocab_size
                   or t == EOS for t in past + future)):
        raise ValueError('Require native bounded prefix and actual sixteen future tokens')
    positions = list(range(1, len(past)))
    if view == 'raw':
        ids = past + future
        positions += list(range(len(past), len(ids)))
    else:
        trace_text, trace = prefix_calculation(past, tokenizer)
        if (trace_text != row['full_calculation_text'] or trace != row['full_calculation_ids']
                or not trace or EOS in trace):
            raise ValueError('Stored full trace differs from exact causal-prefix computation')
        opening, answer = encode(tokenizer, OPEN), encode(tokenizer, ANSWER)
        if not opening or not answer or EOS in opening + answer:
            raise ValueError('Native conditioning and answer markers required')
        if view == 'prior':
            ids = past + opening + trace + answer + future
            # The marker is supplied by the caller; do not teach raw text to
            # predict markup instead of its real next token.
            positions += list(range(len(past) + len(opening), len(ids)))
        else:
            source = past + future
            ids = source + opening + trace + [EOS]
            positions += list(range(len(past), len(source)))
            positions += list(range(len(source) + len(opening), len(ids)))
    return packed(ids, positions, note_positions=(), width=width, target_slots=target_slots)
