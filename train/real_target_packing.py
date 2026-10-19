"""Answer-focused direct and full-trace SFT on verified native real documents.

V15 changes the supervised positions, not the causal prefix or interpreter.
The prefix and supplied calculation opening receive no loss. There is one CE
mean over the remaining targets; no independent auxiliary-loss normalization.
"""
from train.real_trace_packing import OPEN, ANSWER, encode, prefix_calculation
from train.useful_note_sft_data import EOS, packed

ARMS = ('target_direct', 'target_cot')


def training_pack(row, tokenizer, arm, *, width=640, target_slots=128):
    if row.get('split') != 'train' or arm not in ARMS:
        raise ValueError('Declared answer-focused TRAIN arm required')
    past, future = list(row['prefix_ids']), list(row['continuation16_ids'])
    vocab_size = len(tokenizer)
    if (tokenizer.eos_token_id != EOS or row.get('stop_id') != EOS
            or not 128 <= len(past) <= 512 or len(future) != 16
            or any(type(t) is not int or not 0 <= t < vocab_size
                   or t == EOS for t in past + future)):
        raise ValueError('Require original native bounded prefix and sixteen future tokens')
    if arm == 'target_direct':
        ids = past + future
        start = len(past)
    else:
        trace_text, trace = prefix_calculation(past, tokenizer)
        if (trace_text != row.get('full_calculation_text')
                or trace != row.get('full_calculation_ids') or not trace or EOS in trace):
            raise ValueError('Full trace must equal the exact prefix-only calculation')
        opening, answer = encode(tokenizer, OPEN), encode(tokenizer, ANSWER)
        if not opening or not answer or EOS in opening + answer:
            raise ValueError('Require native supplied opening and learned answer boundary')
        context = past + opening
        ids = context + trace + answer + future
        start = len(context)
    # No synthetic document EOS: future16 is an actual source fragment.
    return packed(ids, list(range(start, len(ids))), note_positions=(),
                  width=width, target_slots=target_slots)
