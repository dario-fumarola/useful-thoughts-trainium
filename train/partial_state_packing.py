"""Short causal packs for answer, compact CoT, partial writer and reader.

All targets are shifted exactly once by the shared CE kernel. The CoT loss is
an ordinary mean over its entire completion, not a sum of separate note/answer
means. A writer stop is supervised even though its ID also pads inactive slots.
"""
import torch

from train.partial_state_data import direct_prompt, evaluate, reader_prompt, teacher_partial, writer_prompt
from train.useful_note_sft_data import EOS, packed

COMPONENTS = ('direct', 'cot', 'writer', 'reader_teacher', 'reader_empty')


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def prompt_ids(tree, tokenizer):
    """The deployment writer receives only the available expression."""
    return encode(tokenizer, writer_prompt(tree))


def training_pack(row, tokenizer, component, *, width=128, target_slots=64):
    if row['panel'] != 'TRAIN' or component not in COMPONENTS:
        raise ValueError('Only declared TRAIN components may enter learning')
    if tokenizer.eos_token_id != EOS:
        raise ValueError('Pin the released native base EOS')
    tree = row['expression_tree']
    note = teacher_partial(tree)
    if row['partial_note'] != note or row['answer'] != str(evaluate(tree)):
        raise ValueError('Supervised trace or answer differs from exact prefix computation')
    # Supervision is constructed separately from the prefix-only writer API.
    answer = row['answer'] + '\n'
    if component == 'writer':
        context, completion = writer_prompt(tree), note
    elif component == 'cot':
        context, completion = writer_prompt(tree), note + 'Answer:\n' + answer
    elif component == 'direct':
        context, completion = direct_prompt(tree), answer
    else:
        context = reader_prompt(tree, note if component == 'reader_teacher' else '')
        completion = answer
    past, future = encode(tokenizer, context), encode(tokenizer, completion)
    if (not past or not future or EOS in past + future
            or encode(tokenizer, context + completion) != past + future):
        raise ValueError('Require exact causal native boundary; never retokenize the writer prefix')
    ids = past + future + [EOS]
    positions = list(range(len(past), len(ids)))
    return packed(ids, positions, note_positions=positions if component == 'writer' else (),
                  width=width, target_slots=target_slots)


def batch_packs(packs):
    if not packs:
        raise ValueError('Nonempty batch required')
    keys = set(packs[0])
    if any(set(p) != keys or any(p[k].shape != packs[0][k].shape for k in keys) for p in packs):
        raise ValueError('Fixed common causal shape required')
    return {k: torch.cat([p[k] for p in packs], dim=0) for k in keys}
