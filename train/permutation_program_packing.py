"""Matched answer supervision with either generated states or trained pauses.

Pause positions are supplied inputs, never supervised output tokens. Their
number depends only on the available program length. This is a proposed CPU
packing contract, not a qualified accelerator training recipe.
"""
from train.document_qa_packing import encode
from train.permutation_program_data import (Question, execute, question_from_dict,
                                            render, trace)
from train.useful_note_sft_data import EOS, packed

ARMS = ('direct', 'trace', 'pause')
VIEWS = ('direct', 'empty', 'trace', 'pause')
FINAL = 'Final answer:\n'
WORK = 'Work:\n'
PAUSE_TEXT = '<|fim_pad|>'
PAUSES_PER_STEP = 6


def pause_id(tokenizer):
    value = tokenizer.convert_tokens_to_ids(PAUSE_TEXT)
    if (type(value) is not int or value == EOS
            or tokenizer.convert_ids_to_tokens(value) != PAUSE_TEXT):
        raise ValueError('A supported, distinct existing FIM-pad embedding is required')
    return value


def view_for(arm, epoch, item_index):
    if (arm not in ARMS or type(epoch) is not int or epoch < 0
            or type(item_index) is not int or item_index < 0):
        raise ValueError('Declared arm, epoch and row index required')
    if (epoch + item_index) % 2 == 0:
        return 'direct'
    return 'empty' if arm == 'direct' else arm


def inference_context(question, tokenizer, mode):
    if type(question) is not Question or mode not in VIEWS:
        raise ValueError('Only the available-input Question and a declared mode are accepted')
    text = render(question)
    if mode == 'direct':
        return encode(tokenizer, text + FINAL)
    prefix = encode(tokenizer, text + WORK)
    if mode == 'trace':
        return prefix
    if mode == 'pause':
        prefix += [pause_id(tokenizer)] * (PAUSES_PER_STEP * len(question.program))
    return prefix + encode(tokenizer, FINAL)


def training_pack(row, tokenizer, view, *, width=256, target_slots=64):
    if row.get('split') != 'TRAIN' or view not in VIEWS:
        raise ValueError('Declared TRAIN supervision and view required')
    q = question_from_dict(row['question'])
    answer = 'ABCDEFGHIJKLMNOP'[execute(q)[-1]]
    if row['answer'] != answer or row['full_trace'] != trace(q):
        raise ValueError('Supervision disagrees with the exact prefix-computable execution')
    text = render(q)
    prefix = encode(tokenizer, text if view == 'direct' else text + WORK)
    context = list(prefix)
    work_positions = []
    if view == 'trace':
        full_before_final = encode(tokenizer, text + WORK + row['full_trace'])
        if full_before_final[:len(prefix)] != prefix:
            raise ValueError('Native Work-to-trace token boundary changed')
        context = full_before_final
        work_positions = list(range(len(prefix), len(context)))
    elif view == 'pause':
        context += [pause_id(tokenizer)] * (PAUSES_PER_STEP * len(q.program))
    # The Final field is identically weighted and serialized in every view.
    answer_ids = encode(tokenizer, FINAL + answer) + [EOS]
    if view != 'pause':
        native = encode(tokenizer, text + ('' if view == 'direct' else WORK)
                        + (row['full_trace'] if view == 'trace' else '') + FINAL + answer)
        if native != context + answer_ids[:-1]:
            raise ValueError('Native final-answer boundary changed; never silently retokenize targets')
    answer_positions = list(range(len(context), len(context) + len(answer_ids)))
    result = packed(context + answer_ids, work_positions + answer_positions,
                    note_positions=work_positions, width=width, target_slots=target_slots)
    result['work_mask'] = result.pop('note_close_mask')
    result['answer_mask'] = result.pop('future_mask')
    return result
