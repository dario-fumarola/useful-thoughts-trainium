"""Matched final-answer coefficients with a separately normalized pointer field.

One native causal sequence: optional evidence IDs, then the answer and EOS.
No intermediate factual assertions or root-answer claims precede the final field.
"""
from train.document_evidence_data import render
from train.document_qa_packing import encode, source_positions
from train.useful_note_sft_data import EOS, packed

ARMS = ('evidence_direct', 'evidence_select')
COMPONENTS = ('source', 'answer', 'selection')
FINAL = 'Final answer:\n'
INSTRUCTION = ('Answer the question using the documents. Write Final answer: '
               'on its own line and the answer on the next line.\n')
DIRECT = 'Give the final answer directly.\nResponse:\n'
SELECT = ('You may first list relevant document numbers in an Evidence: field, '
          'then give the final answer.\nResponse:\n')


def prompt(question, mode):
    """Strict question projection only; annotations cannot enter this API."""
    if mode not in ('direct', 'selection'):
        raise ValueError('Declared direct or selection mode required')
    return render(question) + INSTRUCTION + (DIRECT if mode == 'direct' else SELECT)


def selection_field(indices, paragraphs):
    if (type(paragraphs) is not int or paragraphs < 1
            or not isinstance(indices, list)
            or any(type(value) is not int or not 1 <= value <= paragraphs for value in indices)
            or indices != sorted(set(indices))):
        raise ValueError('Unique ascending one-based visible paragraph IDs required')
    return 'Evidence:' + ''.join(f' [{value}]' for value in indices) + '\n'


def view_for(arm, epoch, item_index):
    if (arm not in ARMS or type(epoch) is not int or epoch < 0
            or type(item_index) is not int or item_index < 0):
        raise ValueError('Declared arm, complete-pass index and fixed row index required')
    if (item_index + epoch) % 2 == 0:
        return 'direct'
    return 'empty' if arm == 'evidence_direct' else 'selected'


def training_pack(row, tokenizer, view, *, width=3328, target_slots=512):
    if row.get('split') != 'train' or view not in ('direct', 'empty', 'selected'):
        raise ValueError('Declared TRAIN row and target view required')
    question, answer = row['question'], row['answer']
    if (row.get('item_id') != question['item_id'] or not isinstance(answer, str)
            or not answer.strip() or '\n' in answer or '\r' in answer):
        raise ValueError('Bound question identity and one complete answer line required')
    opening = prompt(question, 'direct' if view == 'direct' else 'selection')
    source = encode(tokenizer, render(question))
    context = encode(tokenizer, opening)
    if context[:len(source)] != source:
        raise ValueError('Native source boundary changed')
    plan = ''
    if view != 'direct':
        indices = row['support_indices'] if view == 'selected' else []
        if view == 'selected' and not indices:
            raise ValueError('Selected teacher view requires actual support annotations')
        plan = selection_field(indices, len(question['contexts']))
    before_final = encode(tokenizer, opening + plan)
    ids = encode(tokenizer, opening + plan + FINAL + answer)
    final = encode(tokenizer, FINAL + answer)
    if (ids[:len(context)] != context or ids[:len(before_final)] != before_final
            or ids[len(before_final):] != final):
        raise ValueError('Native component boundary changed; never repair tokenization silently')
    ids.append(EOS)
    selected = source_positions(question['item_id'], len(source))
    selection = list(range(len(context), len(before_final)))
    answer_positions = list(range(len(before_final), len(ids)))
    result = packed(ids, selected + selection + answer_positions, note_positions=selected,
                    width=width, target_slots=target_slots)
    result['source_mask'] = result.pop('note_close_mask')
    active_task = result.pop('future_mask')
    result['answer_mask'] = active_task & (result['target_positions'] >= len(before_final))
    result['selection_mask'] = active_task & ~result['answer_mask']
    return result
