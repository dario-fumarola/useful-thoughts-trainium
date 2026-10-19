"""Canonical pointer-replay diagnostics, not native-generation likelihood.

The native answer observations stay unchanged. Invalid/missing generated plans
use an explicitly flagged empty replay;no generated answer is pasted back in.
"""
from train.document_evidence_plan_answers import parse_evidence
from train.document_evidence_plan_packing import prompt, FINAL
from train.useful_note_sft_data import EOS

MODES = ('direct', 'empty', 'selected_id_replay', 'other_id_replay', 'oracle')


def selected_ids(attempt, paragraph_count):
    if attempt.get('status') != 'ok':
        return [], dict(fallback=True, reason='generation_error')
    parsed = parse_evidence(attempt.get('raw_text', ''), paragraph_count)
    if parsed.status != 'complete':
        return [], dict(fallback=True, reason='plan_'+parsed.status, parser_reason=parsed.reason)
    return list(parsed.indices), dict(fallback=False, reason='valid_nonempty_plan')


def context_text(question, mode, indices=None):
    if mode not in MODES:
        raise ValueError('Declared diagnostic condition required')
    if mode == 'direct':
        if indices is not None:
            raise ValueError('No pointer labels in the ordinary direct condition')
        return prompt(question, 'direct') + FINAL
    if mode == 'empty':
        if indices is not None:
            raise ValueError('Empty condition cannot inspect supplied pointer labels')
        indices = []
    if (not isinstance(indices, list) or len(indices) != len(set(indices))
            or any(type(value) is not int or not 1 <= value <= len(question['contexts']) for value in indices)):
        raise ValueError('Complete unique in-range pointers required;never clip/repair')
    field = 'Evidence:' + ''.join(f' [{value}]' for value in indices) + '\n'
    return prompt(question, 'selection') + field + FINAL


def scoring_example(question, answer, mode, tokenizer, indices=None):
    text = context_text(question, mode, indices)
    past = tokenizer.encode(text, add_special_tokens=False)
    joint = tokenizer.encode(text + answer, add_special_tokens=False)
    if joint[:len(past)] != past:
        raise ValueError('Native answer boundary changed;never repair silently')
    target = joint[len(past):] + [EOS]
    if (not answer or not past or not 1 < len(target) <= 32
            or len(past) + len(target) > 3584 or EOS in joint):
        raise ValueError('Complete fixed native span required;never shorten/drop a question')
    return dict(item_id=question['item_id'], panel='QA', mode=mode,
                context_ids=past, target_ids=target,
                interpretation='Canonical pointer-ID replay,not the native generation observation')
