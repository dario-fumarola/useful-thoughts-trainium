"""Single-model direct, prior-trace and post-answer-trace causal targets.

    0.5 mean CE on a fixed shared source sample + 0.5 mean task CE.
This is an explicitly sampled source objective, not full-document CE. Every
second pass of each augmented arm uses the SAME ordinary answer path, so a
plain answer path is trained rather than assumed to survive scaffold removal.
"""
import hashlib
import random

from train.document_evidence_data import render
from train.useful_note_sft_data import EOS, packed

ARMS = ('qa_direct', 'qa_prior', 'qa_post')
DIRECT_OPEN = 'Answer:\n'
PRIOR_OPEN = 'Reasoning:\n'
FINAL_OPEN = '\nFinal answer:\n'
POST_OPEN = '\nPost-answer reasoning:\n'
SOURCE_LABELS = 256


def prompt(question, mode):
    if mode not in ('direct', 'prior'):
        raise ValueError('A prefix-only direct or prior prompt is required')
    return render(question) + (DIRECT_OPEN if mode == 'direct' else PRIOR_OPEN)


def encode(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if (not ids or EOS in ids or tokenizer.eos_token_id != EOS
            or tokenizer.decode(ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) != text):
        raise ValueError('Exact native non-EOS fragment encoding required')
    return ids


def source_positions(item_id, source_length):
    if source_length < 2:
        raise ValueError('Source needs causal labels')
    seed = int.from_bytes(hashlib.sha256(('document-qa-v21-source:'+item_id).encode()).digest(), 'big')
    return sorted(random.Random(seed).sample(range(1, source_length), min(SOURCE_LABELS, source_length-1)))


def view_for(arm, epoch):
    if arm not in ARMS or type(epoch) is not int or epoch < 0:
        raise ValueError('Declared arm and complete-pass index required')
    return 'direct' if arm == 'qa_direct' or epoch % 2 == 0 else arm[3:]


def training_pack(row, tokenizer, view, *, width=3328, target_slots=512):
    if row.get('split') != 'train' or view not in ('direct', 'prior', 'post'):
        raise ValueError('Declared TRAIN row and supervised view required')
    # Labels may appear in supervised teacher-forced contexts, never in prompt().
    question, answer, trace = row['question'], row['answer'], row['trace']
    source = encode(tokenizer, render(question))
    opening = prompt(question, 'prior' if view == 'prior' else 'direct')
    context = encode(tokenizer, opening)
    if context[:len(source)] != source:
        raise ValueError('Native source prefix boundary changed')
    target = trace + FINAL_OPEN + answer if view == 'prior' else answer
    ids = encode(tokenizer, opening + target)
    if ids[:len(context)] != context:
        raise ValueError('Native target boundary changed; never repair tokenization silently')
    ids.append(EOS)
    if view == 'post':
        # The answer's EOS is SUPERVISED before the after-answer scaffold.
        # Ordinary inference stops there; no post-answer work is required.
        ids += encode(tokenizer, POST_OPEN + trace) + [EOS]
    task = list(range(len(context), len(ids)))
    selected = source_positions(question['item_id'], len(source))
    return packed(ids, selected+task, note_positions=selected,
                  width=width, target_slots=target_slots)
