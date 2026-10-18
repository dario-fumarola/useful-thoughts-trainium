"""CPU packing only: causal intermediate-note BC and independent full-source CE.

The future treatment accumulates one mean note loss plus one mean source loss;
its control uses the identical source pass alone. No selection/order/optimizer
or training launch is implemented here. ``future_mask`` is the existing CE
helper's second component name; source_only supervises ALL527 source targets.
"""
from __future__ import annotations

import torch

SCHEMA = 'calculation-note-and-fullsource-packer-v1'
NOTE_WIDTH, NOTE_CAP, NOTE_SLOTS = 832, 128, 129
SOURCE_WIDTH, SOURCE_LENGTH, PAD = 544, 528, 151643
INSTRUCTION = (
    'Write one compact useful intermediate calculation for the unfinished calculation in the text. '
    'Give only the intermediate calculation, not the final result of the unfinished calculation.\n'
    'Text:\n(17+28)*9 =\nCalculation:\nCompute 17+28 = 45.\n\n'
    'Text:\n(36/4)+23 =\nCalculation:\nCompute 36/4 = 9.\n\n'
    'Text:\n'
)
SUFFIX = '\nCalculation:\n'


def _ids(values, defined):
    if not isinstance(values,list) or any(type(v) is not int or v not in defined for v in values):
        raise ValueError('Only tokenizer-defined integer IDs are supported')
    return list(values)


def _encode(text, tokenizer, defined):
    if not isinstance(text,str): raise ValueError('Native text string required')
    ids=_ids(tokenizer.encode(text,add_special_tokens=False),defined)
    if tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=text:
        raise ValueError('Native text/token roundtrip required; no normalization or truncation')
    return ids


def writer_prompt(prefix_text, tokenizer):
    """The exact same prefix-only builder must be used by future generation."""
    if not isinstance(prefix_text,str) or not prefix_text.rstrip().endswith('='):
        raise ValueError('Causal unfinished calculation prefix ending in equals required')
    return _encode(INSTRUCTION+prefix_text+SUFFIX,tokenizer,set(tokenizer.get_vocab().values()))


def pack_example(row, tokenizer, component):
    """Return CPU [1,S] inputs and [1,T] target positions/labels/bool masks.

    Note-only reads the causal prefix and externally verified proper-intermediate
    note, never the original source suffix, final RHS or its value. Its canonical
    newline is encoded WITH note text, so fused punctuation/newline stays intact.
    Source-only ignores notes/prompts and supervises original source tokens1..527.
    The curator publication/proper-subexpression proof must be pinned separately
    by a later dataset preparation; this packer does not select or reverify RHSs.
    """
    if row.get('split')!='train': raise ValueError('Arithmetic supervision accepts TRAIN only')
    if component not in ('note_only','source_only'): raise ValueError('Unknown CE component')
    defined=set(tokenizer.get_vocab().values())
    if tokenizer.eos_token_id!=PAD or PAD not in defined: raise ValueError('Native B0 EOS/padding required')
    if component=='source_only':
        ids=_ids(row['ordinary_source_ids'],defined)
        if len(ids)!=SOURCE_LENGTH: raise ValueError('Exactly the original528 source tokens required')
        positions=list(range(1,SOURCE_LENGTH)); labels=ids[1:]
        note_mask,source_mask=[False]*527,[True]*527
        width=SOURCE_WIDTH
    else:
        prefix=row['prefix_text']; note=row['note_text']; proof=row['boundary']['note']
        prefix_ids=_encode(prefix,tokenizer,defined); content=_encode(note,tokenizer,defined)
        if (not 128<=len(prefix_ids)<=512 or prefix_ids!=row['prefix_ids']
                or not note.strip() or len(content)>NOTE_CAP or PAD in content
                or note!=proof['correct_text'] or content!=proof['correct_ids'] or content!=row['note_ids']):
            raise ValueError('Exact causal prefix and bounded externally verified intermediate note required')
        prompt=writer_prompt(prefix,tokenizer); continuation=_encode(note+'\n',tokenizer,defined)
        if not continuation or len(continuation)>NOTE_SLOTS or PAD in continuation:
            raise ValueError('Canonical native newline closure exceeds129 target slots or contains EOS')
        ids=prompt+continuation; active=len(continuation); positions=list(range(len(prompt),len(ids)))
        labels=continuation+[-100]*(NOTE_SLOTS-active); positions += [0]*(NOTE_SLOTS-active)
        note_mask,source_mask=[True]*active+[False]*(NOTE_SLOTS-active),[False]*NOTE_SLOTS
        width=NOTE_WIDTH
    if len(ids)>width: raise ValueError('Packed example exceeds static width; never truncate')
    return dict(input_ids=torch.tensor([ids+[PAD]*(width-len(ids))],dtype=torch.long),
        target_positions=torch.tensor([positions],dtype=torch.long),labels=torch.tensor([labels],dtype=torch.long),
        note_close_mask=torch.tensor([note_mask],dtype=torch.bool),future_mask=torch.tensor([source_mask],dtype=torch.bool))
