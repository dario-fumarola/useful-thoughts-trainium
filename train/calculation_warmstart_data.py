"""CPU packing/preparation: note BC, near-LHS reader CE and full-source CE.

Separate component means/weights belong to the trainer protocol. No optimizer
or training launch is implemented here. ``future_mask`` is the existing CE
helper's second component name; source_only supervises ALL527 source targets,
while reader_only supervises exactly16 natural continuation targets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

SCHEMA = 'calculation-note-reader-and-fullsource-packer-v2'
NOTE_WIDTH, NOTE_CAP, NOTE_SLOTS = 832, 128, 129
SOURCE_WIDTH, SOURCE_LENGTH, PAD = 544, 528, 151643
READER_WIDTH, READER_TARGETS = 1024, 16
COMPONENTS = ('source_only', 'note_only', 'reader_only')
TRAIN_ITEMS = 512
PROSPECTIVE_IDS_SHA256 = 'f6005e96e8ad246ea78fd00cb282aacc0e7d4cd5be57ee46908b4ee502973f9b'
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


def reader_context(row, tokenizer):
    """Exact correct-note near-LHS context; no future field is inspected here."""
    from train.calculation_boundary_probe import exact_lhs_reanchor
    defined=set(tokenizer.get_vocab().values())
    past=_encode(row['prefix_text'],tokenizer,defined); note=_encode(row['note_text'],tokenizer,defined)
    proof=row['boundary']['note']
    if (not 128<=len(past)<=512 or past!=row['prefix_ids'] or past!=row['writer_prefix_ids']
            or row['prefix_text']!=row['writer_prefix_text'] or not note or len(note)>NOTE_CAP or PAD in note
            or row['note_text']!=proof['correct_text'] or note!=proof['correct_ids'] or note!=row['note_ids']
            or _encode('\n\n',tokenizer,defined)!=[271]):
        raise ValueError('Exact native past/note and qualified separator required')
    _,anchor=exact_lhs_reanchor(row,tokenizer)
    return past+[271]+note+[271]+_ids(anchor,defined)


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
    if component not in COMPONENTS: raise ValueError('Unknown CE component')
    defined=set(tokenizer.get_vocab().values())
    if tokenizer.eos_token_id!=PAD or PAD not in defined: raise ValueError('Native B0 EOS/padding required')
    if component=='source_only':
        ids=_ids(row['ordinary_source_ids'],defined)
        if len(ids)!=SOURCE_LENGTH: raise ValueError('Exactly the original528 source tokens required')
        positions=list(range(1,SOURCE_LENGTH)); labels=ids[1:]
        note_mask,source_mask=[False]*527,[True]*527
        width=SOURCE_WIDTH
    elif component=='reader_only':
        context=reader_context(row,tokenizer)
        target=_ids(row['natural_future16_ids'],defined); original=_ids(row['ordinary_source_ids'],defined)
        cut=len(row['prefix_ids'])
        if (len(target)!=READER_TARGETS or len(original)!=SOURCE_LENGTH
                or original[:cut]!=row['prefix_ids'] or target!=original[cut:cut+READER_TARGETS]):
            raise ValueError('Exactly the original natural16 continuation at this causal cut required')
        ids=context+target;positions=list(range(len(context),len(ids)));labels=target
        note_mask,source_mask=[False]*READER_TARGETS,[True]*READER_TARGETS
        width=READER_WIDTH
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


def preparation_sources():
    from train.calculation_boundary_probe import sources
    return {**sources(),'train/calculation_warmstart_data.py':sha256_file(__file__)}


def prepare_training512(directory, prospective_ids, tokenizer_dir, output, deadline_utc):
    """CPU only. Preserve all512 accepted rows/order; never filter or replace."""
    from transformers import AutoTokenizer
    root=Path(__file__).resolve().parents[1]
    directory=Path(directory).resolve();prospective_ids=Path(prospective_ids).resolve()
    tokenizer_dir=Path(tokenizer_dir).resolve();output=Path(output).resolve()
    deadline=require_utc_deadline(deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Finite CPU preparation deadline; no retry')
    check()
    if sha256_file(prospective_ids)!=PROSPECTIVE_IDS_SHA256:raise ValueError('Exact pre-outcome512 ID publication required')
    planned=json.loads(prospective_ids.read_text());manifest_file=directory/'manifest.json';examples_file=directory/'accepted.jsonl'
    manifest=json.loads(manifest_file.read_text());ids=planned['ordered_item_ids']
    if (planned['status']!='SEALED_PROSPECTIVE_IDS_NOT_TRAINING_RELEASE' or planned['count']!=TRAIN_ITEMS
            or planned['split']!='train' or len(ids)!=TRAIN_ITEMS or len(set(ids))!=TRAIN_ITEMS
            or planned['ordered_item_ids_sha256']!=canonical_hash(ids)
            or Path(planned['manifest_file']).resolve()!=manifest_file or sha256_file(manifest_file)!=planned['manifest_sha256']
            or Path(planned['accepted_file']).resolve()!=examples_file or sha256_file(examples_file)!=planned['accepted_sha256']
            or manifest['accepted_sha256']!=planned['accepted_sha256']
            or manifest['schema']!='finemath-arithmetic-boundaries-train-dev-v2'
            or manifest['source_files']!=planned['curator_source_files']
            or manifest['source_sha256']!=planned['source_sha256'] or manifest['source_file']!=planned['source_file']
            or ids!=(manifest['selection']['train']['selected_item_ids']+manifest['selection']['train']['reserve_item_ids'])[:TRAIN_ITEMS]):
        raise ValueError('Immutable curator/source/accepted hash order changed')
    check()
    if sha256_file(manifest['source_file'])!=manifest['source_sha256']:raise ValueError('Original audited source changed')
    for name,digest in manifest['source_files'].items():
        if sha256_file(root/name)!=digest:raise ValueError('Curator source changed')
    check();tok=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True,trust_remote_code=False)
    if canonical_hash(tok.get_vocab())!=manifest['tokenizer_vocab_sha256']:raise ValueError('Native tokenizer differs')
    source_files=preparation_sources();by_id={}
    with examples_file.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['item_id'] in by_id:raise ValueError('Duplicate accepted ID')
            by_id[row['item_id']]=(row,line)
    if any(i not in by_id for i in ids):raise ValueError('Missing sealed training item; no replacement')
    rows=[by_id[i][0] for i in ids];docs=[r['document_id'] for r in rows]
    if (any(r['split']!='train' for r in rows) or len(set(docs))!=TRAIN_ITEMS
            or docs!=planned['ordered_document_ids'] or canonical_hash(docs)!=planned['ordered_document_ids_sha256']):
        raise ValueError('Exact unique TRAIN document inventory required')
    check();output.mkdir(parents=True,exist_ok=False)
    counts={c:dict(examples=0,active_labels=0,note_close_labels=0,future_labels=0) for c in COMPONENTS}
    try:
        with (output/'examples.jsonl').open('x') as stream:
            for row in rows:
                check()
                for component in COMPONENTS:
                    batch=pack_example(row,tok,component);note=batch['note_close_mask'];future=batch['future_mask'];active=note|future
                    if ((note&future).any() or not active.any() or (batch['labels'][~active]!=-100).any()
                            or not torch.equal(batch['labels'][active],batch['input_ids'].gather(1,batch['target_positions'])[active])):
                        raise ValueError('Invalid causal packing metadata')
                    c=counts[component];c['examples']+=1;c['active_labels']+=int(active.sum())
                    c['note_close_labels']+=int(note.sum());c['future_labels']+=int(future.sum())
                stream.write(by_id[row['item_id']][1])  # Original accepted-row bytes; new order only.
        check()
        if preparation_sources()!=source_files or sha256_file(prospective_ids)!=PROSPECTIVE_IDS_SHA256:
            raise ValueError('Preparation source or sealed inventory changed during packing')
        result=dict(schema='calculation-joint-warmstart-training512-v1',status='complete_CPU_preparation_not_training_release',
            created_utc=utc_now(),deadline_utc=deadline_utc,split='train',items=TRAIN_ITEMS,
            examples_file=str(output/'examples.jsonl'),examples_sha256=sha256_file(output/'examples.jsonl'),item_ids_sha256=canonical_hash(ids),
            prospective_ids_file=str(prospective_ids),prospective_ids_sha256=PROSPECTIVE_IDS_SHA256,
            original_dataset_manifest_file=str(manifest_file),original_dataset_manifest_sha256=planned['manifest_sha256'],
            original_accepted_file=str(examples_file),original_accepted_sha256=planned['accepted_sha256'],
            original_source_file=manifest['source_file'],original_source_sha256=manifest['source_sha256'],
            curator_source_files=manifest['source_files'],source_files=source_files,packer_schema=SCHEMA,
            tokenizer_dir=str(tokenizer_dir),tokenizer_vocab_sha256=manifest['tokenizer_vocab_sha256'],
            components=list(COMPONENTS),component_counts=counts,component_label_totals={c:v['active_labels'] for c,v in counts.items()},
            unchanged_full_accepted_rows=True,no_filter_or_replacement=True,
            limitations=['External verified arithmetic supervision; no spontaneous reasoning or training benefit established.',
                'TRAIN only in pre-outcome sealed order; no DEV or official TEST used for training preparation.',
                'Component means/weights,128updates/global8/twoepochs and actual launch belong to a separate reviewed trainer protocol.'])
        atomic_json(output/'manifest.json',result);return result
    except BaseException as error:
        atomic_json(output/'preparation.failed.json',dict(status='partial_not_ready',error=f'{type(error).__name__}: {error}',component_counts=counts))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare',))
    for name in ('dataset-dir','prospective-ids','tokenizer-dir','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True);a=parser.parse_args()
    prepare_training512(a.dataset_dir,a.prospective_ids,a.tokenizer_dir,a.output,a.deadline_utc)


if __name__=='__main__':main()
