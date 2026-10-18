"""CPU inventory for a CLOSED passing GENERAL native-label publication.

Historical filter hashes resolve at the unchanged MAIN source root, separately
from this worktree's training source closure. No model or training release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from train import general_hindsight_filtered_data as native
from train import useful_note_sft_data as packing
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

SCHEMA = 'general-native-supervised-train-v1'
KIND = 'future-aware-general-firstline-native-v1'
ITEMS = 5120
SHAPES = dict(source=dict(width=544,target_slots=544),writer=dict(width=640,target_slots=128),
              reader=dict(width=704,target_slots=16),augmented=dict(width=704,target_slots=704))


def read(path):
    return json.loads(Path(path).read_text())


def verify_pins(pins):
    for name, digest in pins.items():
        if not Path(name).is_absolute() or sha256_file(name) != digest:
            raise ValueError(f'Immutable absolute provenance changed: {name}')


def selected_rows(rows, decisions, tokenizer):
    """Verify the original first accepted order, never rank/re-filter/replace."""
    if ([d['teacher_index'] for d in decisions] != list(range(native.TEACHER_N))
            or len({d['item_id'] for d in decisions}) != native.TEACHER_N):
        raise ValueError('Complete original teacher decision order required')
    for d in decisions:
        path=d['paths']['I3']; ll=path['absolute_ll']; status=path['statuses']
        positive=status['teacher']==status['empty']=='ok'
        if positive:
            if not all(isinstance(ll[k],(int,float)) and math.isfinite(ll[k]) for k in ('teacher','empty')):
                raise ValueError('Finite original teacher/empty evidence required')
            positive=ll['teacher']>ll['empty']
        if (path['strict_teacher_positive']['empty'] is not positive
                or d['accepted'] is not bool(d['eligible_plain_label']['eligible'] and positive)):
            raise ValueError('Original strict gain decision changed')
    chosen=[d for d in decisions if d['accepted']][:ITEMS]
    if (len(chosen)!=ITEMS or len(rows)!=ITEMS
            or [r['item_id'] for r in rows]!=[d['item_id'] for d in chosen]
            or len({r['document_id'] for r in rows})!=ITEMS):
        raise ValueError('Actual complete first5120 unique-document quota required; no fallback')
    defined=set(tokenizer.get_vocab().values()); result=[]
    for r,d in zip(rows,chosen):
        native.validate_native(r,defined)
        if (r.get('teacher_kind')!=KIND or r.get('selection_uses_likelihood') is not True
                or r['document_id']!=d['document_id'] or r['teacher_index']!=d['teacher_index']
                or r['pool_index']!=d['pool_index'] or r['label_decision']!=d
                or r['original_source_ids']!=r['prefix_ids']+r['natural_future16_ids']
                or not native.label_eligibility(dict(status='ok',content_ids=r['teacher_note_ids']),
                    r['natural_future16_ids'],tokenizer,set(tokenizer.all_special_ids)|{151667,151668})['eligible']
                or native.pilot.decode(tokenizer,r['teacher_note_ids'])!=r['teacher_note_text']):
            raise ValueError('Original native teacher eligibility/source/decision changed')
        result.append(dict(r,native_window_cut=512))
    return result


def closed_publication(directory, historical_root):
    directory=Path(directory).resolve(); historical_root=Path(historical_root).resolve()
    p=read(directory/'protocol.json'); m=read(directory/'training.manifest.json'); s=read(directory/'summary.json')
    required=dict(raw_documents=native.RAW_N,teacher_attempts=native.TEACHER_N,
                  reader_attempts=native.TEACHER_N*8,selected=ITEMS,requested=ITEMS)
    if (p['schema']!=native.SCHEMA or m['schema']!='general-hindsight-native-labels-v1'
            or s['status']!='COMPLETE_FIXED5120_NATIVE_LABELS_NOT_TRAINING_RELEASE'
            or m['status']!=s['status'] or m['items']!=ITEMS or m['selection_uses_likelihood'] is not True
            or any(s['counts'][k]!=v for k,v in required.items())
            or m['protocol_sha256']!=sha256_file(directory/'protocol.json')
            or s['protocol_sha256']!=m['protocol_sha256'] or s['source_files']!=p['source_files']
            or m['source_files']!=p['source_files']):
        raise ValueError('Only the actual fully passing original GENERAL publication is preparable')
    pins={str(historical_root/name):digest for name,digest in p['source_files'].items()}
    if any(Path(name).is_absolute() or '..' in Path(name).parts for name in p['source_files']):
        raise ValueError('Historical source names must resolve inside the explicit original root')
    for name in ('protocol.json','manifest.json','training.manifest.json','summary.json'):
        pins[str(directory/name)]=sha256_file(directory/name)
    pins.update(s['input_hashes'])
    for phase in ('raw','teacher','reader'):
        path=Path(p['control_dir'])/f'{phase}.closure.json'; c=read(path)
        if (c['status']!='COMPLETE_PHYSICALLY_CLOSED' or c['protocol_sha256']!=m['protocol_sha256']
                or not c['all16_exit0'] or not c['all48_owned_PID_start_identities_absent']
                or not c['all_sessions_absent'] or not c['assigned_devices_empty'] or c['owned_group_survivors']):
            raise ValueError('All three original whole physical closures required')
        pins[str(path.resolve())]=sha256_file(path)
    for path,digest in ((Path(m['examples_file']),m['examples_sha256']),
                        (Path(s['decisions_file']),s['decisions_sha256'])):
        if not path.is_absolute():raise ValueError('Original closed artifact paths must be absolute')
        pins[str(path)]=digest
    if (m['examples_file']!=s['training_file'] or m['examples_sha256']!=s['training_sha256']
            or m['decisions_sha256']!=s['decisions_sha256']):
        raise ValueError('Original summary/publication bindings differ')
    verify_pins(pins)
    rows=[json.loads(line) for line in Path(m['examples_file']).read_text().splitlines()]
    decisions=[json.loads(line) for line in Path(s['decisions_file']).read_text().splitlines()]
    if canonical_hash([r['item_id'] for r in rows])!=m['item_ids_sha256']:
        raise ValueError('Original native label order changed')
    if any(r.get('teacher_protocol_sha256')!=m['protocol_sha256'] for r in rows):
        raise ValueError('Original teacher protocol lineage changed')
    return p,rows,decisions,pins


def prepare(directory, historical_root, output, deadline_utc):
    from transformers import AutoTokenizer
    from train.useful_note_sft_driver import source_files
    deadline=require_utc_deadline(deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Finite GENERAL CPU preparation; partials retained')
    check(); output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    p,rows,decisions,pins=closed_publication(directory,historical_root)
    tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False)
    rows=selected_rows(rows,decisions,tok); counts={c:dict(examples=0,active_labels=0,note_close_labels=0,non_note_labels=0) for c in SHAPES}
    frozen=source_files(); output.mkdir(parents=True,exist_ok=False)
    try:
        with (output/'training.jsonl').open('x') as stream:
            for row in rows:
                check()
                for c in SHAPES:
                    b=packing.pack_example(row,tok,c,interface='I3',parts=p['parts'],**SHAPES[c])
                    n=b['note_close_mask'];f=b['future_mask'];active=n|f
                    if ((n&f).any() or not torch.equal(b['labels'][active],b['input_ids'].gather(1,b['target_positions'])[active])
                            or (b['labels'][~active]!=-100).any()):raise ValueError('Native causal labels/masks changed')
                    expected=row['teacher_note_ids']+[151668] if c=='writer' else row['natural_future16_ids'] if c=='reader' else row['original_source_ids'][1:] if c=='source' else b['input_ids'][0,1:1+int(active.sum())].tolist()
                    if b['labels'][active].tolist()!=expected or (c=='writer' and f.any()) or (c!='writer' and n.any()):
                        raise ValueError('Native component target role changed')
                    for k,v in dict(examples=1,active_labels=int(active.sum()),note_close_labels=int(n.sum()),non_note_labels=int(f.sum())).items():counts[c][k]+=v
                stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        check();verify_pins(pins)
        if source_files()!=frozen:raise ValueError('Training source closure changed during preparation')
        manifest=dict(schema=SCHEMA,status='complete_CPU_inventory_NOT_Neuron_training_qualification',purpose='science',split='train',
            items=ITEMS,unique_documents=ITEMS,unique_native_window_cuts=ITEMS,created_utc=utc_now(),teacher_kind=KIND,
            selection_uses_likelihood=True,interface='I3',parts=p['parts'],packing=SHAPES,component_counts=counts,
            ordered_item_ids_sha256=canonical_hash([r['item_id'] for r in rows]),ordered_document_ids_sha256=canonical_hash([r['document_id'] for r in rows]),
            training_file=str(output/'training.jsonl'),training_sha256=sha256_file(output/'training.jsonl'),
            tokenizer_vocab_sha256=canonical_hash(tok.get_vocab()),model_dir=p['model_dir'],writer_stop_ids=[151668,151643],
            reader_target='Original natural16',mask_semantics='Writer note+first-close; reader16; source527; augmented every real token afterfirst.',
            input_hashes=pins,filter_input_hashes=pins,source_files=frozen,
            selection='Exact original first5120 strict I3-positive native eligible labels; no new likelihood ranking or replacement.',
            padding='Reuse qualified544/640/704/704 trunk and544/128/16/704 head shapes; extra padding explicit.',
            limitations=['Future-aware generated labels, not deployed prefix-only utility.','Original teacher>empty filter does not require teacher>raw/other or semantic correctness.','CPU inventory does not release qualification/training.'])
        atomic_json(output/'manifest.json',manifest);return manifest
    except BaseException as exc:
        atomic_json(output/'FAILED.json',dict(status='FAILED_CPU_PREPARATION_PARTIALS_RETAINED',error=repr(exc),finished_utc=utc_now()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('publication','historical-source-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True);a=parser.parse_args()
    m=prepare(a.publication,a.historical_source_root,a.output,a.deadline_utc)
    print(json.dumps(dict(schema=SCHEMA,items=m['items'],packing=m['packing'])))


if __name__=='__main__':main()
