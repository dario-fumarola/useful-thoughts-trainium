"""Build disclosed future-selected TRAIN labels and explicit causal loss masks."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import torch

from train import candidate_corpus as corpus
from train.evaluate_local import load_selected
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format

SCHEMA = 'positive-candidate-warmstart-v1'


def pack_example(row, fmt, arm):
    """Canonical rendered closure is supervision, not a fabricated sampled stop."""
    if row.get('split') != 'train':
        raise ValueError('Warmstart supervision accepts TRAIN only')
    prefix, future, note = row['prefix'], row['future'], row['content_ids']
    if len(prefix) != 512 or len(future) != 16 or len(note) > 256:
        raise ValueError('Require prefix512, future16 and at most256 content tokens')
    if any(type(v) is not int or v < 0 for v in prefix+future+note):
        raise ValueError('Token IDs must be nonnegative integers')
    if any(v in {fmt.stop_id, *fmt.additional_stop_ids} for v in note):
        raise ValueError('Content must exclude sampled terminators')
    layout = row.get('training_layout', 'joint_reanchor32')
    if layout not in {'joint_reanchor32', 'writer_only'}:
        raise ValueError('Unknown warmstart training layout')
    if arm == 'warmstart':
        prompt = list(fmt.header_ids)+prefix+list(fmt.open_ids)
        continuation = note+list(fmt.close_ids)
        note_positions = list(range(len(prompt), len(prompt)+len(continuation)))
        if layout == 'writer_only':
            if len(fmt.close_ids) != 2:
                raise ValueError('Writer-only supervision requires the canonical two-token close')
            ids = prompt+continuation
            positions = note_positions+[0]*(258-len(note_positions))
            note_mask = [True]*len(note_positions)+[False]*(258-len(note_positions))
            future_mask, width = [False]*258, 832
        else:
            ids = prompt+continuation+prefix[-32:]+future
            future_positions = list(range(len(ids)-16, len(ids)))
            positions = note_positions+[0]*(258-len(note_positions))+future_positions
            note_mask = [True]*len(note_positions)+[False]*(274-len(note_positions))
            future_mask = [False]*258+[True]*16
            width = 864
    elif arm == 'raw_control':
        ids = prefix+future
        positions = list(range(512,528))
        note_mask, future_mask, width = [False]*16, [True]*16, 544
    else:
        raise ValueError('Unknown short-training arm')
    if len(ids) > width or (arm == 'warmstart' and len(note_positions) > 258):
        raise ValueError('Rendered example exceeds declared static width/target slots')
    active = [a or b for a,b in zip(note_mask,future_mask)]
    labels = [ids[p] if m else -100 for p,m in zip(positions,active)]
    assert all(p >= 1 for p,m in zip(positions,active) if m)
    return dict(input_ids=torch.tensor([ids+[fmt.pad_id]*(width-len(ids))]),
        target_positions=torch.tensor([positions]), labels=torch.tensor([labels]),
        note_close_mask=torch.tensor([note_mask]), future_mask=torch.tensor([future_mask]))


def choose_candidates(protocol, scores, notes):
    """Best own K16, retained only if it beats empty and all matched controls."""
    if protocol['selection'].get('split') != 'train' or any(r.get('split') != 'train' for r in scores):
        raise ValueError('Candidate selection accepts TRAIN only')
    index = {(r['item_id'],r['interface'],r['mode'],r['candidate_index']):r for r in scores}
    if len(index) != len(scores):
        raise ValueError('Duplicate candidate/control scores')
    if any(r.get('status') != 'ok' or len(r.get('token_log_likelihoods', [])) != 16
           or not all(math.isfinite(v) for v in r['token_log_likelihoods']) for r in scores):
        raise ValueError('Every TRAIN score must be finite; no filtering failures')
    chosen, audit = [], []
    for key in protocol['selection']['item_ids']:
        def value(mode, i=None, interface='reanchor32'):
            row = index[key,interface,mode,i]
            values = row['token_log_likelihoods']
            if row['status'] != 'ok' or len(values) != 16 or not all(math.isfinite(v) for v in values):
                raise ValueError('Every TRAIN score must be finite; no filtering failures')
            return math.fsum(values)
        winner = max(range(16), key=lambda i:value('own',i))
        own = value('own',winner)
        contrasts = {f'own_minus_{mode}': own-value(mode,None if mode=='empty' else winner)
                     for mode in ('empty','filler','shuffled','other')}
        contrasts['own_minus_raw'] = own-value('raw',None,'raw')
        contrasts['own_minus_raw_repeat32'] = own-value('repeat32',None,'raw')
        nonempty = bool(notes[key,winner]['content_ids'])
        keep = nonempty and all(contrasts[f'own_minus_{m}'] > 0 for m in ('empty','filler','shuffled','other'))
        audit.append(dict(item_id=key,candidate_index=winner,nonempty=nonempty,selected=keep,
            own_K16_log_likelihood=own,contrasts=contrasts))
        if keep:
            chosen.append((key,winner))
    return chosen,audit


def prepare(directory, corpus_directory):
    p,fmt = corpus.read_protocol(corpus_directory)
    notes,_,barrier = corpus.global_inputs(corpus_directory,p,fmt)
    summary = json.loads((corpus_directory/'summary.json').read_text())
    protocol_sha = sha256_file(corpus_directory/'protocol.json')
    barrier_sha = sha256_file(corpus_directory/'global/COMPLETE.json')
    if (summary['status'] != 'complete' or summary['failures'] or summary['rows'] != 17024
            or summary.get('protocol_sha256') != protocol_sha
            or sorted(r['shard'] for r in summary['scoring']) != list(range(8))):
        raise ValueError('Require complete all128/G16 TRAIN scores')
    scores = []
    for receipt in summary['scoring']:
        d = corpus_directory/'scoring'/f"shard-{receipt['shard']:02d}"
        for filename,key in [('run.json','run_sha256'),('scores.jsonl','scores_sha256')]:
            if sha256_file(d/filename) != receipt[key]:
                raise ValueError('Changed completed scoring shard')
        run = json.loads((d/'run.json').read_text())
        if (run.get('status') != 'complete' or run.get('schema') != corpus.SCHEMA
                or run.get('shard') != receipt['shard'] or run.get('protocol_sha256') != protocol_sha
                or run.get('barrier_sha256') != barrier_sha or run.get('model') != barrier['model']
                or run.get('checkpoint') is not None):
            raise ValueError('Completed scores belong to another corpus/model/barrier')
        scores.extend(json.loads(line) for line in (d/'scores.jsonl').read_text().splitlines())
    checked = corpus.reduce_rows(p,scores)  # Full unique document/control/candidate coverage.
    if checked['failures'] or checked['rows'] != 17024:
        raise ValueError('No incomplete or failed candidate supervision')
    selected,audit = choose_candidates(p,scores,notes)
    if not selected:
        raise ValueError('No candidates meet the declared rule; no fallback dataset')
    records = {r['item_id']:r for r in load_selected(Path(p['selection']['records_file']),
        p['selection'],include_future=True)}
    rows = []
    for key,i in selected:
        record,note = records[key],notes[key,i]
        if canonical_hash(record['prefix']) != note['prefix_sha256']:
            raise ValueError('Training prefix differs from sampled writer input')
        row = dict(item_id=key,document_id=record['document_id'],split='train',
            prefix=record['prefix'],future=record['future'],content_ids=note['content_ids'],
            source_candidate_index=i,sampled_stop_type=note['stop_type'],
            sampled_actions=note['actions'],closure='Canonical rendered close, including newline; synthetic for cap/EOS-ended samples')
        for arm in ('warmstart','raw_control'):
            pack_example(row,fmt,arm)
        rows.append(row)
    directory.mkdir(parents=True,exist_ok=False)
    with (directory/'examples.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    atomic_json(directory/'selection-audit.json',dict(schema=SCHEMA,items=len(audit),
        selected=len(selected),rule='At each TRAIN anchor choose highest own K16 among16; retain nonempty only when strictly above same-format empty and same-index filler/shuffle/other. No raw-win requirement; raw deltas retained.',rows=audit))
    atomic_json(directory/'dataset.json',dict(schema=SCHEMA,created_utc=utc_now(),split='train',
        items=len(rows),item_ids_sha256=canonical_hash([r['item_id'] for r in rows]),
        examples_sha256=sha256_file(directory/'examples.jsonl'),
        audit_sha256=sha256_file(directory/'selection-audit.json'),
        corpus_protocol_sha256=sha256_file(corpus_directory/'protocol.json'),
        corpus_summary_sha256=sha256_file(corpus_directory/'summary.json'),
        corpus_barrier_sha256=sha256_file(corpus_directory/'global/COMPLETE.json'),
        source_sha256=sha256_file(__file__),
        supervision='TRAIN future-selected behavior-cloning content, canonical closure and16 true future targets. A later student receives prefix only when generating.',
        limitations=['Selection uses observed TRAIN continuations and matched controls.',
            'Positive selected-label likelihood does not imply fresh policy utility.',
            'Raw control receives exactly the same selected TRAIN documents/futures/order.']))
    print(json.dumps(dict(dataset=str(directory),items=len(rows))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,required=True)
    args = parser.parse_args()
    prepare(args.output,args.corpus)
