"""Prepare future-selected TRAIN notes for writer-only behavior cloning.

Highest own K16 wins before filtering. It must be nonempty and strictly beat
raw, same-interface empty and same-index content controls, with no fallback.
Futures remain disclosed selection/raw-control provenance, never writer input.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from train import candidate_prepend as prepend, warmstart_data
from train.evaluate_local import load_selected
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now

SCHEMA = 'positive-prepend-writer-train128-v1'
LAYOUT = 'writer_only'
READER = 'prepend_plain'
COMPARATORS = ('raw','empty','filler','shuffled','other')


def sources():
    return {**prepend.sources(), 'train/positive_writer_data.py':sha256_file(__file__),
            'train/warmstart_data.py':sha256_file(warmstart_data.__file__)}


def choose_candidates(p,scores,notes):
    ids=p['selection']['item_ids']
    if p['selection'].get('split')!='train' or any(r.get('split')!='train' for r in scores):
        raise ValueError('Positive writer labels accept TRAIN only')
    index={(r['item_id'],r['interface'],r['mode'],r['candidate_index']):r for r in scores}
    expected={(k,'raw',m,None) for k in ids for m in ('raw','repeat32','repeat64')}
    expected|={(k,READER,'empty',None) for k in ids}
    expected|={(k,READER,m,i) for k in ids for m in ('own','filler','shuffled','other') for i in range(16)}
    if len(index)!=len(scores) or set(index)!=expected:
        raise ValueError('Require every unique TRAIN document/candidate/control score')
    if any(r.get('status')!='ok' or len(r.get('token_log_likelihoods',[]))!=16
           or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in r['token_log_likelihoods']) for r in scores):
        raise ValueError('No failed/nonfinite candidate or control may be filtered')
    documents=dict(zip(ids,p['selection']['document_ids']))
    if any(r['document_id']!=documents[r['item_id']] for r in scores):
        raise ValueError('Score document identity mismatch')
    selected,audit=[],[]
    for key in ids:
        def value(mode,i=None,interface=READER):
            return math.fsum(index[key,interface,mode,i]['token_log_likelihoods'])
        winner=max(range(16),key=lambda i:value('own',i))  # Lowest index breaks exact ties.
        own=value('own',winner)
        baselines=dict(raw=value('raw',None,'raw'),empty=value('empty'),
            **{mode:value(mode,winner) for mode in ('filler','shuffled','other')})
        contrasts={f'own_minus_{mode}':own-baselines[mode] for mode in COMPARATORS}
        contrasts.update(own_minus_raw_repeat32=own-value('repeat32',None,'raw'),
                         own_minus_raw_repeat64=own-value('repeat64',None,'raw'))
        nonempty=bool(notes[key,winner]['content_ids'])
        passed={mode:contrasts[f'own_minus_{mode}']>0 for mode in COMPARATORS}
        keep=nonempty and all(passed.values())
        reasons=([] if nonempty else ['empty_highest_own_candidate'])+[f'not_strictly_above_{mode}' for mode,ok in passed.items() if not ok]
        audit.append(dict(item_id=key,document_id=documents[key],source_candidate_index=winner,
            utility_reader=READER,nonempty=nonempty,selected=keep,own_K16_log_likelihood=own,
            baselines_K16=baselines,contrasts=contrasts,strict_comparator_pass=passed,
            rejection_reasons=reasons,source_candidate_sha256=canonical_hash(notes[key,winner])))
        if keep:selected.append((key,winner))
    return selected,audit


def completed_scores(directory):
    p,fmt=prepend.read_protocol(directory)
    _,notes,_,barrier=prepend.inputs(p)
    summary=json.loads((directory/'summary.json').read_text())
    if (summary['schema']!=prepend.SCHEMA or summary['split']!='train' or summary['status']!='complete'
            or summary['failures']!=0 or summary['rows']!=8704
            or summary['protocol_sha256']!=sha256_file(directory/'protocol.json')
            or summary['corpus_summary_sha256']!=p['hashes']['corpus_summary']
            or [r['shard'] for r in summary['scoring']]!=list(range(8))):
        raise ValueError('All completed prepend TRAIN scoring shards are required')
    scores=[]
    for receipt in summary['scoring']:
        shard=receipt['shard'];d=directory/'scoring'/f'shard-{shard:02d}'
        if (sha256_file(d/'run.json')!=receipt['run_sha256'] or sha256_file(d/'scores.jsonl')!=receipt['scores_sha256']):
            raise ValueError('Changed completed prepend score payload')
        run=json.loads((d/'run.json').read_text())
        if (run['schema']!=prepend.SCHEMA or run['status']!='complete' or run['shard']!=shard
                or run['protocol_sha256']!=sha256_file(directory/'protocol.json')
                or run['corpus_barrier_sha256']!=p['hashes']['corpus_barrier']
                or run['model']!=barrier['model'] or run['checkpoint'] is not None
                or run['rows']!=1088 or run['raw_reference_rows']!=48
                or not 0<=run['raw_reference_max_token_error']<=.002):
            raise ValueError('Prepend score belongs to another source/model/barrier')
        local=[json.loads(line) for line in (d/'scores.jsonl').read_text().splitlines()]
        expected=set(prepend.corpus.shard_selection(p['selection'],shard)['item_ids'])
        if len(local)!=1088 or {r['item_id'] for r in local}!=expected or any(r['shard']!=shard for r in local):
            raise ValueError('Completed score shard coverage mismatch')
        scores.extend(local)
    return p,fmt,notes,scores,summary,barrier


def prepare(output,prepend_directory):
    p,fmt,notes,scores,summary,barrier=completed_scores(prepend_directory)
    selected,audit=choose_candidates(p,scores,notes)
    if len(audit)!=128:raise ValueError('All128 selection decisions are required')
    if not selected:raise ValueError('No notes meet the fixed positive rule; no fallback dataset')
    records={r['item_id']:r for r in load_selected(Path(p['selection']['records_file']),p['selection'],include_future=True)}
    rows=[]
    for key,i in selected:
        record,note=records[key],notes[key,i]
        if canonical_hash(record['prefix'])!=note['prefix_sha256'] or record['document_id']!=note['document_id']:
            raise ValueError('Selected TRAIN record differs from original writer prefix/document')
        row=dict(item_id=key,document_id=record['document_id'],split='train',training_layout=LAYOUT,
            utility_reader=READER,prefix=record['prefix'],future=record['future'],content_ids=note['content_ids'],
            source_candidate_index=i,source_candidate_sha256=canonical_hash(note),
            sampled_actions=note['actions'],sampled_behavior_logps=note['behavior_logps'],sampled_action_mask=note['action_mask'],
            sampled_stop_type=note['stop_type'],sampled_decode_iterations=note['decode_iterations'],
            sampled_group=note['group'],sampled_group_seed=note['group_seed'],sampled_prefix_sha256=note['prefix_sha256'],
            canonical_close_ids=list(fmt.close_ids),
            closure='Both canonical close tokens are synthetic rendered supervision; sampled cap/EOS/stop provenance is retained separately',
            future_labels_used_for_selection=True,future_tokens_used_for_writer_ce=False)
        for arm in ('warmstart','raw_control'):warmstart_data.pack_example(row,fmt,arm)
        rows.append(row)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'examples.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    rule='At each TRAIN document choose highest own K16 among16 (math.fsum; exact ties lowest index), then retain nonempty ONLY if strictly above raw, same prepend-empty, same-index filler/shuffled/other. Zero margin; no fallback candidate.'
    atomic_json(output/'selection-audit.json',dict(schema=SCHEMA,items=128,selected=len(rows),utility_reader=READER,training_layout=LAYOUT,rule=rule,rows=audit))
    source_files=sources()
    atomic_json(output/'dataset.json',dict(schema=SCHEMA,created_utc=utc_now(),split='train',training_layout=LAYOUT,
        utility_reader=READER,items=len(rows),item_ids_sha256=canonical_hash([r['item_id'] for r in rows]),
        examples_sha256=sha256_file(output/'examples.jsonl'),audit_sha256=sha256_file(output/'selection-audit.json'),
        source_sha256=sha256_file(__file__),selector_source='train/positive_writer_data.py',selector_source_sha256=sha256_file(__file__),
        packer_source='train/warmstart_data.py',packer_source_sha256=sha256_file(warmstart_data.__file__),source_files=source_files,
        prepend_protocol_sha256=sha256_file(prepend_directory/'protocol.json'),prepend_summary_sha256=sha256_file(prepend_directory/'summary.json'),
        prepend_scoring=summary['scoring'],corpus_protocol_sha256=p['hashes']['corpus_protocol'],
        corpus_summary_sha256=p['hashes']['corpus_summary'],corpus_barrier_sha256=p['hashes']['corpus_barrier'],
        corpus_generation=barrier['generation'],corpus_controls_sha256=p['hashes']['controls'],
        model=barrier['model'],model_manifest_sha256=p['model_manifest_sha256'],thought_format=p['thought_format'],
        rule=rule,writer_input='header+prefix512+open+selected_note+canonical_close2 ONLY; padding follows, no reanchor/future',
        writer_width=832,writer_target_slots=258,writer_future_supervision_tokens=0,
        raw_control_input='prefix512+future16 only, same selected IDs/order',raw_control_width=544,raw_control_target_slots=16,
        future_labels_used_for_selection=True,future_tokens_used_for_writer_ce=False,canonical_close_tokens_are_synthetic=True,
        supervision='Future-selected behavior cloning of note+canonical close only; future16 retained for disclosed selection and raw_control CE.',
        limitations=['TRAIN future-selected labels are not deployed selection or fresh policy utility.',
            'The original31-label joint_reanchor32 artifact is preserved; this is a new layout and stricter RAW-win rule.',
            'Raw control receives the identical selected TRAIN documents/futures/order.']))
    print(json.dumps(dict(dataset=str(output),items=len(rows),training_layout=LAYOUT,utility_reader=READER)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepend',type=Path,required=True);a=parser.parse_args();prepare(a.output,a.prepend)
