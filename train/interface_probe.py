"""Small development-only reader-interface experiment; no training or oracle selection.

Reuses original prefix-only B0 notes and global length-matched controls. A
re-anchor repeats only already observed prefix tokens, never future tokens.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from train.evaluate_local import load_selected, read_selection, setup_model
from train.evaluate_likelihood import score_continuations
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.core import Format

ROOT = Path(__file__).resolve().parents[1]
INTERFACES = {
    "original": (True, 0),
    "no_header": (False, 0),
    "reanchor32": (True, 32),
    "reanchor64": (True, 64),
    "no_header_reanchor64": (False, 64),
}
MODES = ("empty", "own", "filler", "shuffled", "other")


def reader_context(prefix, content, fmt, *, header, reanchor):
    if content is None:
        return list(prefix) + (list(prefix[-reanchor:]) if reanchor else [])
    return ((list(fmt.header_ids) if header else []) + list(prefix)
            + list(fmt.open_ids) + list(content) + list(fmt.close_ids)
            + (list(prefix[-reanchor:]) if reanchor else []))


def prepare(output, per_quartile):
    output.mkdir(parents=True, exist_ok=False)
    selection_path = ROOT/'local/data/finemath/fixed-local-evaluation-v1.ids.json'
    selection = read_selection(selection_path, qualification=True, split='dev')
    strata_path = ROOT/'local/analysis/local-dev512-v1.strata.json'
    strata = json.loads(strata_path.read_text())
    members = {row['item_id']: row for row in strata['items']}
    counts = Counter()
    ids = []
    for key in selection['item_ids']:
        group = members[key]['B0_surprisal_bin']
        if counts[group] < per_quartile:
            ids.append(key)
            counts[group] += 1
    assert len(counts) == 4 and set(counts.values()) == {per_quartile}
    documents = dict(zip(selection['item_ids'], selection['document_ids']))
    selected = dict(selection, item_ids=ids, document_ids=[documents[key] for key in ids],
                    count=len(ids), item_ids_sha256=canonical_hash(ids))
    paths = dict(
        records=ROOT/'local/data/finemath/production-500000.audit-v2.records.jsonl',
        controls=ROOT/'local/nvme/evaluation/local-dev512-v1-B0-001/scores/controls.jsonl',
        original_scores=ROOT/'local/nvme/evaluation/local-dev512-v1-B0-001/scores/scores.jsonl',
        format=ROOT/'local/science/resolved_format.absolute.json', strata=strata_path)
    protocol = dict(experiment_id='interface-tax-dev32-v1', created_utc=utc_now(),
        purpose='Exploratory DEV reader intervention; original G1 B0 notes, no new writer',
        selection_rule='First fixed DEV IDs in each frozen B0 surprisal quartile',
        per_quartile=per_quartile, selection=selected,
        strata={key: members[key]['B0_surprisal_bin'] for key in ids},
        interfaces=INTERFACES, modes=MODES, horizons=[1, 16], width=1664,
        compute='fp32', candidate_samples_per_item=1,
        limitations='Existing-note writer mismatch under new reader interfaces; not a deployed policy or best-of16 oracle',
        paths={key:str(path) for key,path in paths.items()},
        sha256={key:sha256_file(path) for key,path in paths.items() if key != 'records'},
        source_sha256=sha256_file(Path(__file__)),
        model_dir=str(ROOT/'local/assets/qwen3-1.7b-base'))
    atomic_json(output/'protocol.json', protocol)
    print(json.dumps(dict(prepared=str(output), ids=len(ids), counts=counts)), flush=True)


def summarize(output, protocol, rows):
    results = {}
    for group in ['all', *sorted(set(protocol['strata'].values()))]:
        subset = [row for row in rows if group == 'all' or protocol['strata'][row['item_id']] == group]
        conditions, contrasts = {}, {}
        for horizon in protocol['horizons']:
            index = {(r['item_id'],r['interface'],r['mode']):sum(r['token_log_likelihoods'][:horizon])
                     for r in subset if r['status'] == 'ok'}
            ids = [key for key in protocol['selection']['item_ids']
                   if group == 'all' or protocol['strata'][key] == group]
            for interface, modes in [('raw', ('raw','repeat32','repeat64')),
                                     *((name,MODES) for name in INTERFACES)]:
                for mode in modes:
                    values = [index[(key,interface,mode)] for key in ids if (key,interface,mode) in index]
                    conditions[f'{interface}/{mode}/K{horizon}'] = dict(
                        expected_items=len(ids), successful_items=len(values),
                        mean_log_likelihood=statistics.mean(values) if values else None)
            for interface in INTERFACES:
                for treatment, baseline in [('empty','raw'), ('own','raw'), ('own','empty'),
                                             ('own','filler'), ('own','shuffled'), ('own','other')]:
                    diffs = []
                    for key in ids:
                        a=(key,interface,treatment)
                        b=(key,'raw','raw') if baseline == 'raw' else (key,interface,baseline)
                        if a in index and b in index:
                            diffs.append(index[a]-index[b])
                    contrasts[f'{interface}/{treatment}_minus_{baseline}/K{horizon}'] = dict(
                        expected_items=len(ids), successful_pairs=len(diffs),
                        mean_nats=statistics.mean(diffs) if diffs else None,
                        median_nats=statistics.median(diffs) if diffs else None,
                        positive_pairs=sum(x>0 for x in diffs))
        results[group] = dict(conditions=conditions, contrasts=contrasts)
    return dict(experiment_id=protocol['experiment_id'], status='complete', completed_utc=utc_now(),
        protocol_sha256=sha256_file(output/'protocol.json'), rows=len(rows),
        failures=sum(r['status']!='ok' for r in rows), groups=results,
        interpretation='Exploratory development; candidate oracle and newly sampled writer utility remain unmeasured')


def run(output, device):
    protocol=json.loads((output/'protocol.json').read_text())
    if protocol['source_sha256'] != sha256_file(Path(__file__)):
        raise ValueError('Source changed after preparation')
    for name,digest in protocol['sha256'].items():
        if sha256_file(Path(protocol['paths'][name])) != digest:
            raise ValueError(f'Changed input {name}')
    controls={r['item_id']:r for r in map(json.loads,Path(protocol['paths']['controls']).read_text().splitlines())}
    # Global prefix-only notes and controls already exist before this future read.
    records=load_selected(Path(protocol['paths']['records']),protocol['selection'],include_future=True)
    fmt=Format.from_dict(json.loads(Path(protocol['paths']['format']).read_text())['format'])
    args=SimpleNamespace(model_dir=Path(protocol['model_dir']),checkpoint=None,device=device)
    backend,tokenizer,_,signature,_=setup_model(args)
    atomic_json(output/'started.json',dict(time_utc=utc_now(),device=str(backend.device),
        model=signature,protocol_sha256=sha256_file(output/'protocol.json')))
    old={(r['item_id'],r['mode']):r['token_log_likelihoods'] for r in
         map(json.loads,Path(protocol['paths']['original_scores']).read_text().splitlines()) if r['horizon']==16}
    rows=[]
    max_reference_error=0.0
    with (output/'scores.jsonl').open('x') as stream:
        for item_index,record in enumerate(records):
            contexts=[('raw',name,reader_context(record['prefix'],None,fmt,header=False,reanchor=n))
                      for name,n in [('raw',0),('repeat32',32),('repeat64',64)]]
            for interface,(header,reanchor) in INTERFACES.items():
                for mode in MODES:
                    contexts.append((interface,mode,reader_context(record['prefix'],
                        controls[record['item_id']]['content_ids'][mode],fmt,header=header,reanchor=reanchor)))
            for interface,mode,context in contexts:
                started=time.monotonic()
                row=dict(item_id=record['item_id'],document_id=record['document_id'],
                    interface=interface,mode=mode,context_tokens=len(context))
                try:
                    score=score_continuations(backend,context,[record['future']],
                        total_width=protocol['width'],future_width=16,pad_id=fmt.pad_id)[0]
                    row.update(status='ok',token_log_likelihoods=score.token_log_likelihoods)
                    if interface=='original' or (interface,mode)==('raw','raw'):
                        error=max(abs(x-y) for x,y in zip(score.token_log_likelihoods,old[(record['item_id'],mode)]))
                        max_reference_error=max(error,max_reference_error)
                        row['reference_max_token_error']=error
                        if error>0.002:
                            raise RuntimeError(f'B0 reference parity failed: {error}')
                except Exception as exc:
                    row.update(status='error',error=f'{type(exc).__name__}: {exc}')
                    row.pop('token_log_likelihoods',None)
                row['wall_seconds']=time.monotonic()-started
                rows.append(row)
                stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
                if row['status']!='ok':
                    raise RuntimeError(f'Probe stopped on failure: {row}')
            print(json.dumps(dict(time_utc=utc_now(),items_complete=item_index+1,
                total_items=len(records),rows=len(rows),max_reference_error=max_reference_error)),flush=True)
    report=summarize(output,protocol,rows)
    report['maximum_B0_reference_token_error']=max_reference_error
    report['scores_sha256']=sha256_file(output/'scores.jsonl')
    atomic_json(output/'summary.json',report)
    print(json.dumps(dict(status='complete',summary=str(output/'summary.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--per-quartile',type=int,default=8)
    parser.add_argument('--device',choices=['cpu','xla'],default='xla')
    args=parser.parse_args()
    if args.command=='prepare':
        if args.per_quartile<1 or args.per_quartile>128:
            parser.error('Per-quartile count must be1..128')
        prepare(args.output,args.per_quartile)
    else:
        run(args.output,args.device)
