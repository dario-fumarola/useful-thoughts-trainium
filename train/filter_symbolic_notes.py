"""Frozen-reader TRAIN teacher filtering, all attempts retained; no training.

Score raw, empty and supplied symbolic notes as one question. The arithmetic
filter uses the complete numeral+actual closing-boundary span; natural16 is a
separate diagnostic and the later W reader's stated supervision horizon.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from train import interface_oracle_probe as oracle
from train.evaluate_likelihood import score_continuations
from train.evaluate import asset_provenance
from train.evaluate_local import setup_model
from train.useful_note_sft_data import note_ids
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

MODES = ('raw', 'empty', 'teacher')


def sources():
    return {**oracle.sources(), **{name:sha256_file(name) for name in
        ('train/filter_symbolic_notes.py','train/useful_note_sft_data.py','train/symbolic_note_labels.py')}}


def check(until):
    if datetime.now(timezone.utc) >= require_utc_deadline(until):
        raise TimeoutError('Finite TRAIN filter deadline; no automatic retry')


def read(path):
    return json.loads(Path(path).read_text())


def prepare(candidates, reference_output, output, interface, shards, until):
    check(until)
    if interface not in oracle.INTERFACES or shards not in (1,2,4,8,16,32):
        raise ValueError('One explicit interface and finite shard topology required')
    cm = read(candidates/'manifest.json'); candidate_file = candidates/'candidates.jsonl'
    if (cm['status'] != 'complete_CPU_labels_NOT_final_training_inventory'
            or sha256_file(candidate_file) != cm['candidates_sha256']):
        raise ValueError('Complete immutable TRAIN teacher candidates required')
    p,x = oracle.load(reference_output); ref,_ = oracle.reference(reference_output,p,x)
    summary = read(reference_output/'summary.json')
    if summary['status'] != 'complete' or summary['protocol_sha256'] != sha256_file(reference_output/'protocol.json'):
        raise ValueError('Finish the complete Phase1 question before teacher filtering')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False)
    rows = [json.loads(s) for s in candidate_file.read_text().splitlines() if s.strip()]
    if (len(rows) != cm['candidate_count'] or len({r['item_id'] for r in rows}) != len(rows)
            or len({(r['document_id'],r.get('absolute_source_cut',r['item_id'])) for r in rows}) != len(rows)
            or any(r['split'] != 'train' for r in rows)):
        raise ValueError('Every fixed distinct TRAIN source-cut candidate must be retained')
    contexts = {}
    for r in rows:
        if (note_ids(r,tok) != r['teacher_note_ids'] or len(r['natural_future16_ids']) != 16
                or r['target_ids'] != r['natural_future16_ids'][:len(r['target_ids'])]):
            raise ValueError('Complete symbolic note and original scalar/16 span required')
        contexts[r['item_id']] = {}
        for mode in MODES:
            c = context(r,mode,interface,cm['parts'])
            if len(c)+16 > 1024:raise ValueError('Overwidth filter input; never truncate')
            contexts[r['item_id']][mode] = canonical_hash(c)
    check(until);output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'inputs.json',dict(rows=rows,parts=cm['parts'],context_hashes=contexts))
    atomic_json(output/'protocol.json',dict(schema='symbolic-note-frozen-reader-filter-v1',created_utc=utc_now(),
        source_files=sources(),input_hashes={str(f.resolve()):sha256_file(f) for f in
            (candidates/'manifest.json',candidate_file,reference_output/'protocol.json',reference_output/'summary.json')},
        inputs_sha256=sha256_file(output/'inputs.json'),reference_output=str(reference_output.resolve()),
        model_dir=p['model_dir'],assets=p['assets'],base_signature=ref['signature'],
        interface=interface,phase1_arithmetic_passing=interface in summary['cohorts']['arithmetic']['passing_interfaces'],
        shards=shards,items=len(rows),modes=list(MODES),deadline_utc=until,precision='fp32',width=1024,
        filter='STRICT teacher-minus-empty >0 on numeral+actual closing boundary; retain all raw/empty/teacher natural16 token scores',
        candidate_order_rule=cm.get('candidate_order_rule','Original curator order'),
        unique_documents=len({r['document_id'] for r in rows}),
        final_selection='First5120 positive candidates in the pinned candidate order, no shortfall shrink; distinct cuts may share TRAIN documents',
        requested_training_items=5120))


def context(row,mode,interface,parts):
    return oracle.reader_context(row,row['teacher_note_ids'] if mode=='teacher' else [],
                                 'raw' if mode=='raw' else interface,parts)


def load(output):
    p = read(output/'protocol.json'); x = read(output/'inputs.json')
    if p['source_files'] != sources() or p['inputs_sha256'] != sha256_file(output/'inputs.json'):
        raise ValueError('Frozen filter source or inputs changed')
    for name,digest in p['input_hashes'].items():
        if sha256_file(name) != digest:raise ValueError('Filter input publication changed')
    return p,x


def record_native_check(directory, report, actual, expected):
    """Preserve the actual vector and finite/error diagnostics before gating."""
    actual=actual.cpu();error=float((actual-expected).abs().max())
    torch.save(actual,directory/'forward-logps.pt')
    finite=bool(torch.isfinite(actual).all()) and math.isfinite(error)
    report.update(native_max_full_head_logp_error=error if math.isfinite(error) else None,
                  native_actual_all_finite=bool(torch.isfinite(actual).all()),
                  forward_values_sha256=sha256_file(directory/'forward-logps.pt'))
    atomic_json(directory/'run.json',report,overwrite=True)
    if not finite or error>.002:raise RuntimeError('Native full-head filter gate failed')
    return error


@torch.no_grad()
def score(output,shard,until):
    check(until);p,x = load(output)
    if type(shard) is not int or not 0 <= shard < p['shards'] or until != p['deadline_utc']:
        raise ValueError('Fixed shard/deadline required')
    reference_output = Path(p['reference_output']); rp,rx = oracle.load(reference_output)
    ref,native = oracle.reference(reference_output,rp,rx)
    directory = output/'scores'/f'shard-{shard}'; directory.mkdir(parents=True,exist_ok=False)
    report = dict(status='started',started_utc=utc_now(),rows=0,shard=shard,
                  protocol_sha256=sha256_file(output/'protocol.json'))
    atomic_json(directory/'run.json',report)
    attempt=None
    try:
        if asset_provenance(Path(p['model_dir']))!=p['assets']:
            raise ValueError('Released B0 full asset identity changed before allocation')
        backend,_,checkpoint,signature,_ = setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=None,device='xla'))
        if checkpoint is not None or signature != p['base_signature'] or signature != ref['signature']:
            raise ValueError('Require unchanged released B0 reader')
        index = shard % len(rx['qualification_refs']); q = rx['qualification_refs'][index]
        ids = torch.full((1,1024),151643,dtype=torch.long)
        ids[0,:len(q['context_ids'])] = torch.tensor(q['context_ids'])
        hidden = backend.hidden(ids.to(backend.device))
        actual = backend.model.lm_head(hidden[:,len(q['context_ids'])-1]).float().log_softmax(-1)
        backend.sync(); actual = actual.cpu();report['signature']=signature
        record_native_check(directory,report,actual,native[index:index+1])
        del hidden,actual
        with (directory/'scores.jsonl').open('x') as stream:
            for r in x['rows'][shard::p['shards']]:
                for mode in MODES:
                    check(until);started=time.monotonic()
                    attempt=dict(status='in_progress',item_id=r['item_id'],document_id=r['document_id'],
                        mode=mode,started_utc=utc_now(),context_sha256=x['context_hashes'][r['item_id']][mode],
                        target_token_ids=r['natural_future16_ids'])
                    atomic_json(directory/'active-attempt.json',attempt,overwrite=True)
                    c=context(r,mode,p['interface'],x['parts'])
                    if canonical_hash(c) != x['context_hashes'][r['item_id']][mode]:raise ValueError('Changed filter context')
                    value=score_continuations(backend,c,[r['natural_future16_ids']],total_width=1024,future_width=16,pad_id=151643)[0]
                    record=dict(status='complete',item_id=r['item_id'],document_id=r['document_id'],mode=mode,
                        context_sha256=canonical_hash(c),**asdict(value),
                        primary_span_tokens=len(r['target_ids']),
                        primary_span_ll=sum(value.token_log_likelihoods[:len(r['target_ids'])]),
                        numeric_only_ll=sum(z for z,on in zip(value.token_log_likelihoods,r['numeric_token_mask']) if on),
                        wall_seconds=time.monotonic()-started)
                    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();report['rows']+=1
                    attempt.update(status='complete',finished_utc=utc_now(),wall_seconds=time.monotonic()-started)
                    atomic_json(directory/'active-attempt.json',attempt,overwrite=True);attempt=None
                if report['rows']%300==0:print(json.dumps(dict(utc=utc_now(),shard=shard,rows=report['rows'])),flush=True)
        report.update(status='complete',scores_sha256=sha256_file(directory/'scores.jsonl'))
    except BaseException as error:
        if attempt is not None:
            atomic_json(directory/'failed-attempt.json',dict(attempt,status='failed',
                finished_utc=utc_now(),error=f'{type(error).__name__}: {error}'))
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        report['finished_utc']=utc_now();atomic_json(directory/'run.json',report,overwrite=True)


def summarize(output):
    p,x=load(output);index={};pins={}
    for shard in range(p['shards']):
        d=output/'scores'/f'shard-{shard}';r=read(d/'run.json')
        if (r['status']!='complete' or r['protocol_sha256']!=sha256_file(output/'protocol.json')
                or r['rows']!=len(x['rows'][shard::p['shards']])*len(MODES)
                or r['scores_sha256']!=sha256_file(d/'scores.jsonl')
                or r['signature']!=p['base_signature']
                or not math.isfinite(r['native_max_full_head_logp_error'])
                or not 0<=r['native_max_full_head_logp_error']<=.002
                or r['forward_values_sha256']!=sha256_file(d/'forward-logps.pt')):
            raise ValueError('All immutable qualified filter shards must finish')
        pins[str(d/'scores.jsonl')]=r['scores_sha256']
        for line in (d/'scores.jsonl').read_text().splitlines():
            v=json.loads(line);key=(v['item_id'],v['mode'])
            if key in index:raise ValueError('Duplicate filter cell')
            index[key]=v
    expected={(r['item_id'],m) for r in x['rows'] for m in MODES}
    if set(index)!=expected:raise ValueError('Complete filter denominators required')
    positive=[];decisions=[];full16_positive=0
    for row in x['rows']:
        values={m:index[row['item_id'],m] for m in MODES};n=len(row['target_ids'])
        for mode,v in values.items():
            lp=v['token_log_likelihoods']
            if (v['status']!='complete' or len(lp)!=16 or any(not math.isfinite(z) or z>0 for z in lp)
                    or v['document_id']!=row['document_id'] or v['target_token_ids']!=row['natural_future16_ids']
                    or v['context_sha256']!=x['context_hashes'][row['item_id']][mode]
                    or v['log_likelihood']!=sum(lp) or v['mean_token_log_likelihood']!=sum(lp)/16
                    or v['primary_span_tokens']!=n or v['primary_span_ll']!=sum(lp[:n])):
                raise ValueError('Changed/nonfinite/incomplete filter reduction')
        gain=values['teacher']['primary_span_ll']-values['empty']['primary_span_ll']
        gain16=sum(values['teacher']['token_log_likelihoods'])-sum(values['empty']['token_log_likelihoods'])
        numeric={mode:sum(z for z,on in zip(v['token_log_likelihoods'],row['numeric_token_mask']) if on)
                 for mode,v in values.items()}
        if any(values[mode]['numeric_only_ll']!=ll for mode,ll in numeric.items()):
            raise ValueError('Numeric-only diagnostic changed')
        keep=gain>0;full16_positive+=int(gain16>0)
        if keep:positive.append(row)
        decisions.append(dict(item_id=row['item_id'],document_id=row['document_id'],
            teacher_minus_empty=gain,teacher_minus_empty_natural16=gain16,accepted=keep,
            numeric_teacher_minus_empty=numeric['teacher']-numeric['empty'],
            numeric_teacher_minus_raw=numeric['teacher']-numeric['raw'],raw_numeric_headroom=-numeric['raw']))
    atomic_json(output/'decisions.json',dict(rows=decisions))
    requested=p['requested_training_items'];ready=len(positive)>=requested
    if ready:
        with (output/'training.jsonl').open('x') as stream:
            for row in positive[:requested]:stream.write(json.dumps(row,allow_nan=False)+'\n')
    result=dict(status='complete_filter_training_ready' if ready else 'complete_filter_training_SHORTFALL',
        attempted=len(x['rows']),positive=len(positive),acceptance_rate=len(positive)/len(x['rows']),
        natural16_positive=full16_positive,requested_training_items=requested,shortfall=max(0,requested-len(positive)),
        raw_numeric_headroom_mean=sum(v['raw_numeric_headroom'] for v in decisions)/len(decisions),
        raw_numeric_headroom_under2=sum(v['raw_numeric_headroom']<2 for v in decisions),
        numeric_teacher_minus_raw_mean=sum(v['numeric_teacher_minus_raw'] for v in decisions)/len(decisions),
        selected_item_ids=[r['item_id'] for r in positive[:requested]] if ready else [],
        training_sha256=sha256_file(output/'training.jsonl') if ready else None,
        protocol_sha256=sha256_file(output/'protocol.json'),score_hashes=pins,
        decisions_sha256=sha256_file(output/'decisions.json'),finished_utc=utc_now(),
        interpretation='Future-aware teacher selection on TRAIN only; positive supplied notes do not establish useful prefix-generated thoughts.')
    atomic_json(output/'summary.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','score','summarize'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidates',type=Path);parser.add_argument('--reference-output',type=Path)
    parser.add_argument('--interface',choices=oracle.INTERFACES);parser.add_argument('--shards',type=int,default=4)
    parser.add_argument('--shard',type=int);parser.add_argument('--deadline-utc')
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.candidates,args.reference_output,args.output,args.interface,args.shards,args.deadline_utc)
    elif args.command=='score':score(args.output,args.shard,args.deadline_utc)
    else:print(json.dumps(summarize(args.output)))


if __name__=='__main__':main()
