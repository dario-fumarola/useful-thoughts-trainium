"""Future-aware supervised curation of real numeric equalities, never RL anchors.

Validate every bounded prefix-eligible equation in the original window, then
hash-select ONE accepted equation/document. Source RHS affects the accepted
set explicitly. Writer contexts still contain only exact past source tokens.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import islice
import json
import multiprocessing
from pathlib import Path
import re
import time

from train import computable_equations as base
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

SCHEMA = SALT = 'finemath-computable-equations-supervised-v2'
TOKENIZER = None


def sources():
    return {**base.sources(), 'train/computable_equations_supervised.py':sha256_file(__file__)}


def candidate_at_cut(prefix, cut, tok):
    """Same past-only arithmetic/copy checks as v1; no RHS available here."""
    ids=prefix[:cut];past=base.safe.decode(tok,ids)
    if tok.encode(past,add_special_tokens=False)!=ids:
        raise base.safe.Unsupported('past_token_roundtrip_failed')
    expression=base.safe.maximal_lhs(past);parsed=base.safe.parse_numeric(expression)
    if parsed.binary_ops<1:raise base.safe.Unsupported('no_binary_operation')
    reason=base.safe.whole_value_visible(past,parsed.value)
    if reason:raise base.safe.Unsupported(reason)
    equal=re.search(r'=\s*$',past);left=past[:equal.start()].rstrip()
    assert left.endswith(expression)
    start=len(left)-len(expression)
    return dict(cut=cut,prefix_ids=ids,prefix_text=past,prefix_sha256=canonical_hash(ids),
        expression=expression,normalized_expression=parsed.expression,numeric_value=str(parsed.value),
        binary_ops=parsed.binary_ops,lhs_char_start=start,reanchor_text=past[start:],
        selection_policy='SUPERVISED: validate all eligible cuts128..512, then fixed-hash choose one accepted equation/document; future-aware curation')


def choice_hash(document_id, source_record_id, cut):
    return canonical_hash(dict(salt=SALT,role='accepted_equation_choice',
        document_id=document_id,source_record_id=source_record_id,cut=cut))


def curate_document(header, line, tok, source_file):
    if header['split'] not in {'train','dev'}:raise ValueError('Never parse TEST')
    record=None;valid=[];audit=[];counts=Counter();pieces={}
    for cut in range(128,len(header['prefix'])+1):
        token=header['prefix'][cut-1]
        if token not in pieces:pieces[token]=base.safe.decode(tok,[token])
        if re.search(r'=\s*$',pieces[token]) is None:continue
        entry=dict(cut=cut)
        try:chosen=candidate_at_cut(header['prefix'],cut,tok)
        except base.safe.Unsupported as error:
            entry.update(status='prefix_ineligible',reason=str(error));audit.append(entry);continue
        if record is None:record=json.loads(line)  # Future-aware source curation, after this candidate's prefix checks.
        row,status=base.curate(record,chosen,tok,source_file,base.SOURCE_SHA)
        counts[status['status']]+=1;entry.update(status)
        if row is not None:
            row['item_id']=f"{record['id']}:supervised-equation-v2:{cut}"
            key=choice_hash(header['document_id'],header['item_id'],cut)
            row['supervised_choice_hash']=key;entry['choice_hash']=key
            valid.append(row)
        audit.append(entry)
    result=min(valid,key=lambda r:r['supervised_choice_hash']) if valid else None
    outcome=dict(item_id=header['item_id'],document_id=header['document_id'],split=header['split'],
        status='accepted' if result else 'no_valid_source_equation',
        selected_cut=result['boundary']['cut'] if result else None,
        selected_item_id=result['item_id'] if result else None,
        accepted_candidate_count=len(valid),candidate_audit=audit)
    return result,outcome,dict(counts)


def initialize(tokenizer_dir):
    global TOKENIZER
    from transformers import AutoTokenizer
    TOKENIZER=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True,trust_remote_code=False)


def worker(task):
    header,line,source_file=task
    return curate_document(header,line,TOKENIZER,source_file)


def select_pools(rows):
    result={}
    for split,n in (('train',base.TRAIN_ITEMS),('dev',base.DEV_ITEMS)):
        ordered=sorted((r for r in rows if r['split']==split),key=lambda r:
            canonical_hash(dict(salt=SALT,role='pool',split=split,document_id=r['document_id'],item_id=r['item_id'])))
        result[split]=dict(requested=n,available=len(ordered),selected_item_ids=[r['item_id'] for r in ordered[:n]],
                          reserve_item_ids=[r['item_id'] for r in ordered[n:]])
    return result


def prepare(source,output,tokenizer_dir,deadline_utc,workers=8):
    source=Path(source).resolve();output=Path(output).resolve();tokenizer_dir=Path(tokenizer_dir).resolve()
    deadline=require_utc_deadline(deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Finite supervised CPU deadline; partials retained')
    check()
    if not 1<=workers<=16:raise ValueError('Bounded1..16 CPU workers required')
    pinned=sources()
    if sha256_file(source)!=base.SOURCE_SHA:raise ValueError('Audited source SHA differs')
    check();output.mkdir(parents=True,exist_ok=False)
    rows=[];seen=set();counts=Counter();candidates=Counter();split_counts={k:Counter() for k in ('train','dev')}
    started=time.monotonic()
    try:
        with ((output/'accepted.jsonl').open('x') as accepted,
              (output/'audit.jsonl').open('x') as audit,
              ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize,initargs=(str(tokenizer_dir),)) as pool):
            source_rows=base.safe.prefix_records(source)
            while True:
                check();batch=list(islice(source_rows,1024))
                if not batch:break
                tasks=((header,line,str(source)) for header,line in batch)
                for row,entry,stats in pool.map(worker,tasks,chunksize=16):
                    check();doc=entry['document_id']
                    if doc in seen:raise ValueError('Repeated document or split overlap')
                    seen.add(doc);counts[entry['status']]+=1;split_counts[entry['split']][entry['status']]+=1
                    candidates.update(stats)
                    if row is not None:rows.append(row);accepted.write(json.dumps(row,allow_nan=False)+'\n')
                    audit.write(json.dumps(entry,allow_nan=False)+'\n')
                print(json.dumps(dict(documents=len(seen),counts=counts,candidate_counts=candidates,
                    elapsed_seconds=round(time.monotonic()-started,2))),flush=True)
        check()
        if sources()!=pinned or sha256_file(source)!=base.SOURCE_SHA:raise ValueError('Source changed during pass')
    except BaseException as error:
        atomic_json(output/'preparation.failed.json',dict(status='partial_not_ready',error=f'{type(error).__name__}: {error}',
            documents=len(seen),counts=dict(counts),candidate_counts=dict(candidates),source_files=pinned))
        raise
    # No model calls; tokenizer identity only after the complete candidate pass.
    initialize(str(tokenizer_dir));pools=select_pools(rows);dev=pools['dev']['selected_item_ids']
    manifest=dict(schema=SCHEMA,status='complete_supervised_CPU_preparation_not_model_or_training_qualification',
        created_utc=utc_now(),deadline_utc=deadline_utc,elapsed_seconds=time.monotonic()-started,workers=workers,
        source_file=str(source),source_sha256=base.SOURCE_SHA,source_files=pinned,
        tokenizer_dir=str(tokenizer_dir),tokenizer_vocab_sha256=canonical_hash(TOKENIZER.get_vocab()),
        accepted_file=str(output/'accepted.jsonl'),accepted_sha256=sha256_file(output/'accepted.jsonl'),
        audit_file=str(output/'audit.jsonl'),audit_sha256=sha256_file(output/'audit.jsonl'),
        documents=len(seen),counts=dict(counts),candidate_counts=dict(candidates),
        counts_by_split={k:dict(v) for k,v in split_counts.items()},selection_salt=SALT,selection=pools,
        selected_train_item_ids=pools['train']['selected_item_ids'],selected_dev_item_ids=dev,
        phase1_dev_item_ids=dev[:256],held_post_sft_dev_item_ids=dev[256:512],
        ready_for_requested_pools=all(p['available']>=p['requested'] for p in pools.values()),
        boundary_selection='FUTURE-AWARE SUPERVISED CURATION: all eligible cuts128..512 validated against actual RHS; fixed hash chooses one accepted equation/document',
        target_definition='Exact native complete numeral plus actual closing certificate within16 tokens',
        note_definition='FULL target-copy plumbing oracle including boundary; NOT proper-intermediate or deployed computation',
        limitations=['Not a stopping-time/unconditional/RL anchor sampler: accepted-equation membership uses true future labels.',
            'Each resulting prefix contains only exact original past tokens; no model outcomes or benchmark answers used for selection.',
            'Bounded known-form visible-answer filter is not exhaustive semantic-disclosure detection.',
            'Only original528-token windows; no new raw-document or TEST-array reads.',
            'The raw second DEV half is only a candidate pool; final post-SFT holdout uses the separate prospectively sealed document-exclusion inventory.',
            'One accepted equation/document; fixed5120TRAIN/512DEV quotas never shrunk on shortfall.'])
    atomic_json(output/'manifest.json',manifest)
    print(json.dumps(dict(manifest=str(output/'manifest.json'),pools={k:v['available'] for k,v in pools.items()})),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','tokenizer-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-utc',required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();prepare(a.source,a.output,a.tokenizer_dir,a.deadline_utc,a.workers)


if __name__=='__main__':main()
