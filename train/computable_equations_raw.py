"""Bounded real raw-document extension; explicitly supervised future-aware curation.

Only audited TRAIN/DEV membership lines are JSON/text decoded. Tokenization sees
at most65,536 raw characters and8,208 tokens/document; candidate cuts end8192.
One exact-source equation/document is selected by fixed hash. Existing curated
TRAIN exposure and Phase1 pools are neither replaced nor modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
import multiprocessing
from pathlib import Path
import re
import time

from train import computable_equations as base
from train import computable_equations_supervised as supervised
from train.scalar_completion import parse_initial_scalar_v2
from weekend_thinking.artifacts import atomic_json, canonical_hash, require_utc_deadline, sha256_file, utc_now

SCHEMA = SALT = 'finemath-computable-equations-raw-v3'
RAW_SHA = '2cad3ec923fc71da3a0d8364a63096fbfd1c2c68c773bc5e42e3e01ae6ea8365'
MEMBERSHIP_SHA = '9ee1c7f5dfdad156db8ef34fea2fa62315cf00be0cf6a8bd49c454628ab28cbf'
MAX_CHARACTERS, MAX_TOKENS, MAX_CUT = 65536, 8208, 8192
TOKENIZER = None
MINIMUM_BINARY_OPS = 1


def sources():
    return {**supervised.sources(),'train/computable_equations_raw.py':sha256_file(__file__)}


def text_digest(text):
    """Upstream membership hash normalizes whitespace only; never use as bytes."""
    return hashlib.sha256(' '.join(text.split()).encode('utf-8')).hexdigest()


def choice_hash(document_id, absolute_cut):
    return canonical_hash(dict(salt=SALT,role='accepted_equation_choice',
        document_id=document_id,absolute_source_cut=absolute_cut))


def source_window(tokens,offsets,text,absolute_cut,tok):
    """Exact original528-token chunk and past; reject normalization/UTF8 fragments."""
    start=max(0,absolute_cut-512);end=start+528;cut=absolute_cut-start
    if absolute_cut<128 or absolute_cut>MAX_CUT or end>len(tokens):
        raise base.safe.Unsupported('incomplete_bounded_source_window')
    original=tokens[start:end]
    char_start,char_end=offsets[start][0],offsets[end-1][1]
    past_end=offsets[absolute_cut-1][1]
    if not 0<=char_start<past_end<=char_end<=len(text):
        raise base.safe.Unsupported('invalid_native_source_offsets')
    full=base.safe.decode(tok,original);past=base.safe.decode(tok,original[:cut])
    if (full!=text[char_start:char_end] or past!=text[char_start:past_end]
            or tok.encode(past,add_special_tokens=False)!=original[:cut]):
        raise base.safe.Unsupported('raw_UTF8_native_window_not_lossless')
    return dict(ids=original,cut=cut,token_start=start,token_end=end,
        char_start=char_start,char_end=char_end,past_char_end=past_end)


def curate_document(meta,line,tok,source_file,minimum_binary_ops=1):
    if meta['split'] not in {'train','dev'}:raise ValueError('Never decode raw TEST/unlisted data')
    if type(minimum_binary_ops) is not int or not 1<=minimum_binary_ops<=6:
        raise ValueError('Explicit minimum binary operation count1..6 required')
    raw=json.loads(line);text=raw.get('text')
    if not isinstance(text,str) or raw.get('url','')!=meta['url'] or text_digest(text)!=meta['content_sha256']:
        raise ValueError('Raw line does not match immutable audited membership')
    clipped=text[:MAX_CHARACTERS]
    encoded=tok(clipped,add_special_tokens=False,truncation=True,max_length=MAX_TOKENS,
        return_offsets_mapping=True,return_attention_mask=False)
    tokens=encoded['input_ids'];offsets=encoded['offset_mapping']
    if len(tokens)!=len(offsets) or len(tokens)>MAX_TOKENS:
        raise ValueError('Native tokenizer bounds/offsets invalid')
    best=None;audit=[];counts=Counter();pieces={};valid=0;identity=None
    for absolute_cut in range(128,min(MAX_CUT,len(tokens)-16)+1):
        token=tokens[absolute_cut-1]
        if token not in pieces:pieces[token]=base.safe.decode(tok,[token])
        if re.search(r'=\s*$',pieces[token]) is None:continue
        entry=dict(absolute_cut=absolute_cut)
        try:
            window=source_window(tokens,offsets,clipped,absolute_cut,tok)
            chosen=supervised.candidate_at_cut(window['ids'][:window['cut']],window['cut'],tok)
            if chosen['binary_ops']<minimum_binary_ops:
                raise base.safe.Unsupported('below_declared_prefix_operation_minimum')
        except base.safe.Unsupported as error:
            entry.update(status='prefix_or_serialization_ineligible',reason=str(error));audit.append(entry);continue
        record=dict(id=f"{meta['document_id']}:raw:{meta['source_line']}:window:{window['token_start']}",
            document_id=meta['document_id'],split=meta['split'],prefix=window['ids'][:512],future=window['ids'][512:])
        row,status=base.curate(record,chosen,tok,source_file,RAW_SHA)
        if row is not None:
            # More than the stored16 tokens may be needed to rule out a scale
            # word/expression after a native token fragment. Use bounded actual
            # raw lookahead; never certify an arbitrary cropped endpoint as EOS.
            past_end=window['past_char_end'];lookahead=clipped[past_end:past_end+8192]
            proof=parse_initial_scalar_v2(lookahead,'length_cap')
            if (proof['status']!='valid' or proof.get('value')!=row['numeric_value']
                    or proof.get('literal')!=row['numeric_surface_text']):
                row=None;status=dict(status='unsupported_full_raw_RHS',reason='Native16-source scalar lacks matching complete bounded raw-source proof')
        counts[status['status']]+=1;entry.update(status)
        if row is not None:
            valid+=1;key=choice_hash(meta['document_id'],absolute_cut)
            if identity is None:
                identity=dict(raw_line_sha256=hashlib.sha256(line if isinstance(line,bytes) else line.encode()).hexdigest(),
                    raw_text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
            row.update(item_id=f"{meta['document_id']}:raw-equation-v3:{absolute_cut}",
                supervised_choice_hash=key,raw_source_line=meta['source_line'],source_url=meta['url'],
                **identity,
                upstream_normalized_content_sha256=meta['content_sha256'],
                audited_original_record_id=meta['record_id'],
                source_chunk_token_start=window['token_start'],source_chunk_token_end=window['token_end'],
                absolute_source_cut=absolute_cut,source_chunk_char_start=window['char_start'],
                source_chunk_char_end=window['char_end'],prefix_source_char_end=window['past_char_end'],
                raw_rhs_lookahead_sha256=hashlib.sha256(lookahead.encode()).hexdigest(),
                raw_rhs_lookahead_char_end=past_end+len(lookahead),raw_rhs_validation=proof,
                raw_source_serialization='Exact raw UTF8 substring == native decode for entire528-token chunk and causal past; no byte normalization')
            row['boundary']['absolute_source_cut']=absolute_cut
            row['boundary']['selection_policy']='SUPERVISED bounded raw document: validate all eligible source equations then fixed-hash one/document; future-aware curation'
            entry['choice_hash']=key
            if best is None or key<best['supervised_choice_hash']:best=row
        audit.append(entry)
    return best,dict(document_id=meta['document_id'],split=meta['split'],source_line=meta['source_line'],
        status='accepted' if best else 'no_valid_bounded_raw_equation',
        source_characters=len(text),tokenized_characters=len(clipped),bounded_native_tokens=len(tokens),
        accepted_candidate_count=valid,selected_absolute_cut=best['absolute_source_cut'] if best else None,
        candidate_audit=audit),dict(counts)


def membership(path):
    rows={};docs=set()
    with Path(path).open() as stream:
        for line in stream:
            row=json.loads(line)  # Membership contains identities/metadata, no source text or token arrays.
            if row['split'] not in {'train','dev'}:continue
            n=row['source_line'];doc=row['document_id']
            if n in rows or doc in docs:raise ValueError('Duplicate audited source line/document')
            rows[n]=row;docs.add(doc)
    return rows


def raw_tasks(path,members):
    source_file=str(Path(path).resolve())
    with Path(path).open('rb') as stream:
        for number,line in enumerate(stream,1):
            if number in members:yield members[number],line,source_file
            # All other lines, including TEST, remain opaque bytes.


def initialize(tokenizer_dir,minimum_binary_ops=1):
    global TOKENIZER,MINIMUM_BINARY_OPS
    MINIMUM_BINARY_OPS=minimum_binary_ops
    from transformers import AutoTokenizer
    TOKENIZER=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True,trust_remote_code=False)
    if not TOKENIZER.is_fast:raise ValueError('Native offsets require the existing fast tokenizer')
    if TOKENIZER.truncation_side!='right':raise ValueError('First-token range requires right truncation')


def worker(task):return curate_document(*task[:2],TOKENIZER,task[2],MINIMUM_BINARY_OPS)


def select_pools(rows):
    result={}
    for split,n in (('train',5120),('dev',512)):
        ordered=sorted((r for r in rows if r['split']==split),key=lambda r:
            canonical_hash(dict(salt=SALT,role='pool',split=split,document_id=r['document_id'],item_id=r['item_id'])))
        result[split]=dict(requested=n,available=len(ordered),selected_item_ids=[r['item_id'] for r in ordered[:n]],
            reserve_item_ids=[r['item_id'] for r in ordered[n:]])
    return result


def prepare(raw,members,output,tokenizer_dir,deadline_utc,workers=8,minimum_binary_ops=1):
    raw=Path(raw).resolve();members=Path(members).resolve();output=Path(output).resolve();tokenizer_dir=Path(tokenizer_dir).resolve()
    deadline=require_utc_deadline(deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Finite bounded raw CPU deadline; partials retained')
    check()
    if type(minimum_binary_ops) is not int or not 1<=minimum_binary_ops<=6:
        raise ValueError('Explicit minimum binary operation count1..6 required')
    if not 1<=workers<=16:raise ValueError('Bounded1..16 CPU workers required')
    pinned=sources()
    if sha256_file(raw)!=RAW_SHA or sha256_file(members)!=MEMBERSHIP_SHA:
        raise ValueError('Raw/membership immutable snapshot changed')
    approved=membership(members);assert len(approved)==347895
    check();output.mkdir(parents=True,exist_ok=False)
    rows=[];seen=set();counts=Counter();candidates=Counter();splits={k:Counter() for k in ('train','dev')};started=time.monotonic()
    try:
        with ((output/'accepted.jsonl').open('x') as accepted,(output/'audit.jsonl').open('x') as audit,
              ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(str(tokenizer_dir),minimum_binary_ops)) as pool):
            tasks=raw_tasks(raw,approved)
            while True:
                check();batch=list(islice(tasks,512))
                if not batch:break
                for row,entry,stats in pool.map(worker,batch,chunksize=8):
                    check();doc=entry['document_id']
                    if doc in seen:raise ValueError('Repeated raw document/split')
                    seen.add(doc);counts[entry['status']]+=1;splits[entry['split']][entry['status']]+=1;candidates.update(stats)
                    if row is not None:rows.append(row);accepted.write(json.dumps(row,allow_nan=False)+'\n')
                    audit.write(json.dumps(entry,allow_nan=False)+'\n')
                print(json.dumps(dict(documents=len(seen),counts=counts,candidate_counts=candidates,
                    elapsed_seconds=round(time.monotonic()-started,2))),flush=True)
        check()
        if len(seen)!=len(approved):raise ValueError('Raw membership source-line coverage incomplete')
        if sources()!=pinned or sha256_file(raw)!=RAW_SHA or sha256_file(members)!=MEMBERSHIP_SHA:
            raise ValueError('Source changed during raw pass')
    except BaseException as error:
        atomic_json(output/'preparation.failed.json',dict(status='partial_not_ready',error=f'{type(error).__name__}: {error}',
            documents=len(seen),counts=dict(counts),candidate_counts=dict(candidates),source_files=pinned))
        raise
    initialize(str(tokenizer_dir),minimum_binary_ops);pools=select_pools(rows)
    manifest=dict(schema=SCHEMA,status='complete_bounded_supervised_raw_CPU_preparation_not_training_release',
        created_utc=utc_now(),deadline_utc=deadline_utc,elapsed_seconds=time.monotonic()-started,workers=workers,
        source_file=str(raw),source_sha256=RAW_SHA,membership_file=str(members),membership_sha256=MEMBERSHIP_SHA,
        source_files=pinned,tokenizer_dir=str(tokenizer_dir),tokenizer_vocab_sha256=canonical_hash(TOKENIZER.get_vocab()),
        accepted_file=str(output/'accepted.jsonl'),accepted_sha256=sha256_file(output/'accepted.jsonl'),
        audit_file=str(output/'audit.jsonl'),audit_sha256=sha256_file(output/'audit.jsonl'),
        documents=len(seen),counts=dict(counts),candidate_counts=dict(candidates),counts_by_split={k:dict(v) for k,v in splits.items()},
        max_raw_characters=MAX_CHARACTERS,max_native_tokens=MAX_TOKENS,max_candidate_cut=MAX_CUT,
        minimum_binary_ops=minimum_binary_ops,
        prefix_complexity_rule='Apply declared operation minimum to the causal LHS before validating any RHS; retain existing exact raw-source checks.',
        selection_salt=SALT,selection=pools,ready_for_requested_candidate_pools=all(p['available']>=p['requested'] for p in pools.values()),
        existing_supervised_v2_train_exposure_replaced=False,
        limitations=['Future-aware supervised equation selection, not an unconditional/RL anchor or deployed calculation policy.',
            'Only audited retained TRAIN/DEV raw documents; all TEST/unlisted raw lines remain undecoded bytes.',
            'Range is first65536 raw characters then first8208 native tokens; not an exhaustive whole-document search.',
            'Exact raw bytes are retained; tokenizer normalization or partial-UTF8-window mismatches are rejected.',
            'Copy check covers only the provided last<=512 past tokens and known ambiguous numeric forms, not all semantic disclosures.',
            'Correct note copies the complete verified numeral+boundary target; this is explicitly full-answer plumbing.',
            'Candidate pools do not replace the existing supervised-v2 TRAIN5120 or Phase1 DEV256; fresh held inventory is prepared separately.'])
    atomic_json(output/'manifest.json',manifest)
    print(json.dumps(dict(manifest=str(output/'manifest.json'),pools={k:v['available'] for k,v in pools.items()})),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','members','output','tokenizer-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-utc',required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--minimum-binary-ops',type=int,choices=range(1,7),default=1)
    a=p.parse_args();prepare(a.raw,a.members,a.output,a.tokenizer_dir,a.deadline_utc,a.workers,a.minimum_binary_ops)


if __name__=='__main__':main()
