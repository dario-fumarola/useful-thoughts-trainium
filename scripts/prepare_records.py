#!/usr/bin/env python3
"""Build deterministic document-split records from local raw-document JSONL.

Input: {"text": "...", "url": "optional", "domain": "optional"}, or pretokenized
{"tokens": [...], "document_id": "...", "domain": "..."}. For raw text, supply a
LOCAL tokenizer directory; downloads are not performed by this script.
Pre-tokenized inputs require an explicit tokenizer identity for provenance.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from weekend_thinking.data import digest,make_record,write_records,file_sha256
from weekend_thinking.corpus_audit import page_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tokenizer-dir',type=Path);p.add_argument('--tokenizer-identity',required=True)
    p.add_argument('--prefix-length',type=int,default=512);p.add_argument('--future-length',type=int,default=16)
    p.add_argument('--salt',default='weekend-2026-10-16-v1')
    p.add_argument('--source-revision',required=True)
    a=p.parse_args()
    if a.out.exists() or a.out.with_suffix(".manifest.json").exists():
        raise SystemExit("Refusing to overwrite existing records or their manifest.")
    tok=None
    if a.tokenizer_dir:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(a.tokenizer_dir),local_files_only=True,trust_remote_code=False)
    seen=set();counter=Counter();records=[]
    with a.input.open() as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line);counter['input_documents']+=1
            if 'text' in row:
                if tok is None:raise SystemExit('Raw text requires --tokenizer-dir')
                fingerprint=digest(row['text'])
                # Content-bearing query fields distinguish forum threads and
                # wiki pages. Only named tracking parameters are discarded.
                document_id=page_identity(row.get('url',''),fingerprint)
                tokens=tok.encode(row['text'],add_special_tokens=False)
            else:
                tokens=row['tokens'];document_id=str(row['document_id'])
                fingerprint=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
            if fingerprint in seen:counter['exact_duplicates_removed']+=1;continue
            seen.add(fingerprint)
            rec=make_record(tokens,document_id,domain=row.get('domain','unknown'),
                            prefix_length=a.prefix_length,future_length=a.future_length,salt=a.salt)
            if rec is None:counter['too_short_removed']+=1;continue
            # Multiple versions of a page stay in one split. Keep one below.
            records.append(rec);counter[rec.split]+=1
    unique={}
    for row in records:
        if row.document_id in unique:counter['duplicate_document_removed']+=1;counter[row.split]-=1
        else:unique[row.document_id]=row
    # Hash-sort instead of preserving crawl order. All arms reuse this exact file.
    records=sorted(unique.values(),key=lambda r:hashlib.sha256((a.salt+':order:'+r.id).encode()).hexdigest())
    count=write_records(a.out,records)
    manifest={'records':count,'counts':dict(counter),'tokenizer_identity':a.tokenizer_identity,
              'source_revision':a.source_revision,'salt':a.salt,'prefix_length':a.prefix_length,
              'future_length':a.future_length,'input_sha256':file_sha256(a.input),
              'output_sha256':file_sha256(a.out),
              'limitations':['No near-duplicate or benchmark-overlap filter in this script.',
                  'Use an independently audited decontaminated corpus and run an overlap audit before scientific training.',
                  'No claim about contamination in the released starting checkpoint.']}
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
