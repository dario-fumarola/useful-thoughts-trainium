"""Finite CPU-only fresh-source feasibility; never constructs a language model.

Pinned public TRAIN shards, old source exclusions, unchanged exact curator and
native source windows. This prepares a possible cohort, not a model-run release.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import heapq
from itertools import islice
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import urllib.request

from scripts.prepare_real_cpt_scaffold import PRIOR_PANELS, page, work_row
from train import computable_equations_raw as curator
from train.expression_identity import expression_key
from weekend_thinking.artifacts import sha256_file

REVISION = 'e92b25a616738fe95dc186b64dfb19f9c8525594'
SALT = 'causal-reader-fresh-sources-20261021-v1'
CURATOR_SHA = 'b863cb6e94cbac3489bdc0939a56177436985c5a809895cae410212ec63ac208'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def priority(row, role):
    return hashlib.sha256((SALT+role+row['document_id']).encode()).hexdigest()


def initialize(tokenizer_dir, raw_sha):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    curator.initialize(tokenizer_dir, minimum_binary_ops=2)
    curator.RAW_SHA = raw_sha


def worker(task):
    return curator.curate_document(*task[:2], curator.TOKENIZER, task[2], minimum_binary_ops=2)


def prepare(a):
    plan = read(a.plan); root = a.repo_root.resolve(); out = a.output.resolve()
    deadline = datetime.fromisoformat(a.deadline_utc)
    if deadline.tzinfo is None:
        raise ValueError('UTC deadline required')
    # Four CPU cores including the parent, inherited by tokenizer subprocesses.
    cores = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, cores)
    def check():
        if datetime.now(timezone.utc) >= deadline or (out/'STOP').exists():
            raise TimeoutError('Finite CPU preparation boundary; partials retained')
        if shutil.disk_usage(root).free <= 35*2**30:
            raise RuntimeError('Preserve filesystem reserve; partials retained')
        available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if available < 512*2**30:
            raise RuntimeError('Preserve host-memory reserve')
        if out.exists() and sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 16*2**30:
            raise RuntimeError('Finite16GiB source/output bound')
    check(); out.mkdir(parents=True, exist_ok=False)
    if sha256_file(curator.__file__) != CURATOR_SHA or plan['source_revision'] != REVISION:
        raise ValueError('Exact existing curator and source revision required')
    inventory = root/'local/data/sources/HuggingFaceTB--finemath/SOURCE.json'
    if sha256_file(inventory) != plan['source_inventory_sha256']:
        raise ValueError('Pinned source inventory changed')
    prior = read(root/'local/data/candidate-utility-v30-curation61-v1/plan.json')
    for path, digest in prior['prior_metadata_files'].items():
        if sha256_file(path) != digest:
            raise ValueError('Original exclusion/tokenizer source changed')
    metadata = sorted((root/'local/data/finemath').glob('*.documents.jsonl'))
    metadata += [root/'local/nvme-study/data'/directory/'novel-membership.jsonl' for directory in
                 ('real-source-balanced-v17-fresh-source-v1',
                  'real-source-balanced-v17-fresh-source-shard62-v1')]
    metadata += [root/'local/data/candidate-utility-v30-curation61-v1/TRAIN-membership.jsonl']
    old_pages, old_content = set(), set()
    pins = {str(p): sha256_file(p) for p in metadata}
    for path in metadata:
        for row in rows(path):
            if {'text','prefix','future','answer'} & row.keys():
                raise ValueError('Membership must have no prior source payload')
            old_pages.add(page(row['url']))
            old_content.update(row[k] for k in ('document_id','content_sha256') if row.get(k))
    expressions = [root/'local/data/real-cpt-scaffold-v29-cpu-v2'/name
                   for name in ('TRAIN.jsonl','DEV_calculation.jsonl')]
    expressions += [root/'local/nvme-study/data'/name for name in PRIOR_PANELS]
    expressions += [root/'local/data/candidate-utility-v30-curation61-v1/eligible-new-TRAIN.jsonl']
    old_expressions = set()
    for path in expressions:
        pins[str(path)] = sha256_file(path)
        for row in rows(path):
            if row.get('normalized_expression'):
                old_expressions.add(expression_key(row['normalized_expression']))
    write(out/'plan.json',dict(status='FROZEN_CPU_ONLY_NO_MODEL_RELEASE',
        source_plan=plan, source_plan_sha256=sha256_file(a.plan),
        source_sha256=sha256_file(__file__), curator_sources=curator.sources(),
        exclusions=pins, old_pages=len(old_pages),old_content=len(old_content),
        old_expressions=len(old_expressions), CPU_affinity=cores, deadline_utc=a.deadline_utc))
    url=f'https://huggingface.co/api/datasets/HuggingFaceTB/finemath/tree/{REVISION}/finemath-4plus?recursive=false&limit=1000'
    with urllib.request.urlopen(url, timeout=60) as response:
        public = {r['path']:r for r in json.load(response)}
    write(out/'public-source-metadata.json',public)
    source_files=[]
    for spec in plan['source_files']:
        check(); entry=public[spec['path']]
        if entry['size'] != spec['bytes']:
            raise ValueError('Pinned source size mismatch')
        expected=entry['lfs']['oid']; destination=out/Path(spec['path']).name
        if not re.fullmatch('[0-9a-f]{64}',expected):
            raise ValueError('Actual public LFS SHA required')
        request=f'https://huggingface.co/datasets/HuggingFaceTB/finemath/resolve/{REVISION}/{spec["path"]}'
        partial=destination.with_suffix('.download-part')
        with urllib.request.urlopen(request,timeout=60) as incoming,partial.open('xb') as stream:
            for chunk in iter(lambda:incoming.read(8*2**20),b''):
                check(); stream.write(chunk)
            stream.flush();os.fsync(stream.fileno())
        if partial.stat().st_size!=spec['bytes'] or sha256_file(partial)!=expected:
            raise ValueError('Download size/LFS-SHA/readback mismatch')
        partial.rename(destination);source_files.append((spec['path'],destination,expected))
        print('SOURCE_DOWNLOADED',spec['path'],flush=True)
    write(out/'downloads.json',[dict(filename=n,path=str(p),sha256=h) for n,p,h in source_files])
    import pyarrow as pa
    import pyarrow.parquet as pq
    counts=Counter();seen_pages=set();seen_content=set();kept=0
    raw=out/'fresh-TRAIN-raw.jsonl';membership=out/'fresh-TRAIN-membership.jsonl'
    with raw.open('xb') as stream,membership.open('x') as members:
        for filename,path,_ in source_files:
            position=0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512,columns=['url','text']):
                check(); selected=[]
                for index,url in enumerate(batch.column('url').to_pylist()):
                    if not isinstance(url,str):counts['unsupported_url']+=1;continue
                    key=page(url)
                    if key in old_pages:counts['old_page']+=1;continue
                    if key in seen_pages:counts['duplicate_page']+=1;continue
                    seen_pages.add(key);selected.append(index)
                values=batch.take(pa.array(selected,type=pa.int64())).to_pylist()
                for index,row in zip(selected,values):
                    text=row['text'];content=hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
                    if content in old_content:counts['old_content']+=1;continue
                    if content in seen_content:counts['duplicate_content']+=1;continue
                    seen_content.add(content);kept+=1
                    payload=json.dumps(dict(text=text,url=row['url']),ensure_ascii=False).encode()+b'\n'
                    meta=dict(source_line=kept,document_id=content,content_sha256=content,
                        url=row['url'],split='train',record_id=f'gate-feasibility:{filename}:{position+index}',
                        parquet_filename=filename,parquet_row=position+index,
                        raw_offset=stream.tell(),raw_bytes=len(payload))
                    stream.write(payload);members.write(json.dumps(meta)+'\n')
                position+=len(batch);counts['parquet_rows']+=len(batch)
            print('SOURCE_FILTERED',filename,'retained_total',kept,flush=True)
        for file in (stream,members):file.flush();os.fsync(file.fileno())
    raw_sha=sha256_file(raw);membership_sha=sha256_file(membership)
    write(out/'filter.json',dict(counts=dict(counts),retained=kept,raw_sha256=raw_sha,membership_sha256=membership_sha))
    def tasks():
        with raw.open('rb') as source,membership.open() as meta:
            for line,header in zip(source,meta,strict=True):
                yield json.loads(header),line,str(raw)
    processed=0;accepted=[]
    with (out/'accepted.jsonl').open('x') as accept,(out/'curation-audit.jsonl').open('x') as audit:
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(str(root/'local/assets/qwen3-1.7b-base'),raw_sha)) as pool:
            iterator=tasks()
            while True:
                check();batch=list(islice(iterator,512))
                if not batch:break
                for row,entry,_ in pool.map(worker,batch,chunksize=4):
                    processed+=1;counts['curation:'+entry['status']]+=1
                    if row is not None:accepted.append(row);accept.write(json.dumps(row)+'\n')
                    audit.write(json.dumps(entry)+'\n')
                if processed%4096==0:print('CPU_CURATION',processed,'accepted',len(accepted),flush=True)
        for file in (accept,audit):file.flush();os.fsync(file.fileno())
    if processed!=kept:raise ValueError('Incomplete source coverage')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(root/'local/assets/qwen3-1.7b-base',local_files_only=True)
    pool=[];accepted_sha=sha256_file(out/'accepted.jsonl')
    for row in accepted:
        check()
        if len(row['prefix_ids'])!=512:counts['short_prefix']+=1;continue
        if expression_key(row['boundary']['normalized_expression']) in old_expressions:
            counts['old_expression']+=1;continue
        try:pool.append(work_row(row,tokenizer,split='train',source_sha=accepted_sha))
        except ValueError:counts['unsupported_pack']+=1
    pool.sort(key=lambda r:priority(r,'calculation'))
    selected=[];seen_expressions=set()
    for row in pool:
        key=expression_key(row['normalized_expression'])
        if key in seen_expressions:counts['duplicate_expression']+=1;continue
        seen_expressions.add(key);selected.append(row)
    quota=16+2560+4096
    write(out/'arithmetic-availability.json',dict(eligible=len(selected),required=quota,counts=dict(counts)))
    if len(selected)<quota:
        raise ValueError(f'Finite source inventory insufficient: {len(selected)} < {quota}; do not weaken filters')
    selected=selected[:quota];math_ids={r['document_id'] for r in selected}
    candidates=heapq.nsmallest(40000,(r for r in rows(membership) if r['document_id'] not in math_ids),
                              key=lambda r:priority(r,'ordinary'))
    ordinary=[]
    with raw.open('rb') as source:
        for meta in candidates:
            check();source.seek(meta['raw_offset']);line=source.read(meta['raw_bytes'])
            text=json.loads(line)['text'][:curator.MAX_CHARACTERS]
            encoded=tokenizer(text,add_special_tokens=False,truncation=True,max_length=curator.MAX_TOKENS,
                return_offsets_mapping=True,return_attention_mask=False)
            tokens,offsets=encoded['input_ids'],encoded['offset_mapping']
            if len(tokens)<528:counts['ordinary_short']+=1;continue
            cut=512+int(priority(meta,'ordinary-cut')[:16],16)%(min(8192,len(tokens)-16)-512+1)
            try:window=curator.source_window(tokens,offsets,text,cut,tokenizer)
            except curator.base.safe.Unsupported:counts['ordinary_native_rejection']+=1;continue
            row=dict(item_id=meta['document_id']+':ordinary:'+str(cut),document_id=meta['document_id'],
                kind='ordinary',split='train',prefix_ids=window['ids'][:512],continuation16_ids=window['ids'][512:],
                stop_id=151643,source_file_sha256=raw_sha,
                provenance=dict(source_url=meta['url'],raw_source_line=meta['source_line'],
                                absolute_source_cut=cut,raw_line_sha256=hashlib.sha256(line).hexdigest()))
            ordinary.append(row)
            if len(ordinary)==quota:break
    if len(ordinary)!=quota:raise ValueError('Finite ordinary source pool insufficient')
    cohorts={}
    for kind,values in [('calculation',selected),('ordinary',ordinary)]:
        for name,start,end in [('qualification',0,16),('TRAIN',16,2576),('DEV_'+kind,2576,6672)]:
            for row in values[start:end]:
                row=dict(row);row['split']='train' if name in ('TRAIN','qualification') else 'followup_dev'
                cohorts.setdefault(name,[]).append(row)
    from train.real_cpt_scaffold import validate_source
    required_ids=math_ids|{r['document_id'] for r in ordinary}
    all_ids=set();all_pages=set()
    metadata_by_id={r['document_id']:r for r in rows(membership) if r['document_id'] in required_ids}
    files={}
    for name,values in cohorts.items():
        with (out/(name+'.jsonl')).open('x') as stream,(out/(name+'-questions.jsonl')).open('x') as questions:
            for row in values:
                validate_source(row,tokenizer,training=name in ('TRAIN','qualification'))
                meta=metadata_by_id[row['document_id']];key=page(meta['url'])
                if row['document_id'] in all_ids or key in all_pages:raise ValueError('Cohort source overlap')
                all_ids.add(row['document_id']);all_pages.add(key)
                row['source_locator']=dict(revision=REVISION,filename=meta['parquet_filename'],parquet_row=meta['parquet_row'])
                stream.write(json.dumps(row)+'\n')
                questions.write(json.dumps({k:row[k] for k in ('item_id','document_id','prefix_ids')})+'\n')
            for file in (stream,questions):file.flush();os.fsync(file.fileno())
        files[name]=dict(rows=len(values),sha256=sha256_file(out/(name+'.jsonl')),
                        questions_sha256=sha256_file(out/(name+'-questions.jsonl')))
    for path,digest in pins.items():
        if sha256_file(path)!=digest:raise ValueError('Original exclusions changed during preparation')
    write(out/'complete.json',dict(status='CPU_SOURCE_FEASIBILITY_COMPLETE_NOT_MODEL_RELEASE',
        cohorts=files,total_unique_pages=len(all_pages),raw_sha256=raw_sha,membership_sha256=membership_sha,
        eligible_calculation_documents=len(seen_expressions),counts=dict(counts),
        limits='No model calls/features/head fitting. Exact source/native independent cohort check remains required.'))
    print('CPU_SOURCE_FEASIBILITY_COMPLETE',json.dumps(files),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','plan','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True)
    prepare(parser.parse_args())


if __name__=='__main__':main()
