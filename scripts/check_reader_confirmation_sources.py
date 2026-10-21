"""Check fresh native windows against pinned Parquet sources, without models."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
import json
import os
from pathlib import Path
import time


def read(path): return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*2**20),b''): h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','data','downloads','used-source-plan','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True);a=parser.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);start=time.monotonic()
    deadline=datetime.fromisoformat(a.deadline_utc)
    def check():
        if datetime.now(timezone.utc)>=deadline: raise TimeoutError('Finite source validation')
    p=read(a.plan);complete=read(a.data/'complete.json')
    if digest(a.plan)!=complete['plan_sha256']: raise ValueError('Frozen preparation plan')
    used=set()
    for spec in read(a.used_source_plan)['used_source_projections']:
        if digest(spec['path'])!=spec['sha256']: raise ValueError('Used-source metadata changed')
        used.update(json.loads(line)['document_id'] for line in Path(spec['path']).open())
    if digest(p['census_projection']['path'])!=p['census_projection']['sha256']:
        raise ValueError('Frozen population projection')
    with gzip.open(p['census_projection']['path'],'rt') as stream:
        candidates=(dict(document_id=r['document_id'],chosen_cut=r['chosen_cut']) for line in stream
                    if not (r:=json.loads(line))['used'] and r['positions']>0)
        natural=heapq.nsmallest(4096,candidates,key=lambda r:hashlib.sha256(
            ('reader-fresh-confirmation-20261021:natural:'+r['document_id']).encode()).hexdigest())
    groups=defaultdict(list);seen=set();all_rows=[]
    for cohort,expected in [('CONFIRM_natural',4096),('CONFIRM_calculation',1024)]:
        spec=complete['cohorts'][cohort]
        if (digest(a.data/(cohort+'.jsonl'))!=spec['labels_sha256']
                or digest(a.data/(cohort+'-questions.jsonl'))!=spec['questions_sha256']):
            raise ValueError('Complete frozen source views')
        rows=[json.loads(line) for line in (a.data/(cohort+'.jsonl')).open()]
        questions=[json.loads(line) for line in (a.data/(cohort+'-questions.jsonl')).open()]
        if len(rows)!=expected or len(questions)!=expected: raise ValueError('All chosen sources required')
        if cohort=='CONFIRM_natural' and [r['document_id'] for r in rows]!=[r['document_id'] for r in natural]:
            raise ValueError('Natural source selection used more than identity/length')
        for i,(r,q) in enumerate(zip(rows,questions)):
            if q!={k:r[k] for k in ('item_id','document_id','prefix_ids')}:
                raise ValueError('Writer projection contains labels or differs from source')
            if r['document_id'] in used or r['document_id'] in seen: raise ValueError('Source overlap')
            seen.add(r['document_id'])
            if cohort=='CONFIRM_natural' and r['provenance']['absolute_source_cut']!=natural[i]['chosen_cut']:
                raise ValueError('A natural cut was replaced after selection')
            if len(r['prefix_ids'])!=512 or len(r['continuation16_ids'])!=16: raise ValueError('Native widths')
            groups[r['provenance']['source_locator']['filename']].append(r);all_rows.append(r)
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    from train.calculation_boundaries import maximal_lhs,parse_numeric
    from train.scalar_completion import parse_initial_scalar_v2
    tokenizer=AutoTokenizer.from_pretrained(p['tokenizer_directory'],local_files_only=True,trust_remote_code=False)
    bindings={r['filename']:r for r in read(a.downloads)};checked=0;equations=0
    for name,rows in groups.items():
        check();spec=bindings[name]
        if digest(spec['path'])!=spec['sha256']: raise ValueError('Pinned source-Parquet digest')
        parquet=pq.ParquetFile(spec['path']);ends=[];end=0;by_group=defaultdict(list)
        for index in range(parquet.num_row_groups):
            end+=parquet.metadata.row_group(index).num_rows;ends.append(end)
        for r in rows:
            position=r['provenance']['source_locator']['parquet_row']
            index=next(i for i,end in enumerate(ends) if position<end)
            by_group[index].append((position-(ends[index-1] if index else 0),r))
        for index,wanted in by_group.items():
            check();table=parquet.read_row_group(index,columns=['url','text'])
            for position,r in wanted:
                source=table.slice(position,1).to_pylist()[0];text=source['text']
                if (source['url']!=r['provenance']['source_url']
                        or hashlib.sha256(' '.join(text.split()).encode()).hexdigest()!=r['document_id']):
                    raise ValueError('Source URL/content identity')
                tokens=tokenizer(text[:65536],add_special_tokens=False,truncation=True,
                    max_length=8208,return_offsets_mapping=True,return_attention_mask=False)
                ids=tokens['input_ids'];cut=r['provenance']['absolute_source_cut']
                if ids[cut-512:cut+16]!=r['prefix_ids']+r['continuation16_ids']:
                    raise ValueError('Original native source window mismatch')
                if r['kind']=='calculation':
                    past=tokenizer.decode(r['prefix_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    lhs=parse_numeric(maximal_lhs(past));offset=tokens['offset_mapping'][cut-1][1]
                    rhs=parse_initial_scalar_v2(text[:65536][offset:offset+8192],'length_cap')
                    if lhs.binary_ops<2 or rhs['status']!='valid' or str(lhs.value)!=r['numeric_value'] or rhs['value']!=r['numeric_value']:
                        raise ValueError('Independent arithmetic source support')
                    equations+=1
                checked+=1
        print('SOURCE_VERIFIED',name,checked,flush=True)
    if checked!=5120 or equations!=1024: raise ValueError('Incomplete source validation')
    result=dict(status='PASS_FRESH_READER_PINNED_PARQUET_SOURCE_CHECK',documents=checked,
        verified_equations=equations,source_disjoint_from_prior_documents=len(used),
        natural_selection='Reconstructed independently from identity/length only; no replacements',
        plan_sha256=digest(a.plan),preparation_sha256=digest(a.data/'complete.json'),
        checker_sha256=digest(__file__),wall_seconds=time.monotonic()-start,model_calls=0,
        limits='Shared tokenizer and arithmetic grammar; independently reads every selected pinned-Parquet row. No predictor execution or scientific outcome scoring.')
    with a.output.open('x') as stream: stream.write(json.dumps(result,indent=2)+'\n')
    print(result['status'],flush=True)


if __name__=='__main__': main()
