"""Independent native/source checks of every proposed reader-gate document.

Reads pinned TRAIN Parquet rows and tokenizer files only. No language-model
weights, feature extraction, source acquisition or scientific outcome scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

from scripts.prepare_real_cpt_scaffold import page
from train.calculation_boundaries import maximal_lhs, parse_numeric
from train.expression_identity import expression_key
from train.scalar_completion import parse_initial_scalar_v2
from weekend_thinking.artifacts import sha256_file


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','tokenizer','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True)
    a=parser.parse_args();deadline=datetime.fromisoformat(a.deadline_utc)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
    def check():
        if datetime.now(timezone.utc)>=deadline:
            raise TimeoutError('Finite independent source-check deadline')
    plan=json.loads((a.data/'plan.json').read_text())
    complete=json.loads((a.data/'complete.json').read_text())
    if complete['status']!='CPU_SOURCE_FEASIBILITY_COMPLETE_NOT_MODEL_RELEASE':
        raise ValueError('Complete original preparation required')
    reused=plan.get('reused_pool')
    if reused:
        for name,digest in reused['files'].items():
            if sha256_file(Path(reused['directory'])/name)!=digest:
                raise ValueError('Closed reused pool changed')
    downloads=json.loads((a.data/'downloads.json').read_text())
    expected_downloads={r['path']:r.get('sha256') for r in plan['source_plan']['source_files']}
    if reused:
        expected_downloads.update({r['filename']:r['sha256'] for r in
            json.loads((Path(reused['directory'])/'downloads.json').read_text())})
    if (len(downloads)!=len(expected_downloads)
            or {r['filename'] for r in downloads}!=set(expected_downloads)):
        raise ValueError('Exact frozen additional and reused source shards required')
    for binding in downloads:
        digest=expected_downloads[binding['filename']]
        if digest is not None and binding['sha256']!=digest:
            raise ValueError('Download differs from frozen public LFS SHA')
    old_pages=set();old_contents=set();old_expressions=set()
    for filename,digest in plan['exclusions'].items():
        check()
        if sha256_file(filename)!=digest:raise ValueError('Original exclusion file changed')
        for row in rows(filename):
            if row.get('url'):
                old_pages.add(page(row['url']))
                old_contents.update(row[k] for k in ('document_id','content_sha256') if row.get(k))
            if row.get('normalized_expression'):
                old_expressions.add(expression_key(row['normalized_expression']))
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    special=set(tokenizer.all_special_ids)
    groups=defaultdict(list);seen_docs=set();seen_pages=set();seen_expressions=set();counts=Counter()
    expected={'TRAIN':5120,'DEV_calculation':4096,'DEV_ordinary':4096,'qualification':32}
    for name,n in expected.items():
        path=a.data/(name+'.jsonl');questions=a.data/(name+'-questions.jsonl')
        if sha256_file(path)!=complete['cohorts'][name]['sha256']:
            raise ValueError('Frozen cohort changed')
        if sha256_file(questions)!=complete['cohorts'][name]['questions_sha256']:
            raise ValueError('Frozen writer projection changed')
        values=list(rows(path));q=list(rows(questions))
        if len(values)!=n or len(q)!=n:raise ValueError('Wrong prospective cohort count')
        for row,question in zip(values,q):
            if set(question)!={'item_id','document_id','prefix_ids'}:
                raise ValueError('Writer projection contains forbidden label fields')
            if question!={k:row[k] for k in question}:
                raise ValueError('Writer prefix differs from scored source')
            if (len(row['prefix_ids'])!=512 or len(row['continuation16_ids'])!=16
                    or set(row['prefix_ids']+row['continuation16_ids'])&special):
                raise ValueError('Native nonspecial complete512+16 source required')
            if row['document_id'] in seen_docs:raise ValueError('Cross-cohort document overlap')
            seen_docs.add(row['document_id']);counts[name+':'+row['kind']]+=1
            loc=row['source_locator']
            if loc['revision']!=plan['source_plan']['source_revision']:
                raise ValueError('Original source revision changed')
            groups[loc['filename']].append(row)
    verified_equations=0;checked=0
    for filename,values in groups.items():
        check()
        binding=next(r for r in downloads if r['filename']==filename)
        path=Path(binding['path']).resolve()
        allowed={a.data.resolve()}
        if reused:allowed.add(Path(reused['directory']).resolve())
        if path.parent not in allowed or path.name!=Path(filename).name:
            raise ValueError('Parquet must belong to a pinned original or extension pool')
        if sha256_file(path)!=binding['sha256']:raise ValueError('Pinned original Parquet changed')
        parquet=pq.ParquetFile(path);ends=[];end=0
        for index in range(parquet.num_row_groups):
            end+=parquet.metadata.row_group(index).num_rows;ends.append(end)
        by_group=defaultdict(list)
        for row in values:
            original=row['source_locator']['parquet_row']
            group=next(i for i,end in enumerate(ends) if original<end)
            by_group[group].append(row)
        for group,selected in by_group.items():
            check();table=parquet.read_row_group(group,columns=['url','text'])
            start=ends[group-1] if group else 0
            for row in selected:
                check();original=table.slice(row['source_locator']['parquet_row']-start,1).to_pylist()[0]
                text=original['text'];doc=hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
                key=page(original['url'])
                if (doc!=row['document_id'] or doc in old_contents
                        or key in old_pages or key in seen_pages):
                    raise ValueError('Actual original source identity/exclusion/uniqueness failure')
                seen_pages.add(key);text=text[:65536]
                native=tokenizer(text,add_special_tokens=False,truncation=True,max_length=8208,
                    return_offsets_mapping=True,return_attention_mask=False)
                ids,offsets=native['input_ids'],native['offset_mapping']
                cut=row['provenance']['absolute_source_cut'];first=cut-512;end=cut+16
                if ids[first:end]!=row['prefix_ids']+row['continuation16_ids']:
                    raise ValueError('Stored prefix/future differs from original native stream')
                past=tokenizer.decode(row['prefix_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                entire=tokenizer.decode(ids[first:end],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                if (past!=text[offsets[first][0]:offsets[cut-1][1]]
                        or entire!=text[offsets[first][0]:offsets[end-1][1]]
                        or tokenizer.encode(past,add_special_tokens=False)!=row['prefix_ids']):
                    raise ValueError('Original UTF8/native boundary reconstruction failure')
                if row['kind']=='calculation':
                    expression=maximal_lhs(past);identity=expression_key(expression)
                    if (identity in seen_expressions or identity in old_expressions
                            or identity!=expression_key(row['normalized_expression'])):
                        raise ValueError('Exact arithmetic identity overlap/mismatch')
                    seen_expressions.add(identity)
                    root=parse_numeric(expression).value
                    if root!=Fraction(row['numeric_value']):raise ValueError('Prefix-computable root differs')
                    trace=row['full_calculation_text'];last=None
                    for line in trace.splitlines():
                        lhs,rhs=line.split(' = ')
                        left,right=parse_numeric(lhs).value,parse_numeric(rhs).value
                        if left!=right:raise ValueError('Supplied equation is false')
                        last=right;verified_equations+=1
                    if last!=root or tokenizer.encode(trace,add_special_tokens=False)!=row['full_calculation_ids']:
                        raise ValueError('Trace/native root mismatch')
                    source_rhs=text[offsets[cut-1][1]:offsets[cut-1][1]+8192]
                    proof=parse_initial_scalar_v2(source_rhs,'length_cap')
                    if proof['status']!='valid' or Fraction(proof['value'])!=root:
                        raise ValueError('Actual original RHS does not prove the computed root')
                checked+=1
        print('SOURCE_VERIFIED',filename,'total',checked,flush=True)
    if checked!=13344 or len(seen_expressions)!=6672:
        raise ValueError('Incomplete independent cohort reconstruction')
    result=dict(status='PASS_ALL_SELECTED_SOURCE_NATIVE_EXCLUSION_AND_EXACT_EQUATION_CHECKS',
        documents=checked,unique_pages=len(seen_pages),exact_expressions=len(seen_expressions),
        independently_true_equations=verified_equations,counts=dict(counts),
        source_sha256=sha256_file(__file__),preparation_sha256=sha256_file(a.data/'complete.json'),
        limits='Within-repository independent reconstruction only. No model/feature/head/scientific result or pretraining novelty claim.')
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
