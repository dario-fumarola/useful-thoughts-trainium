"""Read-only count check of a completed, insufficient source inventory.

Does not rerun curation, acquire sources, build replacement cohorts, or load
language-model weights. Reconstructs eligibility from saved accepted records
without calling the producer's work_row/selection routine.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path

from train.expression_identity import expression_key
from train.real_cpt_scaffold import ANSWER,OPEN,WIDTH,native_metadata,verified_work
from weekend_thinking.artifacts import atomic_json,sha256_file


def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open() as stream:
        for line in stream:yield json.loads(line)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','tokenizer','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-utc',required=True)
    a=p.parse_args();deadline=datetime.fromisoformat(a.deadline_utc)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
    def check():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Finite independent count check')
    root=Path(__file__).resolve().parents[1];plan=read(a.data/'plan.json')
    reported=read(a.data/'arithmetic-availability.json');filtered=read(a.data/'filter.json')
    if (a.data/'complete.json').exists():raise ValueError('This checker is only for the insufficient inventory')
    if sha256_file(root/'scripts/prepare_causal_reader_data.py')!=plan['source_sha256']:
        raise ValueError('Original frozen producer source changed')
    for name,digest in plan['curator_sources'].items():
        if sha256_file(root/name)!=digest:raise ValueError('Original exact curator changed')
    old=set()
    for filename,digest in plan['exclusions'].items():
        check()
        if sha256_file(filename)!=digest:raise ValueError('Original exclusions changed')
        for row in rows(filename):
            if row.get('normalized_expression'):old.add(expression_key(row['normalized_expression']))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    vocab,special=native_metadata(tok)
    markers=len(tok.encode(OPEN,add_special_tokens=False))+len(tok.encode(ANSWER,add_special_tokens=False))
    accepted={};keys=set();counts=Counter();reasons=Counter()
    for row in rows(a.data/'accepted.jsonl'):
        check();item=row['document_id']
        if item in accepted:raise ValueError('Repeated accepted source')
        accepted[item]=row['absolute_source_cut'];counts['accepted']+=1
        past=row['prefix_ids'];future=row['natural_future16_ids']
        if len(past)!=512:counts['short_prefix']+=1;continue
        key=expression_key(row['boundary']['normalized_expression'])
        if key in old:counts['old_expression']+=1;continue
        try:
            _,trace=verified_work(past,tok)
            if 512+markers+len(trace)+16>WIDTH:raise ValueError('complete640pack')
            if (len(future)!=16 or any(type(t) is not int or not 0<=t<vocab
                                      or t in special for t in past+future)):
                raise ValueError('native_source')
        except ValueError as error:
            counts['unsupported_pack']+=1;reasons[str(error)]+=1;continue
        counts['packable_pre_dedup']+=1
        if key in keys:counts['duplicate_expression']+=1
        else:keys.add(key)
    audited=Counter();matched=set()
    for index,(entry,meta) in enumerate(zip(rows(a.data/'curation-audit.jsonl'),
                                          rows(a.data/'fresh-TRAIN-membership.jsonl'),strict=True)):
        if index%8192==0:check()
        if entry['document_id']!=meta['document_id'] or entry['source_line']!=meta['source_line']:
            raise ValueError('Original curation audit/source coverage differs')
        audited[entry['status']]+=1
        if entry['status']=='accepted':
            if accepted.get(entry['document_id'])!=entry['selected_absolute_cut']:
                raise ValueError('Accepted saved equation differs from source audit')
            matched.add(entry['document_id'])
    if (sum(audited.values())!=filtered['retained'] or len(matched)!=counts['accepted']
            or audited['accepted']!=counts['accepted'] or len(keys)!=reported['eligible']
            or reported['required']!=6672):raise ValueError('Availability count or complete coverage mismatch')
    for field in ('short_prefix','old_expression','unsupported_pack','duplicate_expression'):
        if counts[field]!=reported['counts'][field]:raise ValueError('Eligibility exclusion count differs')
    if len(keys)>=6672:raise ValueError('Reported source insufficiency not reproduced')
    atomic_json(a.output,dict(status='PASS_SAVED_POOL_INSUFFICIENCY_COUNT_NOT_COHORT_QUALIFICATION',
        source_clean_documents=sum(audited.values()),curation_status_counts=dict(audited),
        accepted_equation_documents=counts['accepted'],eligibility_counts=dict(counts),
        pack_rejection_reasons=dict(reasons),unique_eligible_calculations=len(keys),required=6672,
        shortfall=6672-len(keys),source_sha256=sha256_file(__file__),
        artifacts={name:sha256_file(a.data/name) for name in ('plan.json','filter.json','accepted.jsonl',
            'curation-audit.jsonl','fresh-TRAIN-membership.jsonl','arithmetic-availability.json')},
        limits='Independent control flow/count reconstruction, shared exact parsing/verified-work definitions. No Parquet/native audit of a complete selected cohort, source expansion, replacement cohort, model call, feature or head fitting. Does not prove all FineMath or other data sources insufficient.'))
    print('SAVED_POOL_INSUFFICIENCY_VERIFIED',len(keys),'required6672',flush=True)


if __name__=='__main__':main()
