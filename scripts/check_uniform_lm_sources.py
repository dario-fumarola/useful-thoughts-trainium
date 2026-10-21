"""Recount the split and verify every LM window against pinned Parquet rows."""
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

from weekend_thinking.artifacts import atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'data', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True)
    a = parser.parse_args(); start = time.monotonic()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    deadline = datetime.fromisoformat(a.deadline_utc)
    def check():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError('Finite independent source verification')
    def bound(spec):
        path = Path(spec['path'])
        if sha256_file(path) != spec['sha256']:
            raise ValueError('Pinned source changed: '+str(path))
        return path
    p = json.loads(a.plan.read_text()); complete = json.loads((a.data/'complete.json').read_text())
    if (complete['status'] != 'COMPLETE_UNIFORM_LM_SOURCE_PREPARATION'
            or complete['plan_sha256'] != sha256_file(a.plan)
            or p['checker_sha256'] != sha256_file(__file__)):
        raise ValueError('Original preparation and independent checker bindings required')
    blocked = set()
    for spec in p['additional_used_sources']:
        blocked.update(json.loads(s)['document_id'] for s in bound(spec).open())
    with gzip.open(bound(p['population']), 'rt') as stream:
        candidates = (dict(document_id=r['document_id'], positions=r['positions'])
            for line in stream if (r := json.loads(line))['positions'] > 0
            and not r['used'] and r['document_id'] not in blocked)
        selected = heapq.nsmallest(sum(p['cohorts'].values()), candidates,
            key=lambda r: hashlib.sha256(f"{p['salt']}:split:{r['document_id']}".encode()).hexdigest())
    if sha256_file(a.data/'selection.json') != complete['selection_sha256']:
        raise ValueError('Original identity-only freeze changed')
    frozen = json.loads((a.data/'selection.json').read_text())
    expected = []; cursor = 0
    for cohort, count in p['cohorts'].items():
        for r in selected[cursor:cursor+count]:
            cut = 512 + int(hashlib.sha256(f"{p['salt']}:cut:{r['document_id']}".encode()).hexdigest(),16) % r['positions']
            expected.append(dict(r, cohort=cohort, cut=cut))
        cursor += count
    if frozen != expected or len(frozen) != sum(p['cohorts'].values()):
        raise ValueError('The split/cut was changed after source selection')
    sources = defaultdict(list); seen = set()
    for cohort, count in p['cohorts'].items():
        check(); path = a.data/(cohort+'.jsonl'); spec = complete['files'][cohort]
        if sha256_file(path) != spec['sha256']:
            raise ValueError('A prepared native source window changed')
        rows = [json.loads(s) for s in path.open()]
        chosen = [r for r in frozen if r['cohort'] == cohort]
        if len(rows) != count or len(chosen) != count or spec['predicted_tokens'] != count*527:
            raise ValueError('Complete fixed cohort required')
        for row, choice in zip(rows, chosen):
            doc = row['document_id']; cut = choice['cut']
            if (doc != choice['document_id'] or doc in seen or doc in blocked
                    or row['cohort'] != cohort or len(row['ids']) != 528
                    or row['schema'] != p['identity'] or row['provenance']['native_cut'] != cut
                    or row['split'] != ('train' if cohort in ('TRAIN','QUALIFICATION') else cohort.lower())):
                raise ValueError('Unique separated sources and exact original window required')
            seen.add(doc); sources[row['provenance']['filename']].append(row)
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    tokenizer = AutoTokenizer.from_pretrained(p['tokenizer_directory'], local_files_only=True,
                                              trust_remote_code=False)
    downloads = {s['filename']: s for s in json.loads(bound(p['parquet_downloads']).read_text())}
    checked = 0
    for name, rows in sources.items():
        check(); spec = downloads[name]
        if sha256_file(spec['path']) != spec['sha256']:
            raise ValueError('Pinned original Parquet changed')
        parquet = pq.ParquetFile(spec['path']); ends = []; total = 0
        for group in range(parquet.num_row_groups):
            total += parquet.metadata.row_group(group).num_rows; ends.append(total)
        groups = defaultdict(list)
        for row in rows:
            position = row['provenance']['parquet_row']
            group = next(i for i, end in enumerate(ends) if position < end)
            groups[group].append((position-(ends[group-1] if group else 0), row))
        for group, wanted in groups.items():
            check(); text_column = parquet.read_row_group(group, columns=['text']).column('text')
            for offset, row in wanted:
                text = text_column[offset].as_py()
                if hashlib.sha256(' '.join(text.split()).encode()).hexdigest() != row['document_id']:
                    raise ValueError('Original source identity mismatch')
                ids = tokenizer(text[:65536], add_special_tokens=False, truncation=True,
                    max_length=8208, return_attention_mask=False)['input_ids']
                cut = row['provenance']['native_cut']
                if ids[cut-512:cut+16] != row['ids']:
                    raise ValueError('Original native token window mismatch')
                checked += 1
        print('SOURCE_VERIFIED', name, checked, flush=True)
    if checked != sum(p['cohorts'].values()):
        raise ValueError('Independent checks must cover every chosen source')
    atomic_json(a.output, dict(status='PASS_UNIFORM_LM_PINNED_SOURCE_AND_SPLIT_CHECK',
        documents=checked, cohorts=p['cohorts'], prior_source_exclusions=len(blocked),
        plan_sha256=sha256_file(a.plan), preparation_sha256=sha256_file(a.data/'complete.json'),
        checker_sha256=sha256_file(__file__), wall_seconds=time.monotonic()-start,
        model_calls=0, accelerator_chip_hours=0,
        limits='Independent identity/cut reconstruction and every original Parquet window; shared tokenizer, no model execution or pretraining-novelty guarantee.'))
    print('PASS_UNIFORM_LM_PINNED_SOURCE_AND_SPLIT_CHECK', flush=True)


if __name__ == '__main__':
    main()
