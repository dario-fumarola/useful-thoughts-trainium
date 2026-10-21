"""Select and reconstruct disjoint, unenriched native-token LM cohorts on CPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import time

from weekend_thinking.artifacts import sha256_file, atomic_json
from train.uniform_lm import SCHEMA


def priority(salt, role, document_id):
    return hashlib.sha256(f'{salt}:{role}:{document_id}'.encode()).hexdigest()


def select_sources(records, counts, blocked, salt):
    """Only identity, complete-window length and prior-source membership select."""
    total = sum(counts.values())
    eligible = (dict(document_id=r['document_id'], positions=r['positions'])
                for r in records if not r['used'] and r['positions'] > 0
                and r['document_id'] not in blocked)
    selected = heapq.nsmallest(total, eligible,
        key=lambda r: priority(salt, 'split', r['document_id']))
    if len(selected) != total or len({r['document_id'] for r in selected}) != total:
        raise ValueError('Insufficient distinct untouched source documents')
    output = []; first = 0
    for cohort, count in counts.items():
        for row in selected[first:first+count]:
            cut = 512 + int(priority(salt, 'cut', row['document_id']), 16) % row['positions']
            output.append(dict(row, cohort=cohort, cut=cut))
        first += count
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True)
    a = parser.parse_args(); p = json.loads(a.plan.read_text())
    deadline = datetime.fromisoformat(a.deadline_utc)
    if deadline.tzinfo is None or p['identity'] != SCHEMA:
        raise ValueError('An explicit source plan and UTC deadline are required')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    start = time.monotonic()
    def check():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError('Finite uniform-LM source preparation')
        if shutil.disk_usage(a.output.parent).free < 35*2**30:
            raise RuntimeError('Preserve the filesystem reserve')
        memory = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                      if s.startswith('MemAvailable:'))
        if memory < 512*2**30:
            raise RuntimeError('Preserve host-memory reserve')
        if a.output.exists() and sum(x.stat().st_size for x in a.output.iterdir()
                                    if x.is_file()) > p['maximum_output_MiB']*2**20:
            raise RuntimeError('Finite dataset output budget')
    def bound(spec):
        path = Path(spec['path'])
        if sha256_file(path) != spec['sha256']:
            raise ValueError('Source binding changed: '+str(path))
        return path
    if sha256_file(__file__) != p['preparer_sha256']:
        raise ValueError('Freeze the preparer before source selection')
    check(); a.output.mkdir(exist_ok=False)
    blocked = set()
    for spec in p['additional_used_sources']:
        blocked.update(json.loads(s)['document_id'] for s in bound(spec).open())
    with gzip.open(bound(p['population']), 'rt') as stream:
        selection = select_sources((json.loads(s) for s in stream), p['cohorts'], blocked, p['salt'])
    atomic_json(a.output/'selection.json', selection)
    selected = {r['document_id']: r for r in selection}
    # No source, window or split can be replaced after this identity-only freeze.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(p['tokenizer_directory'], local_files_only=True,
                                              trust_remote_code=False)
    rows = {}
    for pool in p['source_pools']:
        check(); membership = bound(pool['membership']); raw = bound(pool['raw'])
        metadata = [r for s in membership.open() if (r := json.loads(s))['document_id'] in selected]
        with raw.open('rb') as source:
            for m in metadata:
                if len(rows) % 128 == 0: check()
                doc = m['document_id']; choice = selected[doc]
                if doc in rows: raise ValueError('A selected source occurs twice')
                source.seek(m['raw_offset']); line = source.read(m['raw_bytes']); original = json.loads(line)
                text = original['text']
                if (original['url'] != m['url'] or m['split'] != 'train'
                        or hashlib.sha256(' '.join(text.split()).encode()).hexdigest() != doc):
                    raise ValueError('Original source identity changed')
                ids = tokenizer(text[:65536], add_special_tokens=False, truncation=True,
                    max_length=8208, return_attention_mask=False)['input_ids']
                positions = max(0, min(8192, len(ids)-16)-512+1)
                if positions != choice['positions']:
                    raise ValueError('Preselected native-window population changed')
                cut = choice['cut']; window = ids[cut-512:cut+16]
                if len(window) != 528:
                    raise ValueError('Retain the chosen source and stop; never replace a window')
                cohort = choice['cohort']
                rows[doc] = dict(schema=SCHEMA, item_id=f'{doc}:uniform:{cut}', document_id=doc,
                    split='train' if cohort in ('TRAIN', 'QUALIFICATION') else cohort.lower(),
                    cohort=cohort, ids=window,
                    provenance=dict(pool=pool['name'], filename=m['parquet_filename'],
                        parquet_row=m['parquet_row'], native_cut=cut,
                        raw_line_sha256=hashlib.sha256(line).hexdigest()))
    if set(rows) != set(selected):
        raise ValueError('Every selected source must reconstruct')
    files = {}
    for cohort, count in p['cohorts'].items():
        path = a.output/(cohort+'.jsonl')
        with path.open('x') as out:
            for choice in selection:
                if choice['cohort'] == cohort:
                    out.write(json.dumps(rows[choice['document_id']], separators=(',', ':'))+'\n')
        files[cohort] = dict(rows=count, sha256=sha256_file(path), predicted_tokens=count*527)
    check()
    atomic_json(a.output/'complete.json', dict(status='COMPLETE_UNIFORM_LM_SOURCE_PREPARATION',
        schema=SCHEMA, plan_sha256=sha256_file(a.plan), selection_sha256=sha256_file(a.output/'selection.json'),
        files=files, original_tokens_per_window=528, predicted_tokens_per_window=527,
        wall_seconds=time.monotonic()-start, model_calls=0, accelerator_chip_hours=0,
        limits='Unenriched eligible-document sample in a source-clean finite frame, not all-corpus token weighting or proven novelty relative to base pretraining. CPU preparation alone does not qualify native training.'))
    print('COMPLETE_UNIFORM_LM_SOURCE_PREPARATION', flush=True)


if __name__ == '__main__':
    main()
