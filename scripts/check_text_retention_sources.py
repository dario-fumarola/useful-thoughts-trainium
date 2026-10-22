"""Independently reconstruct every selected cross-domain source window."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scripts.prepare_real_cpt_scaffold import page
from weekend_thinking.artifacts import sha256_file


def check(plan_file):
    started = time.monotonic()
    plan = json.loads(Path(plan_file).read_text())
    root = Path(plan['data_directory'])
    complete = json.loads((root/'complete.json').read_text())
    source, data = root/'source.parquet', root/'ENGLISH_RETENTION.jsonl'
    if (sha256_file(source) != plan['source_sha256']
            or sha256_file(data) != complete['data_sha256']):
        raise ValueError('Original source or complete selected data changed')
    selected = [json.loads(line) for line in data.open()]
    eligible = [json.loads(line) for line in (root/'eligible-index.jsonl').open()]
    expected = sorted(eligible, key=lambda row: hashlib.sha256(
        (plan['salt']+':select:'+row['document_id']).encode()).hexdigest())[:1024]
    if ([v['document_id'] for v in expected] != [v['document_id'] for v in selected]
            or len({v['document_id'] for v in selected}) != 1024
            or len({v['canonical_page'] for v in selected}) != 1024):
        raise ValueError('Fixed hash selection and unique sources must be preserved')
    pages = {v['canonical_page'] for v in selected}
    content = {v['document_id'] for v in selected}
    for name, digest in plan['exclusion_membership_files'].items():
        if sha256_file(name) != digest:
            raise ValueError('Pinned exclusion membership changed')
        for line in Path(name).open():
            row = json.loads(line)
            if (page(row['url']) in pages
                    or any(row.get(key) in content for key in ('document_id', 'content_sha256'))):
                raise ValueError('A selected source overlaps known mathematical data')
    tokenizer = AutoTokenizer.from_pretrained(plan['tokenizer'], local_files_only=True,
                                              trust_remote_code=False)
    by_row = {v['parquet_row']: v for v in selected}
    found = set()
    offset = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=256, columns=['text', 'url', 'id']):
        for number, row in enumerate(batch.to_pylist(), offset):
            if number not in by_row:
                continue
            saved = by_row[number]
            text = row['text']
            doc = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
            ids = tokenizer(text[:65536], add_special_tokens=False, truncation=True,
                max_length=8208, return_attention_mask=False)['input_ids']
            positions = max(0, min(8192, len(ids)-16)-512+1)
            cut = 512+int(hashlib.sha256((plan['salt']+':cut:'+doc).encode()).hexdigest(), 16) % positions
            if (doc != saved['document_id'] or page(row['url']) != saved['canonical_page']
                    or row['id'] != saved['source_record_id']
                    or hashlib.sha256(text.encode()).hexdigest() != saved['source_text_sha256']
                    or saved['ids'] != ids[cut-512:cut+16] or saved['cut'] != cut
                    or saved['positions'] != positions or len(saved['ids']) != 528):
                raise ValueError('Selected original source or native token window differs')
            found.add(doc)
        offset += batch.num_rows
        if offset >= plan['maximum_source_rows']:
            break
    if found != content:
        raise ValueError('Every selected source must independently reconstruct')
    result = dict(status='PASS_INDEPENDENT_GENERAL_WEB_SOURCES',
        documents=len(found), data_sha256=sha256_file(data),
        source_sha256=sha256_file(source), selection_recomputed=True,
        all_original_windows_match=True, known_math_sources_excluded=True,
        wall_seconds=time.monotonic()-started, model_calls=0,
        scope='Separate original-record reconstruction and hash selection; shared tokenizer and canonical-page definition.')
    (root/'independent-source-check.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    print(json.dumps(check(parser.parse_args().plan), indent=2))
