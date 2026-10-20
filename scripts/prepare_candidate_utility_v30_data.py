"""Bounded fresh TRAIN source preparation; no model or accelerator execution.

Keep the first 65,536 pinned Parquet rows, exclude prior retained pages/content
before converting text to Python, and reuse the exact verified raw arithmetic
curator. Selection uses source truth and exact arithmetic, never model outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import islice
import json
import multiprocessing
import os
from pathlib import Path
import time

from scripts.prepare_real_cpt_scaffold import PRIOR_PANELS, page, work_row
from train import computable_equations_raw as curator
from train.expression_identity import expression_key
from weekend_thinking.artifacts import atomic_json, require_utc_deadline, sha256_file, utc_now

REVISION = 'e92b25a616738fe95dc186b64dfb19f9c8525594'
CURATOR_SHA = 'b863cb6e94cbac3489bdc0939a56177436985c5a809895cae410212ec63ac208'
SALT = 'candidate-utility-v30-fresh-TRAIN-source61-v1'


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    for line in Path(path).open():
        yield json.loads(line)


def initialize(tokenizer_dir, raw_sha):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    curator.initialize(tokenizer_dir, minimum_binary_ops=2)
    # The constant is used only to bind provenance in the reused callable.
    curator.RAW_SHA = raw_sha


def worker(task):
    return curator.curate_document(*task[:2], curator.TOKENIZER, task[2], minimum_binary_ops=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root', 'source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    assert 1 <= args.workers <= 8
    deadline = require_utc_deadline(args.deadline_utc)
    def check():
        assert datetime.now(timezone.utc) < deadline, 'Finite CPU deadline; partials retained.'
        assert __import__('shutil').disk_usage(args.output.parent).free > 35 * 2**30
    check()
    assert sha256_file(curator.__file__) == CURATOR_SHA
    download = read(args.source / 'download-complete.json')
    assert download['revision'] == REVISION and download['split'] == 'train'
    assert download['filename'] == 'finemath-4plus/train-00061-of-00064.parquet'
    parquet = Path(download['path'])
    assert sha256_file(parquet) == download['sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.repo_root
    anchor_path = root / 'local/data/real-cpt-scaffold-v29-cpu-v2/manifest.json'
    assert sha256_file(anchor_path) == '77c8774d3fdeb283728878c44522dab541ec61a213d3f89ece471b3dcc45fb3f'
    anchor = read(anchor_path)
    tokenizer_pins = {}
    for name, info in anchor['local_tokenizer_files'].items():
        path = root / 'local/assets/qwen3-1.7b-base' / name
        assert sha256_file(path) == info['sha256']
        tokenizer_pins[str(path)] = info['sha256']
    metadata = sorted((root / 'local/data/finemath').glob('*.documents.jsonl'))
    for folder in ('real-source-balanced-v17-fresh-source-v1', 'real-source-balanced-v17-fresh-source-shard62-v1'):
        source = read(root / 'local/nvme-study/data' / folder / 'filtered-complete.json')
        path = Path(source['membership_file'])
        assert sha256_file(path) == source['membership_sha256']
        metadata.append(path)
    pins = {str(path): sha256_file(path) for path in metadata}
    pins.update(tokenizer_pins)
    pins[str(anchor_path)] = sha256_file(anchor_path)
    for path in metadata:
        if str(path) in anchor['inputs']:
            assert pins[str(path)] == anchor['inputs'][str(path)]['sha256']
    old_pages, old_content = set(), set()
    for path in metadata:
        for row in lines(path):
            assert not {'text', 'prefix', 'future', 'answer'} & row.keys()
            old_pages.add(page(row['url']))
            old_content.update(row[k] for k in ('document_id', 'content_sha256') if row.get(k))
    expression_files = [root / 'local/data/real-cpt-scaffold-v29-cpu-v2' / name
                        for name in ('TRAIN.jsonl', 'DEV_calculation.jsonl')]
    expression_files += [root / 'local/nvme-study/data' / name for name in PRIOR_PANELS]
    old_expressions = set()
    for path in expression_files:
        pins[str(path)] = sha256_file(path)
        if path.parent == anchor_path.parent:
            assert pins[str(path)] == anchor['files'][path.name]['sha256']
        for row in lines(path):
            if 'normalized_expression' in row:
                old_expressions.add(expression_key(row['normalized_expression']))
    plan = dict(status='FROZEN_CPU_TRAIN_DATA_PREPARATION_NOT_MODEL_RELEASE', utc=utc_now(),
        dataset='HuggingFaceTB/finemath', revision=REVISION, split='train', source=download,
        max_parquet_rows=65536, prior_metadata_files=pins, old_pages=len(old_pages),
        old_content_keys=len(old_content), old_exact_expressions=len(old_expressions),
        curator_source_files=curator.sources(), script_sha256=sha256_file(__file__),
        workers=args.workers, deadline_utc=args.deadline_utc,
        bounds=dict(raw_characters=65536, native_tokens=8208, max_cut=8192, min_binary_ops=2,
                    selected_prefix_tokens=512, work_actions_including_stop=128, reader_width=672),
        limits=['CPU source preparation only; model sampling protocol and qualification remain unreleased.',
                'One fixed source shard/row range; prior source metadata used only for exclusions, including reserved IDs, never old TEST payloads.',
                'Source RHS verifies eligibility; exact work is reconstructed only from the prefix. This is supervised curation.',
                'Exact ordered-AST/page/content exclusion is not semantic or pretraining novelty.'])
    atomic_json(args.output / 'plan.json', plan)
    import pyarrow as pa
    import pyarrow.parquet as pq
    counts = Counter(); seen_pages, seen_content = set(), set()
    raw, membership = args.output / 'TRAIN-raw.jsonl', args.output / 'TRAIN-membership.jsonl'
    source_position = kept = 0
    with raw.open('x') as stream, membership.open('x') as members:
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=512, columns=['url', 'text']):
            check()
            batch = batch.slice(0, min(len(batch), 65536 - source_position))
            if not len(batch): break
            urls = batch.column(batch.schema.get_field_index('url')).to_pylist()
            eligible = []
            for index, url in enumerate(urls):
                if not isinstance(url, str): counts['unsupported_URL'] += 1; continue
                key = page(url)
                if key in old_pages: counts['prior_page_excluded_before_text'] += 1; continue
                if key in seen_pages: counts['within_slice_duplicate_page'] += 1; continue
                seen_pages.add(key); eligible.append(index)
            # Text for excluded prior URLs is never converted to Python.
            selected = batch.take(pa.array(eligible, type=pa.int64())).to_pylist()
            counts['new_TRAIN_texts_decoded'] += len(selected)
            for index, row in zip(eligible, selected):
                text = row['text']; assert isinstance(text, str)
                key = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
                if key in old_content: counts['prior_content_excluded'] += 1; continue
                if key in seen_content: counts['within_slice_duplicate_content'] += 1; continue
                seen_content.add(key); kept += 1; position = source_position + index
                stream.write(json.dumps(dict(text=text, url=row['url'], source=dict(
                    dataset='HuggingFaceTB/finemath', revision=REVISION,
                    filename=download['filename'], parquet_row=position)), ensure_ascii=False) + '\n')
                members.write(json.dumps(dict(source_line=kept, document_id=key,
                    content_sha256=key, url=row['url'], split='train',
                    record_id=f'finemath-v30-TRAIN:61:{position}', parquet_row=position)) + '\n')
            source_position += len(batch)
            if source_position == 65536: break
        for file in (stream, members): file.flush(); os.fsync(file.fileno())
    counts['parquet_rows_scanned'] = source_position; counts['source_clean_TRAIN_documents'] = kept
    assert source_position == min(65536, pq.ParquetFile(parquet).metadata.num_rows)
    raw_sha, membership_sha = sha256_file(raw), sha256_file(membership)
    atomic_json(args.output / 'filtered.json', dict(status='COMPLETE_BOUNDED_SOURCE_TRAIN_FILTER',
        counts=dict(counts), raw_sha256=raw_sha, membership_sha256=membership_sha))
    print('FILTER_COMPLETE', json.dumps(dict(counts)), flush=True)
    def tasks():
        with raw.open('rb') as incoming, membership.open() as meta:
            for number, (line, header) in enumerate(zip(incoming, meta, strict=True), 1):
                value = json.loads(header)
                assert value['source_line'] == number and value['split'] == 'train'
                yield value, line, str(raw.resolve())
    started = time.monotonic(); processed = 0; accepted = []
    try:
        with (args.output / 'accepted.jsonl').open('x') as accept, (args.output / 'audit.jsonl').open('x') as audit:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize, initargs=(str(root / 'local/assets/qwen3-1.7b-base'), raw_sha)) as pool:
                iterator = tasks()
                while True:
                    check(); batch = list(islice(iterator, 256))
                    if not batch: break
                    for row, entry, statistics in pool.map(worker, batch, chunksize=4):
                        processed += 1; counts['curation:' + entry['status']] += 1
                        if row is not None:
                            assert row['split'] == 'train' and row['boundary']['binary_ops'] >= 2
                            accepted.append(row); accept.write(json.dumps(row) + '\n')
                        audit.write(json.dumps(entry) + '\n')
                    accept.flush(); audit.flush()
                    print('CPU_CURATION', processed, 'accepted', len(accepted), 'seconds', round(time.monotonic()-started, 1), flush=True)
            for file in (accept, audit): file.flush(); os.fsync(file.fileno())
        assert processed == kept
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(root / 'local/assets/qwen3-1.7b-base', local_files_only=True)
        pool = []; accepted_sha = sha256_file(args.output / 'accepted.jsonl')
        for row in accepted:
            check()
            if len(row['prefix_ids']) != 512: counts['pool:short_prefix'] += 1; continue
            identity = expression_key(row['boundary']['normalized_expression'])
            if identity in old_expressions: counts['pool:previous_expression'] += 1; continue
            try:
                value = work_row(row, tokenizer, split='train', source_sha=accepted_sha)
            except ValueError:
                counts['pool:unsupported_native_work_pack'] += 1; continue
            assert Fraction(value['full_calculation_text'].strip().rsplit(' = ', 1)[1]) == Fraction(value['numeric_value'])
            pool.append(value)
        pool.sort(key=lambda row: hashlib.sha256((SALT + row['document_id']).encode()).hexdigest())
        seen_expressions = set(); eligible = []
        for row in pool:
            key = expression_key(row['normalized_expression'])
            if key in seen_expressions: counts['pool:duplicate_expression'] += 1; continue
            seen_expressions.add(key); eligible.append(row)
        with (args.output / 'eligible-new-TRAIN.jsonl').open('x') as stream:
            for row in eligible: stream.write(json.dumps(row) + '\n')
            stream.flush(); os.fsync(stream.fileno())
        for path, digest in pins.items(): assert sha256_file(path) == digest
        assert sha256_file(raw) == raw_sha and sha256_file(membership) == membership_sha
        assert curator.sources() == plan['curator_source_files']
        result = dict(status='COMPLETE_CPU_FRESH_TRAIN_POOL_NOT_MODEL_RELEASE', utc=utc_now(),
            counts=dict(counts), eligible_documents=len(eligible),
            eligible_operations=dict(Counter(len(row['full_calculation_text'].splitlines()) for row in eligible)),
            files={name: dict(bytes=(args.output/name).stat().st_size, sha256=sha256_file(args.output/name))
                   for name in ('TRAIN-raw.jsonl', 'TRAIN-membership.jsonl', 'accepted.jsonl', 'audit.jsonl', 'eligible-new-TRAIN.jsonl')},
            plan_sha256=sha256_file(args.output / 'plan.json'), script_sha256=sha256_file(__file__),
            limits=plan['limits'] + ['No fixed diagnostic cohort or scientific/model release follows from pool availability.'])
        atomic_json(args.output / 'complete.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        atomic_json(args.output / 'failed.json', dict(status='INCOMPLETE_CPU_PREPARATION_PARTIALS_RETAINED',
            error=f'{type(error).__name__}: {error}', processed=processed, accepted=len(accepted), counts=dict(counts)))
        raise


if __name__ == '__main__':
    main()
