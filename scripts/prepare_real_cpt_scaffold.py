"""Freeze V29 source cohorts and audit CPU packing. No model/accelerator calls.

Reuses legitimate local FineMath assets. Reserved TEST payloads are never parsed.
New arithmetic DEV comes from previously acquired, page/content-excluded source
TRAIN shards, after excluding previous expressions and evaluation documents.
Arithmetic eligibility was verified against source RHS by the original curator;
computation targets themselves are reconstructed from the causal prefix alone.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from transformers import AutoTokenizer

from train.expression_identity import expression_key
from train import real_cpt_scaffold as scaffold
from weekend_thinking.artifacts import atomic_json, sha256_file


RECORD_SHA = 'bd81972fff861f497ce61479079d6904e1baef0a335dfc848cc33e4d90d069ff'
MEMBERSHIP_SHA = '9ee1c7f5dfdad156db8ef34fea2fa62315cf00be0cf6a8bd49c454628ab28cbf'
TRAIN_SHA = '437c64bc9ce4b451f4d5da07fccca9e108ee9add75b3c955bc54d2a2eca6b915'
SALT = 'real-cpt-scaffold-v29-source-freeze-v1'
HEADER = re.compile(rb'^\{"id":"([^"\\]+)","document_id":"([^"\\]+)",'
                    rb'"split":"(train|dev|test)","prefix":')
PRIOR_PANELS = [
    'real-prefix-partial-labels-v9/DEV.jsonl',
    'real-trace-cpt-v10-held-panels-v1/DEV_novel_expression256.jsonl',
    'real-trace-cpt-v10-held-panels-v1/DEV_ordinary256.jsonl',
    'real-trace-cpt-v10-held-panels-v2/DEV_novel_expression256.jsonl',
    'real-trace-cpt-v10-held-panels-v2/DEV_ordinary256.jsonl',
    'real-target-sft-v15-held-v1/DEV_fresh384.jsonl',
    'real-source-balanced-v17-confirmation-v1/DEV_confirm512.jsonl',
    'real-source-balanced-v17-confirmation-v1/DEV_ordinary256.jsonl',
]
FRESH_CURATIONS = [
    ('real-source-balanced-v17-fresh-curation-v1',
     '6f6958e229d344f2976a241418d06169a2b15256647440ce908b9b4c3c154db5'),
    ('real-source-balanced-v17-fresh-curation-shard62-v1',
     '99b26935acf45fa9b52ba31c3cbd4370f7146852beb858c8aeec84093a5eb90b'),
]


def page(url):
    parsed = urlsplit(url)
    host = (parsed.hostname or '').casefold().removeprefix('www.')
    return host + unquote(parsed.path).casefold().rstrip('/')


def priority(row, role):
    return hashlib.sha256(f'{SALT}:{role}:{row["document_id"]}'.encode()).hexdigest()


def read_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def receipt(path):
    return dict(bytes=path.stat().st_size, sha256=sha256_file(path))


def write_rows(output, name, rows):
    path = output / name
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',', ':'), ensure_ascii=True) + '\n')
    return dict(rows=len(rows), **receipt(path))


def work_row(row, tokenizer, *, split, source_sha):
    past = list(row['prefix_ids'])
    text, ids = scaffold.verified_work(past, tokenizer)
    markers = (len(tokenizer.encode(scaffold.OPEN, add_special_tokens=False))
               + len(tokenizer.encode(scaffold.ANSWER, add_special_tokens=False)))
    if len(past) + markers + len(ids) + 16 > scaffold.WIDTH:
        raise ValueError('Complete auxiliary view exceeds640; never truncate')
    result = dict(item_id=row['item_id'], document_id=row['document_id'], kind='calculation',
        split=split, prefix_ids=past,
        continuation16_ids=list(row.get('continuation16_ids', row.get('natural_future16_ids', []))),
        full_calculation_text=text, full_calculation_ids=ids, stop_id=tokenizer.eos_token_id,
        normalized_expression=row.get('normalized_expression', row.get('boundary', {}).get('normalized_expression')),
        source_file_sha256=source_sha,
        provenance=row.get('source_provenance', {k: row[k] for k in (
            'raw_source_line', 'source_url', 'raw_line_sha256', 'raw_text_sha256',
            'absolute_source_cut') if k in row}))
    for key in ('numeric_surface_ids', 'numeric_surface_text', 'closing_boundary_ids', 'numeric_value'):
        result[key] = row[key]
    scaffold.validate_source(result, tokenizer, training=split == 'train')
    if row.get('full_calculation_ids', ids) != ids or row.get('full_calculation_text', text) != text:
        raise ValueError('Existing teacher disagrees with pure prefix reconstruction')
    return result


def freeze(root, output, *, arithmetic_dev=768, ordinary_dev=4096):
    if arithmetic_dev < 256 or ordinary_dev < 256:
        raise ValueError('Meaningful document-level cohorts required')
    source = root / 'local/data/finemath'
    study = root / 'local/nvme-study/data'
    records = source / 'production-500000.audit-v2.records.jsonl'
    membership = source / 'production-500000.audit-v2.records.documents.jsonl'
    train_path = study / 'real-prefix-partial-labels-v9/TRAIN.jsonl'
    inputs = {}
    for path, expected in ((membership, MEMBERSHIP_SHA), (train_path, TRAIN_SHA)):
        inputs[str(path)] = receipt(path)
        if inputs[str(path)]['sha256'] != expected:
            raise ValueError(f'Frozen input changed: {path}')
    tokenizer_dir = root / 'local/assets/qwen3-1.7b-base'
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    train = [work_row(r, tok, split='train', source_sha=TRAIN_SHA) for r in read_rows(train_path)]
    print(json.dumps(dict(stage='prefix_only_TRAIN_labels', documents=len(train))), flush=True)
    if len(train) != len({r['document_id'] for r in train}) or len(train) != 5120:
        raise ValueError('Original5120 unique calculation training documents required')

    previous, missing_panels = [], []
    for rel in PRIOR_PANELS:
        path = study / rel
        if not path.exists():
            missing_panels.append(str(path))
            continue
        inputs[str(path)] = receipt(path)
        previous += read_rows(path)
    if missing_panels:
        raise ValueError(f'Cannot establish documented previous panel exclusions: {missing_panels}')
    fixed_path = source / 'fixed-local-evaluation-v1.ids.json'
    fixed = json.loads(fixed_path.read_text())
    inputs[str(fixed_path)] = receipt(fixed_path)
    previous_documents = {r['document_id'] for r in previous}
    previous_documents.update(fixed['splits']['dev']['document_ids'])
    previous_expressions = {expression_key(r['normalized_expression']) for r in train + previous
                            if 'normalized_expression' in r}
    blocked_documents = previous_documents | {r['document_id'] for r in train}

    metadata_rows = read_rows(membership)
    metadata = {r['document_id']: r for r in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError('One retained source record per document required')
    prior_pages = {page(r['url']) for r in metadata_rows}
    prior_content = {r['content_sha256'] for r in metadata_rows}
    previous_pages = {page(r['url']) for r in metadata_rows if r['document_id'] in previous_documents}
    previous_content = {r['content_sha256'] for r in metadata_rows if r['document_id'] in previous_documents}
    pool, excluded = [], Counter()
    for directory, expected in FRESH_CURATIONS:
        path = study / directory / 'accepted.jsonl'
        inputs[str(path)] = receipt(path)
        if inputs[str(path)]['sha256'] != expected:
            raise ValueError('Existing fresh source curation changed')
        for row in read_rows(path):
            excluded['source_accepted'] += 1
            # These source TRAIN shards were projected as DEV by the existing
            # curator. Reject any incompatible split instead of relabeling TEST.
            if row['split'] != 'dev':
                raise ValueError('Existing source-TRAIN-derived DEV projection required')
            if row['document_id'] in blocked_documents:
                excluded['previous_document'] += 1; continue
            if expression_key(row['boundary']['normalized_expression']) in previous_expressions:
                excluded['previous_expression'] += 1; continue
            if page(row['source_url']) in prior_pages:
                excluded['prior_source_page'] += 1; continue
            if row['upstream_normalized_content_sha256'] in prior_content:
                excluded['prior_source_content'] += 1; continue
            try:
                candidate = work_row(row, tok, split='followup_dev', source_sha=expected)
            except ValueError as exc:
                # Shape/trace eligibility is decided without model outcomes.
                excluded['native_pack_or_prefix_work_rejection'] += 1
                excluded['reason:' + str(exc)] += 1
                continue
            pool.append(candidate)
    pool.sort(key=lambda r: priority(r, 'arithmetic-dev'))
    selected, seen_documents, seen_expressions, seen_pages = [], set(), set(), set()
    for row in pool:
        key = expression_key(row['normalized_expression'])
        p = page(row['provenance']['source_url'])
        if row['document_id'] in seen_documents or key in seen_expressions or p in seen_pages:
            excluded['duplicate_candidate_document_page_or_expression'] += 1; continue
        seen_documents.add(row['document_id']); seen_expressions.add(key); seen_pages.add(p)
        selected.append(row)
    eligible_arithmetic = len(selected)
    print(json.dumps(dict(stage='fresh_arithmetic_inventory', eligible=eligible_arithmetic,
                          exclusions=dict(excluded))), flush=True)
    if eligible_arithmetic < arithmetic_dev:
        raise ValueError(f'Fresh fitting arithmetic inventory {eligible_arithmetic} < {arithmetic_dev}')
    dev_calculation = selected[:arithmetic_dev]
    blocked_documents.update(r['document_id'] for r in dev_calculation)

    ordinary_ids = {}
    for split, count in (('train', len(train) * scaffold.ORDINARY_PER_CALCULATION), ('dev', ordinary_dev)):
        train_pages = {page(r['url']) for r in metadata_rows if r['split'] == 'train'}
        train_content = {r['content_sha256'] for r in metadata_rows if r['split'] == 'train'}
        eligible = [r for r in metadata_rows if r['split'] == split
                    and r['document_id'] not in blocked_documents
                    and page(r['url']) not in previous_pages
                    and r['content_sha256'] not in previous_content
                    and (split == 'train' or (page(r['url']) not in train_pages
                         and r['content_sha256'] not in train_content))]
        eligible.sort(key=lambda r: priority(r, f'ordinary-{split}'))
        if len(eligible) < count:
            raise ValueError(f'Insufficient source-separated ordinary {split} documents')
        ordinary_ids[split] = {r['record_id']: r for r in eligible[:count]}
    ordinary = {'train': {}, 'dev': {}}
    scan = Counter()
    digest = hashlib.sha256()
    with records.open('rb') as f:
        for line in f:
            digest.update(line)
            match = HEADER.match(line)
            if not match:
                raise ValueError('Source header changed; never fall back to parsing unknown split payloads')
            record_id, document_id, split = (v.decode('ascii') for v in match.groups())
            scan['headers:' + split] += 1
            if split == 'test':
                scan['reserved_payloads_skipped'] += 1
                continue
            if record_id not in ordinary_ids[split]: continue
            raw = json.loads(line)
            if raw['id'] != record_id or raw['document_id'] != document_id or raw['split'] != split:
                raise ValueError('Native record/header disagreement')
            meta = ordinary_ids[split][record_id]
            if document_id != meta['document_id'] or meta['split'] != split:
                raise ValueError('Membership/native source disagreement')
            row = dict(item_id=record_id, document_id=document_id, split=split, kind='ordinary',
                prefix_ids=raw['prefix'], continuation16_ids=raw['future'], stop_id=tok.eos_token_id,
                source_file_sha256=RECORD_SHA, source_record_sha256=hashlib.sha256(line).hexdigest(),
                provenance={k:meta[k] for k in ('source_line', 'url', 'content_sha256', 'component_root_source_line')})
            scaffold.validate_source(row, tok, training=split == 'train')
            if record_id in ordinary[split]: raise ValueError('Repeated source record')
            ordinary[split][record_id] = row
            scan['payloads_parsed:' + split] += 1
    if digest.hexdigest() != RECORD_SHA:
        raise ValueError('Original source records changed')
    inputs[str(records)] = dict(bytes=records.stat().st_size, sha256=digest.hexdigest())
    print(json.dumps(dict(stage='native_source_readback', counts=dict(scan))), flush=True)
    for split in ordinary:
        if ordinary[split].keys() != ordinary_ids[split].keys():
            raise ValueError('Selected native records are missing')
    training = train + list(ordinary['train'].values())
    training.sort(key=lambda r: priority(r, 'training-storage-order'))
    dev_ordinary = sorted(ordinary['dev'].values(), key=lambda r: priority(r, 'ordinary-dev'))
    train_docs = {r['document_id'] for r in training}
    held_docs = {r['document_id'] for r in dev_calculation + dev_ordinary}
    if (len(train_docs) != len(training) or len(held_docs) != len(dev_calculation) + len(dev_ordinary)
            or train_docs & held_docs or held_docs & previous_documents):
        raise ValueError('Unique, previous-panel-separated held documents required')
    def identity(r):
        p = r['provenance']
        return page(p.get('url', p.get('source_url', '')))
    if set(map(identity, training)) & set(map(identity, dev_calculation + dev_ordinary)):
        raise ValueError('Canonical source page overlaps training and held cohorts')

    inventory = {arm: Counter() for arm in scaffold.ARMS}
    for number, row in enumerate(training, 1):
        clean_reference = None
        for arm in scaffold.ARMS:
            view = scaffold.training_views(row, tok, arm)
            clean = view.clean
            clean_sha = hashlib.sha256(clean['input_ids'].numpy().tobytes()
                + clean['labels'].numpy().tobytes() + clean['component_masks']['source'].numpy().tobytes()).hexdigest()
            if clean_reference is not None and clean_sha != clean_reference:
                raise ValueError('Clean LM views differ across arms')
            clean_reference = clean_sha
            counts = inventory[arm]
            counts['documents'] += 1
            counts['source_targets'] += int(clean['component_masks']['source'].sum())
            counts['source_allocated_positions'] += scaffold.WIDTH
            if view.auxiliary:
                counts['auxiliary_examples'] += 1
                counts['auxiliary_allocated_positions'] += scaffold.WIDTH
                for name, mask in view.auxiliary['component_masks'].items():
                    counts[name + '_targets'] += int(mask.sum())
        if number % 2048 == 0:
            print(json.dumps(dict(stage='all_arm_native_pack_audit', documents=number)), flush=True)
    for seed in (2026102029, 2026102030):
        order = scaffold.grouped_order(training, seed=seed, passes=2)
        if Counter(order) != Counter({i:2 for i in range(len(training))}):
            raise ValueError('Incorrect source exposures')
        for start in range(0, len(order), 32):
            if sum(training[i]['kind'] == 'calculation' for i in order[start:start+32]) != 8:
                raise ValueError('Nonconstant auxiliary batch fraction')

    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, values in [('TRAIN.jsonl', training), ('DEV_calculation.jsonl', dev_calculation),
                         ('DEV_ordinary.jsonl', dev_ordinary)]:
        files[name] = write_rows(output, name, values)
    files['DEV_calculation-questions.jsonl'] = write_rows(output, 'DEV_calculation-questions.jsonl', [
        {k:r[k] for k in ('item_id', 'document_id', 'prefix_ids')} for r in dev_calculation])
    manifest = dict(schema=scaffold.SCHEMA, status='CPU_DATA_AND_PACKING_COMPLETE_NOT_TRAINING_RELEASE',
        utc=datetime.now(timezone.utc).isoformat(), selection_salt=SALT,
        source_files={str(Path(__file__).resolve()):receipt(Path(__file__)),
                      'train/real_cpt_scaffold.py':receipt(Path(scaffold.__file__))},
        inputs=inputs, files=files, per_pass={k:dict(v) for k,v in inventory.items()},
        source_scan=dict(scan), reserved_TEST_payloads_parsed=0,
        previous_panel_documents=len(previous_documents), previous_expressions=len(previous_expressions),
        fresh_arithmetic_eligible=eligible_arithmetic, arithmetic_exclusions=dict(excluded),
        arithmetic_operations=dict(Counter(len(r['full_calculation_text'].splitlines()) for r in dev_calculation)),
        train_held_document_overlap=len(train_docs & held_docs), held_previous_panel_document_overlap=0,
        train_held_canonical_page_overlap=0,
        training_seeds=[2026102029, 2026102030], passes_proposed=2, updates_proposed=1280,
        local_tokenizer_files={p.name:receipt(p) for p in sorted(tokenizer_dir.iterdir())
            if p.is_file() and p.suffix == '.json'},
        limits=[
            'CPU data/packing only: no weight loading, model forward, accelerator work, learning result, or training release.',
            'FineMath remains mathematical text; this is not a representative general-language pretraining corpus.',
            'Calculation TRAIN5120 and raw ordinary TRAIN15360 form a deliberately enriched25% calculation mixture.',
            'Arithmetic candidates were selected by source-RHS verification; pure prefix labels do not make selection autonomous.',
            'Fresh arithmetic is unused source-TRAIN-derived DEV, not officialTEST; ordered exact-value AST and canonical-page exclusions are not semantic/pretraining novelty.',
            'Ordinary DEV excludes documented fixed and real-document panels; earlier corpus inventory is not claimed to be universally unseen text.',
            'Existing raw-corpus near-duplicate screening is probabilistic. Cross-source page/content exclusions do not prove absence of semantic duplicates.',
            'Source exposures/clean LM are identical, but work labels, auxiliary lengths, gradient difficulty and actual compute differ among arms.',
            'Extra raw continuation loss upweights selected spans in every arm; the raw control is weighted continued NTP, not uniform whole-corpus NTP.',
        ])
    atomic_json(output/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arithmetic-dev', type=int, default=768)
    parser.add_argument('--ordinary-dev', type=int, default=4096)
    args = parser.parse_args()
    result = freeze(args.repo_root.resolve(), args.output.resolve(),
                    arithmetic_dev=args.arithmetic_dev, ordinary_dev=args.ordinary_dev)
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs', 'local_tokenizer_files')}, indent=2))


if __name__ == '__main__':
    main()
