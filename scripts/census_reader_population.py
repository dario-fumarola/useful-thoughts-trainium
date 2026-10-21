"""Count prediction-position prevalence from frozen curation logs, without models.

The population is bounded, source-clean FineMath TRAIN documents already scanned
by the exact curator. This is a census of that frame, not all FineMath, and not
an estimate of the prevalence of every kind of useful reasoning.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

SALT = 'reader-population-census-20261021'
POST_PREFIX = {'accepted', 'unsupported_source_label', 'source_numeric_mismatch',
               'unsupported_full_raw_RHS'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            h.update(block)
    return h.hexdigest()


def position_summary(row):
    """Choose a position by identity/length only; never search for an answer.

    All native cuts with 512 past and 16 future tokens are in the denominator,
    including cuts not ending in '=' and cuts rejected by serialization checks.
    Accepted numerators are exactly the frozen curator's verified class before
    auxiliary packing and expression-novelty filters.
    """
    if row['split'] != 'train':
        raise ValueError('Only frozen TRAIN source logs')
    if row['status'] not in {'accepted', 'no_valid_bounded_raw_equation'}:
        raise ValueError('Unknown document outcome')
    length = row['bounded_native_tokens']
    if type(length) is not int or not 0 <= length <= 8208:
        raise ValueError('Original native token bound')
    end = min(8192, length-16)
    positions = max(0, end-512+1)
    sampled = (512 + int(hashlib.sha256((SALT+':cut:'+row['document_id']).encode()).hexdigest(), 16)
               % positions) if positions else None
    accepted, pre_label, seen = set(), set(), set()
    for candidate in row['candidate_audit']:
        cut, status = candidate['absolute_cut'], candidate['status']
        if type(cut) is not int or not 128 <= cut <= min(8192, length-16) or cut in seen:
            raise ValueError('Duplicate or out-of-range logged candidate')
        seen.add(cut)
        if status not in POST_PREFIX | {'prefix_or_serialization_ineligible'}:
            raise ValueError('Unknown original curator state: '+status)
        if status == 'accepted':
            accepted.add(cut)
        if cut >= 512 and status in POST_PREFIX:
            pre_label.add(cut)
    if (len(accepted) != row['accepted_candidate_count']
            or (row['status'] == 'accepted') != bool(accepted)
            or (accepted and row['selected_absolute_cut'] not in accepted)
            or (not accepted and row['selected_absolute_cut'] is not None)):
        raise ValueError('Original curation coverage/accounting disagrees')
    valid = {cut for cut in accepted if cut >= 512}
    return dict(document_id=row['document_id'], positions=positions,
                verified_boundaries=len(valid), pre_label_boundaries=len(pre_label),
                chosen_cut=sampled, chosen_verified=int(sampled in valid),
                chosen_pre_label=int(sampled in pre_label))


def aggregate(rows):
    """Sufficient counts for both position-weighted and document-weighted views."""
    count = Counter()
    sum_verified_fraction = sum_pre_label_fraction = 0.
    for row in rows:
        count['documents'] += 1
        n, k, p = row['positions'], row['verified_boundaries'], row['pre_label_boundaries']
        count['positions'] += n
        count['verified_boundaries'] += k
        count['pre_label_boundaries'] += p
        if n:
            count['eligible_documents'] += 1
            count['documents_containing_verified_boundary'] += int(k > 0)
            count['chosen_verified'] += row['chosen_verified']
            count['chosen_pre_label'] += row['chosen_pre_label']
            sum_verified_fraction += k/n
            sum_pre_label_fraction += p/n
    eligible = count['eligible_documents']
    return dict(counts=dict(count),
        position_weighted_verified_prevalence=count['verified_boundaries']/count['positions'] if count['positions'] else None,
        document_weighted_expected_verified_prevalence=sum_verified_fraction/eligible if eligible else None,
        document_contains_verified_prevalence=count['documents_containing_verified_boundary']/eligible if eligible else None,
        fixed_hash_position_verified_prevalence=count['chosen_verified']/eligible if eligible else None,
        position_weighted_pre_label_incidence=count['pre_label_boundaries']/count['positions'] if count['positions'] else None,
        document_weighted_expected_pre_label_incidence=sum_pre_label_fraction/eligible if eligible else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None or plan['identity'] != SALT:
        raise ValueError('Frozen plan and finite UTC deadline required')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    started = time.monotonic()
    def check():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError('Finite census deadline; retain partial evidence')
        if shutil.disk_usage(args.output.parent).free < 35*2**30:
            raise RuntimeError('Preserve free disk reserve')
        available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if available < 512*2**30:
            raise RuntimeError('Preserve host-memory reserve')
        if args.output.exists() and sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > 256*2**20:
            raise RuntimeError('Finite census output ceiling')
    check()
    args.output.mkdir(exist_ok=False)
    if digest(__file__) != plan['script_sha256']:
        raise ValueError('Census source changed after freeze')
    used = set()
    for spec in plan['used_source_projections']:
        if digest(spec['path']) != spec['sha256']:
            raise ValueError('Used-source identity projection changed')
        number = 0
        for number, line in enumerate(Path(spec['path']).open(), 1):
            row = json.loads(line)
            if set(row) != {'item_id', 'document_id', 'prefix_ids'}:
                raise ValueError('Use prefix-only identity projections, never saved outcomes')
            used.add(row['document_id'])
        if number != spec['documents']:
            raise ValueError('Complete original source identities required')
    if len(used) != sum(s['documents'] for s in plan['used_source_projections']):
        raise ValueError('Original cohorts must have disjoint source identities')
    groups = {'all': [], 'unused': []}
    seen = set(); source_summaries = {}; bindings = {}
    # Each projection has six integer columns, source identity and a used flag;
    # no text, target tokens, hidden features or likelihoods are read or saved.
    projection = args.output/'document-counts.jsonl.gz'
    with gzip.open(projection, 'wt') as output:
        for spec in plan['audit_sources']:
            check(); h = hashlib.sha256(); pool = []; number = 0
            with Path(spec['path']).open('rb') as stream:
                for number, line in enumerate(stream, 1):
                    if number % 8192 == 1:
                        check()
                    h.update(line); row = position_summary(json.loads(line))
                    doc = row['document_id']
                    if doc in seen:
                        raise ValueError('Sources must be unique across both frozen pools')
                    seen.add(doc); row['used'] = doc in used
                    row['pool'] = spec['name']
                    output.write(json.dumps(row, separators=(',', ':'))+'\n')
                    # Store only count tuples for aggregation, not full records.
                    compact = {k: row[k] for k in ('positions', 'verified_boundaries',
                        'pre_label_boundaries', 'chosen_verified', 'chosen_pre_label')}
                    pool.append(compact)
                    groups['all'].append(compact)
                    if not row['used']:
                        groups['unused'].append(compact)
                    if number % 131072 == 0:
                        print('CENSUS', spec['name'], number, flush=True)
            if h.hexdigest() != spec['sha256'] or number != spec['documents']:
                raise ValueError('Frozen audit digest/count differs')
            source_summaries[spec['name']] = aggregate(pool)
            bindings[spec['name']] = dict(sha256=h.hexdigest(), documents=number)
    result = dict(status='COMPLETE_BOUNDED_READER_POPULATION_CENSUS', identity=SALT,
        plan_sha256=digest(args.plan), sources=bindings, by_source=source_summaries,
        populations={k: aggregate(v) for k, v in groups.items()},
        source_projections_used_only_for_identity=True, used_source_documents=len(used),
        projection_sha256=digest(projection), wall_seconds=time.monotonic()-started,
        accelerator_chip_hours=0, model_calls=0,
        limits=plan['limits'], completed_utc=datetime.now(timezone.utc).isoformat())
    (args.output/'census.json').write_text(json.dumps(result, indent=2)+'\n')
    print('COMPLETE_BOUNDED_READER_POPULATION_CENSUS', flush=True)


if __name__ == '__main__':
    main()
