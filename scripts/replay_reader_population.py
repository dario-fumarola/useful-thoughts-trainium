"""Source-free census recount; no production census imports or model libraries."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('counts', 'census', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.census.read_text())
    digest = hashlib.sha256()
    with args.counts.open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            digest.update(block)
    if digest.hexdigest() != report['projection_sha256']:
        raise ValueError('Published projection digest differs')
    groups = defaultdict(lambda: dict(counts=defaultdict(int), fractions=[], pre_fractions=[]))
    seen = set()
    with gzip.open(args.counts, 'rt') as stream:
        for line in stream:
            r = json.loads(line); doc = r['document_id']
            n, k, p = r['positions'], r['verified_boundaries'], r['pre_label_boundaries']
            if doc in seen or not 0 <= k <= p <= n <= 7681:
                raise ValueError('Invalid or repeated document count')
            seen.add(doc)
            cut = (512+int(hashlib.sha256(('reader-population-census-20261021:cut:'+doc).encode())
                          .hexdigest(), 16) % n) if n else None
            if cut != r['chosen_cut']:
                raise ValueError('A position must be chosen independently of candidate labels')
            if r['chosen_verified'] not in (0, 1) or r['chosen_pre_label'] not in (0, 1):
                raise ValueError('Invalid selected-position indicators')
            if r['chosen_verified'] > min(k, r['chosen_pre_label']) or r['chosen_pre_label'] > p:
                raise ValueError('Selected-position incidence contradicts document counts')
            for name in ('all', r['pool'], *(['unused'] if not r['used'] else [])):
                group = groups[name]; count = group['counts']
                for key, value in dict(documents=1, positions=n, verified_boundaries=k,
                        pre_label_boundaries=p, eligible_documents=int(n > 0),
                        documents_containing_verified_boundary=int(k > 0),
                        chosen_verified=r['chosen_verified'], chosen_pre_label=r['chosen_pre_label']).items():
                    count[key] += value
                if n:
                    group['fractions'].append(k/n)
                    group['pre_fractions'].append(p/n)
    expected = {**report['populations'], **report['by_source']}
    if set(groups) != set(expected):
        raise ValueError('All populations and source partitions required')
    for name, group in groups.items():
        c = group['counts']; e = expected[name]
        if dict(c) != e['counts']:
            raise ValueError('Count mismatch: '+name)
        values = dict(position_weighted_verified_prevalence=c['verified_boundaries']/c['positions'],
            document_weighted_expected_verified_prevalence=math.fsum(group['fractions'])/c['eligible_documents'],
            document_contains_verified_prevalence=c['documents_containing_verified_boundary']/c['eligible_documents'],
            fixed_hash_position_verified_prevalence=c['chosen_verified']/c['eligible_documents'],
            position_weighted_pre_label_incidence=c['pre_label_boundaries']/c['positions'],
            document_weighted_expected_pre_label_incidence=math.fsum(group['pre_fractions'])/c['eligible_documents'])
        if any(abs(value-e[key]) > 1e-14 for key, value in values.items()):
            raise ValueError('Incidence mismatch: '+name)
    output = dict(status='PASS_PUBLIC_READER_POPULATION_COUNT_REPLAY', documents=len(seen),
        positions=groups['all']['counts']['positions'], verified_boundaries=groups['all']['counts']['verified_boundaries'],
        projection_sha256=digest.hexdigest(),
        limits='Recounts projected source identities and candidate counts, not omitted source text, tokenization or semantic curation. No model calls.')
    with args.output.open('x') as stream:
        stream.write(json.dumps(output, indent=2)+'\n')
    print(output['status'], flush=True)


if __name__ == '__main__':
    main()
