"""Report the declared ordinary continuation16 endpoint from public V29 scores.

This is secondary to the frozen full527 retention gate. No threshold, parser,
model execution, cohort change, or primary-decision change is introduced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence
    index, receipt = read(root / 'index-portable.json'), read(root / 'projection-receipt.json')
    primary = read(root / 'results/readout.json')
    scores, absolute, pins = {}, [], {}
    for model, paths in index['models'].items():
        relative = paths['ordinary']
        assert sha(root / relative) == receipt['files'][relative]['projected_sha256']
        pins[relative] = sha(root / relative)
        values = {}
        for line in (root / relative).open():
            row = json.loads(line)
            assert len(row['token_logps']) == 527 and row['suffix16_logps'] == row['token_logps'][-16:]
            assert abs(math.fsum(row['suffix16_logps']) - row['suffix16_logp']) < 1e-8
            assert row['item_id'] not in values
            values[row['item_id']] = (row['document_id'], -math.fsum(row['suffix16_logps']) / 16)
        assert len(values) == len({v[0] for v in values.values()}) == 4096
        scores[model] = values
        absolute.append(dict(model=model, documents=4096, tokens_per_document=16,
            mean_suffix16_NLL=math.fsum(v[1] for v in values.values()) / 4096))
    pairs = [(model, 'B0') for model in scores if model != 'B0']
    for seed in (2026102029, 2026102030):
        pairs.extend((f'computation-seed{seed}', f'{arm}-seed{seed}') for arm in ('raw', 'format'))
    contrasts = []
    for treatment, reference in pairs:
        a, b = scores[reference], scores[treatment]
        assert set(a) == set(b) and all(a[k][0] == b[k][0] for k in a)
        delta = np.asarray([b[k][1] - a[k][1] for k in sorted(a)], dtype=np.float64)
        rng = np.random.default_rng(2026102029)
        sample = []
        for start in range(0, 10000, 128):
            draw = rng.integers(0, 4096, size=(min(128, 10000 - start), 4096))
            sample.extend(delta[draw].sum(axis=1) / 4096)
        contrasts.append(dict(reference_model=reference, treatment_model=treatment,
            documents=4096, NLL_cost=float(delta.mean()), ci95=np.quantile(sample, [.025, .975]).tolist(),
            bootstrap_repeats=10000, bootstrap_seed=2026102029))
    result = dict(status='COMPLETE_DECLARED_ORDINARY_SUFFIX16_SECONDARY_NO_NEW_GATES',
        absolute=absolute, contrasts=contrasts, primary_decisions_unchanged=primary['decision'],
        primary_readout_sha256=sha(root / 'results/readout.json'), public_score_files=pins,
        source_sha256=sha(__file__),
        limits=['Declared continuation16 secondary; the retention threshold applies to full527, not this slice.',
                'Paired document intervals conditional on fixed checkpoints; no training-order pooling.',
                'Same source slice across all seven models; no comparison with earlier experiments using other cohorts.'])
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'ordinary-suffix16.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    for name, values in [('ordinary-suffix16-absolute.csv', absolute), ('ordinary-suffix16-paired.csv', contrasts)]:
        with (args.output / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    print(json.dumps(dict(status=result['status'], contrasts=len(contrasts))), flush=True)


if __name__ == '__main__':
    main()
