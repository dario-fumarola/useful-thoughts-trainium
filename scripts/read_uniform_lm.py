"""Select a terminal baseline on DEV, or read its sealed confirmation."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from weekend_thinking.artifacts import atomic_json, sha256_file, utc_now
from weekend_thinking.stats import cluster_bootstrap


def collect(directory, count):
    records = {}
    for rank in range(8):
        path = directory/f'rank-{rank:05d}.jsonl'
        receipt = json.loads((directory/f'complete-rank-{rank:05d}.json').read_text())
        if receipt['sha256'] != sha256_file(path):
            raise ValueError('Original native score export differs')
        rows = [json.loads(s) for s in path.open()]
        if len(rows) != receipt['rows']:
            raise ValueError('Incomplete rank export')
        for row in rows:
            if (row['item_id'] in records or len(row['token_logps']) != 527
                    or not all(math.isfinite(v) for v in row['token_logps'])):
                raise ValueError('Duplicate or incomplete target score')
            records[row['item_id']] = row
    if len(records) != count or len({r['document_id'] for r in records.values()}) != count:
        raise ValueError('Complete unique source cohort required')
    keys = sorted(records)
    return keys, [records[k]['document_id'] for k in keys], np.array(
        [records[k]['token_logps'] for k in keys], dtype=np.float64)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--release', type=Path, required=True)
    a.add_argument('--phase', choices=('select', 'confirm'), required=True)
    a.add_argument('--output', type=Path, required=True)
    args = a.parse_args()
    release = json.loads(args.release.read_text())
    cohort = 'DEV' if args.phase == 'select' else 'CONFIRM'
    count = 1024 if args.phase == 'select' else 4096
    names = ['B0']+list(release['all_terminal_models'])
    if args.phase == 'confirm':
        selection = release['selection']
        if sha256_file(selection['file']) != selection['sha256']:
            raise ValueError('The pre-confirmation selection record differs')
        chosen = json.loads(Path(selection['file']).read_text())['recipe']
        names = ['B0', chosen]
    values, keys, documents = {}, None, None
    for name in names:
        job = release['named_jobs'][cohort+'-'+name]
        k, d, v = collect(Path(job['output']), count)
        if keys is not None and (k != keys or d != documents):
            raise ValueError('Strict source pairing failed')
        keys, documents = k, d
        values[name] = v
    absolute = {name: dict(full_window_NLL=float(-v.mean()),
        suffix16_NLL=float(-v[:, -16:].mean()),
        suffix16_log_probability=float(v[:, -16:].sum(1).mean()))
        for name, v in values.items()}
    contrasts = {}
    for name in names[1:]:
        difference = values[name]-values['B0']
        b = cluster_bootstrap(np.column_stack((difference.mean(1),
            difference[:, -16:].sum(1))), documents, unit='unique source document',
            repeats=5000, seed=17)
        contrasts[name] = dict(full_window_gain_nats_per_token=float(b['mean'][0]),
            full_window_CI95=b['CI95'][0].tolist(),
            suffix16_gain_nats=float(b['mean'][1]), suffix16_CI95=b['CI95'][1].tolist(),
            statistics=b['metadata'])
    result = dict(utc=utc_now(), phase=args.phase, cohort=cohort, documents=count,
                  absolute=absolute, contrasts=contrasts, release_sha256=sha256_file(args.release),
                  uncertainty='Document sampling conditional on the fixed checkpoints; not training replication')
    if args.phase == 'select':
        selected = min(names[1:], key=lambda n: (absolute[n]['full_window_NLL'],
            release['all_terminal_models'][n]['lr'], n))
        result.update(status='FROZEN_DEV_TERMINAL_SELECTION_BEFORE_CONFIRMATION', recipe=selected,
                      checkpoint=release['all_terminal_models'][selected],
                      selection_rule='Lowest DEV full527 NLL; exact ties lower peak LR then lexical name')
    else:
        passed = contrasts[chosen]['full_window_CI95'][0] > 0
        result.update(status='COMPLETE_UNIFORM_LM_CONFIRMATION', selected_recipe=chosen,
                      uniform_baseline_improves=passed,
                      verdict='PASS' if passed else 'FAIL',
                      scope='Uniform NTP on the declared source frame; no intermediate thoughts trained')
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
