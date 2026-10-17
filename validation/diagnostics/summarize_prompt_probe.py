"""Validate and summarize the fixed-note instruction/boundary reader probe."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from validation.diagnostics.summarize_interface_probe import digest, estimate


def summarize(directory):
    p = json.loads((directory/'protocol.json').read_text())
    run = json.loads((directory/'summary.json').read_text())
    assert run['status'] == 'complete'
    assert run['protocol_sha256'] == digest(directory/'protocol.json')
    assert run['scores_sha256'] == digest(directory/'scores.jsonl')
    rows = [json.loads(line) for line in (directory/'scores.jsonl').read_text().splitlines()]
    ids = p['selection']['item_ids']
    modes = ('empty', 'own', 'filler', 'shuffled', 'other')
    expected = {(k, 'raw', m) for k in ids for m in ('raw', 'repeat32')}
    expected |= {(k, i, m) for k in ids for i in p['configurations'] for m in modes}
    index = {(r['item_id'], r['interface'], r['mode']): r for r in rows}
    assert len(rows) == len(index) == len(expected) and set(index) == expected
    assert all(r['status'] == 'ok' and len(r['token_log_likelihoods']) == 16 and
               all(math.isfinite(x) for x in r['token_log_likelihoods']) for r in rows)
    report = dict(experiment_id=p['experiment_id'], status='complete', items=len(ids),
        rows=len(rows), failures=0, protocol_sha256=digest(directory/'protocol.json'),
        scores_sha256=digest(directory/'scores.jsonl'),
        original_reference_max_token_error=run['maximum_reference_error'],
        candidates_per_document=1, compute=p['compute'], prompt_provenance=p['prompt_provenance'],
        serialization=p['serialization'], limitation=p['limitation'],
        uncertainty='5000 paired document bootstrap draws, seed17, percentile95%; fixed B0/notes, exploratory adaptive development',
        groups={})
    for group in ['all', *sorted(set(p['strata'].values()))]:
        keys = [k for k in ids if group == 'all' or p['strata'][k] == group]
        cells = {}
        for h in p['horizons']:
            def values(interface, mode):
                return [math.fsum(index[k, interface, mode]['token_log_likelihoods'][:h]) for k in keys]
            for interface, config in p['configurations'].items():
                raw_repeat = 'repeat32' if config['repeat'] else 'raw'
                own, empty = values(interface, 'own'), values(interface, 'empty')
                comparisons = dict(empty_minus_raw=(empty, values('raw', 'raw')),
                    own_minus_raw=(own, values('raw', 'raw')),
                    empty_minus_raw_repeat=(empty, values('raw', raw_repeat)),
                    own_minus_raw_repeat=(own, values('raw', raw_repeat)))
                comparisons.update({f'own_minus_{m}': (own, values(interface, m)) for m in modes if m != 'own'})
                cells[f'{interface}/K{h}'] = dict(
                    absolute_means={m:math.fsum(values(interface, m))/len(keys) for m in modes},
                    contrasts={label:estimate([a-b for a,b in zip(left,right)])
                               for label,(left,right) in comparisons.items()})
        report['groups'][group] = cells
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.input)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=report['status'], rows=report['rows'], items=report['items'])))
