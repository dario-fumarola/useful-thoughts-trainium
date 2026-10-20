"""Secondary V29 scalar-accuracy pairs; no new gates or model execution.

Run only after the complete, independently checked likelihood readout exists.
All original attempts remain in the denominator, including execution failures.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path

from train.real_cpt_readout import SCALAR_MODES
from train.scalar_completion import parse_initial_scalar_v2
from weekend_thinking.artifacts import atomic_json, sha256_file
from weekend_thinking.stats import paired_cluster_bootstrap


def outcomes(attempts, sources):
    """Strict one-to-one source pairing, with failure indicators kept as zeros."""
    by_mode = {mode: {} for mode in SCALAR_MODES}
    for row in attempts:
        mode, key = row['mode'], row['item_id']
        if mode not in by_mode or key not in sources or key in by_mode[mode]:
            raise ValueError('Unexpected or duplicate scalar attempt')
        source = sources[key]
        if row['document_id'] != source['document_id']:
            raise ValueError('Original source-document pairing changed')
        parsed = parse_initial_scalar_v2(row['text'], row['stop_type'])
        if parsed != row['parsed']:
            raise ValueError('Frozen parser differs from original stored outcome')
        correct = (row['status'] == 'ok' and row['source_note_status'] == 'ok'
                   and parsed['status'] == 'valid'
                   and Fraction(parsed['value']) == Fraction(source['numeric_value']))
        by_mode[mode][key] = dict(item_id=key, document_id=row['document_id'],
                                  score=int(correct))
    if any(set(values) != set(sources) for values in by_mode.values()):
        raise ValueError('Every original question and all three modes required')
    return {mode: [values[key] for key in sorted(values)]
            for mode, values in by_mode.items()}


def summarize(index, readout, independent_check, output):
    read = lambda p: json.loads(Path(p).read_text())
    config, report, checked = read(index), read(readout), read(independent_check)
    if report['status'] != 'COMPLETE_V29_ALL_FIXED_MODEL_READOUT':
        raise ValueError('Complete primary readout required; no partial-score analysis')
    if (checked.get('status') != 'PASS_ALL_PAIRED_DOCUMENT_CONTRASTS_DECISIONS_AND_FIXED_HF_SPOTS'
            or checked.get('index_sha256') != sha256_file(index)
            or checked.get('report_sha256') != sha256_file(readout)
            or checked.get('source_sha256') != 'e5bc573b25ac07f4a3ae23e8fbbb0ab98f070ae39698ddd224e20778c2110c28'
            or not checked.get('all_contexts_and_writer_prefix_only_inputs_reconstructed')):
        raise ValueError('Independent primary-readout check must pass first')
    if report['source_pins'].get(str(index.resolve())) != sha256_file(index):
        raise ValueError('Original primary index binding changed')
    source_file = Path(config['data']) / 'DEV_calculation.jsonl'
    manifest_file = source_file.parent / 'manifest.json'
    if sha256_file(manifest_file) != '77c8774d3fdeb283728878c44522dab541ec61a213d3f89ece471b3dcc45fb3f':
        raise ValueError('Original frozen data manifest required')
    manifest = read(manifest_file)
    if sha256_file(source_file) != manifest['files'][source_file.name]['sha256']:
        raise ValueError('Original calculation cohort changed')
    sources = [json.loads(line) for line in source_file.open()]
    source_index = {row['item_id']: row for row in sources}
    if len(sources) != 768 or len(source_index) != 768:
        raise ValueError('All 768 unique source documents required')
    inventory = {}
    expected = {'B0'} | {f'{arm}-seed{seed}' for arm in ('raw', 'format', 'computation')
                        for seed in (2026102029, 2026102030)}
    if set(config['models']) != expected:
        raise ValueError('All seven original fixed models required')
    for model, jobs in config['models'].items():
        attempts = []
        if len(jobs['scalar']) != 4:
            raise ValueError('All four original scalar shards required')
        for folder in jobs['scalar']:
            path = Path(folder) / 'answers.jsonl'
            if report['source_pins'].get(str(path.resolve())) != sha256_file(path):
                raise ValueError('Original checked scalar attempts changed')
            attempts.extend(json.loads(line) for line in path.open())
        inventory[model] = outcomes(attempts, source_index)
        for mode, values in inventory[model].items():
            old = [r for r in report['scalar_secondary']
                   if r['model'] == model and r['mode'] == mode]
            if len(old) != 1 or old[0]['correct'] != sum(r['score'] for r in values):
                raise ValueError('Secondary counts must equal the frozen primary report')
    contrasts = []
    for seed in (2026102029, 2026102030):
        t, r, f = (f'{arm}-seed{seed}' for arm in ('computation', 'raw', 'format'))
        pairs = [(r, 'raw', 'raw'), (f, 'raw', 'raw'), ('B0', 'raw', 'raw'),
                 (t, 'raw', 'own'), (t, 'empty', 'own'), (r, 'raw', 'own'),
                 (f, 'raw', 'own'), ('B0', 'raw', 'own')]
        for reference, ref_mode, treatment_mode in pairs:
            a, b = inventory[reference][ref_mode], inventory[t][treatment_mode]
            value = paired_cluster_bootstrap(a, b, repeats=10000, seed=2026102029)
            value.update(order=seed, reference_model=reference, reference_mode=ref_mode,
                         treatment_model=t, treatment_mode=treatment_mode,
                         reference_correct=sum(v['score'] for v in a),
                         treatment_correct=sum(v['score'] for v in b),
                         fixes=sum(x['score'] == 0 and y['score'] == 1 for x, y in zip(a, b)),
                         breaks=sum(x['score'] == 1 and y['score'] == 0 for x, y in zip(a, b)))
            contrasts.append(value)
    result = dict(status='COMPLETE_SECONDARY_SCALAR_PAIRED_REPORT_NO_NEW_GATES',
                  contrasts=contrasts, primary_decisions_unchanged=report['decision'],
                  pins={str(p.resolve()): sha256_file(p)
                        for p in (index, readout, independent_check, source_file, Path(__file__))},
                  limits=['Secondary descriptive comparisons, not a rescue of a failed primary gate.',
                          'Unadjusted paired 95% document-bootstrap intervals conditional on fixed checkpoints.',
                          'Two orders reported separately; no seed pooling or outcome-based selection.',
                          'Errors, unsupported outputs and missing-note executions remain in all denominators.'])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'scalar-paired.json', result)
    with (output / 'scalar-paired.csv').open('x', newline='') as stream:
        fields = ['order', 'reference_model', 'reference_mode', 'treatment_model',
                  'treatment_mode', 'reference_correct', 'treatment_correct', 'fixes',
                  'breaks', 'items', 'gain_pp', 'lower95_pp', 'upper95_pp']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in contrasts:
            writer.writerow({**{k: row[k] for k in fields[:-3]},
                             'gain_pp': 100 * row['difference_b_minus_a'],
                             'lower95_pp': 100 * row['ci95'][0],
                             'upper95_pp': 100 * row['ci95'][1]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'readout', 'independent-check', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    summarize(args.index, args.readout, args.independent_check, args.output)


if __name__ == '__main__':
    main()
