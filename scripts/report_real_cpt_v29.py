"""Render checked V29 tables, figures and outcome-independent fixed examples."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).open()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def table(path, values):
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def label(name):
    if name == 'B0':
        return name
    arm, seed = name.split('-seed')
    return {'raw': 'R', 'format': 'F', 'computation': 'T'}[arm] + ('1' if seed == '2026102029' else '2')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'readout', 'independent-check', 'scalar', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    index, report, check = read(args.index), read(args.readout), read(args.independent_check)
    secondary = read(args.scalar)
    assert check['status'] == 'PASS_ALL_PAIRED_DOCUMENT_CONTRASTS_DECISIONS_AND_FIXED_HF_SPOTS'
    assert check['report_sha256'] == sha(args.readout) and check['index_sha256'] == sha(args.index)
    assert secondary['primary_decisions_unchanged'] == report['decision']
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in [('absolute-likelihood.csv', report['absolute_math']),
                        ('ordinary-NLL.csv', report['ordinary_absolute']),
                        ('scalar-answers-and-errors.csv', report['scalar_secondary']),
                        ('writer-actions-and-errors.csv', report['writer_costs']),
                        ('all-paired-likelihood-contrasts.csv', report['all_paired_contrasts']),
                        ('all-paired-scalar-contrasts.csv', secondary['contrasts'])]:
        table(args.output / name, value)
    for path, name in [(args.readout, 'readout.json'), (args.independent_check, 'independent-check.json'),
                       (args.scalar, 'scalar-paired.json')]:
        shutil.copyfile(path, args.output / name)
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharex=True)
    for axis, seed in zip(axes, ('2026102029', '2026102030')):
        c = report['by_order'][seed]
        values = [c[k] for k in ('raw_control', 'format_control', 'base')]
        x = [v['difference_b_minus_a'] for v in values]
        axis.errorbar(x, range(3), xerr=[[a - v['ci95'][0] for a, v in zip(x, values)],
                                       [v['ci95'][1] - a for a, v in zip(x, values)]],
                      fmt='o', capsize=4, color='#215c93')
        axis.axvline(0, color='0.4', linewidth=1)
        axis.axvline(.25, color='#b26d00', linestyle='--', linewidth=1, label='Point threshold +0.25')
        axis.set_yticks(range(3), ['T raw − R raw', 'T raw − F raw', 'T raw − B0 raw'])
        axis.invert_yaxis()
        axis.set_title('Data order ' + seed)
        axis.set_xlabel('Full 16-token log-likelihood gain (nats)')
        axis.grid(axis='x', alpha=.2)
    fig.suptitle('Direct continuation prediction; paired 95% document intervals')
    fig.tight_layout()
    fig.savefig(args.output / 'direct-prediction-contrasts.svg')
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8.5, 3.5))
    values = report['ordinary_absolute']
    axis.bar([label(v['model']) for v in values], [v['mean_NLL'] for v in values], color='#447599')
    base = next(v['mean_NLL'] for v in values if v['model'] == 'B0')
    axis.axhline(base, color='black', linestyle='--', linewidth=1, label='Untouched base')
    axis.set_ylabel('NLL (nats/token), lower is better')
    axis.set_title('Ordinary FineMath pages: 4,096 documents × 527 real transitions')
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'ordinary-NLL.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
    for axis, seed in zip(axes, ('2026102029', '2026102030')):
        models = ['B0'] + [arm + '-seed' + seed for arm in ('raw', 'format', 'computation')]
        for offset, mode in enumerate(('raw', 'empty', 'own')):
            selected = [next(r for r in report['scalar_secondary'] if r['model'] == m and r['mode'] == mode) for m in models]
            axis.bar([n + (offset - 1) * .24 for n in range(4)],
                     [100 * r['correct'] / 768 for r in selected], width=.24,
                     label={'raw': 'Raw', 'empty': 'Empty format', 'own': 'Generated work'}[mode])
        axis.set_xticks(range(4), [label(m) for m in models])
        axis.set_title('Data order ' + seed)
        axis.set_ylim(0, 100)
        axis.set_ylabel('Initial numeric answer correct (%)')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Secondary scalar accuracy; all 768 attempts retained in every bar')
    fig.tight_layout()
    fig.savefig(args.output / 'scalar-accuracy.svg')
    plt.close(fig)
    costs = []
    E = args.index.parent
    for model in index['models']:
        for panel in ('writer', 'ordinary', 'math', 'scalar'):
            path = E / 'supervision' / model / panel / 'closure.json'
            value = read(path)
            assert value['status'] == 'SESSION_AND_PHYSICAL_DEVICE_CLOSED' and value['exit_code'] == 0
            costs.append(dict(model=model, panel=panel,
                              reserved_chip_hours=value['reserved_chip_hours'],
                              original_closure_sha256=sha(path)))
    table(args.output / 'evaluation-reservations.csv', costs)
    write(args.output / 'costs.json', dict(training_reserved_chip_hours=28.4274855470983,
        evaluation_reserved_chip_hours=math.fsum(v['reserved_chip_hours'] for v in costs),
        evaluation_groups=28, actual_workers=49,
        limits='Whole physical-chip reservations, not elapsed wall time or FLOPs. Scalar groups use four logical shards within one charged chip. Separate qualifications excluded.'))
    sources = {r['item_id']: r for r in rows(Path(index['data']) / 'DEV_calculation.jsonl')}
    selected = sorted(sources, key=lambda key: hashlib.sha256(('v29-public-fixed-examples:' + key).encode()).hexdigest())[:8]
    examples = {key: {k: sources[key][k] for k in ('item_id', 'document_id', 'normalized_expression',
                 'numeric_value', 'full_calculation_text')} for key in selected}
    for value in examples.values():
        value['models'] = {}
    for model, paths in index['models'].items():
        notes = {r['item_id']: r for r in rows(Path(paths['writer']) / 'notes.jsonl')}
        answers = [r for path in paths['scalar'] for r in rows(Path(path) / 'answers.jsonl')]
        math_rows = rows(Path(paths['math']) / 'logps.jsonl')
        for key in selected:
            note = notes[key]
            examples[key]['models'][model] = dict(
                generated_note={k: v for k, v in note.items() if k != 'prompt_ids'},
                scalar_attempts=[{k: v for k, v in r.items() if k != 'prompt_ids'} for r in answers if r['item_id'] == key],
                likelihoods=[{k: r[k] for k in ('mode', 'full_logp', 'token_logps', 'component_logps', 'note_execution_status')}
                             for r in math_rows if r['item_id'] == key])
    write(args.output / 'fixed-eight-examples.json', dict(
        selection='Lowest SHA256(v29-public-fixed-examples: + item_id), eight IDs; no output or score used.',
        limits='Source prose omitted; expressions and all model outputs retained. Supplied full traces include the computed answer.',
        examples=[examples[key] for key in selected]))
    print(json.dumps(read(args.output / 'costs.json')), flush=True)


if __name__ == '__main__':
    main()
