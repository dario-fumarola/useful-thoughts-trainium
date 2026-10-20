"""Replay V29 public statistics without models, source text, or accelerator use.

Extract all three evidence archives into one folder, then pass its v29-evidence
directory and the downloaded release manifest. Requires NumPy and this repository
at the published commit. This verifies public numerical projections; omitted
source-context reconstruction and HF forwards require the original local assets.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.scalar_completion import parse_initial_scalar_v2


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).open()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def close(a, b):
    assert abs(a - b) < 1e-10, (a, b)


def pair(a, b, reported):
    assert set(a) == set(b) and len(a) == reported['items'] == reported['documents']
    assert len({v[0] for v in a.values()}) == len(a)
    assert all(a[k][0] == b[k][0] for k in a)
    delta = np.asarray([b[k][1] - a[k][1] for k in sorted(a)], dtype=np.float64)
    assert np.isfinite(delta).all()
    rng = np.random.default_rng(2026102029)
    sample = []
    for start in range(0, 10000, 128):
        draw = rng.integers(0, len(delta), size=(min(128, 10000 - start), len(delta)))
        sample.extend(delta[draw].sum(axis=1) / len(delta))
    close(float(delta.mean()), reported['difference_b_minus_a'])
    for actual, expected in zip(np.quantile(sample, [.025, .975]), reported['ci95']):
        close(float(actual), expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence
    manifest = read(args.manifest)
    assert manifest['status'] == 'PASS_ALL_ARCHIVES_FULL_MEMBER_READBACK_AND_FSYNC'
    files = {}
    for archive in manifest['archives']:
        for relative, digest in archive['contents'].items():
            assert relative not in files and sha(root / relative) == digest
            files[relative] = digest
    index = read(root / 'index-portable.json')
    report = read(root / 'results/readout.json')
    secondary = read(root / 'results/scalar-paired.json')
    assert sha(root / 'results/readout.json') == manifest['readout_sha256']
    release = read(root / 'results/release.json')
    parser_path = Path(__file__).resolve().parents[1] / 'train/scalar_completion.py'
    assert sha(parser_path) == release['source_files']['train/scalar_completion.py']
    sources = {r['item_id']: r for r in rows(root / index['cohorts']['calculation'])}
    assert len(sources) == 768 and len(index['models']) == 7
    inventory = {}
    for model, paths in index['models'].items():
        value = {panel: rows(root / paths[panel]) for panel in ('math', 'ordinary', 'writer')}
        assert len(value['math']) == 4608 and len(value['ordinary']) == 4096 and len(value['writer']) == 768
        value['scalar'] = [row for n in range(4) for row in rows(root / paths['scalar-' + str(n)])]
        assert len(value['scalar']) == 2304
        for row in value['scalar']:
            parsed = parse_initial_scalar_v2(row['text'], row['stop_type'])
            assert parsed == row['parsed']
            row['correct'] = int(row['status'] == 'ok' and row['source_note_status'] == 'ok'
                and parsed['status'] == 'valid'
                and Fraction(parsed['value']) == Fraction(sources[row['item_id']]['numeric_value']))
        inventory[model] = value
        for row in value['math'] + value['ordinary']:
            assert len(row['token_logps']) == row['tokens'] == (527 if 'mean_NLL' in row else 16)
            assert all(math.isfinite(v) and v <= 1e-5 for v in row['token_logps'])
            close(math.fsum(row['token_logps']), row['full_logp'])
    def scores(model, mode, metric):
        ordinary = metric == 'mean_document_NLL_delta'
        result = {}
        for row in inventory[model]['ordinary' if ordinary else 'math']:
            if row['mode'] != mode:
                continue
            if ordinary:
                score = -math.fsum(row['token_logps']) / 527
            elif metric == 'mean_document_span16_logp_gain':
                score = math.fsum(row['token_logps'])
            else:
                name = metric.removeprefix('mean_document_').removesuffix('_logp_gain')
                mask = row['component_masks'][name]
                assert len(mask) == 16
                score = math.fsum(v for v, use in zip(row['token_logps'], mask) if use)
            assert row['item_id'] not in result
            result[row['item_id']] = (row['document_id'], score)
        return result
    for contrast in report['all_paired_contrasts']:
        pair(scores(contrast['reference_model'], contrast['reference_mode'], contrast['metric']),
             scores(contrast['treatment_model'], contrast['treatment_mode'], contrast['metric']), contrast)
    for original in report['absolute_math']:
        selected = [r for r in inventory[original['model']]['math'] if r['mode'] == original['mode']]
        close(math.fsum(r['full_logp'] for r in selected) / 768, original['mean_span16_logp'])
        close(math.fsum(r['K1_logp'] for r in selected) / 768, original['mean_K1_logp'])
    for original in report['ordinary_absolute']:
        close(math.fsum(-math.fsum(r['token_logps']) / 527
                       for r in inventory[original['model']]['ordinary']) / 4096, original['mean_NLL'])
    scalar = {}
    for model, value in inventory.items():
        for mode in ('raw', 'empty', 'own'):
            selected = [r for r in value['scalar'] if r['mode'] == mode]
            assert len(selected) == 768
            scalar[model, mode] = {r['item_id']: (r['document_id'], r['correct']) for r in selected}
            assert len(scalar[model, mode]) == 768
            original = next(r for r in report['scalar_secondary'] if r['model'] == model and r['mode'] == mode)
            counts = dict(correct=sum(r['correct'] for r in selected),
                unsupported=sum(r['parsed']['status'] == 'unsupported' for r in selected),
                incomplete=sum(r['parsed']['status'] == 'incomplete' for r in selected),
                format_errors=sum(r['parsed']['status'] == 'format_error' for r in selected),
                execution_errors=sum(r['status'] != 'ok' or r['source_note_status'] != 'ok' for r in selected),
                source_note_errors=sum(r['source_note_status'] != 'ok' for r in selected),
                parser_errors=sum(r['parsed']['status'] == 'error' for r in selected),
                capped=sum(r['stop_type'] == 'length_cap' for r in selected),
                generated_actions=sum(r['n_actions'] for r in selected))
            assert all(original[k] == v for k, v in counts.items())
    for contrast in secondary['contrasts']:
        a = scalar[contrast['reference_model'], contrast['reference_mode']]
        b = scalar[contrast['treatment_model'], contrast['treatment_mode']]
        pair(a, b, contrast)
        assert sum(a[k][1] == 0 and b[k][1] == 1 for k in a) == contrast['fixes']
        assert sum(a[k][1] == 1 and b[k][1] == 0 for k in a) == contrast['breaks']
    decisions = {}
    for seed, c in report['by_order'].items():
        good = lambda r: r['difference_b_minus_a'] >= .25 and r['ci95'][0] > 0
        recipe = good(c['raw_control']) and good(c['format_control'])
        base = good(c['base'])
        retention = c['ordinary_base']['ci95'][1] <= .03
        inference = c['inference_execution_complete'] and all(good(c[k]) for k in ('own_raw', 'own_empty', 'own_raw_control'))
        decisions[seed] = dict(training_recipe_pass=recipe, absolute_base_pass=base,
            retention_target_met=retention, overall_CPT_criterion=recipe and base and retention,
            inference_gate_pass=inference)
    assert decisions == report['decision']['by_order']
    result = dict(status='PASS_PUBLIC_FILE_HASHES_ALL_PAIRED_STATISTICS_AND_SCALAR_PARSER',
        files_checked=len(files), likelihood_contrasts=len(report['all_paired_contrasts']),
        scalar_contrasts=len(secondary['contrasts']), decisions=decisions,
        report_sha256=sha(root / 'results/readout.json'), source_sha256=sha(__file__),
        limits='No model execution or omitted source-context verification. Those were separately checked locally before publication.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
