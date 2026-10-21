"""Read-only objective/exposure/loss attribution for the closed V29 controls.

No checkpoint is loaded and no evaluation score is recomputed. This diagnoses
what was optimized; it cannot identify the cause of held-document regression.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from train.real_cpt_scaffold import grouped_order
from weekend_thinking.artifacts import atomic_json, sha256_file


def rows(path):
    return [json.loads(s) for s in Path(path).open() if s.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = a.root/'local/data/real-cpt-scaffold-v29-cpu-v2/TRAIN.jsonl'
    assert sha256_file(data) == 'ef15ff29e4c972ed6162f05d293bdc1059967e2914a0e9973e7b0f95f71bd509'
    source = rows(data); by_id = {r['item_id']: r for r in source}
    assert len(source) == len(by_id) == 20480
    assert len({r['document_id'] for r in source}) == len(source)
    ratios = []; clean_targets = 0; future_targets = 0
    for r in source:
        n = len(r['prefix_ids']) + len(r['continuation16_ids']) - 1
        clean_targets += n
        if r['kind'] == 'calculation':
            assert len(r['continuation16_ids']) == 16
            future_targets += 16
            ratios.append(1 + .5 * n / 16)
    result = dict(status='READ_ONLY_CLOSED_V29_RAW_OBJECTIVE_AND_EXPOSURE_AUDIT',
        data_sha256=sha256_file(data), unique_documents=len(source),
        source_kinds=dict(Counter(r['kind'] for r in source)),
        clean_target_positions_per_pass=clean_targets,
        selected_future_positions_per_pass=future_targets,
        selected_fraction_of_clean_positions=future_targets/clean_targets,
        per_document_objective='meanCE(all original source transitions) + I_calculation * 0.5 * meanCE(original future16)',
        mean_extra_auxiliary_coefficient_per_source_document=.5*.25,
        selected_future_to_other_clean_token_coefficient_ratio=dict(
            minimum=min(ratios), maximum=max(ratios), mean=statistics.mean(ratios),
            full512prefix=1 + .5*527/16),
        runs=[], limits=[
            'This raw control is continuation-reweighted NTP, not uniform vanilla NTP.',
            'Coefficients are explicit loss coefficients, not measured gradient contributions.',
            'A decreasing repeated-source loss is not evidence of fresh-source prediction improvement.',
            'No intervention isolates weighting, corpus selection, constant learning rate, or repetition as the regression cause.',
            'The original comparisons and failed gates remain unchanged; no new training is released.'])
    for seed in (2026102029, 2026102030):
        run = a.root/f'local/science-runs/real-cpt-scaffold-v29/cpt-raw-seed{seed}-v1'
        metric = rows(run/'metrics.jsonl')
        assert len(metric) == 1280 and [r['step'] for r in metric] == list(range(1, 1281))
        order = grouped_order(source, seed=seed, passes=2)
        actual = [i for r in metric for i in r['ordered_item_ids']]
        assert actual == [source[i]['item_id'] for i in order]
        assert set(Counter(actual).values()) == {2} and len(actual) == 40960
        assert all(Counter(by_id[i]['kind'] for i in r['ordered_item_ids']) ==
                   {'ordinary': 24, 'calculation': 8} for r in metric)
        max_reconstruction_error = max(abs(r['loss'] - (
            r['loss_components']['source_mean_loss'] + .125 *
            r['loss_components']['continuation_mean_loss'])) for r in metric)
        assert max_reconstruction_error < 1e-6
        blocks = []
        for start in range(0, 1280, 160):
            block = metric[start:start+160]
            blocks.append(dict(first_step=start+1, last_step=start+160,
                source_mean_loss=statistics.mean(r['loss_components']['source_mean_loss'] for r in block),
                continuation_mean_loss=statistics.mean(r['loss_components']['continuation_mean_loss'] for r in block),
                mean_gradient_norm=statistics.mean(r['gradient_norm'] for r in block)))
        result['runs'].append(dict(seed=seed, metrics_sha256=sha256_file(run/'metrics.jsonl'),
            ordered_exposure_checks_passed=True, unique_documents=20480, exposures=40960,
            maximum_loss_reconstruction_error=max_reconstruction_error, blocks=blocks))
    a.output.mkdir(parents=True, exist_ok=False)
    atomic_json(a.output/'raw-objective-audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
