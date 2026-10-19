"""Offline canonical answer and pointer diagnostics for the frozen v23 pilot.

No generation or prompt construction. Complete original question coverage is
required; execution errors remain zero-scored observations, not exclusions.
"""
from collections import Counter
from dataclasses import asdict
import math

import numpy as np

from train.document_evidence_plan_answers import parse_evidence, annotated_support_overlap
from train.document_qa_answers import ParsedAnswer, parse_answer, score

SEED = 2026101923
DRAWS = 20000


def score_condition(attempts, questions, gold, *, mode):
    if (not questions or len(questions) != len({q['item_id'] for q in questions})
            or set(gold) != {q['item_id'] for q in questions}):
        raise ValueError('Unique original questions and separate bound gold required')
    by_id = {}
    for attempt in attempts:
        key = attempt['item_id']
        if key in by_id or key not in gold or attempt['mode'] != mode:
            raise ValueError('Duplicate, unknown or mismatched native observation')
        by_id[key] = attempt
    if set(by_id) != set(gold):
        raise ValueError('Every original attempt including errors is required')
    observations = []
    for question in questions:
        key = question['item_id']
        original = by_id[key]
        if original['status'] not in ('ok', 'error') or original['parser_mode'] != 'prior':
            raise ValueError('Native execution status and common anchored parser required')
        stop = original['stop_type']
        parsed = (parse_answer(original['text'], 'prior',
            terminated=stop.startswith('token:'), capped=stop == 'length_cap')
            if original['status'] == 'ok' else ParsedAnswer('execution_error', reason='execution_error'))
        annotation = gold[key]
        metrics = score(parsed, annotation['answer'], annotation.get('answer_aliases', []))
        row = dict(item_id=key, mode=mode, parsed=asdict(parsed), **metrics,
            status=original['status'], stop_type=stop,
            generated_actions=len(original['actions']))
        if mode == 'selection':
            plan = (parse_evidence(original['raw_text'], len(question['contexts']))
                    if original['status'] == 'ok' else None)
            row['selection'] = asdict(plan) if plan else dict(status='execution_error')
            row['valid_nonempty_selection'] = bool(plan and plan.status == 'complete')
            row['annotated_support_overlap'] = (annotated_support_overlap(plan, annotation['support_indices'])
                if plan else dict(precision=0., recall=0., exact_match=0, f1=0.))
        observations.append(row)
    count = len(observations)
    summary = dict(items=count, correct=sum(v['canonical_em'] for v in observations),
        **{metric: sum(v[metric] for v in observations)/count for metric in
           ('canonical_em', 'canonical_f1', 'attributed_alias_em', 'attributed_alias_f1')},
        parsed_statuses=dict(Counter(v['parsed']['status'] for v in observations)),
        execution_errors=sum(v['status'] != 'ok' for v in observations),
        capped=sum(v['stop_type'] == 'length_cap' for v in observations),
        generated_actions=sum(v['generated_actions'] for v in observations))
    if mode == 'selection':
        summary.update(valid_nonempty_selections=sum(v['valid_nonempty_selection'] for v in observations),
            selection_statuses=dict(Counter(v['selection']['status'] for v in observations)),
            annotated_support_overlap={metric: sum(v['annotated_support_overlap'][metric] for v in observations)/count
                for metric in ('precision', 'recall', 'exact_match', 'f1')})
    return observations, summary


def paired_contrast(control, treatment, *, draws=DRAWS, seed=SEED):
    if (not control or control.keys() != treatment.keys()
            or type(draws) is not int or draws < 1):
        raise ValueError('Complete aligned original question observations required')
    ids = sorted(control)
    delta = np.asarray([treatment[key]-control[key] for key in ids], dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite observations cannot be silently omitted')
    generator = np.random.default_rng(seed)
    means = []
    for start in range(0, draws, 500):
        indices = generator.integers(0, len(ids), size=(min(500, draws-start), len(ids)))
        means.extend(delta[indices].mean(1).tolist())
    return dict(items=len(ids), difference_treatment_minus_control=float(delta.mean()),
        ci95=np.quantile(means, [.025, .975]).tolist(), bootstrap_draws=draws,
        bootstrap_seed=seed, unit='paired original questions;checkpoint-conditional,unadjusted95%')


def pilot_decisions(primary, same_model_direct, same_model_empty, valid_nonempty, items):
    """Keep net recipe gain, serialization and inference assistance distinct."""
    if (type(items) is not int or items != 255 or type(valid_nonempty) is not int
            or not 0 <= valid_nonempty <= items
            or any(value['items'] != items for value in (primary, same_model_direct, same_model_empty))):
        raise ValueError('The original complete fresh255 pilot population is required')
    for value in (primary, same_model_direct, same_model_empty):
        if (value['bootstrap_draws'] != DRAWS or value['bootstrap_seed'] != SEED
                or not all(math.isfinite(v) for v in [value['difference_treatment_minus_control'], *value['ci95']])):
            raise ValueError('Unchanged prospective finite bootstrap is required')
    recipe = primary['difference_treatment_minus_control'] >= .03 and primary['ci95'][0] > 0
    integrity = valid_nonempty/items >= .95
    assistance = recipe and integrity and same_model_direct['ci95'][0] > 0 and same_model_empty['ci95'][0] > 0
    return dict(net_recipe_primary_pass=bool(recipe), selection_integrity_pass=integrity,
        same_model_direct_lower_bound_positive=bool(same_model_direct['ci95'][0] > 0),
        same_model_empty_lower_bound_positive=bool(same_model_empty['ci95'][0] > 0),
        inference_assistance_pass=bool(assistance),
        eligible_for_same_protocol_replication=bool(recipe and integrity),
        two_seed_success_established=False)
