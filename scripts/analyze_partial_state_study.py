"""Descriptive v7 trace completeness, cost and baseline analysis after full audit.

This adds no training or primary endpoint and never changes the complete audit.
All source rows, including unsupported outputs, stay in the denominator.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re

from scripts.audit_partial_state_panels import exact_trace, independent_state, paired, sha


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def analyze(root):
    root = Path(root).resolve()
    source = root/'validation/partial_state_v7_matched5120_complete_20261018.json'
    report = json.loads(source.read_text())
    if report['status'] != 'COMPLETE_BOTH_SEEDS_ALL_FOUR_PANELS_NO_FILTERING':
        raise ValueError('Complete audited study required before descriptive analysis')
    registry = json.loads((root/'local/diagnostics/partial-state-v7-matched5120-evaluation-registry-20261018.json').read_text())
    payload_root = source.with_suffix('')
    results = {}
    trace_pattern = re.compile(r'\s*([+-]?\d+)\s*([+*\-])\s*([+-]?\d+)\s*=\s*([+-]?\d+)\s*')
    for seed, panels in report['results'].items():
        results[seed] = {}
        for panel, primary in panels.items():
            files = {kind: payload_root/f'seed{seed}-{panel}-all-original-{kind}.jsonl.gz'
                     for kind in ('answers', 'notes')}
            for file in files.values():
                if sha(file) != report['public_payloads'][str(file.relative_to(root))]['sha256']:
                    raise ValueError('Original public payload changed')
            answers = rows(files['answers'])
            cells = {(v['arm'], v['item_id'], v['mode']): v for v in answers}
            note_rows = rows(files['notes'])
            question_file = Path(registry['jobs'][seed]['reader']['outputs'][panel])/'questions.jsonl'
            questions = [json.loads(line) for line in question_file.read_text().splitlines()]
            notes = {v['item_id']: v for v in note_rows}
            if len(notes) != 512 or len(cells) != 5632:
                raise ValueError('Complete fixed cohorts required')
            buckets = defaultdict(lambda: dict(items=0, own_correct=0, teacher_correct=0))
            expected_lengths, observed_lengths = Counter(), Counter()
            first_examples = {}
            for q in questions:
                key = q['item_id']
                note = notes[key]
                state = independent_state(note, q['expression_tree'])
                expected = exact_trace(q['expression_tree'])[1][:-1]
                matches = [trace_pattern.fullmatch(line) for line in note['text'].strip().splitlines()]
                actual = [(int(m[1]), m[2], int(m[3]), int(m[4])) for m in matches] if state['equations_supported'] else []
                if state['canonical_nonroot_exact']:
                    bucket = 'exact_canonical'
                elif not state['equations_supported']:
                    bucket = 'unsupported'
                elif not state['all_stated_equations_correct']:
                    bucket = 'arithmetic_error'
                elif Counter(actual) == Counter(expected):
                    bucket = 'same_required_equations_different_order'
                elif not (Counter(actual)-Counter(expected)):
                    bucket = 'proper_subset_of_required_equations'
                else:
                    bucket = 'correct_arithmetic_other_trace'
                own, teacher = cells['reader', key, 'own'], cells['reader', key, 'teacher']
                values = buckets[bucket]
                values['items'] += 1
                values['own_correct'] += own['score']
                values['teacher_correct'] += teacher['score']
                expected_lengths[state['expected_nonroot_statements']] += 1
                observed_lengths[state['observed_statements']] += 1
                if bucket not in first_examples:
                    first_examples[bucket] = dict(question=q, note=note, state=state,
                                                 student_answer=own, teacher_answer=teacher)
            if sum(v['items'] for v in buckets.values()) != 512:
                raise ValueError('No error category may disappear')
            metric = primary['results']
            costs = {}
            for name, arm, mode in [('split', 'reader', 'own'), ('direct', 'direct', 'direct'),
                                    ('compact_cot', 'cot', 'cot'), ('full_cot', 'cot_full', 'cot')]:
                v = metric[arm][mode]
                costs[name] = dict(mean_generated_actions=(v['generated_actions']+v['writer_actions'])/512,
                                   mean_recorded_decode_seconds=(v['decode_wall_seconds']+v['writer_wall_seconds'])/512,
                                   unsupported=v['unsupported'], correct=v['correct'])
            bfile = payload_root/f'B0-{panel}-all1536-original-answers.jsonl.gz'
            if sha(bfile) != report['public_payloads'][str(bfile.relative_to(root))]['sha256']:
                raise ValueError('Untouched reference payload changed')
            base = {(v['item_id'], v['mode']): v for v in rows(bfile)}
            extra_contrasts = {}
            for arm, mode in [('direct', 'direct'), ('cot', 'cot'), ('cot_full', 'cot')]:
                extra_contrasts[arm+'_minus_untouched_B0_cot'] = paired([
                    cells[arm, q['item_id'], mode]['score']-base[q['item_id'], 'cot']['score']
                    for q in questions])
            results[seed][panel] = dict(trace_categories=dict(buckets),
                expected_statement_counts=dict(expected_lengths), observed_statement_counts=dict(observed_lengths),
                costs=costs, additional_posthoc_contrasts=extra_contrasts,
                examples_first_in_each_observed_category=first_examples)
    training_lengths = Counter()
    with (root/'local/nvme/data/partial-state-arithmetic-v6/TRAIN.jsonl').open() as stream:
        for line in stream:
            q = json.loads(line)
            training_lengths[len(exact_trace(q['expression_tree'])[1])-1] += 1
    result = dict(schema='partial-state-v7-descriptive-error-cost-analysis-v1',
        status='POSTHOC_DESCRIPTIVE_PRIMARY_RESULTS_UNCHANGED', source_report_sha256=sha(source),
        analysis_script_sha256=sha(__file__), results=results,
        TRAIN_expected_nonroot_statement_counts=dict(training_lengths),
        limitations=['Descriptive analysis after the complete primary question; extra baseline contrasts are post hoc.',
                     'Trace categories compare exact equation multisets, not every algebraically equivalent computation.',
                     'Correct arithmetic alone does not imply a relevant or complete computation.',
                     'Recorded decode times are separate concurrent per-call measurements, not an end-to-end service benchmark. Initial model loading/native/cache qualification is outside them; compilation inside a decode may be included.',
                     'Generation uses FP32; checkpoints were trained with BF16 parameters/gradients and FP32 masters.',
                     'Paired intervals measure question uncertainty within each panel and seed; no pooling of two seeds or shared B0.'])
    output = root/'validation/partial_state_v7_matched5120_descriptive_20261019.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return dict(output=str(output), training_nonroot_statements=dict(training_lengths))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2))
