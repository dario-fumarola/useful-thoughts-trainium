"""Independently check published native scores, signs, counts and literal gates."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import numpy as np

ROOT = Path('/home/ubuntu/dario')
path = ROOT / 'validation/real_trace_cpt_v10_complete_20261019.json'
report = json.loads(path.read_text())
assert report['status'] == 'COMPLETE_TWO_SEED_REAL_DOCUMENT_STUDY_ALL_FIXED_PANELS'
directory = path.with_suffix('')
cells = {}
BOOTSTRAP_INDICES = np.random.default_rng(2026101915).integers(0,256,size=(20000,256))
score_rows = note_rows = 0

def load_payload(name, kind):
    global score_rows, note_rows
    source = ROOT / name
    metadata = report['payloads'][name]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata['sha256']
    values = [json.loads(line) for line in gzip.decompress(source.read_bytes()).splitlines()]
    assert len(values) == metadata['rows']
    if kind == 'score':
        score_rows += len(values)
    else:
        note_rows += len(values)
    return values

for model, panels in report['absolute_likelihood'].items():
    cells[model] = {}
    for panel, modes in panels.items():
        names = [name for name in report['payloads']
                 if Path(name).name.startswith(f'{model}-{panel}-all') and name.endswith('-logps.jsonl.gz')]
        assert len(names) == 1
        values = load_payload(names[0], 'score')
        cells[model][panel] = {}
        for mode, summary in modes.items():
            selected = [row for row in values if row['mode'] == mode]
            assert len(selected) == len({row['document_id'] for row in selected}) == 256
            assert len({row['item_id'] for row in selected}) == 256
            for row in selected:
                assert len(row['logps16']) == 16
                assert all(math.isfinite(v) and v <= 1e-6 for v in row['logps16'])
                assert math.isclose(math.fsum(row['logps16']), row['span16_logprob'], abs_tol=1e-10)
            mean = math.fsum(math.fsum(row['logps16']) for row in selected) / 256
            assert math.isclose(mean, summary['mean_span16_logprob'], abs_tol=1e-10)
            assert math.isclose(-mean/16, summary['mean_token_NLL'], abs_tol=1e-10)
            assert summary['generation_error_rows'] == sum(row['note_execution_status'] != 'ok' for row in selected)
            cells[model][panel][mode] = {row['document_id']: row for row in selected}

def difference(reference, treatment, panel, reference_mode, treatment_mode, result, nll=False):
    a, b = cells[reference][panel][reference_mode], cells[treatment][panel][treatment_mode]
    assert set(a) == set(b)
    deltas = []
    for document in sorted(a, key=lambda d:a[d]['item_id']):
        assert a[document]['item_id'] == b[document]['item_id']
        value = math.fsum(b[document]['logps16']) - math.fsum(a[document]['logps16'])
        deltas.append(-value/16 if nll else value)
    assert math.isclose(math.fsum(deltas)/256, result['difference_b_minus_a'], abs_tol=1e-10)
    assert len(result['ci95']) == 2 and all(math.isfinite(v) for v in result['ci95'])
    assert result['ci95'][0] <= result['ci95'][1]
    assert result['items']==result['documents']==256 and result['bootstrap_draws']==20000 and result['bootstrap_seed']==2026101915
    independent_ci=np.quantile(np.asarray(deltas,dtype=np.float64)[BOOTSTRAP_INDICES].mean(axis=1),[.025,.975])
    assert np.allclose(independent_ci,result['ci95'],rtol=0,atol=1e-10),(independent_ci,result['ci95'])

for model, panels in report['within_model_contrasts'].items():
    for panel, values in panels.items():
        for name, am, bm in [('empty_minus_raw','raw','empty'),('oracle_minus_raw','raw','oracle'),('own_minus_raw','raw','own'),('own_minus_empty','empty','own'),('own_minus_other','other','own')]:
            difference(model, model, panel, am, bm, values[name])

for seed, results in report['seeds'].items():
    raw, prior, post = (f'{arm}-seed{seed}' for arm in ('real_raw', 'real_prior', 'real_post'))
    comparisons = {
        'post_raw_minus_raw_raw': (raw, post, 'raw', 'raw'),
        'prior_own_minus_raw_raw': (raw, prior, 'raw', 'own'),
        'prior_own_minus_prior_raw': (prior, prior, 'raw', 'own'),
        'prior_own_minus_prior_empty': (prior, prior, 'empty', 'own'),
        'prior_own_minus_prior_other': (prior, prior, 'other', 'own'),
        'prior_own_minus_raw_own': (raw, prior, 'own', 'own'),
        'post_raw_minus_B0raw': ('B0', post, 'raw', 'raw'),
        'raw_raw_minus_B0raw': ('B0', raw, 'raw', 'raw'),
        'prior_own_minus_B0own': ('B0', prior, 'own', 'own'),
        'prior_own_minus_B0raw': ('B0', prior, 'raw', 'own'),
    }
    for key, (a, b, am, bm) in comparisons.items():
        difference(a, b, 'DEV_AST256', am, bm, results['primary_AST'][key])
    for arm, model in [('post', post), ('prior', prior)]:
        difference(raw, model, 'DEV_ordinary256', 'raw', 'raw', results['ordinary_token_NLL_regression'][arm], nll=True)
    for key in results['legacy_secondary']:
        a, b, am, bm = comparisons[key]
        difference(a, b, 'DEV_legacy256', am, bm, results['legacy_secondary'][key])
    primary = results['primary_AST']
    gain, ordinary = primary['post_raw_minus_raw_raw'], results['ordinary_token_NLL_regression']['post']
    training = (gain['difference_b_minus_a'] >= .25 and gain['ci95'][0] > 0
                and ordinary['difference_b_minus_a'] <= .02 and ordinary['ci95'][1] <= .05)
    inference_gain = primary['prior_own_minus_raw_raw']
    execution = all(row['note_execution_status'] == 'ok'
                    for mode in ('own', 'other') for row in cells[prior]['DEV_AST256'][mode].values())
    inference = (execution and inference_gain['difference_b_minus_a'] >= .25 and inference_gain['ci95'][0] > 0
                 and primary['prior_own_minus_prior_raw']['ci95'][0] > 0
                 and primary['prior_own_minus_prior_empty']['ci95'][0] > 0)
    assert training == results['declared_decisions']['training_benefit_pass']
    assert inference == results['declared_decisions']['inference_benefit_pass']

for name in report['payloads']:
    if name.endswith('-generated-notes.jsonl.gz'):
        values = load_payload(name, 'notes')
        assert len(values) == len({row['document_id'] for row in values}) == 256
        assert len({row['item_id'] for row in values}) == 256
        for row in values:
            assert row['n_actions'] == len(row['actions']) <= 128
            assert row['actions'][:len(row['content_ids'])] == row['content_ids']

assert score_rows == 19712 and note_rows == 3584 and len(report['payloads']) == 35
result = dict(status='PASS_PUBLIC_SCORES_INDEPENDENT_VECTOR_BOOTSTRAP_COUNTS_SIGNS_AND_LITERAL_GATES',
              utc=datetime.now(timezone.utc).isoformat(), score_rows=score_rows, note_rows=note_rows,
              report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              limits='Independent arithmetic/completeness and vectorized paired-document bootstrap check; actual model forwards are not rerun. Intervals remain conditional on fixed checkpoints, not training-seed uncertainty.')
with (directory/'independent-readout-check.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
