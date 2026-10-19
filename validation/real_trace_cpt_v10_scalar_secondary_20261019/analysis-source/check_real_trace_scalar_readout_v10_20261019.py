"""Independent exact-value/count/vector-bootstrap check of complete public outputs."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import numpy as np

R = Path('/home/ubuntu/dario')
path = R/'validation/real_trace_cpt_v10_scalar_secondary_20261019.json'
report = json.loads(path.read_text())
assert report['status'] == 'COMPLETE_ALL_SEVEN_MODELS_BOTH_FIXED_SCALAR_PANELS'
indices = np.random.default_rng(2026101915).integers(0, 256, size=(20000, 256))
cells = {}
total = 0
for model, panels in report['tables'].items():
    cells[model] = {}
    for panel, summaries in panels.items():
        name = f'validation/real_trace_cpt_v10_scalar_secondary_20261019/{model}-{panel}-all1280-scalar-continuations.jsonl.gz'
        file = R/name
        assert hashlib.sha256(file.read_bytes()).hexdigest() == report['payloads'][name]['sha256']
        values = [json.loads(line) for line in gzip.decompress(file.read_bytes()).splitlines()]
        assert len(values) == report['payloads'][name]['rows'] == 1280
        total += len(values)
        cells[model][panel] = {}
        for value in values:
            expected = Fraction(value['expected_numeric_value'])
            parsed = value['parsed']
            valid = parsed['status'] == 'valid'
            numeric = None
            if valid:
                literal = parsed['literal']
                assert value['text'].lstrip(' \t\r\n').startswith(literal)
                parts = literal.split('/')
                assert len(parts) in (1, 2)
                numeric = Fraction(parts[0].strip())
                if len(parts) == 2:
                    numeric /= Fraction(parts[1].strip())
                assert numeric == Fraction(parsed['value'])
            execution_error = value['status'] != 'ok' or value['source_note_status'] != 'ok'
            correct = valid and not execution_error and numeric == expected
            assert bool(value['correct']) == correct
            assert value['category'] == ('execution_error' if execution_error else parsed['status'])
            actions = value['actions']
            assert len(actions) == value['n_actions'] <= 32
            stopped = bool(actions and actions[-1] == 151643)
            assert 151643 not in actions[:-1]
            assert value['content_ids'] == (actions[:-1] if stopped else actions)
            if value['status'] == 'ok':
                assert value['stop_type'] == ('token:151643' if stopped else 'length_cap')
                assert stopped or len(actions) == 32
        for mode, summary in summaries.items():
            selected = [v for v in values if v['mode'] == mode]
            assert len(selected) == len({v['item_id'] for v in selected}) == len({v['document_id'] for v in selected}) == 256
            assert summary['documents'] == 256 and summary['correct'] == sum(v['correct'] for v in selected)
            assert summary['categories'] == dict(Counter(v['category'] for v in selected))
            assert sum(summary['categories'].values()) == 256
            assert summary['capped'] == sum(v['stop_type'] == 'length_cap' for v in selected)
            assert summary['generated_actions'] == sum(v['n_actions'] for v in selected)
            cells[model][panel][mode] = {v['item_id']: v for v in selected}

for seed, panels in report['contrasts'].items():
    raw, prior, post = (f'{arm}-seed{seed}' for arm in ('real_raw', 'real_prior', 'real_post'))
    comparisons = dict(
        post_raw_minus_raw_raw=(raw, post, 'raw', 'raw'),
        prior_own_minus_raw_raw=(raw, prior, 'raw', 'own'),
        prior_own_minus_prior_raw=(prior, prior, 'raw', 'own'),
        prior_own_minus_prior_empty=(prior, prior, 'empty', 'own'),
        prior_own_minus_prior_other=(prior, prior, 'other', 'own'),
        prior_own_minus_raw_own=(raw, prior, 'own', 'own'),
        post_raw_minus_B0raw=('B0', post, 'raw', 'raw'),
        raw_raw_minus_B0raw=('B0', raw, 'raw', 'raw'),
        prior_own_minus_B0raw=('B0', prior, 'raw', 'own'),
        prior_own_minus_B0own=('B0', prior, 'own', 'own'))
    for panel, results in panels.items():
        for key, result in results.items():
            a, b, amode, bmode = comparisons[key]
            av, bv = cells[a][panel][amode], cells[b][panel][bmode]
            assert set(av) == set(bv)
            for item in av:
                assert av[item]['document_id'] == bv[item]['document_id']
                assert av[item]['expected_numeric_value'] == bv[item]['expected_numeric_value']
            delta = np.array([int(bv[k]['correct'])-int(av[k]['correct']) for k in sorted(av)], dtype=np.float64)
            assert math.isclose(delta.mean(), result['difference_b_minus_a'], abs_tol=1e-12)
            ci = np.quantile(delta[indices].mean(1), [.025, .975])
            assert np.allclose(ci, result['ci95'], rtol=0, atol=1e-12)
            assert result['items'] == result['documents'] == 256
            assert result['bootstrap_seed'] == 2026101915 and result['bootstrap_draws'] == 20000

assert total == report['scalar_continuations'] == 17920 and len(report['payloads']) == 14
result = dict(status='PASS_PUBLIC_SCALAR_VALUES_NATIVE_STOPS_ALL_COUNTS_AND_INDEPENDENT_VECTOR_BOOTSTRAP',
              utc=datetime.now(timezone.utc).isoformat(), scalar_continuations=total,
              report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              limits='Checks exact leading literal/rational values, errors, counts and intervals independently. Syntax-validity flags come from the frozen parser; no model forwards rerun.')
with (path.with_suffix('')/'independent-readout-check.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
