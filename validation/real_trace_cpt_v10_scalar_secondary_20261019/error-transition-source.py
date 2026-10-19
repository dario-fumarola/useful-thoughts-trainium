"""Reproduce the descriptive transition partition; no new observations or gate.

Run from the repository root with Python 3. The original result stays immutable.
"""
from collections import Counter
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path.cwd()
REPORT = ROOT / 'validation/real_trace_cpt_v10_scalar_secondary_20261019.json'
assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == 'c25b6a06f855684622084a9e4744b2c55ef8f7290f31ee108e0aca8537e1855f'
report = json.loads(REPORT.read_text())

def rows(model, panel):
    name = f'validation/real_trace_cpt_v10_scalar_secondary_20261019/{model}-{panel}-all1280-scalar-continuations.jsonl.gz'
    data = (ROOT / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == report['payloads'][name]['sha256']
    values = [json.loads(line) for line in gzip.decompress(data).splitlines()]
    assert len(values) == 1280
    return {(v['item_id'], v['mode']): v for v in values}

contrasts = {}
for seed in ['2026101913', '2026101914']:
    contrasts[seed] = {}
    for panel in ['DEV_AST256', 'DEV_legacy256']:
        prior = rows(f'real_prior-seed{seed}', panel)
        raw = rows(f'real_raw-seed{seed}', panel)
        base = rows('B0', panel)
        contrasts[seed][panel] = {}
        for name, reference, mode in [('raw_control', raw, 'raw'), ('same_model_raw', prior, 'raw'),
                                      ('same_model_empty', prior, 'empty'), ('B0raw', base, 'raw')]:
            count = Counter(documents=256, both_valid=0, wrong_to_correct_with_both_valid=0,
                            correct_to_wrong_with_both_valid=0, correct_gain_when_reference_not_valid=0,
                            correct_loss_when_treatment_not_valid=0)
            keys = [key for key, m in prior if m == 'own']
            assert len(keys) == len(set(keys)) == 256
            net = 0
            for key in keys:
                a, b = reference[key, mode], prior[key, 'own']
                av, bv = a['category'] == 'valid', b['category'] == 'valid'
                ac, bc = bool(a['correct']), bool(b['correct'])
                assert not ac or av
                assert not bc or bv
                net += int(bc) - int(ac)
                if av and bv:
                    count['both_valid'] += 1
                    count['wrong_to_correct_with_both_valid'] += int(bc and not ac)
                    count['correct_to_wrong_with_both_valid'] += int(ac and not bc)
                elif bc and not av:
                    count['correct_gain_when_reference_not_valid'] += 1
                elif ac and not bv:
                    count['correct_loss_when_treatment_not_valid'] += 1
            count['total_correct_gain_all256'] = net
            assert net == (count['wrong_to_correct_with_both_valid'] - count['correct_to_wrong_with_both_valid']
                           + count['correct_gain_when_reference_not_valid'] - count['correct_loss_when_treatment_not_valid'])
            contrasts[seed][panel][name] = dict(count)

original = json.loads((REPORT.with_suffix('') / 'error-transitions.json').read_text())
assert contrasts == original['contrasts']
print(json.dumps(dict(status='PASS_DESCRIPTIVE_TRANSITIONS_REPRODUCED_ALL256_RETAINED', contrasts=contrasts), indent=2))
