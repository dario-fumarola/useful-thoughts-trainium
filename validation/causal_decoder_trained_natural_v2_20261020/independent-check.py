"""Independent CPU readout of saved natural-stop batching witnesses; no model calls."""
from pathlib import Path
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone

import numpy as np
import torch

C = Path('/home/ubuntu/dario/local/diagnostics/causal-decoder-trained-natural-v2')
read = lambda name: json.loads((C / name).read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

plan, result, closure = (read(n) for n in ('plan.json', 'worker-result.json', 'closure.json'))
assert sha(C / 'CPU-native.pt') == plan['cpu_native_sha256']
assert sha(C / 'inputs.json') == plan['inputs_sha256']
assert sha(C / 'plan.json') == result['plan_sha256']
assert all(sha(Path(p)) == digest for p, digest in plan['sources'].items())
reference = torch.load(C / 'CPU-native.pt', map_location='cpu', weights_only=True).double().numpy()
assert reference.shape == (8, 151936) and np.isfinite(reference).all()
r = reference - np.logaddexp.reduce(reference, axis=1)[:, None]
checks = []
for size in (1, 4, 8):
    actual = torch.load(C / f'XLA-native-B{size}.pt', map_location='cpu', weights_only=True).double().numpy()
    assert actual.shape == r.shape and np.isfinite(actual).all()
    a = actual - np.logaddexp.reduce(actual, axis=1)[:, None]
    delta = a - r
    stored = read(f'native-B{size}.json')
    assert stored == result['native'][str(size)]
    for i in range(8):
        forward = float(np.sum(np.exp(r[i]) * -delta[i]))
        reverse = float(np.sum(np.exp(a[i]) * delta[i]))
        assert min(forward, reverse) >= -1e-10
        values = dict(reference_to_device_kl_nats=max(0., forward),
                      device_to_reference_kl_nats=max(0., reverse),
                      max_bidirectional_kl_nats=max(0., forward, reverse),
                      max_logp_error=float(np.max(np.abs(delta[i]))),
                      rms_logp_error=float(np.sqrt(np.mean(delta[i] ** 2))),
                      relative_l2_logp_error=float(np.linalg.norm(delta[i]) / np.linalg.norm(r[i])))
        for key, value in values.items():
            assert math.isclose(value, stored[i][key], rel_tol=0., abs_tol=1e-10), (size, i, key)
        assert values['max_bidirectional_kl_nats'] <= .01 and values['max_logp_error'] <= .001
        assert int(np.argmax(a[i])) == int(np.argmax(r[i])) and stored[i]['argmax_equal'] and stored[i]['passed']
        checks.append(dict(batch_size=size, row=i, **values))
expected = read('original-serial-attempts.json')
assert len(expected) == len(read('inputs.json')) == 8
for size in (4, 8):
    assert read(f'batched-B{size}-attempts.json') == expected
lengths = [len(row['actions']) for row in expected]
assert lengths == result['action_lengths']
for row in expected:
    assert 0 < len(row['actions']) < 128
    assert row['stop_type'] == f"token:{row['actions'][-1]}"
    assert row['actions'][-1] in plan['stop_ids']
    assert row['content_ids'] == row['actions'][:-1]
    assert row['text'] == row['raw_text']
assert result['natural_stops'] == 8 and result['distinct_action_lengths'] == len(set(lengths)) == 8
times = read('timings.json')
assert times == result['timed_seconds']
medians = {key: statistics.median(values) for key, values in times.items()}
assert all(len(v) == 3 and all(t > 0 and math.isfinite(t) for t in v) for v in times.values())
assert medians == result['median_seconds']
assert {key: medians['1'] / value for key, value in medians.items()} == result['speedup_vs_serial']
timed_logs = []
for line in (C / 'worker.log').read_text().splitlines():
    if line.startswith('{"batch":'):
        timed_logs.append(json.loads(line))
assert len(timed_logs) == 9
for size in (1, 4, 8):
    assert [v['seconds'] for v in timed_logs if v['batch'] == size] == times[str(size)]
assert closure['exit_code'] == 0 and closure['actual_owned_device_observed']
assert closure['assigned_physical_device_empty'] and closure['all_original_session_processes_absent']
assert not closure['host_bounds_stop'] and 0 < closure['reserved_chip_hours'] < plan['maximum_reserved_chip_hours']
output = dict(status='PASS_INDEPENDENT_SAVED_NATIVE24_OUTPUTS8_TIMING9_AND_CLOSURE',
    utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(Path(__file__)), native_checks=checks,
    action_lengths=lengths, natural_stops=8, stored_output_conditions=3,
    original_serial_and_batched_outputs_exact=True, median_seconds=medians,
    speedup_vs_serial=result['speedup_vs_serial'], reserved_chip_hours=closure['reserved_chip_hours'],
    witness_files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in C.iterdir() if p.is_file()},
    limits=['No new model calls or capability scores.',
            'The three timed repeats attest output parity in the frozen producer; saved complete arrays cover the original serial and two batched conditions.',
            'Eight equal-length TRAIN prompts and one trained model; descriptive warm timing, not a serving benchmark.'])
with (C / 'independent-check.json').open('x') as f:
    json.dump(output, f, indent=2); f.write('\n')
print(json.dumps({k:v for k,v in output.items() if k not in ('native_checks','witness_files')}, indent=2))
