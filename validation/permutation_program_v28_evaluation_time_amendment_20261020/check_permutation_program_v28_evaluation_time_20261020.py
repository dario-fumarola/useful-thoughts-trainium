"""CPU-only allowance/ownership checks, not model execution or science validation."""
from pathlib import Path
import copy
import importlib.util
import json
import os
import subprocess
import sys
import time

from permutation_program_v28_eval_budget_20261020 import MAX_CHIPS, MAX_CHIP_HOURS, STOP_CHIP_HOURS, reserved_hours, liveness
from release_permutation_program_v28_fixed_evaluation_20261020 import apply_time_only, DESIGN_SHA
from weekend_thinking.artifacts import atomic_json, sha256_file

R = Path('/home/ubuntu/dario')
C = R / 'local/diagnostics/permutation-program-v28-evaluation-time-amendment-v1'
HELPER = R / 'local/diagnostics/permutation-program-v28-world4-qualification-v1/qualify.py'


def main():
    C.mkdir(parents=True, exist_ok=False)
    assert MAX_CHIPS == 8 and MAX_CHIP_HOURS == 30 and STOP_CHIP_HOURS == 29.5
    # Synthetic bookkeeping only: the real all-six release is deliberately not
    # callable until all actual terminal model audits exist.
    original = dict(maximum_evaluation_reserved_chip_hours=18, prospective_design_sha256=DESIGN_SHA,
        evaluation_source_files={'fixture': 'unchanged'}, model_specs={'fixture': {'step': 4096}},
        decoding=dict(shards=4, max_actions=128, prefill_width=256, compute='FP32'),
        source='fixture, not an actual model release')
    amendment = dict(status='PROSPECTIVE_V28_EVALUATION_TIME_ONLY_AMENDMENT',
        prospective_design_sha256=DESIGN_SHA, evaluation_source_files={'fixture': 'unchanged'},
        held_model_outcomes_at_amendment=0, new_maximum_reserved_chip_hours=30,
        maximum_concurrent_chips=8, maximum_wall_hours=8)
    before = copy.deepcopy(original)
    adjusted = apply_time_only(original, amendment)
    assert original == before and adjusted['maximum_evaluation_reserved_chip_hours'] == 30
    adjusted['maximum_evaluation_reserved_chip_hours'] = 18
    assert adjusted == original
    rejected = []
    for key, value in [('held_model_outcomes_at_amendment', 1), ('new_maximum_reserved_chip_hours', 31),
                       ('maximum_concurrent_chips', 9), ('maximum_wall_hours', 9),
                       ('evaluation_source_files', {'fixture': 'changed'})]:
        bad = copy.deepcopy(amendment)
        bad[key] = value
        try:
            apply_time_only(original, bad)
        except AssertionError:
            rejected.append(key)
        else:
            raise AssertionError('Invalid amendment was accepted: ' + key)
    records = [dict(started_monotonic=10., closed_monotonic=3610.), dict(started_monotonic=1810.)]
    assert reserved_hours(records, 3610.) == 1.5
    # Even a conservative150-second simultaneous shutdown of all8 groups uses
    # less than the preserved0.5-chip-hour stop reserve.
    assert MAX_CHIPS * 150 / 3600 < MAX_CHIP_HOURS - STOP_CHIP_HOURS
    spec = importlib.util.spec_from_file_location('v28_time_check_owned_helpers', HELPER)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(20)'], start_new_session=True)
    try:
        identity = h.identity(process.pid)
        assert identity and h.same(identity)
        heartbeat = C / 'heartbeat-fixture.json'
        now = time.monotonic()
        beat = dict(status='ACTIVE_FIXED_V28_EVALUATION', owner=identity, maximum_chips=8,
                    maximum_reserved_chip_hours=30., reserved_chip_hours=29.49, monotonic=now)
        atomic_json(heartbeat, beat)
        assert liveness(heartbeat, now, h.same)
        assert not liveness(heartbeat, now + 61, h.same)
        for key, value in [('status', 'STOP_FIXED_V28_EVALUATION'), ('maximum_reserved_chip_hours', 18.),
                           ('reserved_chip_hours', 29.5), ('maximum_chips', 9)]:
            bad = dict(beat)
            bad[key] = value
            atomic_json(heartbeat, bad, overwrite=True)
            assert not liveness(heartbeat, now, h.same), key
        atomic_json(heartbeat, beat, overwrite=True)
        process.terminate()
        process.wait(timeout=5)
        assert not liveness(heartbeat, now, h.same)
        assert h.same(h.identity(os.getpid()))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    result = dict(status='PASS_CPU_TIME_ONLY_PRESERVATION_ACCOUNTING_AND_ACTUAL_OWNER_DEATH',
        model_calls=0, accelerator_calls=0, actual_terminal_releases=0,
        preserved_original_scientific_fields=True, invalid_amendments_rejected=rejected,
        maximum_chips=MAX_CHIPS, maximum_reserved_chip_hours=MAX_CHIP_HOURS,
        stop_at_chip_hours=STOP_CHIP_HOURS, shutdown_reserve_chip_hours=MAX_CHIP_HOURS - STOP_CHIP_HOURS,
        actual_owned_process_death_rejected=True, original_process_remains_alive=True,
        checker_sha256=sha256_file(__file__),
        limits=['Host bookkeeping and one actual disposable CPU process only, not eight-chip runtime qualification.',
                'The actual all-six checkpoint/model release remains gated on independent terminal audits.'])
    atomic_json(C / 'checks.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
