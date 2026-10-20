"""Apply the prospective time-only allowance to the unchanged all-six-model release.

The frozen builder writes and preserves its original18-hour record. The amended
operational record points to it and to the public pre-outcome amendment. Model
identities, source, data, decoding and scientific rules are identical.
"""
from pathlib import Path
import copy
import json

from scripts.release_permutation_program_evaluation import release as frozen_release
from train.permutation_program_science_evaluate import sources, verify_release
from weekend_thinking.artifacts import atomic_json, sha256_file

R = Path('/home/ubuntu/dario')
C = R / 'local/diagnostics/permutation-program-v28-fixed-evaluation-v1'
STUDY = R / 'local/diagnostics/permutation-program-v28-study-v1'
AMENDMENT = R / 'validation/permutation_program_v28_evaluation_time_amendment_20261020.json'
DESIGN_SHA = 'd01498ec6eec98fedad28bfa5b5426059ccceff196ef7522bb44ea27d98ebba6'
read = lambda path: json.loads(Path(path).read_text())


def apply_time_only(original, amendment):
    assert original['maximum_evaluation_reserved_chip_hours'] == 18
    assert original['prospective_design_sha256'] == amendment['prospective_design_sha256'] == DESIGN_SHA
    assert original['evaluation_source_files'] == amendment['evaluation_source_files']
    assert amendment['status'] == 'PROSPECTIVE_V28_EVALUATION_TIME_ONLY_AMENDMENT'
    assert amendment['held_model_outcomes_at_amendment'] == 0
    assert amendment['new_maximum_reserved_chip_hours'] == 30
    assert amendment['maximum_concurrent_chips'] == 8
    assert amendment['maximum_wall_hours'] == 8
    result = copy.deepcopy(original)
    result['maximum_evaluation_reserved_chip_hours'] = 30
    # Every original scientific/numerical field is byte-equivalent after JSON
    # serialization; no task, model or score can be changed by this function.
    restored = copy.deepcopy(result)
    restored['maximum_evaluation_reserved_chip_hours'] = 18
    assert restored == original
    return result


def main():
    amendment = read(AMENDMENT)
    assert amendment['evaluation_source_files'] == sources()
    for relative, digest in amendment['host_source_files'].items():
        assert sha256_file(R / relative) == digest
    assert not (C / 'original-release18.json').exists() and not (C / 'release.json').exists()
    assert not (C / 'EXECUTION').exists()
    assert not (R / 'local/nvme-cpt/evaluation/permutation-program-v28-fixed-v1').exists()
    original_path = C / 'original-release18.json'
    original = frozen_release(STUDY, original_path)
    result = apply_time_only(original, amendment)
    result['operational_budget_amendment'] = dict(path=str(AMENDMENT), sha256=sha256_file(AMENDMENT))
    result['original_six_terminal_release'] = dict(path=str(original_path), sha256=sha256_file(original_path))
    result['operational_release_source_sha256'] = sha256_file(__file__)
    atomic_json(C / 'release.json', result)
    verify_release(C / 'release.json')
    print(json.dumps(dict(status=result['status'], models=len(result['model_specs']),
                          maximum_evaluation_reserved_chip_hours=30, numerical_changes=False)))


if __name__ == '__main__':
    main()
