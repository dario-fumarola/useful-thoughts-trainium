import importlib.util
import json
from pathlib import Path

import pytest

from weekend_thinking.artifacts import canonical_hash, sha256_file

spec = importlib.util.spec_from_file_location('summary_script', Path(__file__).resolve().parents[1] / 'scripts/summarize_downstream.py')
summary_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary_script)


@pytest.fixture
def conditions(tmp_path):
    paths = {}
    for name, (arm, mode) in summary_script.CONDITIONS.items():
        path = tmp_path / (name + '.jsonl')
        identity = dict(arm=arm, seed=1234, spec_hash='spec', step=256,
                        order_hash='order', base_assets_sha256='base', model_sha256=arm)
        condition = dict(arm=arm, mode=mode, seed=1234, spec_hash='spec', step=256)
        rows = [dict(item_id=f'gsm8k:test:{item:05d}', document_id=str(item), score=float(item % 2),
                     condition=condition, model_identity_sha256=canonical_hash(identity),
                     protocol_sha256=mode, scoring_manifest_sha256='gold') for item in range(4)]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        summary = dict(qualification=False, items=4, condition=condition, model_identity=identity,
                       protocol_sha256=mode, scoring_manifest_sha256='gold', scores_sha256=sha256_file(path),
                       inputs_sha256=dict(frozen_protocol='freeze', scoring_manifest='gold'))
        path.with_suffix('.summary.json').write_text(json.dumps(summary))
        paths[name] = path
    return paths


def test_correct_four_conditions_retain_provenance(conditions):
    rows, summaries = summary_script.read_final_conditions(conditions)
    assert len(rows) == 4 and all(len(value) == 4 for value in rows.values())
    assert summaries['r_thought']['condition']['mode'] == 'one_thought'


def test_swapped_condition_file_is_rejected(conditions):
    conditions['c_direct'], conditions['r_direct'] = conditions['r_direct'], conditions['c_direct']
    with pytest.raises(ValueError, match='condition binding'):
        summary_script.read_final_conditions(conditions)


def test_score_file_corruption_is_rejected(conditions):
    path = conditions['r_thought']
    path.write_text(path.read_text().replace('"score": 0.0', '"score": 1.0'))
    with pytest.raises(ValueError, match='condition binding'):
        summary_script.read_final_conditions(conditions)


@pytest.mark.parametrize('change', ['seed', 'checkpoint', 'protocol', 'gold', 'row'])
def test_incompatible_pair_is_rejected_even_with_valid_score_hash(conditions, change):
    path = conditions['r_thought']
    summary_path = path.with_suffix('.summary.json')
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == 'seed':
        summary['condition']['seed'] = summary['model_identity']['seed'] = 5678
    if change == 'checkpoint':
        summary['model_identity']['model_sha256'] = 'different-model'
    if change == 'protocol':
        summary['protocol_sha256'] = 'different-protocol'
    if change == 'gold':
        summary['scoring_manifest_sha256'] = 'different-gold'
    for row in rows:
        row.update(condition=summary['condition'], model_identity_sha256=canonical_hash(summary['model_identity']),
                   protocol_sha256=summary['protocol_sha256'], scoring_manifest_sha256=summary['scoring_manifest_sha256'])
    if change == 'row':
        rows[0]['condition'] = dict(rows[0]['condition'], arm='vanilla')
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    summary['scores_sha256'] = sha256_file(path)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        summary_script.read_final_conditions(conditions)
