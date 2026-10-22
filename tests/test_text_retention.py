"""Protect the reused worker and the fixed cross-domain source boundary."""
import json

import pytest

from scripts.run_uniform_lm import evaluation_entrypoint, prelaunch_memory_floor
from train import text_retention_evaluate as retention
from weekend_thinking.artifacts import sha256_file


@pytest.fixture
def frozen(tmp_path):
    data = tmp_path/'data.jsonl'
    data.write_text('{}\n')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(status='COMPLETE_GENERAL_WEB_SOURCE_PREPARATION',
        documents=1024, data_sha256=sha256_file(data))))
    check = tmp_path/'check.json'
    check.write_text(json.dumps(dict(status='PASS_INDEPENDENT_GENERAL_WEB_SOURCES',
                                    data_sha256=sha256_file(data))))
    p = dict(schema='frozen-text-retention-v1', cohort='ENGLISH_RETENTION',
        world_size=8, micro_batch=4, items=1024,
        source_files=retention.driver.source_files(),
        evaluator_sha256=sha256_file(retention.__file__),
        worker_evaluator_sha256=sha256_file(retention.original_evaluator.__file__),
        model_name='B0', checkpoint=None)
    for name, path in [('data', data), ('data_manifest', manifest),
                       ('independent_source_check', check)]:
        p[name+'_file'] = str(path)
        p[name+'_sha256'] = sha256_file(path)
    output = tmp_path/'output'
    release = dict(maximum_reserved_chip_hours=1.5,
        jobs={str(output): dict(protocol_sha256='frozen-protocol')},
        all_terminal_models={k: dict(step=128) for k in retention.BASELINE_MODELS})
    return p, release, output


def test_original_entrypoint_and_memory_defaults_stay_fixed():
    assert evaluation_entrypoint({}) == 'train.uniform_lm_evaluate'
    assert prelaunch_memory_floor({}) == 768*2**30
    assert evaluation_entrypoint(dict(evaluation_module='train.text_retention_evaluate')) == 'train.text_retention_evaluate'
    assert prelaunch_memory_floor(dict(minimum_prelaunch_host_available_GiB=672)) == 672*2**30


@pytest.mark.parametrize('module', ['os', 'train.partial_state_driver', '', None])
def test_unreviewed_evaluator_is_rejected(module):
    with pytest.raises(ValueError):
        evaluation_entrypoint(dict(evaluation_module=module))


@pytest.mark.parametrize('floor', [0, 511, 512, 671, 673, True, 768.])
def test_staging_floor_cannot_be_arbitrarily_lowered(floor):
    with pytest.raises(ValueError):
        prelaunch_memory_floor(dict(minimum_prelaunch_host_available_GiB=floor))


def test_valid_frozen_baseline_cohort(frozen):
    retention.validate(*frozen)


@pytest.mark.parametrize('field,value', [('cohort', 'DEV'), ('items', 1023),
    ('worker_evaluator_sha256', 'changed'), ('evaluator_sha256', 'changed'),
    ('micro_batch', 8)])
def test_changed_execution_contract_rejected(frozen, field, value):
    p, release, output = frozen
    p[field] = value
    with pytest.raises(ValueError):
        retention.validate(p, release, output)


def test_replacing_a_selected_source_is_rejected(frozen):
    p, release, output = frozen
    from pathlib import Path
    Path(p['data_file']).write_text('replacement')
    with pytest.raises(ValueError):
        retention.validate(p, release, output)


def test_every_baseline_terminal_must_precede_retention(frozen):
    p, release, output = frozen
    release['all_terminal_models'].pop(next(iter(retention.BASELINE_MODELS)))
    with pytest.raises(ValueError):
        retention.validate(p, release, output)


def test_all_six_scaffold_models_must_freeze_together(frozen, tmp_path):
    p, release, output = frozen
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'fixed checkpoint identity fixture')
    name = 'scaffold_trace-seed11'
    p.update(model_name=name, checkpoint=str(checkpoint),
             checkpoint_sha256=sha256_file(checkpoint))
    release['all_terminal_models'][name] = dict(step=128,
        checkpoint_sha256=p['checkpoint_sha256'])
    with pytest.raises(ValueError):
        retention.validate(p, release, output)
    for key in retention.SCAFFOLD_MODELS:
        release['all_terminal_models'][key] = dict(step=128,
            checkpoint_sha256=p['checkpoint_sha256'])
    retention.validate(p, release, output)
    checkpoint.write_bytes(b'altered checkpoint')
    with pytest.raises(ValueError):
        retention.validate(p, release, output)


def test_independent_source_failure_cannot_be_ignored(frozen):
    from pathlib import Path
    p, release, output = frozen
    path = Path(p['independent_source_check_file'])
    path.write_text(json.dumps(dict(status='FAIL', data_sha256=p['data_sha256'])))
    p['independent_source_check_sha256'] = sha256_file(path)
    with pytest.raises(ValueError):
        retention.validate(p, release, output)


def test_sources_keep_native_eos_and_reject_duplicate_pages(tmp_path):
    rows = [dict(item_id='item'+str(i), document_id=str(i), canonical_page='page'+str(i),
                 ids=[151643]+[3]*526+[151643]) for i in range(1024)]
    path = tmp_path/'sources.jsonl'
    def write():
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write()
    assert retention.read_sources(path)[0]['ids'] == rows[0]['ids']
    rows[-1]['canonical_page'] = rows[0]['canonical_page']
    write()
    with pytest.raises(ValueError):
        retention.read_sources(path)


def named_cohort(frozen):
    from pathlib import Path
    p, release, output = frozen
    p.update(schema='frozen-original-text-cohorts-v1', cohort='MATH_CONFIRM', items=2048)
    manifest = dict(status='COMPLETE_UNIFORM_LM_SOURCE_PREPARATION', identity='fixed-source-mixture',
        files={'MATH_CONFIRM': dict(rows=2048, sha256=p['data_sha256'], role='confirmation')})
    check = dict(status='PASS_INDEPENDENT_UNIFORM_MIXTURE_SOURCES',
                 files={'MATH_CONFIRM': dict(rows=2048, sha256=p['data_sha256'])})
    for name, value in [('data_manifest', manifest), ('independent_source_check', check)]:
        path = Path(p[name+'_file'])
        path.write_text(json.dumps(value))
        p[name+'_sha256'] = sha256_file(path)
    release.update(source_identity='fixed-source-mixture',
                   required_terminal_models=['mixed-seed11', 'mixed-seed12', 'uniform_reference'],
                   all_terminal_models={n: dict(step=128) for n in
                       ['mixed-seed11', 'mixed-seed12', 'uniform_reference']})
    return p, release, output


def test_named_fresh_cohorts_reuse_the_identical_worker(frozen):
    retention.validate(*named_cohort(frozen))


def test_training_role_cannot_be_used_as_confirmation(frozen):
    from pathlib import Path
    p, release, output = named_cohort(frozen)
    path = Path(p['data_manifest_file'])
    manifest = json.loads(path.read_text())
    manifest['files']['MATH_CONFIRM']['role'] = 'training'
    path.write_text(json.dumps(manifest))
    p['data_manifest_sha256'] = sha256_file(path)
    with pytest.raises(ValueError):
        retention.validate(p, release, output)


def test_named_study_freezes_both_orders_even_before_B0_scoring(frozen):
    p, release, output = named_cohort(frozen)
    release['all_terminal_models'].pop('mixed-seed12')
    with pytest.raises(ValueError):
        retention.validate(p, release, output)


@pytest.mark.parametrize('field,value', [('items', 2049), ('cohort', 'TRAIN'),
    ('worker_evaluator_sha256', 'different-worker')])
def test_named_cohort_rejects_changed_boundary(frozen, field, value):
    p, release, output = named_cohort(frozen)
    p[field] = value
    with pytest.raises(ValueError):
        retention.validate(p, release, output)


@pytest.mark.parametrize('budget', [True, 0, float('inf'), float('nan'), 2])
def test_score_budget_is_finite_and_bounded(frozen, budget):
    p, release, output = named_cohort(frozen)
    release['maximum_reserved_chip_hours'] = budget
    with pytest.raises(ValueError):
        retention.validate(p, release, output)
