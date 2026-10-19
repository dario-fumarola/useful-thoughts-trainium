"""A format-only control must match the selector coefficient, not its labels."""
from copy import deepcopy
import json
import sys
import pytest
from train.partial_state_driver import (
    EVIDENCE_FORMAT_ID, main, validate, validate_evidence_format_control,
)
from weekend_thinking.artifacts import sha256_file
from test_document_evidence_weight import protocols


def compared_protocols(qualification=False):
    control, selector = protocols()
    candidate = deepcopy(control)
    candidate.update(experiment_id=EVIDENCE_FORMAT_ID, selection_loss_weight=1.,
        purpose=('document_evidence_format_weight_qualification' if qualification
                 else 'document_evidence_format_weight_control'),
        control_protocol_file='direct.json', control_protocol_sha256='d'*64,
        selector_protocol_file='selector.json', selector_protocol_sha256='e'*64)
    if qualification:
        candidate.update(items=128, steps=8, save_every=2)
    return control, selector, candidate


@pytest.mark.parametrize('qualification', [False, True])
def test_versioned_format_control_preserves_both_comparisons(qualification):
    control, selector, candidate = compared_protocols(qualification)
    validate(candidate)
    validate_evidence_format_control(candidate, control, selector)
    for update in ({'experiment_id': 'old-study'}, {'selection_loss_weight': .1},
                   {'selection_loss_weight': True}, {'selector_protocol_sha256': ''},
                   {'control_protocol_file': None}):
        with pytest.raises(ValueError):
            validate({**candidate, **update})
    # Neither old protocol can silently adopt the new weight/identity.
    with pytest.raises(ValueError):
        validate({**control, 'selection_loss_weight': 1.})
    with pytest.raises(ValueError):
        validate_evidence_format_control(selector, control, selector)


@pytest.mark.parametrize('key,value', [
    ('seed', 2026101924), ('source_loss_weight', .25),
    ('answer_loss_weight', .25), ('lr', 2e-5), ('items', 1024),
    ('steps', 128), ('save_every', 64), ('data_sha256', 'f'*64),
    ('view_schedule', 'whole_pass'),
])
def test_format_control_rejects_unmatched_training_settings(key, value):
    control, selector, candidate = compared_protocols()
    with pytest.raises(ValueError):
        validate_evidence_format_control({**candidate, key: value}, control, selector)


def test_format_control_cannot_borrow_changed_kernel_or_wrong_selector():
    control, selector, candidate = compared_protocols()
    changed = deepcopy(candidate)
    changed['source_files']['train/masked_causal_ce.py'] = 'f'*64
    with pytest.raises(ValueError):
        validate_evidence_format_control(changed, control, selector)
    with pytest.raises(ValueError):
        validate_evidence_format_control(candidate, control,
                                         {**selector, 'selection_loss_weight': .1})


def argv(tmp_path, protocol, arm):
    path = tmp_path/'candidate.json'
    path.write_text(json.dumps(protocol))
    return ['driver', '--protocol', str(path), '--output', str(tmp_path/'output'),
            '--stop-file', str(tmp_path/'STOP'), '--arm', arm,
            '--deadline-utc', '2030-01-01T00:00:00+00:00']


def test_format_control_cannot_launch_the_selector_arm(tmp_path, monkeypatch):
    _, _, candidate = compared_protocols(qualification=True)
    monkeypatch.setattr(sys, 'argv', argv(tmp_path, candidate, 'evidence_select'))
    with pytest.raises(ValueError, match='Only the coefficient-matched format control'):
        main()


def test_science_cannot_borrow_the_selector_qualification(tmp_path, monkeypatch):
    control, selector, candidate = compared_protocols()
    for key, value in [('control_protocol', control), ('selector_protocol', selector)]:
        path = tmp_path/(key+'.json')
        path.write_text(json.dumps(value))
        candidate[key+'_file'] = str(path)
        candidate[key+'_sha256'] = sha256_file(path)
    qualification = tmp_path/'qualification.json'
    qualification.write_text(json.dumps(dict(
        status='PASS_ACTUAL_DOCUMENT_EVIDENCE_WEIGHT_FULL_RESTORE',
        source_files=candidate['source_files'], qualified_training_shape={k:candidate[k]
            for k in ('world_size','micro_batch','global_batch','width','target_slots','head_chunk','accumulation')})))
    candidate.update(hardware_qualification_file=str(qualification),
                     hardware_qualification_sha256=sha256_file(qualification))
    monkeypatch.setattr(sys, 'argv', argv(tmp_path, candidate, 'evidence_direct'))
    with pytest.raises(ValueError, match='Actual same-source and same-shape'):
        main()
