"""A coefficient experiment must preserve the existing data and loss pathways."""
from copy import deepcopy
import pytest
from train.partial_state_driver import (
    EVIDENCE_SCHEMA, EVIDENCE_WEIGHT_ID, source_files, validate,
    validate_evidence_weight_control,
)


def protocols(qualification=False):
    control = dict(schema=EVIDENCE_SCHEMA, model_id='Qwen/Qwen3-1.7B-Base',
        world_size=8, micro_batch=1, global_batch=32, accumulation=4,
        width=3328, target_slots=512, head_chunk=32, schedule='constant',
        parameter_dtype='bf16', optimizer_dtype='fp32', items=2048,
        purpose='document_evidence_plan_pilot', steps=256, save_every=128,
        seed=2026101923, lr=1e-5, source_sample_labels=256,
        view_schedule='interleaved_fixed_item_plus_epoch_parity',
        loss='separate_mean_source_answer_selection_per_example',
        source_loss_weight=.5, answer_loss_weight=.5, selection_loss_weight=.1,
        data_sha256='a'*64, data_manifest_sha256='b'*64,
        source_files=source_files(), hardware_qualification_file='old.json',
        hardware_qualification_sha256='c'*64)
    treatment = deepcopy(control)
    treatment.update(experiment_id=EVIDENCE_WEIGHT_ID, selection_loss_weight=1.,
        purpose='document_evidence_weight_qualification' if qualification else 'document_evidence_weight_ablation',
        control_protocol_file='control.json', control_protocol_sha256='d'*64)
    if qualification:
        treatment.update(items=128, steps=8, save_every=2)
    return control, treatment


@pytest.mark.parametrize('qualification', [False, True])
def test_new_weight_has_distinct_identity_and_preserves_control(qualification):
    control, treatment = protocols(qualification)
    validate(control)
    validate(treatment)
    validate_evidence_weight_control(treatment, control)
    with pytest.raises(ValueError):
        validate({**control, 'selection_loss_weight': 1.})
    with pytest.raises(ValueError):
        validate({**treatment, 'selection_loss_weight': .1})
    with pytest.raises(ValueError):
        validate({**treatment, 'experiment_id': 'old-study'})
    with pytest.raises(ValueError):
        validate({**treatment, 'control_protocol_sha256': ''})


@pytest.mark.parametrize('key,value', [
    ('source_loss_weight', .25), ('answer_loss_weight', .25),
    ('seed', 2026101924), ('data_sha256', 'f'*64),
    ('view_schedule', 'whole_pass'), ('lr', 2e-5),
    ('items', 1024), ('steps', 128), ('save_every', 64),
])
def test_control_comparison_rejects_other_training_changes(key, value):
    control, treatment = protocols()
    with pytest.raises(ValueError):
        validate_evidence_weight_control({**treatment, key: value}, control)


def test_changed_kernel_cannot_borrow_coefficient_only_identity():
    control, treatment = protocols()
    treatment['source_files']['train/masked_causal_ce.py'] = 'f'*64
    with pytest.raises(ValueError):
        validate_evidence_weight_control(treatment, control)
