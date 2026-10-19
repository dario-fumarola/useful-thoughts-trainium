"""Release the fixed readout of the actual original protected terminal model."""
from pathlib import Path
from datetime import datetime, timezone
import json
from train.document_qa_evaluate import ROOT
import document_evidence_weight_v25_TRAIN_fit_20261019 as fit
import document_evidence_weight_v25_held_20261019 as held
from weekend_thinking.artifacts import atomic_json, sha256_file

R = Path('/home/ubuntu/dario')
C = R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
assert ROOT == R/'local/worktrees/document-evidence-eval-v23'
read = lambda p: json.loads(p.read_text())
release = read(C/'release.json')
assert release['status'] == 'RELEASED_DOCUMENT_EVIDENCE_WEIGHT_V25_TREATMENT_ONLY'
binding_file = C/'frozen-models/evidence_select-weight1-seed2026101923.json'
model = read(binding_file)
assert model['status'] == 'AUDITED_EVIDENCE_WEIGHT_V25_TERMINAL256' and model['step'] == 256
assert sha256_file(model['checkpoint']) == model['checkpoint_sha256']
assert sha256_file(model['audit_file']) == model['audit_sha256']
assert sha256_file(model['protection_file']) == model['protection_sha256']
assert read(Path(model['audit_file']))['status'] == 'PASS_SINGLE_TERMINAL256_EVIDENCE_WEIGHT_TRAINING_STATES_NATIVE8_ORDER_COUNTERS_AND_LABELS'
assert read(Path(model['protection_file']))['status'] == 'PASS_SINGLE_TERMINAL_MODEL_AND_LATEST_FULL256_PROTECTED'
for name, digest in release['pipeline_sources'].items():
    assert sha256_file(R/name) == digest
assert sha256_file(R/'validation/document_evidence_weight_v25_readout_addendum_20261019.json') == release['readout_addendum_sha256']
assert sha256_file(R/'validation/document_evidence_weight_v25_omission_control_20261019.json') == release['omission_control_contract_sha256']
assert sha256_file(R/'validation/document_evidence_plan_v23_pilot_complete_20261019.json') == 'ca7cc615495d0cdade77ef70cce0f19055e3456b1c79c460a7f84008dec809b3'
assert not fit.O.exists() and not held.NATIVE.exists() and not held.REPLAY.exists()
review = dict(status='RELEASED_WEIGHT_V25_TERMINAL_FIXED_EVALUATION',
    utc=datetime.now(timezone.utc).isoformat(), model_binding_sha256=sha256_file(binding_file),
    model_checkpoint_sha256=model['checkpoint_sha256'],
    data_manifest_sha256=release['data_manifest_sha256'],
    fit_source_files=fit.sources(), held_source_files=held.sources(),
    readout_addendum_sha256=release['readout_addendum_sha256'],
    omission_control_contract_sha256=release['omission_control_contract_sha256'],
    total_maximum_reserved_chip_hours=.6,
    evaluation=['Unchanged TRAIN256 selection fitting.',
        'Unchanged held255 native direct/selection, regardless of fitting outcome.',
        'Held255 canonical empty/own/oracle pointer answer regeneration with supplied Final header.',
        'Ordinary256 prediction cost; optional canonical span diagnostics only within remaining bound.'],
    checkpoint_rule='Original terminal256, no checkpoint selection, retry or prompt/parser adjustment.',
    limits=['Held255 was already inspected; developmental only.',
        'Exact support labels can be imperfect; no semantic-gold claim.',
        'All errors/caps/unsupported/fallbacks retained; all original scores/gates unchanged.'])
atomic_json(C/'evaluation-release.json', review)
print(json.dumps(dict(status=review['status'], model=model['id'], max_chip_hours=.6)))
