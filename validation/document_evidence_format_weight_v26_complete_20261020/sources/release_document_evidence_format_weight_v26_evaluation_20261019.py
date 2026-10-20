"""Release the two fixed answer paths and ordinary cost of the protected control."""
from pathlib import Path
import json
from train.document_qa_evaluate import ROOT
import document_evidence_format_weight_v26_held_20261019 as held
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-format-weight-v26-control-v1'
OLD=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
read=lambda p:json.loads(p.read_text())
assert ROOT==R/'local/worktrees/document-evidence-eval-v23'
release=read(C/'release.json');assert release['status']=='RELEASED_DOCUMENT_EVIDENCE_FORMAT_WEIGHT_V26_TREATMENT_ONLY'
binding=C/'frozen-models/evidence_direct-format1-seed2026101923.json';model=read(binding)
assert model['status']=='AUDITED_EVIDENCE_FORMAT_WEIGHT_V26_TERMINAL256' and model['step']==256
assert sha256_file(model['checkpoint'])==model['checkpoint_sha256']
for key in ('audit','protection'):assert sha256_file(model[key+'_file'])==model[key+'_sha256']
assert read(Path(model['audit_file']))['status']=='PASS_SINGLE_TERMINAL256_EVIDENCE_FORMAT_WEIGHT_TRAINING_STATES_NATIVE8_ORDER_COUNTERS_AND_LABELS'
assert read(Path(model['protection_file']))['status']=='PASS_SINGLE_TERMINAL_MODEL_AND_LATEST_FULL256_PROTECTED'
for name,digest in release['pipeline_sources'].items():assert sha256_file(R/name)==digest
assert sha256_file(R/'validation/document_evidence_format_weight_v26_design_20261019.json')==release['prospective_design_sha256']
assert read(OLD/'held-independent-check.json')['status']=='PASS_INDEPENDENT_NATIVE_TOKEN_ANSWER_COST_COUNT_AND_PAIRED_HELD_READOUT'
assert not held.O.exists()
review=dict(status='RELEASED_FORMAT_WEIGHT_V26_TERMINAL_FIXED_EVALUATION',utc=utc_now(),model_binding_sha256=sha256_file(binding),model_checkpoint_sha256=model['checkpoint_sha256'],
    data_manifest_sha256=release['data_manifest_sha256'],held_source_files=held.sources(),comparison_scored_sha256=sha256_file(OLD/'held-native-and-replay-scored.json'),
    total_maximum_reserved_chip_hours=.3,stage_caps=dict(generation=.2,ordinary=.1),
    evaluation=['Native direct and canonical closed-empty answers on unchanged inspected held255.','Same ordinary256/32targets; current-model CPU references and actual calibration required.'],
    primary='Selector1 direct minus format1 direct; predeclared developmental criterion only.',checkpoint_rule='Original terminal256, no checkpoint selection/retry/prompt/parser adjustment.',
    limits=['One order seed and inspected panel; any positive still needs separate fresh powered confirmation.','All errors/caps/unsupported kept in denominators; old results unchanged.'])
atomic_json(C/'evaluation-release.json',review)
print(json.dumps(dict(status=review['status'],model=model['id'],max_chip_hours=.3)))
