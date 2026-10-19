"""Serial release of the concrete, qualified single-coefficient treatment."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import shutil

R = Path('/home/ubuntu/dario')
C = R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
Q = R/'local/diagnostics/document-evidence-weight-v25-world8-qualification-v1'
read = lambda p: json.loads(Path(p).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

plan = read(C/'plan.json')
assert plan['status'] == 'PREPARED_SINGLE_WEIGHT_TREATMENT_PENDING_ROOT_RELEASE'
assert len(plan['jobs']) == 1
job = plan['jobs'][0]
assert job['arm'] == 'evidence_select' and job['seed'] == 2026101923
assert not Path(job['output']).exists()
p = read(job['protocol'])
assert sha(job['protocol']) == job['protocol_sha256']
q = read(plan['qualification_file'])
assert q['status'] == 'PASS_ACTUAL_DOCUMENT_EVIDENCE_WEIGHT_FULL_RESTORE'
assert q['actual_updates'] == {'evidence_select': 8} and len(q['actual_native_checks']) == 16
assert read(Q/'independent-audit-exit.json')['exit_code'] == 0
assert plan['qualification_sha256'] == sha(plan['qualification_file'])
assert q['source_files'] == p['source_files']
assert p['experiment_id'] == 'document-evidence-weight-v25' and p['selection_loss_weight'] == 1.
assert p['source_loss_weight'] == p['answer_loss_weight'] == .5
assert p['items'] == 2048 and p['steps'] == 256 and p['save_every'] == 128
assert sha(p['control_protocol_file']) == p['control_protocol_sha256'] == plan['original_control_protocol_sha256']
control_binding = R/'local/diagnostics/document-evidence-plan-v23-pilot-v1/frozen-models/evidence_select-seed2026101923.json'
control = read(control_binding)
assert control['status'] == 'AUDITED_EVIDENCE_PLAN_V23_TERMINAL256'
assert control['training_protocol_sha256'] == p['control_protocol_sha256']
assert sha(control['checkpoint']) == control['checkpoint_sha256']
assert sha(control['audit_file']) == control['audit_sha256']
assert sha(R/'validation/document_evidence_weight_v25_design_20261019.json') == plan['prospective_design_sha256']
assert sha(R/'validation/document_evidence_weight_v25_readout_addendum_20261019.json') == plan['readout_addendum_sha256']
assert sha(R/'validation/document_selector_fit_v24_complete_20261019.json') == plan['prior_TRAIN_fit_report_sha256']
assert (datetime.fromisoformat(plan['deadline_utc'])-datetime.now(timezone.utc)).total_seconds() >= 80*60
full = plan['budget']['measured_full_state_bytes']
assert shutil.disk_usage(R).free > 35*2**30 + full + 5*2**30
assert shutil.disk_usage(R/'local/nvme-cpt').free > 35*2**30 + 2*(full+2**30)
names = [
    'document_evidence_weight_v25_state_audit_20261019.py',
    'audit_document_evidence_weight_v25_terminals_20261019.py',
    'protect_document_evidence_weight_v25_first_full128_20261019.py',
    'protect_document_evidence_weight_v25_terminals_20261019.py',
    'document_evidence_weight_v25_TRAIN_fit_20261019.py',
    'check_document_evidence_weight_v25_TRAIN_fit_20261019.py',
    'supervise_document_evidence_weight_v25_TRAIN_fit_20261019.py',
    'document_evidence_weight_v25_held_20261019.py',
    'analyze_document_evidence_weight_v25_held_20261019.py',
    'bind_document_evidence_weight_v25_terminal_20261019.py',
    'supervise_document_evidence_weight_v25_held_20261019.py',
    'document_evidence_v23_likelihood_contexts.py',
    'real_source_state_audit_v17_20261019.py',
]
release = dict(status='RELEASED_DOCUMENT_EVIDENCE_WEIGHT_V25_TREATMENT_ONLY',
    utc=datetime.now(timezone.utc).isoformat(), plan_sha256=sha(C/'plan.json'),
    supervisor_sha256=sha(C/'supervise-job.py'), helper_sha256=sha(Q/'qualify.py'),
    qualification_file=plan['qualification_file'], qualification_sha256=plan['qualification_sha256'],
    source_commit=plan['source_commit'], control_binding_sha256=sha(control_binding),
    prospective_design_sha256=plan['prospective_design_sha256'],
    readout_addendum_sha256=plan['readout_addendum_sha256'],
    omission_control_contract_sha256=sha(R/'validation/document_evidence_weight_v25_omission_control_20261019.json'),
    pipeline_sources={str((R/'local/analysis'/n).relative_to(R)): sha(R/'local/analysis'/n) for n in names},
    data_manifest_sha256=p['data_manifest_sha256'],
    model_spec=dict(id='evidence_select-weight1-seed2026101923', arm='evidence_select',
        seed=job['seed'], step=256, run_id=Path(job['output']).name,
        training_protocol_sha256=job['protocol_sha256']),
    decision='Qualified actual8 updates/full restore4; same computational path and unchanged data/order/source-answer coefficients. Release one fresh treatment, terminal256 only. No second seed, model-size change, RL or parameter sweep.',
    budget=plan['budget'], deadline_utc=plan['deadline_utc'],
    evaluation='After actual terminal audit/protection and root review: fixed TRAIN fit and fixed held255 native/closed-field replay. Held panel is developmental regardless of fit outcome, never untouched confirmation.',
    limitations=['Author support labels can be insufficient or ambiguous.',
                'A scalar coefficient is not a measured gradient fraction.',
                'Changing the coefficient also changes clipping/optimizer dynamics.',
                'All old primary reports/scores/gates remain unchanged; no prompt or parser rescue.'])
with (C/'release.json').open('x') as stream:
    json.dump(release, stream, indent=2)
print(json.dumps(dict(status=release['status'], plan_sha256=release['plan_sha256'],
    job=job['key'], deadline_utc=release['deadline_utc']), indent=2))
