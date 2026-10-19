"""Prepare one concrete weight treatment after qualification; never launch it."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import shutil
import subprocess
from train.partial_state_driver import ROOT, source_files, validate, order_for, validate_evidence_weight_control

R = Path('/home/ubuntu/dario')
W = R/'local/worktrees/document-evidence-weight-v25'
Q = R/'local/diagnostics/document-evidence-weight-v25-world8-qualification-v1'
C = R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
assert ROOT == W
read = lambda p: json.loads(Path(p).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()
def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


assert not (C/'plan.json').exists() and not (C/'release.json').exists()
qualification = read(Q/'qualification-independent-audit.json')
assert qualification['status'] == 'PASS_ACTUAL_DOCUMENT_EVIDENCE_WEIGHT_FULL_RESTORE'
assert read(Q/'independent-audit-exit.json')['exit_code'] == 0
assert qualification['actual_updates'] == {'evidence_select': 8}
assert qualification['qualified_arms'] == ['evidence_select']
assert qualification['source_files'] == source_files()
assert len(qualification['actual_native_checks']) == 16
assert qualification['audit_source_sha256'] == sha(Q/'independent-audit.py')
assert read(Q/'qualification-release.json')['auditor_sha256'] == sha(Q/'independent-audit.py')
assert read(Q/'select-resume8/closure.json')['assigned_physical12_13_empty']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True).strip()
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip()
assert commit.startswith('5414f8d')
control_file = R/'local/diagnostics/document-evidence-plan-v23-pilot-v1/protocol-evidence_select-seed2026101923.json'
control = read(control_file)
design = R/'validation/document_evidence_weight_v25_design_20261019.json'
p = dict(control, purpose='document_evidence_weight_ablation', experiment_id='document-evidence-weight-v25',
    selection_loss_weight=1., control_protocol_file=str(control_file),
    control_protocol_sha256=sha(control_file), source_files=source_files(), visible_cores='8-15',
    hardware_qualification_file=str(Q/'qualification-independent-audit.json'),
    hardware_qualification_sha256=sha(Q/'qualification-independent-audit.json'),
    prospective_design_sha256=sha(design))
p['readout_addendum_sha256'] = sha(R/'validation/document_evidence_weight_v25_readout_addendum_20261019.json')
validate(p)
validate_evidence_weight_control(p, control)
assert qualification['qualified_training_shape'] == {k: p[k] for k in (
    'world_size', 'micro_batch', 'global_batch', 'width', 'target_slots', 'head_chunk', 'accumulation')}
assert sha(p['data_file']) == p['data_sha256']
assert Counter(order_for(p)) == Counter({i: 4 for i in range(2048)})
assert order_for(p) == order_for(control)
manifest = read(p['data_manifest_file'])
assert sha(p['data_manifest_file']) == p['data_manifest_sha256']
for item in manifest['files'].values():
    assert sha(item['path']) == item['sha256']
full = read(R/'local/nvme-study/diagnostics/document-evidence-weight-v25-world8-qualification-v1/step-0000008/COMPLETE.json')
full_bytes = sum(v['bytes'] for v in full['files'].values())
# Keep the old global science recovery copy until a newer full copy is verified.
# Two successive full copies leave the older model weights, so budget those too.
assert shutil.disk_usage(R).free > 35*2**30 + full_bytes + 5*2**30
assert shutil.disk_usage(R/'local/nvme-cpt').free > 35*2**30 + 2*(full_bytes+2**30)
path = C/'protocol-evidence_select-weight1-seed2026101923.json'
write(path, p)
out = R/'local/nvme-cpt/runs/document-evidence-weight-v25-selection1-TRAIN2048-seed2026101923-v1'
assert not out.exists()
job = dict(arm='evidence_select', seed=2026101923, key='evidence_select-weight1-seed2026101923',
    protocol=str(path), protocol_sha256=sha(path), output=str(out), physical_devices=[2, 3])
now = datetime.now(timezone.utc)
deadline = min(now+timedelta(minutes=84), datetime(2026, 10, 20, 4, tzinfo=timezone.utc))
assert (deadline-now).total_seconds() >= 80*60
write(C/'plan.json', dict(status='PREPARED_SINGLE_WEIGHT_TREATMENT_PENDING_ROOT_RELEASE',
    utc=now.isoformat(), source_root=str(W), source_commit=commit, jobs=[job],
    deadline_utc=deadline.isoformat(), qualification_file=p['hardware_qualification_file'],
    qualification_sha256=p['hardware_qualification_sha256'],
    prospective_design_sha256=sha(design), readout_addendum_sha256=p['readout_addendum_sha256'], held_primary_manifest_sha256=p['held_primary_manifest_sha256'],
    original_control_protocol_sha256=sha(control_file),
    prior_TRAIN_fit_report_sha256=sha(R/'validation/document_selector_fit_v24_complete_20261019.json'),
    budget=dict(max_reserved_chips=2, hard_training_chip_hour_bound_including_termination_grace=3.,
        soft_wall_minutes=84, termination_grace_minutes=5, evaluation_chip_hour_bound_separate=.6,
        measured_full_state_bytes=full_bytes, root_free_bytes=shutil.disk_usage(R).free,
        NVMe_free_bytes=shutil.disk_usage(R/'local/nvme-cpt').free),
    checkpoint_policy='Full128/256 retained on NVMe. Independently audit/protect128 before retiring ONLY previous EBS optimizer duplicates. Protect256 before retiring128 optimizer backup duplicates. Keep all original states and terminal models.',
    evaluation='First unchanged TRAIN256 selector-fit panel, terminal256 only. Fixed held255 readout after protection/root review regardless of fit outcome; developmental, not untouched confirmation.',
    no_automatic_retry=True, no_checkpoint_selection=True))
print(json.dumps(dict(status='PREPARED_SINGLE_WEIGHT_TREATMENT_PENDING_ROOT_RELEASE', job=job, deadline_utc=deadline.isoformat()), indent=2))
