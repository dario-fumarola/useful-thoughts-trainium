"""Prepare the single missing format1.0 control after its actual mechanics audit."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import shutil
import subprocess
from train.partial_state_driver import ROOT, source_files, validate, order_for, validate_evidence_format_control

R=Path('/home/ubuntu/dario')
W=R/'local/worktrees/document-evidence-format-weight-v26'
Q=R/'local/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1'
C=R/'local/diagnostics/document-evidence-format-weight-v26-control-v1'
assert ROOT==W
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)

assert not (C/'plan.json').exists() and not (C/'release.json').exists()
audit=read(Q/'qualification-independent-audit.json')
assert audit['status']=='PASS_ACTUAL_DOCUMENT_EVIDENCE_FORMAT_WEIGHT_FULL_RESTORE'
assert read(Q/'independent-audit-exit.json')['exit_code']==0
assert audit['actual_updates']=={'evidence_direct':8} and audit['qualified_arms']==['evidence_direct']
assert audit['source_files']==source_files() and len(audit['actual_native_checks'])==16
assert audit['audit_source_sha256']==sha(Q/'independent-audit.py')==read(Q/'qualification-release.json')['audit_sha256']
assert read(Q/'direct-resume8/closure.json')['assigned_physical12_13_empty']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
assert commit.startswith('29606b0')
control_file=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1/protocol-evidence_direct-seed2026101923.json'
selector_file=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1/protocol-evidence_select-weight1-seed2026101923.json'
control,selector=read(control_file),read(selector_file)
design=R/'validation/document_evidence_format_weight_v26_design_20261019.json'
assert sha(design)==read(Q/'qualification-release.json')['prospective_design_sha256']
p=dict(control,purpose='document_evidence_format_weight_control',experiment_id='document-evidence-format-weight-v26',
    selection_loss_weight=1.,control_protocol_file=str(control_file),control_protocol_sha256=sha(control_file),
    selector_protocol_file=str(selector_file),selector_protocol_sha256=sha(selector_file),
    source_files=source_files(),visible_cores='8-15',hardware_qualification_file=str(Q/'qualification-independent-audit.json'),
    hardware_qualification_sha256=sha(Q/'qualification-independent-audit.json'),prospective_design_sha256=sha(design))
validate(p)
validate_evidence_format_control(p,control,selector)
assert audit['qualified_training_shape']=={k:p[k] for k in ('world_size','micro_batch','global_batch','width','target_slots','head_chunk','accumulation')}
assert sha(p['data_file'])==p['data_sha256'] and Counter(order_for(p))==Counter({i:4 for i in range(2048)})
assert order_for(p)==order_for(control)==order_for(selector)
assert sha(p['data_manifest_file'])==p['data_manifest_sha256']
for info in read(p['data_manifest_file'])['files'].values():assert sha(info['path'])==info['sha256']
full=read(R/'local/nvme/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1/step-0000008/COMPLETE.json')
full_bytes=sum(v['bytes'] for v in full['files'].values())
assert shutil.disk_usage(R).free>35*2**30+full_bytes+5*2**30
assert shutil.disk_usage(R/'local/nvme-cpt').free>35*2**30+2*(full_bytes+2**30)
path=C/'protocol-evidence_direct-format1-seed2026101923.json';write(path,p)
out=R/'local/nvme-cpt/runs/document-evidence-format-weight-v26-format1-TRAIN2048-seed2026101923-v1'
assert not out.exists()
job=dict(arm='evidence_direct',seed=2026101923,key='evidence_direct-format1-seed2026101923',protocol=str(path),protocol_sha256=sha(path),output=str(out),physical_devices=[2,3])
now=datetime.now(timezone.utc);deadline=now+timedelta(minutes=84)
write(C/'plan.json',dict(status='PREPARED_SINGLE_FORMAT_CONTROL_PENDING_ROOT_RELEASE',utc=now.isoformat(),source_root=str(W),source_commit=commit,jobs=[job],deadline_utc=deadline.isoformat(),
    qualification_file=p['hardware_qualification_file'],qualification_sha256=p['hardware_qualification_sha256'],prospective_design_sha256=sha(design),held_primary_manifest_sha256=p['held_primary_manifest_sha256'],
    original_control_protocol_sha256=sha(control_file),completed_selector_protocol_sha256=sha(selector_file),
    budget=dict(max_reserved_chips=2,hard_training_chip_hour_bound_including_termination_grace=3.,soft_wall_minutes=84,termination_grace_minutes=5,evaluation_chip_hour_bound_separate=.3,
        measured_full_state_bytes=full_bytes,root_free_bytes=shutil.disk_usage(R).free,NVMe_free_bytes=shutil.disk_usage(R/'local/nvme-cpt').free),
    checkpoint_policy='Full128/256 retained on NVMe. Independently audit/protect128 before retiring ONLY previous V25 EBS optimizer duplicates. Protect256 before retiring128 optimizer backup duplicates. Keep all original states and model weights.',
    evaluation='Terminal256 only: native direct and canonical closed-empty answer paths on inspected held255; ordinary256. Reuse completed selector1 and historical controls. Fresh confirmation remains separate.',
    no_automatic_retry=True,no_checkpoint_selection=True))
print(json.dumps(dict(status='PREPARED_SINGLE_FORMAT_CONTROL_PENDING_ROOT_RELEASE',job=job,deadline_utc=deadline.isoformat()),indent=2))
