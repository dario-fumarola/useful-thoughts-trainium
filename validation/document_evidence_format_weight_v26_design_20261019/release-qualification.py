"""Release only the bounded coefficient-matched format qualification."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, importlib.util, json, shutil
from train.partial_state_driver import ROOT, source_files, validate, validate_evidence_format_control
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-evidence-format-weight-v26'
C=R/'local/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1'
assert ROOT==W
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
p=read(C/'protocol.json');validate(p)
validate_evidence_format_control(p,read(p['control_protocol_file']),read(p['selector_protocol_file']))
assert p['source_files']==source_files() and sha(p['data_file'])==p['data_sha256']
assert '1823 passed' in (W/'local/checks/tests.kIGiyp/tests.txt').read_text()
relocation=R/'local/diagnostics/document-evidence-format-weight-v26-CPU-v1/inactive-backup-relocation.json'
assert read(relocation)['status']=='COMPLETE_ONLY_TWO_INACTIVE_EBS_DUPLICATES_RELOCATED_ALL_ORIGINALS_RETAINED'
assert (R/'local/nvme').is_mount() and shutil.disk_usage(R/'local/nvme').free>35*2**30+4*24091475979+2**30
assert shutil.disk_usage(R).free>35*2**30+24091475979+4*2**30
assert shutil.disk_usage(R/'local/nvme-cpt').free>35*2**30+2*24091475979
out=R/'local/nvme/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1';assert not out.exists()
spec=importlib.util.spec_from_file_location('format_qualification_helpers',C/'qualify.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
top=h.topology();assert all(not z['neuron_processes'] for z in top if z['neuron_device'] in (12,13)) and h.available()>768*2**30
science=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1';protected=read(science/'terminal-protection.json')
assert protected['status']=='PASS_SINGLE_TERMINAL_MODEL_AND_LATEST_FULL256_PROTECTED'
assert sha(Path(protected['latest_full']['EBS'])/'COMPLETE.json')==protected['latest_full']['COMPLETE_sha256']
F=R/'validation/document_evidence_format_weight_v26_design_20261019.json'
watcher=R/'local/analysis/watch_document_evidence_format_v26_qualification_audit_20261019.py'
now=datetime.now(timezone.utc)
release=dict(status='RELEASED_BOUNDED_FORMAT_WEIGHT_MECHANICS_ONLY',utc=now.isoformat(),source_root=str(W),source_commit='29606b0',
 protocol_sha256=sha(C/'protocol.json'),supervisor_sha256=sha(C/'qualify.py'),audit_sha256=sha(C/'independent-audit.py'),
 watcher_sha256=sha(watcher),release_source_sha256=sha(__file__),prospective_design_file=str(F),prospective_design_sha256=sha(F),
 CPU_checks='1823 pass; smoke/compile pass',numerical_equivalence_sha256=sha(C/'numerical-source-equivalence.json'),
 inactive_backup_relocation_file=str(relocation),inactive_backup_relocation_sha256=sha(relocation),physical_devices=[12,13],logical_cores='48-55',
 deadline_utc=(now+timedelta(minutes=80)).isoformat(),maximum_reserved_chip_hours=3.,output=str(out),original_full_states=[2,4,6,8],
 decision='Only actual8-update/fresh-process restore mechanics are released. Fresh B0 full256 study/evaluation not released.',
 no_automatic_retry=True,not_capability_evidence=True,
 storage_free_bytes={str(d):shutil.disk_usage(d).free for d in [R,R/'local/nvme',R/'local/nvme-cpt',R/'local/nvme-study']})
with (C/'qualification-release.json').open('x') as f:json.dump(release,f,indent=2)
print(json.dumps({k:release[k] for k in ['status','source_commit','deadline_utc','maximum_reserved_chip_hours','output']}))
