"""Publish and readback-protect the concrete single-control release before launch."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-format-weight-v26-control-v1'
Q=R/'local/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def syncdir(p):
    fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')

release=read(C/'release.json');plan=read(C/'plan.json');audit=read(Q/'qualification-independent-audit.json')
assert release['status']=='RELEASED_DOCUMENT_EVIDENCE_FORMAT_WEIGHT_V26_TREATMENT_ONLY'
assert release['plan_sha256']==sha(C/'plan.json') and release['qualification_sha256']==sha(Q/'qualification-independent-audit.json')
assert audit['status']=='PASS_ACTUAL_DOCUMENT_EVIDENCE_FORMAT_WEIGHT_FULL_RESTORE' and read(Q/'independent-audit-exit.json')['exit_code']==0
assert not Path(plan['jobs'][0]['output']).exists()
name='document_evidence_format_weight_v26_science_release_20261019'
A=R/'validation'/name;A.mkdir(exist_ok=False)
sources={
    'plan.json':C/'plan.json','release.json':C/'release.json','protocol.json':Path(plan['jobs'][0]['protocol']),
    'supervise-job.py':C/'supervise-job.py','CPU-helper-readiness.json':C/'CPU-helper-readiness.json','CPU-helper-final-review.json':C/'CPU-helper-final-review.json',
    'qualification-independent-audit.json':Q/'qualification-independent-audit.json','qualification-audit-exit.json':Q/'independent-audit-exit.json',
    'qualification-audit.log':Q/'independent-audit.log','qualification-completion.json':Q/'qualification-completion.json'}
for path,digest in release['pipeline_sources'].items():
    src=R/path;assert sha(src)==digest;sources[src.name]=src
for name2 in ['prepare_document_evidence_format_weight_v26_control_20261019.py','release_document_evidence_format_weight_v26_control_20261019.py',Path(__file__).name]:sources[name2]=R/'local/analysis'/name2
files={}
for name2,src in sources.items():
    dst=A/name2;shutil.copyfile(src,dst);assert sha(dst)==sha(src);files[name2]=dict(bytes=dst.stat().st_size,sha256=sha(dst))
F=R/'validation'/(name+'.json')
report=dict(experiment_id='document-evidence-format-weight-v26',status=release['status'],utc=datetime.now(timezone.utc).isoformat(),
    question='Does selector1.0 direct answering improve beyond matched format-only auxiliary1.0?',source_commit=release['source_commit'],
    qualification=dict(status=audit['status'],actual_updates=audit['actual_updates'],actual_native_checks=len(audit['actual_native_checks']),full_states=list(audit['full_states']),limits=audit['limits']),
    training=plan,release_sha256=sha(C/'release.json'),files=files,
    qualification_cost='Separate engineering allocation; actual reserved-device cost computed from recorded launch/closure times, not training-update durations.',
    evaluation='Separate release after original terminal256 is independently audited, protected and bound. Direct/canonical-empty held255 plus ordinary256 only.',
    primary='Completed selector1.0 direct minus new format1.0 direct; >=3pp and positive paired95 lower bound is developmental evidence only.',
    prior_reports_unchanged=True,not_fresh_confirmation=True)
write(F,report)
B=R/'local/backups/science/document-evidence-format-weight-v26-control-v1/RELEASE';B.mkdir(parents=True,exist_ok=False)
protected={}
for src in [F,*sorted(A.iterdir())]:
    dest=B/src.relative_to(R/'validation');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    with dest.open('rb') as f:os.fsync(f.fileno())
    assert sha(dest)==sha(src);protected[str(dest.relative_to(B))]=dict(bytes=dest.stat().st_size,sha256=sha(dest))
for p in sorted([B,*[p for p in B.rglob('*') if p.is_dir()]],key=lambda p:len(p.parts),reverse=True):syncdir(p)
write(C/'science-release-protection.json',dict(status='PASS_FROZEN_SCIENCE_RELEASE_SOURCE_AND_ACTUAL_QUALIFICATION_READBACK_FSYNC',destination=str(B),files=protected,same_instance=True))
print(json.dumps(dict(public_report=str(F),status=release['status'],protected_files=len(protected))))
