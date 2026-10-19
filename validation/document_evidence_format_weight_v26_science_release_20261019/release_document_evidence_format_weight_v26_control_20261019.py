"""Release one actual-qualified fresh format-only model; evaluation remains separate."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-format-weight-v26-control-v1'
Q=R/'local/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()

plan=read(C/'plan.json');assert plan['status']=='PREPARED_SINGLE_FORMAT_CONTROL_PENDING_ROOT_RELEASE' and len(plan['jobs'])==1
job=plan['jobs'][0];assert job['arm']=='evidence_direct' and job['seed']==2026101923 and not Path(job['output']).exists()
p=read(job['protocol']);assert sha(job['protocol'])==job['protocol_sha256']
q=read(plan['qualification_file']);assert q['status']=='PASS_ACTUAL_DOCUMENT_EVIDENCE_FORMAT_WEIGHT_FULL_RESTORE'
assert q['actual_updates']=={'evidence_direct':8} and len(q['actual_native_checks'])==16 and read(Q/'independent-audit-exit.json')['exit_code']==0
assert plan['qualification_sha256']==sha(plan['qualification_file']) and q['source_files']==p['source_files']
assert p['experiment_id']=='document-evidence-format-weight-v26' and p['selection_loss_weight']==1.
assert p['source_loss_weight']==p['answer_loss_weight']==.5 and p['items']==2048 and p['steps']==256 and p['save_every']==128
assert sha(p['control_protocol_file'])==p['control_protocol_sha256']==plan['original_control_protocol_sha256']
assert sha(p['selector_protocol_file'])==p['selector_protocol_sha256']==plan['completed_selector_protocol_sha256']
bindings={
    'format0.1':R/'local/diagnostics/document-evidence-plan-v23-pilot-v1/frozen-models/evidence_direct-seed2026101923.json',
    'selector1':R/'local/diagnostics/document-evidence-weight-v25-ablation-v1/frozen-models/evidence_select-weight1-seed2026101923.json'}
for name,path in bindings.items():
    b=read(path);assert b['step']==256
    assert b['training_protocol_sha256']==p['control_protocol_sha256' if name=='format0.1' else 'selector_protocol_sha256']
    assert sha(b['checkpoint'])==b['checkpoint_sha256'] and sha(b['audit_file'])==b['audit_sha256']
assert sha(R/'validation/document_evidence_format_weight_v26_design_20261019.json')==plan['prospective_design_sha256']
assert (datetime.fromisoformat(plan['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()>=80*60
full=plan['budget']['measured_full_state_bytes']
assert shutil.disk_usage(R).free>35*2**30+full+5*2**30
assert shutil.disk_usage(R/'local/nvme-cpt').free>35*2**30+2*(full+2**30)
prefix='document_evidence_format_weight_v26'
names=[prefix+'_state_audit_20261019.py','audit_'+prefix+'_terminals_20261019.py','protect_'+prefix+'_first_full128_20261019.py','protect_'+prefix+'_terminals_20261019.py',
    'bind_'+prefix+'_terminal_20261019.py','watch_'+prefix+'_terminal_20261019.py','run_'+prefix+'_first_protection_20261019.py',
    prefix+'_held_20261019.py','analyze_'+prefix+'_held_20261019.py','check_'+prefix+'_held_20261019.py','supervise_'+prefix+'_held_20261019.py',
    prefix+'_ordinary_20261019.py','check_'+prefix+'_ordinary_20261019.py','supervise_'+prefix+'_ordinary_20261019.py',
    'release_'+prefix+'_evaluation_20261019.py','document_evidence_v23_likelihood_contexts.py','real_source_state_audit_v17_20261019.py']
release=dict(status='RELEASED_DOCUMENT_EVIDENCE_FORMAT_WEIGHT_V26_TREATMENT_ONLY',utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(C/'plan.json'),
    supervisor_sha256=sha(C/'supervise-job.py'),helper_sha256=sha(Q/'qualify.py'),qualification_file=plan['qualification_file'],qualification_sha256=plan['qualification_sha256'],source_commit=plan['source_commit'],
    comparison_bindings={name:dict(file=str(path),sha256=sha(path)) for name,path in bindings.items()},prospective_design_sha256=plan['prospective_design_sha256'],
    pipeline_sources={str((R/'local/analysis'/n).relative_to(R)):sha(R/'local/analysis'/n) for n in names},data_manifest_sha256=p['data_manifest_sha256'],
    model_spec=dict(id='evidence_direct-format1-seed2026101923',arm='evidence_direct',seed=job['seed'],step=256,run_id=Path(job['output']).name,training_protocol_sha256=job['protocol_sha256']),
    decision='Qualified actual8 updates/full restore4; release one fresh empty-format1.0 model. No new data, second seed, size change, RL, retry or sweep.',budget=plan['budget'],deadline_utc=plan['deadline_utc'],
    evaluation='Only after terminal audit/protection and separate root release: fixed direct/canonical-empty held255 and ordinary256. No fresh confirmation claim.',
    limitations=['Same scalar weights do not match gradient magnitudes or semantic information.','One data-order seed,2048questions,inspected held255; imperfect author annotations.','All historical scores/gates unchanged.'])
with (C/'release.json').open('x') as f:json.dump(release,f,indent=2)
print(json.dumps(dict(status=release['status'],job=job['key'],deadline_utc=release['deadline_utc']),indent=2))
