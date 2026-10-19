"""Explicit IO-only second scoring attempt;reuse exact closed original inputs."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os,subprocess
from train.document_evidence_plan_likelihood import ROOT,sources
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-evidence-span-v23-io-v2';assert ROOT==W
C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1';D=C/'likelihood-io-repair-v2'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for b in iter(lambda:stream.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
assert read(C/'likelihood-v1-failure-preserved.json')['status']=='PRESERVED_ORIGINAL_CALIBRATION_WRITE_FAILURE_ZERO_HELD_ROWS_PHYSICALLY_CLOSED'
D.mkdir(exist_ok=False)
registry=[];changed=sources()
for group in read(C/'likelihood-registry.json')['groups']:
 old=Path(group['output']);p=read(old/'protocol.json')
 assert [name for name,value in p['source_files'].items() if changed[name]!=value]==['train/document_evidence_plan_likelihood.py']
 output=old.with_name(old.name.replace('-likelihood-v1','-likelihood-io-v2'));output.mkdir(exist_ok=False)
 for name,value in p['files'].items():
  assert sha(old/name)==value;os.link(old/name,output/name);assert sha(output/name)==value
 original=dict(p)
 p.update(created_utc=datetime.now(timezone.utc).isoformat(),source_files=changed,
  execution_attempt='EXPLICIT_ROOT_REVIEWED_IO_REPAIR_V2',parent_failed_protocol_sha256=sha(old/'protocol.json'),
  repair='Immutableper-panel calibrationwitnesses plus atomicreplace ofthisattempt cumulative summary. Numericalkernel,inputs,CPUreferences,models,shapes,bounds andsciencegates unchanged.')
 assert p['model']==original['model'] and p['files']==original['files'] and p['inventories']==original['inventories']
 with (output/'protocol.json').open('x') as stream:json.dump(p,stream,indent=2)
 registry.append(dict(model=group['model'],arm=group['arm'],output=str(output),control=str(C/('LL-IO-V2-'+group['model'])),physical=group['physical'],original=group['original'],protocol_sha256=sha(output/'protocol.json'),parent_failed_output=group['output'],parent_failed_protocol_sha256=sha(old/'protocol.json')))
controller=R/'local/analysis/supervise_document_evidence_v23_likelihood_io_v2_20261019.py'
s=(R/'local/analysis/supervise_document_evidence_v23_likelihood_group_20261019.py').read_text()
s=s.replace('local/worktrees/document-evidence-span-v23\'','local/worktrees/document-evidence-span-v23-io-v2\'')
s=s.replace('port=59000+physical*100','port=61000+physical*100')
needle="assert p['release_sha256']==h.sha(p['release_file'])"
extra="repair=h.read(R/'local/diagnostics/document-evidence-plan-v23-pilot-v1/likelihood-io-repair-v2/release.json');assert repair['status']=='RELEASED_EXPLICIT_IO_ONLY_LIKELIHOOD_REPAIR_V2' and repair['controller_sha256']==h.sha(__file__) and repair['protocols'][p['model']['id']]==h.sha(O/'protocol.json')\n"
assert needle in s;s=s.replace(needle,extra+needle)
ast.parse(s);controller.write_text(s)
with (C/'likelihood-registry-v2.json').open('x') as stream:json.dump(dict(status='EXPLICIT_IO_ONLY_REPAIR_EXACT_ORIGINAL_INPUTS_NO_HELD_ROWS_REVIEWED',groups=registry),stream,indent=2)
plan=dict(status='PREPARED_IO_ONLY_SCORING_REPAIR_V2_PENDING_ROOT_SERIAL_REVIEW',utc=datetime.now(timezone.utc).isoformat(),
 source_root=str(W),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),
 original_failure_receipt_sha256=sha(C/'likelihood-v1-failure-preserved.json'),original_registry_sha256=sha(C/'likelihood-registry.json'),
 new_registry_sha256=sha(C/'likelihood-registry-v2.json'),controller_sha256=sha(controller),
 protocols={g['model']:g['protocol_sha256'] for g in registry},groups=registry,
 only_changed_numerical_source_file='NONE;scoringkernel/loaders unchanged. WorkerIOonly.',
 preservation='Allv1failed outputs/logs/source retained andEBSreadbackverified;all8originalsessions absent,physical6/7 empty. Noheldscorefile created.',
 reuse='Identical hardlinkednativecontext/targetfiles andown-modelCPUreferencefiles,bytehash equality. Both modelcheckpoints andall1530nativegenerationobservations unchanged.',
 bounds='Original.001max-token andabsmeanspanTRAINcalibration;all255perQAcondition/ordinary256,FP32batch4,width3584/672,target32,fourshards,samehardware6/7. No scores/gates/threshold/data/prompt changes.',
 deadline_utc='2026-10-20T06:00:00+00:00',no_automatic_retry=True)
with (D/'plan.json').open('x') as stream:json.dump(plan,stream,indent=2)
print(json.dumps(dict(status=plan['status'],source_commit=plan['source_commit'],changed_source=['calibration witness IO only']),indent=2))
