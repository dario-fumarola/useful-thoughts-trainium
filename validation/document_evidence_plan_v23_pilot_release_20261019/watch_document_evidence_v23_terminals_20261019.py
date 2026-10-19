"""Wait for the two physically closed pilots,then audit/protect/bind exactly once.

No evaluation or automatic training retry is performed by this watcher.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,os,subprocess,sys,time,hashlib
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-evidence-v23'
C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(C/'plan.json');release=read(C/'release.json')
assert release['plan_sha256']==sha(C/'plan.json')
assert release['pipeline_sources'][str(Path(__file__).relative_to(R))]==sha(__file__)
end=datetime.fromisoformat(plan['deadline_utc'])+timedelta(hours=1)
while not all((C/j['key']/'closure.json').exists() for j in plan['jobs']):
 if any((C/j['key']/'failure.json').exists() for j in plan['jobs']):raise RuntimeError('A pilot failed;preserve original outputs,no automaticretry/evaluation')
 if datetime.now(timezone.utc)>=end:raise TimeoutError('Pilot closure deadline exhausted')
 time.sleep(10)
while not (C/'first-full128-protection.json').exists():
 if datetime.now(timezone.utc)>=end:raise TimeoutError('First recovery protection unavailable;no binding fabricated')
 time.sleep(10)
assert read(C/'first-full128-protection.json')['status']=='PASS_FIRST128_INDEPENDENT_FULL_STATE_PROTECTED_PREVIOUS_DUPLICATES_RETIRED'
pipeline=C/'terminal-pipeline';pipeline.mkdir(exist_ok=False)
env=os.environ.copy();env.update(PYTHONPATH=str(W)+':'+str(R/'local/analysis')+':'+str(R/'local/test-deps'),USE_TORCH_XLA='0',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
for label,name,timeout in [('audit','audit_document_evidence_v23_terminals_20261019.py',3600),('protect','protect_document_evidence_v23_terminals_20261019.py',1800),('bind','bind_document_evidence_v23_terminals_20261019.py',300)]:
 source=R/'local/analysis'/name
 assert release['pipeline_sources'][str(source.relative_to(R))]==sha(source)
 with (pipeline/(label+'.log')).open('xb') as log:
  result=subprocess.run([sys.executable,'-u',str(source)],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 with (pipeline/(label+'-exit.json')).open('x') as stream:json.dump(dict(exit_code=result.returncode,utc=datetime.now(timezone.utc).isoformat()),stream)
 if result.returncode:raise RuntimeError(label+' failed;original outputs retained,no retry or binding')
with (C/'terminal-pipeline-completion.json').open('x') as stream:json.dump(dict(status='BOTH_PILOT_TERMINALS_AUDITED_PROTECTED_BOUND_NO_HELD_READOUT',utc=datetime.now(timezone.utc).isoformat()),stream)
