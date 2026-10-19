"""Run one fixed complete readout after every native/likelihood closure.

Root still reviews the figure/result and any conditional replication. No
result-driven rerun,automatic scientific release or Git push occurs here.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(C/'readout-execution-plan.json')
assert p['source_files'][str(Path(__file__).relative_to(R))]==sha(__file__)
D=C/'readout-pipeline';D.mkdir(exist_ok=False)
end=datetime.fromisoformat(p['deadline_utc'])
while not all((C/name).exists() for name in ['ALL_GENERATION_CLOSED_NO_READOUT.json','ALL_LIKELIHOOD_CLOSED_NO_READOUT.json']):
 if datetime.now(timezone.utc)>=end:raise TimeoutError('Complete readout unavailable;no automaticretry')
 for folder in ['generation-pipeline','likelihood-pipeline']:
  for child in (C/folder).glob('*-exit.json'):
   if read(child)['exit_code']:raise RuntimeError('Fixed evaluation failed;no complete readout fabricated')
 time.sleep(10)
env=os.environ.copy();env.update(PYTHONPATH=str(R)+':'+str(R/'local/test-deps'),USE_TORCH_XLA='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
for label,name in [('analyze','analyze_document_evidence_v23_20261019.py'),('check','check_document_evidence_v23_readout_20261019.py'),('render','render_document_evidence_v23_20261019.py'),('protect','protect_document_evidence_v23_readout_20261019.py')]:
 source=R/'local/analysis'/name
 assert p['source_files'][str(source.relative_to(R))]==sha(source)
 assert all(sha(R/name)==value for name,value in p['scoring_source_files'].items())
 with (D/(label+'.log')).open('xb') as stream:result=subprocess.run([sys.executable,'-u',str(source)],cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
 with (D/(label+'-exit.json')).open('x') as stream:json.dump(dict(exit_code=result.returncode,source_sha256=sha(source),utc=datetime.now(timezone.utc).isoformat()),stream)
 if result.returncode:raise RuntimeError(label+' failed;preserve original evidence,no automaticretry')
with (C/'readout-pipeline-completion.json').open('x') as stream:json.dump(dict(status='PILOT_ANALYZED_CHECKED_RENDERED_PROTECTED_READY_FOR_ROOT_REVIEW',utc=datetime.now(timezone.utc).isoformat()),stream,indent=2)
