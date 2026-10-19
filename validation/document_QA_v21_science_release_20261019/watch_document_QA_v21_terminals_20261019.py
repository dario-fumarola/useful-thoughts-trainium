"""Finish full training audit,full-first protection,and exact bindings once;no retry."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,time
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-qa-v21';C=R/'local/diagnostics/document-QA-v21-study-v1';plan=json.loads((C/'plan.json').read_text());D=C/'terminal-pipeline';D.mkdir(exist_ok=False)
while not all((C/j['key']/'closure.json').exists() for j in plan['jobs']):
 failed=[j['key'] for j in plan['jobs'] if (C/j['key']/'failure.json').exists()]
 if failed:raise RuntimeError('Preserve failed science jobs,no automatic retry: '+str(failed))
 if datetime.now(timezone.utc)>=datetime.fromisoformat(plan['deadline_utc']):raise TimeoutError('Declared deadline,no automatic retry')
 time.sleep(10)
while not (C/'first-full128-protection.json').exists():
 if datetime.now(timezone.utc)>=datetime.fromisoformat(plan['deadline_utc']):raise TimeoutError('128 protection not closed;keep last full safe')
 time.sleep(10)
env=os.environ.copy();env.update(USE_TORCH_XLA='0',PYTHONPATH=str(W)+':'+str(R/'local/analysis')+':'+str(R/'local/test-deps'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
for stage,name in [('audit','audit_document_QA_v21_terminals_20261019.py'),('protection','protect_document_QA_v21_terminals_20261019.py'),('bindings','bind_document_QA_v21_terminals_20261019.py')]:
 with (D/(stage+'.log')).open('xb') as f:p=subprocess.run(['/home/ubuntu/ema-trainium/.venv/bin/python','-u',str(R/'local/analysis'/name)],cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT)
 (D/(stage+'-exit.json')).write_text(json.dumps(dict(exit_code=p.returncode,utc=datetime.now(timezone.utc).isoformat()),indent=2))
 if p.returncode:raise SystemExit('Terminal '+stage+' failed;no automatic retry')
(C/'terminal-pipeline-completion.json').write_text(json.dumps(dict(status='ALL_SIX_TERMINAL256_MODELS_AUDITED_PROTECTED_AND_BOUND',utc=datetime.now(timezone.utc).isoformat()),indent=2))
