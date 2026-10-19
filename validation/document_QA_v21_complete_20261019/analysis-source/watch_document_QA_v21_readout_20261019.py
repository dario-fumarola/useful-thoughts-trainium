"""Once-only readout after complete fixed generation/likelihood closure."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-QA-v21-study-v1';D=C/'readout-pipeline';D.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
end=datetime.fromisoformat('2026-10-20T13:00:00+00:00')
while not all((C/n).exists() for n in ['ALL_GENERATION_CLOSED_NO_READOUT.json','ALL_LIKELIHOOD_CLOSED_NO_READOUT.json']):
 if datetime.now(timezone.utc)>=end:raise TimeoutError('No complete fixedevaluation beforedeadline;no retry/partialreadout')
 time.sleep(10)
python='/home/ubuntu/ema-trainium/.venv/bin/python';env=os.environ.copy();env.update(PYTHONPATH=str(R/'local/worktrees/document-qa-span-v21')+':'+str(R/'local/analysis')+':'+str(R/'local/test-deps'),USE_TORCH_XLA='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
for label,name in [('analyze','analyze_document_QA_v21_20261019.py'),('independent-check','check_document_QA_v21_readout_20261019.py'),('render','render_document_QA_v21_20261019.py'),('protect','protect_document_QA_v21_readout_20261019.py')]:
 source=R/'local/analysis'/name
 with (D/(label+'.log')).open('xb') as f:p=subprocess.run([python,'-u',str(source)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 (D/(label+'-exit.json')).write_text(json.dumps(dict(exit_code=p.returncode,source_sha256=sha(source))))
 if p.returncode:raise RuntimeError(label+' failed;no automaticretry or publication')
(C/'final-readout-completion.json').write_text(json.dumps(dict(status='ALL_SEVEN_MODELS_COMPLETE_ANALYZED_INDEPENDENTLY_CHECKED_RENDERED_AND_PROTECTED',utc=datetime.now(timezone.utc).isoformat(),report=str(R/'validation/document_QA_v21_complete_20261019.json'),report_sha256=sha(R/'validation/document_QA_v21_complete_20261019.json')),indent=2))
