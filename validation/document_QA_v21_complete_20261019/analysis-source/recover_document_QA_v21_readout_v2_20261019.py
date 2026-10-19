"""Explicit derived-check repair;no model/data/score/metric/gate changes or reruns."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-QA-v21-study-v1';D=C/'readout-pipeline-repair-v2';D.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(C/'readout-pipeline/analyze-exit.json')['exit_code']==0 and read(C/'readout-pipeline/independent-check-exit.json')['exit_code']==1
assert (C/'ALL_GENERATION_CLOSED_NO_READOUT.json').exists() and (C/'ALL_LIKELIHOOD_CLOSED_NO_READOUT.json').exists() and not (C/'final-readout-completion.json').exists()
F=R/'validation/document_QA_v21_complete_20261019.json';original_sha=sha(F)
plan=dict(status='EXPLICIT_DERIVED_CHECK_REPAIR_ORIGINAL_FAILED_ATTEMPT_RETAINED',utc=datetime.now(timezone.utc).isoformat(),report_sha256=original_sha,reason='One of60 native summary comparisons differs at<=2e-12;both original gates pass. Exact Pythonfloat equality was inappropriate for recomputed FP64 reductions. V2 uses1e-10 summarycomparison only;all original scientific/native/prefill thresholds unchanged.',scope='No model forwards,recomputed science scores,prompt/data/backend changes or skipped rows. Re-run derived independent checks,then previouslyunexecuted render/protection.',original_failed_checker_sha256=sha(R/'local/analysis/check_document_QA_v21_readout_20261019.py'),original_failed_log_sha256=sha(C/'readout-pipeline/independent-check.log'),diagnostic_sha256=sha(C/'readout-pipeline/native-comparison-discrepancies.json'))
(D/'plan.json').write_text(json.dumps(plan,indent=2))
python='/home/ubuntu/ema-trainium/.venv/bin/python';env=os.environ.copy();env.update(PYTHONPATH=str(R/'local/worktrees/document-qa-span-v21')+':'+str(R/'local/analysis')+':'+str(R/'local/test-deps'),USE_TORCH_XLA='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
for label,name in [('independent-check','check_document_QA_v21_readout_v2_20261019.py'),('render','render_document_QA_v21_20261019.py'),('protect','protect_document_QA_v21_readout_v2_20261019.py')]:
 source=R/'local/analysis'/name
 with (D/(label+'.log')).open('xb') as f:p=subprocess.run([python,'-u',str(source)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
 (D/(label+'-exit.json')).write_text(json.dumps(dict(exit_code=p.returncode,source_sha256=sha(source))))
 assert sha(F)==original_sha
 if p.returncode:raise RuntimeError('Derivedrepair '+label+' failed;preserveall,noautomaticretry')
(C/'final-readout-completion.json').write_text(json.dumps(dict(status='ALL_SEVEN_MODELS_COMPLETE_ANALYZED_INDEPENDENTLY_CHECKED_RENDERED_AND_PROTECTED',utc=datetime.now(timezone.utc).isoformat(),report=str(F),report_sha256=sha(F),derived_checker_repair_plan_sha256=sha(D/'plan.json'),original_failed_attempt_preserved=str(C/'readout-pipeline')),indent=2))
