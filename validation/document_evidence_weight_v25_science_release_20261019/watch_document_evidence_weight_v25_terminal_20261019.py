"""After physical closure, audit/protect/bind once. No model evaluation or retry."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import hashlib
import os
import subprocess
import sys
import time

R = Path('/home/ubuntu/dario')
C = R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
W = R/'local/worktrees/document-evidence-weight-v25'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan, release = read(C/'plan.json'), read(C/'release.json')
assert release['plan_sha256'] == sha(C/'plan.json') and len(plan['jobs']) == 1
job = plan['jobs'][0]
deadline = datetime.fromisoformat(plan['deadline_utc'])+timedelta(minutes=20)
marker = C/(job['key']+'-completion.json')
while not marker.exists():
    if (C/job['key']/'failure.json').exists():
        raise SystemExit('Original treatment failed; preserve evidence, no automatic retry')
    if datetime.now(timezone.utc) >= deadline:
        raise SystemExit('No physical closure within the bounded post-job wait')
    time.sleep(10)
assert read(marker)['status'] == 'SINGLE_FROZEN_JOB_CLOSED_PENDING_INDEPENDENT_AUDIT'
env = os.environ.copy()
env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps')+':'+str(R/'local/analysis'),
    USE_TORCH_XLA='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
scripts = [('terminal-audit', 'audit_document_evidence_weight_v25_terminals_20261019.py'),
           ('terminal-protection', 'protect_document_evidence_weight_v25_terminals_20261019.py'),
           ('terminal-binding', 'bind_document_evidence_weight_v25_terminal_20261019.py')]
for label, name in scripts:
    if label == 'terminal-protection':
        while not (C/'first-full128-protection.json').exists():
            exit_file = C/'first-full128-protection-exit.json'
            if exit_file.exists() and read(exit_file)['exit_code'] != 0:
                raise SystemExit('First full128 protection failed; do not retire any further duplicate')
            if datetime.now(timezone.utc) >= deadline:
                raise SystemExit('Full128 protection not complete by bounded post-job wait')
            time.sleep(10)
    source = R/'local/analysis'/name
    for path, digest in release['pipeline_sources'].items():
        assert sha(R/path) == digest
    with (C/(label+'.log')).open('xb') as output:
        result = subprocess.run([sys.executable, '-u', str(source)], cwd=W,
            env=env, stdout=output, stderr=subprocess.STDOUT, timeout=1800)
    with (C/(label+'-exit.json')).open('x') as stream:
        json.dump(dict(exit_code=result.returncode, source_sha256=sha(source),
            utc=datetime.now(timezone.utc).isoformat()), stream, indent=2)
    if result.returncode:
        raise SystemExit('Original terminal pipeline failed; preserve outputs, no retry')
with (C/'terminal-pipeline-completion.json').open('x') as stream:
    json.dump(dict(status='TERMINAL_AUDITED_PROTECTED_BOUND_NO_EVALUATION_LAUNCHED',
        utc=datetime.now(timezone.utc).isoformat()), stream, indent=2)
print('TERMINAL_AUDITED_PROTECTED_BOUND_NO_EVALUATION_LAUNCHED', flush=True)
