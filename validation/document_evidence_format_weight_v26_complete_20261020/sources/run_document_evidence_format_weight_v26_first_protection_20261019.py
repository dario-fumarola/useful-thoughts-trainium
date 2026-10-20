"""Record the one-shot independent full128 protection outcome."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import os
import subprocess
import sys

R = Path('/home/ubuntu/dario')
C = R/'local/diagnostics/document-evidence-format-weight-v26-control-v1'
W = R/'local/worktrees/document-evidence-format-weight-v26'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan, release = read(C/'plan.json'), read(C/'release.json')
assert release['plan_sha256'] == sha(C/'plan.json')
name = 'local/analysis/protect_document_evidence_format_weight_v26_first_full128_20261019.py'
assert sha(R/name) == release['pipeline_sources'][name]
env = os.environ.copy()
env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps')+':'+str(R/'local/analysis'),
    USE_TORCH_XLA='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
seconds = int((datetime.fromisoformat(plan['deadline_utc'])-datetime.now(timezone.utc)).total_seconds())+15*60
assert seconds > 0
with (C/'first-full128-protection.log').open('xb') as output:
    result = subprocess.run([sys.executable, '-u', str(R/name)], cwd=W, env=env,
        stdout=output, stderr=subprocess.STDOUT, timeout=seconds)
with (C/'first-full128-protection-exit.json').open('x') as stream:
    json.dump(dict(exit_code=result.returncode, source_sha256=sha(R/name),
        utc=datetime.now(timezone.utc).isoformat()), stream, indent=2)
raise SystemExit(result.returncode)
