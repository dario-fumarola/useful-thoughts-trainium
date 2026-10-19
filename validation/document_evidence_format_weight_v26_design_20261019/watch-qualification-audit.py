"""Wait for physical closure, then run the CPU audit exactly once; no retries."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

R=Path('/home/ubuntu/dario')
C=R/'local/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1'
W=R/'local/worktrees/document-evidence-format-weight-v26'
audit=C/'independent-audit.py'
release=json.loads((C/'qualification-release.json').read_text())
assert release['status']=='RELEASED_BOUNDED_FORMAT_WEIGHT_MECHANICS_ONLY'
expected=release['audit_sha256']
assert hashlib.sha256(audit.read_bytes()).hexdigest()==expected
deadline=datetime.fromisoformat(release['deadline_utc'])+timedelta(minutes=10)
while not (C/'qualification-completion.json').exists():
    if list(C.glob('*/failure.json')):
        raise SystemExit('Original qualification failed; no audit or retry')
    if datetime.now(timezone.utc)>=deadline:
        raise SystemExit('Qualification completion not available by bounded audit deadline')
    time.sleep(5)
env=os.environ.copy()
env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),USE_TORCH_XLA='0',
           OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
with (C/'independent-audit.log').open('xb') as output:
    result=subprocess.run([sys.executable,'-u',str(audit)],cwd=W,env=env,
                          stdout=output,stderr=subprocess.STDOUT,timeout=1800)
target=C/'independent-audit-exit.json'
with target.open('x') as stream:
    json.dump(dict(exit_code=result.returncode,audit_sha256=expected,
                   utc=datetime.now(timezone.utc).isoformat()),stream,indent=2)
raise SystemExit(result.returncode)
