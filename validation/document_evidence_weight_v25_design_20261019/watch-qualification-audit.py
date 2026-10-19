"""Wait for physical closure, then run the CPU audit exactly once; no retries."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

R=Path('/home/ubuntu/dario')
C=R/'local/diagnostics/document-evidence-weight-v25-world8-qualification-v1'
W=R/'local/worktrees/document-evidence-weight-v25'
audit=C/'independent-audit.py'
expected='3ccb603af517fd7646cda87407734a5bec2ac7b61ed0dbf91305d8037d07ef09'
assert hashlib.sha256(audit.read_bytes()).hexdigest()==expected
deadline=datetime(2026,10,19,22,5,tzinfo=timezone.utc)
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
