"""Finite,once-only complete-readout release;never inspects partial correctness."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import time

R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/real-source-eval-v17-study-v1'
plan=json.loads((C/'plan.json').read_text());W=Path(plan['source_root'])
deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()+3600
def read(p):return json.loads(Path(p).read_text())
def ready():
    if not (C/'score-queue-completion.json').exists():return False
    assert read(C/'score-queue-completion.json')['status']=='ALL_ELEVEN_FIXED_NATIVE_SCORE_JOBS_CLOSED'
    for item in plan['models']:
        if not Path(item['registry']).exists():return False
        reg=read(item['registry']);p=Path(reg['control'])/'completion.json'
        if not p.exists():return False
        assert read(p)['status']=='COMPLETE_FIXED_V17_MODEL_NOTES_AND_SCALAR'
    return True
while not ready():
    if time.time()>deadline:raise SystemExit('Finite complete-readout wait ended;no partial review or automatic retry.')
    time.sleep(20)
env=os.environ.copy();env.update(PYTHONPATH=f'{W}:{R}/local/analysis:{R}/local/test-deps',USE_TORCH_XLA='0',MPLCONFIGDIR=str(R/'local/matplotlib'))
for label,script in [('training','publish_real_source_training_v17_20261019.py'),('analyze','analyze_real_source_v17_20261019.py'),('independent','check_real_source_readout_v17_20261019.py'),
                     ('render','render_real_source_v17_20261019.py'),('examples','save_real_source_examples_v17_20261019.py')]:
    log=C/f'final-{label}.log'
    with log.open('xb') as stream:
        code=subprocess.call([sys.executable,'-u',str(R/'local/analysis'/script)],cwd=W,env=env,
            stdout=stream,stderr=subprocess.STDOUT,timeout=3600)
    with (C/f'final-{label}-exit.json').open('x') as stream:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),exit_code=code),stream,indent=2)
    if code:raise SystemExit('Complete-readout stage failed;preserve outputs,no automatic retry.')
with (C/'final-readout-completion.json').open('x') as stream:
    json.dump(dict(status='ALL_ELEVEN_COMPLETE_ANALYZED_INDEPENDENTLY_CHECKED_RENDERED_AND_EXAMPLED',utc=datetime.now(timezone.utc).isoformat()),stream,indent=2)

for label,script in [('transitions','account_real_source_scalar_errors_v17_20261019.py'),('protection','protect_real_source_readout_v17_20261019.py')]:
    with (C/f'final-{label}.log').open('xb') as stream:
        code=subprocess.call([sys.executable,'-u',str(R/'local/analysis'/script)],cwd=W,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=3600)
    with (C/f'final-{label}-exit.json').open('x') as stream:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),exit_code=code),stream,indent=2)
    if code:raise SystemExit('Complete-readout protectionstage failed;preservealloutputs,noautomaticretry.')
with (C/'final-publication-ready.json').open('x') as stream:
    json.dump(dict(status='ALL_ELEVEN_COMPLETE_RESULTS_CHECKED_AND_READBACK_PROTECTED',utc=datetime.now(timezone.utc).isoformat()),stream,indent=2)
