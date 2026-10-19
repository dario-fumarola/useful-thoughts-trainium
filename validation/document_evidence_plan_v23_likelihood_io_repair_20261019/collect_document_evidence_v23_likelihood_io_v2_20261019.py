"""Observe the explicitly launched IO-repair groups;never launch/retry a job."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1';D=C/'likelihood-io-repair-v2'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(D/'release.json');assert p['status']=='RELEASED_EXPLICIT_IO_ONLY_LIKELIHOOD_REPAIR_V2'
groups=read(C/'likelihood-registry-v2.json')['groups'];end=datetime.fromisoformat('2026-10-20T06:00:00+00:00')
while not all((Path(g['control'])/'closure.json').exists() for g in groups):
 if any((Path(g['control'])/'STOP_REASON.json').exists() for g in groups):raise RuntimeError('ExplicitIOrepair failed;preserve,noautomaticretry')
 if datetime.now(timezone.utc)>=end:raise TimeoutError('IOrepair deadline exhausted')
 time.sleep(5)
for g in groups:
 q=read(Path(g['control'])/'closure.json')
 assert q['status']=='ALL_FROZEN_LIKELIHOOD_SHARDS_PHYSICALLY_CLOSED_NO_READOUT' and q['actual_rows']==1531 and q['assigned_physical_device_empty']
with (C/'ALL_LIKELIHOOD_CLOSED_NO_READOUT.json').open('x') as stream:json.dump(dict(status='ALL3062_ORIGINAL_POINTER_REPLAY_AND_ORDINARY_ROWS_CLOSED_NO_READOUT',utc=datetime.now(timezone.utc).isoformat(),groups=groups,execution_attempt='EXPLICIT_IO_ONLY_REPAIR_V2',original_failure_preserved_sha256=sha(C/'likelihood-v1-failure-preserved.json'),repair_release_sha256=sha(D/'release.json')),stream,indent=2)
print('ALL3062_ORIGINAL_ROWS_CLOSED_FROM_EXPLICIT_IO_REPAIR_V2')
