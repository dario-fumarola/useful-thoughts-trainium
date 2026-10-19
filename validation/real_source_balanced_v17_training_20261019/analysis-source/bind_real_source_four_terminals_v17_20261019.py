"""Freeze exactly the four audited,protected terminal models for evaluation."""
from pathlib import Path
import json
from real_source_state_audit_v17_20261019 import read,sha
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/real-source-balanced-v17-study-v1'
audit=read(C/'terminal-independent-audit.json');protection=read(C/'terminal-protection.json');plan=read(C/'plan.json')
assert audit['status']=='PASS_ALL_FOUR_TERMINAL320_TRAINING_STATES_NATIVE32_ORDER_AND_COUNTERS'
assert protection['status']=='PASS_ALL_FOUR_TERMINAL_MODELS_AND_LATEST_FULL320_PROTECTED'
assert protection['audit_sha256']==sha(C/'terminal-independent-audit.json')
assert sha(R/'local/backups/science/real-source-balanced-v17-study-v1/TERMINALS/ARCHIVE.json')==protection['EBS_archive_sha256']
models={(v['arm'],v['seed']):v for v in audit['models']}
D=C/'frozen-models';D.mkdir(exist_ok=False)
for job in plan['jobs']:
 entry=models[(job['arm'],job['seed'])]['checkpoints']['320'];key=f'{job["arm"]}-seed{job["seed"]}'
 checkpoint=Path(job['output'])/'step-0000320/model.pt'
 assert sha(checkpoint)==protection['models'][key]['sha256']==entry['model_sha256']
 assert sha(protection['models'][key]['EBS'])==entry['model_sha256']
 binding=dict(status='FROZEN_AUDITED_REAL_SOURCE_BALANCED_TERMINAL320',id=key+'-step320',arm=job['arm'],seed=job['seed'],
   checkpoint=str(checkpoint),checkpoint_sha256=entry['model_sha256'],step=320,run_id=entry['run_id'],spec_hash=entry['spec_hash'],
   training_protocol=job['protocol'],training_protocol_sha256=job['protocol_sha256'],
   audit_file=str(C/'terminal-independent-audit.json'),audit_sha256=sha(C/'terminal-independent-audit.json'),
   protection_file=str(C/'terminal-protection.json'),protection_sha256=sha(C/'terminal-protection.json'))
 with (D/(key+'.json')).open('x') as f:json.dump(binding,f,indent=2)
 print('Frozen audited protected terminal',key,flush=True)
