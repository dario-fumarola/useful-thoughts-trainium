"""Freeze exactly the two audited,protected terminal models for evaluation."""
from pathlib import Path
import json
from real_source_state_audit_v17_20261019 import read,sha
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
audit=read(C/'terminal-independent-audit.json');protection=read(C/'terminal-protection.json');plan=read(C/'plan.json')
assert audit['status']=='PASS_BOTH_TERMINAL256_EVIDENCE_TRAINING_STATES_NATIVE16_ORDER_COUNTERS_AND_LABELS'
assert protection['status']=='PASS_BOTH_TERMINAL_MODELS_AND_LATEST_FULL256_PROTECTED'
assert protection['audit_sha256']==sha(C/'terminal-independent-audit.json')
assert sha(R/'local/backups/science/document-evidence-plan-v23-pilot-v1/TERMINALS/ARCHIVE.json')==protection['EBS_archive_sha256']
models={(v['arm'],v['seed']):v for v in audit['models']}
D=C/'frozen-models';D.mkdir(exist_ok=False)
for job in plan['jobs']:
 entry=models[(job['arm'],job['seed'])]['checkpoints']['256'];key=f'{job["arm"]}-seed{job["seed"]}'
 checkpoint=Path(job['output'])/'step-0000256/model.pt'
 assert sha(checkpoint)==protection['models'][key]['sha256']==entry['model_sha256']
 assert sha(protection['models'][key]['EBS'])==entry['model_sha256']
 binding=dict(status='AUDITED_EVIDENCE_PLAN_V23_TERMINAL256',id=key,arm=job['arm'],seed=job['seed'],
   checkpoint=str(checkpoint),checkpoint_sha256=entry['model_sha256'],step=256,run_id=entry['run_id'],spec_hash=entry['spec_hash'],
   training_protocol=job['protocol'],training_protocol_sha256=job['protocol_sha256'],
   audit_file=str(C/'terminal-independent-audit.json'),audit_sha256=sha(C/'terminal-independent-audit.json'),
   protection_file=str(C/'terminal-protection.json'),protection_sha256=sha(C/'terminal-protection.json'))
 with (D/(key+'.json')).open('x') as f:json.dump(binding,f,indent=2)
 print('Frozen audited protected terminal',key,flush=True)
