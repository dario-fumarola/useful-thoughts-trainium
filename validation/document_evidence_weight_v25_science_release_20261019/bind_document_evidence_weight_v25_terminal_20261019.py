"""Bind the original audited/protected terminal treatment, never a selected step."""
from pathlib import Path
from datetime import datetime, timezone
import json
import torch
from real_source_state_audit_v17_20261019 import read, sha

R = Path('/home/ubuntu/dario')
C = R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
plan, release = read(C/'plan.json'), read(C/'release.json')
assert release['plan_sha256'] == sha(C/'plan.json')
audit = read(C/'terminal-independent-audit.json')
protection = read(C/'terminal-protection.json')
assert audit['status'] == 'PASS_SINGLE_TERMINAL256_EVIDENCE_WEIGHT_TRAINING_STATES_NATIVE8_ORDER_COUNTERS_AND_LABELS'
assert protection['status'] == 'PASS_SINGLE_TERMINAL_MODEL_AND_LATEST_FULL256_PROTECTED'
assert audit['plan_sha256'] == sha(C/'plan.json')
assert protection['audit_sha256'] == sha(C/'terminal-independent-audit.json')
assert len(plan['jobs']) == len(audit['models']) == 1
job = plan['jobs'][0]
source = Path(job['output'])/'step-0000256'
manifest = read(source/'COMPLETE.json')
assert manifest['step'] == 256 and sha(source/'model.pt') == manifest['files']['model.pt']['sha256']
weights = torch.load(source/'model.pt', weights_only=True, mmap=True, map_location='cpu')
assert weights['step'] == 256 and weights['run_id'] == Path(job['output']).name
key = f"{job['arm']}-seed{job['seed']}"
assert sha(protection['models'][key]['EBS']) == sha(source/'model.pt')
spec = release['model_spec']
assert spec['run_id'] == weights['run_id'] and spec['training_protocol_sha256'] == job['protocol_sha256']
binding = dict(status='AUDITED_EVIDENCE_WEIGHT_V25_TERMINAL256', utc=datetime.now(timezone.utc).isoformat(),
    id=spec['id'], arm=job['arm'], seed=job['seed'], step=256,
    checkpoint=str(source/'model.pt'), checkpoint_sha256=sha(source/'model.pt'),
    run_id=weights['run_id'], spec_hash=weights['spec_hash'],
    training_protocol=job['protocol'], training_protocol_sha256=job['protocol_sha256'],
    audit_file=str(C/'terminal-independent-audit.json'), audit_sha256=sha(C/'terminal-independent-audit.json'),
    protection_file=str(C/'terminal-protection.json'), protection_sha256=sha(C/'terminal-protection.json'))
folder = C/'frozen-models'; folder.mkdir(exist_ok=False)
with (folder/(spec['id']+'.json')).open('x') as stream:
    json.dump(binding, stream, indent=2)
print(json.dumps({k: v for k, v in binding.items() if k != 'spec_hash'}, indent=2))
