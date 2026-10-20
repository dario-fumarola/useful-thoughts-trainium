"""Consolidate the declared control only after both independent readout checks."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-format-weight-v26-control-v1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
h=read(C/'held-result.json');o=read(C/'ORDINARY/result.json')
assert h['status']=='COMPLETE_DEVELOPMENTAL_FORMAT_WEIGHT_V26_ANSWERS' and o['status']=='COMPLETE_FIXED_ORDINARY256_COST'
checks={name:read(C/path) for name,path in [('held','held-independent-check.json'),('ordinary','ORDINARY/independent-check.json')]}
assert all(v['status'].startswith('PASS_') for v in checks.values())
assert checks['held']['result_sha256']==sha(C/'held-result.json') and checks['ordinary']['result_sha256']==sha(C/'ORDINARY/result.json')
assert checks['held']['new_attempts']==510 and checks['ordinary']['items']==256
assert sha(R/'validation/document_evidence_weight_v25_complete_20261019.json')=='ae7d3eacd701acb20eb969721dea531782c18f73d87bd79dd5bfacc8d6d86837'
assert sha(R/'validation/document_evidence_plan_v23_pilot_complete_20261019.json')=='ca7cc615495d0cdade77ef70cce0f19055e3456b1c79c460a7f84008dec809b3'
terminal=read(C/'frozen-models/evidence_direct-format1-seed2026101923.json')
assert terminal['status']=='AUDITED_EVIDENCE_FORMAT_WEIGHT_V26_TERMINAL256' and terminal['step']==256
assert terminal['audit_sha256']==sha(C/'terminal-independent-audit.json') and terminal['protection_sha256']==sha(C/'terminal-protection.json')
assert read(C/'terminal-pipeline-completion.json')['status']=='TERMINAL_AUDITED_PROTECTED_BOUND_NO_EVALUATION_LAUNCHED'
assert (C/'fixed-eight-examples.json').exists() and len(list((C/'figures').glob('*.svg')))==3
cost=read(C/'actual-cost.json');assert cost['status']=='COMPLETE_ACTUAL_RESERVED_CHIP_TIME_ALL_STAGES'
result=dict(status='COMPLETE_FIXED_V26_READOUT_PENDING_ORIGINAL_ARCHIVE',utc=datetime.now(timezone.utc).isoformat(),experiment_id='document-evidence-format-weight-v26',
    question='Does evidence-content supervision improve direct answering beyond coefficient-matched empty-format supervision?',
    new_control=dict(fresh_B0=True,model='Qwen3-1.7B-Base',arm='evidence_direct',selection_loss_weight=1.,source_loss_weight=.5,answer_loss_weight=.5,questions=2048,passes=4,updates=256,data_order_seed=2026101923),
    comparison='Completed original V25 selector1.0; historical V23 format0.1 retained. Same data/order/exposures/source-answer coefficients/numerical kernels; auxiliary content differs.',
    terminal_model=terminal,training_integrity=read(C/'terminal-independent-audit.json'),developmental_held_answers=h,ordinary_cost=o,
    actual_new_generation_attempts=510,ordinary_likelihood_rows=256,independent_checks=checks,actual_cost=cost,
    primary_developmental_decision='PASS_DEVELOPMENTAL_CRITERION' if h['primary_developmental_criterion_pass'] else 'CRITERION_NOT_MET',
    operational_controls=h['operational_controls'],key_input_hashes={name:sha(C/name) for name in ('release.json','plan.json','evaluation-release.json')},
    readout_files={name:sha(C/name) for name in ('held-result.json','held-scored.json','ORDINARY/result.json','actual-cost.json','fixed-eight-examples.json')},
    boundaries=['Previously inspected held255 and one data-order seed. Passing the diagnostic criterion is not fresh confirmation; failing it is not proof of zero effect.',
        'Primary direct-answer transfer and added inference-time ID generation are different hypotheses.',
        'Equal scalar coefficients are not matched gradient magnitudes, label counts or information. All20 passages remain visible.',
        'Support annotations can be semantically incomplete. Neither annotation fitting nor oracle utility establishes autonomous computation.',
        'No general-language pretraining gain, pretraining novelty, calibration or ordinary-CoT superiority claim.',
        'All old scores/gates remain unchanged. No automatic scale-up, sweep or RL follows.'])
with (C/'complete-readout-before-protection.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(status=result['status'],primary=result['primary_developmental_decision'])))
