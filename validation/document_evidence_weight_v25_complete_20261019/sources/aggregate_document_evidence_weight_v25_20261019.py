"""Consolidate fixed results only after original terminal and all readout checks."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
fit=read(C/'TRAIN-fit/result.json');held=read(C/'held-native-and-replay-result.json');ordinary=read(C/'ORDINARY/result.json')
assert fit['status']=='COMPLETE_WEIGHT_V25_FIXED_TRAIN_FIT'
assert held['status']=='COMPLETE_DEVELOPMENTAL_WEIGHT_V25_NATIVE_AND_CLOSED_FIELD_ANSWERS'
assert ordinary['status']=='COMPLETE_FIXED_ORDINARY256_COST'
checks={name:read(C/path) for name,path in [('TRAIN','TRAIN-fit/independent-check.json'),('held','held-independent-check.json'),('ordinary','ORDINARY/independent-check.json')]}
assert all(v['status'].startswith('PASS_') for v in checks.values())
assert checks['TRAIN']['report_sha256']==sha(C/'TRAIN-fit/result.json')
assert checks['held']['result_sha256']==sha(C/'held-native-and-replay-result.json')
assert checks['ordinary']['result_sha256']==sha(C/'ORDINARY/result.json')
assert sha(R/'validation/document_evidence_plan_v23_pilot_complete_20261019.json')=='ca7cc615495d0cdade77ef70cce0f19055e3456b1c79c460a7f84008dec809b3'
assert sha(R/'validation/document_selector_fit_v24_complete_20261019.json')=='b98ca968835f5f93232ed018e336dfb42f338146cef34fc05de450177212bcb7'
terminal=read(C/'frozen-models/evidence_select-weight1-seed2026101923.json')
assert terminal['status']=='AUDITED_EVIDENCE_WEIGHT_V25_TERMINAL256' and terminal['step']==256
assert terminal['audit_sha256']==sha(C/'terminal-independent-audit.json') and terminal['protection_sha256']==sha(C/'terminal-protection.json')
assert read(C/'terminal-pipeline-completion.json')['status']=='TERMINAL_AUDITED_PROTECTED_BOUND_NO_EVALUATION_LAUNCHED'
all_attempts=checks['TRAIN']['actual_attempts']+checks['held']['attempts'];assert all_attempts==1531
assert checks['ordinary']['items']==256
assert (C/'fixed-eight-examples.json').exists() and len(list((C/'figures').glob('*.svg')))==3
cost=read(C/'actual-cost.json');assert cost['status']=='COMPLETE_ACTUAL_RESERVED_CHIP_TIME_ALL_STAGES'
result=dict(status='COMPLETE_FIXED_V25_READOUT_PENDING_ORIGINAL_ARCHIVE',utc=datetime.now(timezone.utc).isoformat(),
 experiment_id='document-evidence-weight-v25',question='Does increasing selector supervision alone repair TRAIN set fitting and improve developmental answering?',
 treatment=dict(fresh_B0=True,model='Qwen3-1.7B-Base',selection_loss_weight=1.,source_loss_weight=.5,answer_loss_weight=.5,questions=2048,passes=4,updates=256,data_order_seed=2026101923),
 control='Completed original selection0.1 model. Same documents/order/exposures and source/final-answer coefficients. This is one coefficient ablation, not a new architecture or two-seed study.',
 terminal_model=terminal,training_integrity=read(C/'terminal-independent-audit.json'),
 primary_TRAIN_fitting=fit,developmental_held_answers=held,ordinary_cost=ordinary,
 descriptive_selection=read(C/'held-selection-descriptive.json'),
 actual_generation_attempts=1531,ordinary_likelihood_rows=256,independent_checks=checks,
 actual_cost=cost,
 key_input_hashes={name:sha(C/name) for name in ('release.json','plan.json','evaluation-release.json')},
 readout_files={name:sha(C/name) for name in ('TRAIN-fit/result.json','TRAIN-fit/native-and-scored.jsonl','held-native-and-replay-result.json','held-native-and-replay-scored.json','ORDINARY/result.json','actual-cost.json','fixed-eight-examples.json','held-selection-descriptive.json')},
 primary_decision='PASS' if fit['fitting_target_pass'] else 'FAIL',
 boundaries=['The fitting gate is unchanged. Exact annotation-set agreement is not semantic entailment; reviewed source labels contain entity collisions.',
 'Held255 was inspected before choosing the coefficient. All answering comparisons are developmental and conditional on these checkpoints.',
 'The model sees all20 passages before IDs; missing an annotated ID is not the same as withholding the passage.',
 'Native generation and closed-field answer regeneration are different interfaces. A direct path that generates IDs must be qualified.',
 'A1.0 loss coefficient is not a gradient-share measurement; clipping and Adam dynamics can change.',
 'Only one data-order seed. No broad reasoning, general-language pretraining gain, calibration or CoT-superiority claim.',
 'No old V23/V24 score, gate, parser or primary report is changed; no scale-up or RL release follows automatically.'])
with (C/'complete-readout-before-protection.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(status=result['status'],primary_fitting=result['primary_decision'])))
