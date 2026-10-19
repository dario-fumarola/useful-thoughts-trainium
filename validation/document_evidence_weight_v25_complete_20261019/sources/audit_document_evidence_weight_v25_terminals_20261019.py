"""Independently inspect both full training trajectories/states after closure."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,math,random,statistics
import torch
from train.partial_state_driver import ROOT
from train.useful_note_precision import compare_distributions
from weekend_thinking.artifacts import canonical_hash
from weekend_thinking.pretrained import load_base
from document_evidence_weight_v25_state_audit_20261019 import read,sha,equal,audit_checkpoint,independent_counts,audit_metrics
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-evidence-weight-v25';assert ROOT==W
C=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1';plan=read(C/'plan.json');release=read(C/'release.json')
assert release['status']=='RELEASED_DOCUMENT_EVIDENCE_WEIGHT_V25_TREATMENT_ONLY' and release['plan_sha256']==sha(C/'plan.json')
assert len(plan['jobs'])==1
for job in plan['jobs']:
 d=C/job['key'];close=read(d/'closure.json')
 assert close['status']=='COMPLETE_TRAINING_PHYSICALLY_CLOSED_FULL_STATES_VERIFIED'
 assert close['all_original_session_processes_absent'] and close['assigned_devices_empty']
 assert read(d/'exit.json')['exit_code']==0 and read(d/'exit.json')['external_stop'] is None
first=read(plan['jobs'][0]['protocol']);model,tok=load_base(first['model_dir']);data=[json.loads(l) for l in Path(first['data_file']).open()]
assert len(data)==2048 and len({r['item_id'] for r in data})==2048 and sha(first['data_file'])==first['data_sha256']
counts=independent_counts(data,tok)
reference=torch.load(Path(read(first['base_protocol'])['native_reference_output'])/'next-logps.pt',weights_only=True,map_location='cpu')
models=[]
for job in plan['jobs']:
 p=read(job['protocol']);O=Path(job['output']);D=C/job['key'];close=read(D/'closure.json')
 assert sha(job['protocol'])==job['protocol_sha256'] and all(sha(W/n)==v for n,v in p['source_files'].items())
 assert p['steps']==256 and p['items']==2048 and p['seed']==job['seed'] and p['data_sha256']==first['data_sha256'] and p['source_files']==first['source_files']
 assert sha(p['held_primary_manifest'])==p['held_primary_manifest_sha256']==plan['held_primary_manifest_sha256']
 run=read(O/'run.json');completion=read(O/'completion.json');assert completion['status']=='complete' and completion['actual_steps']==256 and close['completion_sha256']==sha(O/'completion.json')
 assert run['protocol_sha256']==job['protocol_sha256'] and len(read(D/'first-live.json')['actual8'])==8
 native=[]
 for rank in range(8):
  receipt=read(O/f'qualification-rank-{rank:05d}.json');i=receipt['reference_index'];actual=torch.load(O/f'forward-rank-{rank:05d}.pt',map_location='cpu',weights_only=True)
  check=compare_distributions(reference[i:i+1],actual)[0];assert check['passed'] and receipt['status']=='PASS';native.append(check)
  witness=torch.load(O/f'initial-master-rank-{rank:05d}.pt',map_location='cpu',weights_only=True);assert equal(witness['expected_cpu'],witness['actual_fp32_masters'])
 p,order,observed=audit_metrics(job,256,counts)
 assert len([json.loads(line) for line in (O/'metrics.jsonl').open()])==256
 expected_answer=sum(v['loss_components']['answer_labels'] for v in observed)
 expected_selection=sum(v['loss_components']['selection_labels'] for v in observed)
 assert expected_answer==67408 and expected_selection==dict(evidence_direct=8192,evidence_select=39930)[job['arm']]
 assert sum(v['loss_components']['source_labels'] for v in observed)==2097152
 checkpoints={}
 for step in (128,256):
  directory=O/f'step-{step:07d}';assert sha(directory/'COMPLETE.json')==close['checkpoint_manifests'][str(step)]
  checkpoints[str(step)]=audit_checkpoint(directory,p,job['arm'],O.name,order,job['protocol_sha256'],observed[step-1]['counters'],model)
 models.append(dict(arm=job['arm'],seed=job['seed'],output=str(O),checkpoints=checkpoints,native_checks=native,actual_updates=256,actual_initial_master_witnesses=8,
  terminal_counters=observed[-1]['counters'],source_labels=2097152,answer_labels=expected_answer,selection_labels=expected_selection,warm_last60_median_update_seconds=statistics.median(v['elapsed_seconds'] for v in observed[-60:]),
  completion_sha256=sha(O/'completion.json'),closure_sha256=sha(D/'closure.json'),metrics_sha256=sha(O/'metrics.jsonl'),protocol_sha256=sha(job['protocol'])))
 print('Verified256updates/native8/full128and256',job['key'],flush=True)
report=dict(status='PASS_SINGLE_TERMINAL256_EVIDENCE_WEIGHT_TRAINING_STATES_NATIVE8_ORDER_COUNTERS_AND_LABELS',utc=datetime.now(timezone.utc).isoformat(),models=models,source_files=first['source_files'],plan_sha256=sha(C/'plan.json'),audit_source_sha256=sha(__file__),generic_state_audit_source_sha256=sha(R/'local/analysis/real_source_state_audit_v17_20261019.py'),independent_inventory_source_sha256=sha(R/'local/analysis/document_evidence_weight_v25_state_audit_20261019.py'),
 limits=['Training/state integrity,not held capability. Native8 checks repeat four canonical contexts.','All2 original full states retained;allterminalmodels/latestfull need separate EBS readback before binding evaluation.','No uninterrupted/resumed bitwise parity orstochasticRL claim.'])
with (C/'terminal-independent-audit.json').open('x') as f:json.dump(report,f,indent=2)
print(report['status'],flush=True)
