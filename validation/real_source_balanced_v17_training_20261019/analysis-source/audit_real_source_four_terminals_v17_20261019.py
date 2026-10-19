"""One independent audit after all four predeclared scientific terminals close."""
from pathlib import Path
import json,math,statistics
from datetime import datetime,timezone
import torch
from train.partial_state_driver import ROOT
from train.useful_note_precision import compare_distributions
from weekend_thinking.pretrained import load_base
from weekend_thinking.artifacts import canonical_hash
from real_source_state_audit_v17_20261019 import read,sha,equal,expected_metrics,audit_checkpoint
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/real-source-balanced-v17';assert ROOT==W
C=R/'local/diagnostics/real-source-balanced-v17-study-v1';plan=read(C/'plan.json')
assert read(C/'release.json')['plan_sha256']==sha(C/'plan.json')
assert len(plan['jobs'])==4
for job in plan['jobs']:
 d=C/f'{job["arm"]}-seed{job["seed"]}';assert read(d/'closure.json')['status']=='COMPLETE_TRAINING_PHYSICALLY_CLOSED_FULL_STATES_VERIFIED'
 assert read(d/'exit.json')['exit_code']==0 and read(d/'exit.json')['external_stop'] is None
 assert read(d/'closure.json')['all_original_session_processes_absent'] and read(d/'closure.json')['assigned_devices_empty']
first=read(plan['jobs'][0]['protocol']);model,tok=load_base(first['model_dir'])
data=[json.loads(l) for l in Path(first['data_file']).open()];assert len(data)==5120
assert sha(first['data_file'])==first['data_sha256']
data_by_id={row['item_id']:row for row in data};assert len(data_by_id)==5120
reference=torch.load(Path(read(first['base_protocol'])['native_reference_output'])/'next-logps.pt',weights_only=True,map_location='cpu')
models=[]
for job in plan['jobs']:
 p=read(job['protocol']);O=Path(job['output']);D=C/f'{job["arm"]}-seed{job["seed"]}'
 assert sha(job['protocol'])==job['protocol_sha256'] and all(sha(W/k)==v for k,v in p['source_files'].items())
 assert p['steps']==320 and p['items']==5120 and p['seed']==job['seed']
 assert p['data_sha256']==first['data_sha256'] and p['source_files']==first['source_files']
 assert sha(p['held_primary_manifest'])==p['held_primary_manifest_sha256']==plan['held_primary_manifest_sha256']
 run=read(O/'run.json');completion=read(O/'completion.json');close=read(D/'closure.json')
 assert completion['status']=='complete' and completion['actual_steps']==320 and close['completion_sha256']==sha(O/'completion.json')
 assert run['protocol_sha256']==job['protocol_sha256']
 captured=read(D/'first-live.json');assert len(captured['actual8'])==8
 native=[]
 for rank in range(8):
  report=read(O/f'qualification-rank-{rank:05d}.json');index=report['reference_index']
  actual=torch.load(O/f'forward-rank-{rank:05d}.pt',map_location='cpu',weights_only=True)
  check=compare_distributions(reference[index:index+1],actual)[0]
  assert check['passed'] and report['status']=='PASS';native.append(check)
  witness=torch.load(O/f'initial-master-rank-{rank:05d}.pt',map_location='cpu',weights_only=True)
  assert equal(witness['expected_cpu'],witness['actual_fp32_masters'])
 order,expected=expected_metrics(p,job['arm'],data)
 observed=[json.loads(l) for l in (O/'metrics.jsonl').open()]
 assert len(observed)==320 and run['order_hash']==canonical_hash(order)
 for actual,wanted in zip(observed,expected):
  assert all(actual[k]==v for k,v in wanted.items())
  assert math.isfinite(actual['loss']) and math.isfinite(actual['gradient_norm']) and actual['gradient_norm']>0
  indices=[data_by_id[k] for k in actual['ordered_item_ids']]
  source=sum(len(row['prefix_ids'])-1 for row in indices)
  task=sum(16 if job['arm']=='source_direct' else len(row['full_calculation_ids'])+18 for row in indices)
  component=actual['loss_components']
  assert component['source_labels']==source and component['task_labels']==task
  assert component['source_weight']==component['task_weight']==0.5
  assert all(math.isfinite(component[k]) for k in ('source_mean_loss','task_mean_loss'))
  assert abs(actual['loss']-.5*component['source_mean_loss']-.5*component['task_mean_loss'])<1e-5
 checkpoints={}
 for step in (160,320):
  directory=O/f'step-{step:07d}'
  assert sha(directory/'COMPLETE.json')==close['checkpoint_manifests'][str(step)]
  checkpoints[str(step)]=audit_checkpoint(directory,p,job['arm'],O.name,order,job['protocol_sha256'],expected[step-1]['counters'],model)
 result=dict(arm=job['arm'],seed=job['seed'],output=str(O),checkpoints=checkpoints,
   native_checks=native,actual_initial_master_witnesses=8,actual_updates=320,
   terminal_counters=observed[-1]['counters'],
   source_labels=sum(m['loss_components']['source_labels'] for m in observed),task_labels=sum(m['loss_components']['task_labels'] for m in observed),warm_last60_median_update_seconds=statistics.median(m['elapsed_seconds'] for m in observed[-60:]),
   completion_sha256=sha(O/'completion.json'),closure_sha256=sha(D/'closure.json'),
   metrics_sha256=sha(O/'metrics.jsonl'),protocol_sha256=sha(job['protocol']))
 models.append(result);print('Independentlyverified320updates/native8/full160and320',job['arm'],job['seed'],flush=True)
report=dict(status='PASS_ALL_FOUR_TERMINAL320_TRAINING_STATES_NATIVE32_ORDER_AND_COUNTERS',utc=datetime.now(timezone.utc).isoformat(),
 models=models,source_files=first['source_files'],plan_sha256=sha(C/'plan.json'),
 audit_source_sha256=sha(__file__),state_audit_source_sha256=sha(R/'local/analysis/real_source_state_audit_v17_20261019.py'),
 limits=['Trainingmechanicsandcompleteartifactintegrity,notheldoutcapability.','Evaluationbindingsrequireseparateprotectionofallterminalmodelsandlatestfullstate.','Noadditionalindependentobservationsfromrecheckingtheearlierprotectioncheckpoint.'])
with (C/'terminal-independent-audit.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({k:v for k,v in report.items() if k not in ('models','source_files')},indent=2),flush=True)
