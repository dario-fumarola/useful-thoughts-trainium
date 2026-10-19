"""Once-only secondary pointer replay after all native attempts physically close.

Own-model TRAIN CPU references precede each shard's held scoring. No native
answer score inspection, silent backend fallback or automatic retry.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-evidence-span-v23'
C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
execution=read(C/'likelihood-execution-plan.json')
assert execution['status']=='RELEASED_V23_SECONDARY_POINTER_REPLAY_EXECUTION'
assert execution['source_files'][str(Path(__file__).relative_to(R))]==sha(__file__)
assert execution['pilot_release_sha256']==sha(C/'release.json')
D=C/'likelihood-pipeline';D.mkdir(exist_ok=False)
end=datetime.fromisoformat(execution['deadline_utc'])
while not (C/'ALL_GENERATION_CLOSED_NO_READOUT.json').exists():
 if datetime.now(timezone.utc)>=end:raise TimeoutError('Native generation incomplete;no partial likelihood preparation')
 for child in (C/'generation-pipeline').glob('*-exit.json'):
  if read(child)['exit_code']:raise RuntimeError('Generation failed;no replay retry')
 time.sleep(10)
originals=read(C/'ALL_GENERATION_CLOSED_NO_READOUT.json')
assert originals['status']=='ALL1530_FROZEN_NATIVE_ATTEMPTS_CLOSED_NO_READOUT'
env=os.environ.copy();env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),USE_TORCH_XLA='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
registry=[]
for original in originals['groups']:
 key=original['model']
 output=R/'local/nvme-cpt/evaluation'/f'document-evidence-plan-v23-{key}-likelihood-v1'
 command=[sys.executable,'-u','-m','train.document_evidence_plan_likelihood','prepare',
  '--output',str(output),'--original',original['output'],
  '--data-dir',str(R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'),
  '--ordinary-manifest',str(R/'local/nvme-cpt/data/document-QA-v21-ordinary256-v2/manifest.json'),
  '--calibration-dir',str(R/'local/diagnostics/document-QA-v21-span-qualification-v1')]
 with (D/(key+'-prepare.log')).open('xb') as log:
  result=subprocess.run(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 with (D/(key+'-prepare-exit.json')).open('x') as stream:json.dump(dict(exit_code=result.returncode,utc=datetime.now(timezone.utc).isoformat()),stream)
 if result.returncode:raise RuntimeError('Replay preparation failed;preserve evidence,no automaticretry')
 registry.append(dict(model=key,arm=original['arm'],output=str(output),control=str(C/('LL-'+key)),
  physical=execution['physical_by_arm'][original['arm']],original=original['output'],protocol_sha256=sha(output/'protocol.json')))
with (C/'likelihood-registry.json').open('x') as stream:json.dump(dict(status='BOTH_FROZEN_POINTER_REPLAY_PANELS_PREPARED_BEFORE_HELD_SCORING',groups=registry),stream,indent=2)
controller=R/'local/analysis/supervise_document_evidence_v23_likelihood_group_20261019.py'
assert execution['source_files'][str(controller.relative_to(R))]==sha(controller)
children=[]
for group in registry:
 log=(D/(group['model']+'-controller.log')).open('xb')
 process=subprocess.Popen([sys.executable,'-u',str(controller),group['output'],str(group['physical']),group['control']],cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
 children.append((group,process,log))
while any(process.poll() is None for _,process,_ in children):
 if datetime.now(timezone.utc)>=end:raise TimeoutError('Scoring deadline;owned child supervisors enforce closure')
 time.sleep(10)
for group,process,log in children:
 log.close()
 with (D/(group['model']+'-controller-exit.json')).open('x') as stream:json.dump(dict(exit_code=process.returncode,utc=datetime.now(timezone.utc).isoformat()),stream)
 if process.returncode:raise RuntimeError('Scoring failed;retain errors,no partial complete-estimate claim orautomaticretry')
for group in registry:
 closure=read(Path(group['control'])/'closure.json')
 assert closure['status']=='ALL_FROZEN_LIKELIHOOD_SHARDS_PHYSICALLY_CLOSED_NO_READOUT'
 assert closure['actual_rows']==1531 and closure['assigned_physical_device_empty']
with (C/'ALL_LIKELIHOOD_CLOSED_NO_READOUT.json').open('x') as stream:json.dump(dict(status='ALL3062_ORIGINAL_POINTER_REPLAY_AND_ORDINARY_ROWS_CLOSED_NO_READOUT',utc=datetime.now(timezone.utc).isoformat(),groups=registry),stream,indent=2)
