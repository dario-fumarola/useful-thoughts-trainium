"""Once-only fixed evaluation after both pilot terminal states are protected.

Only prompt preparation, native generation and physical closure; no answer
scores, no old held panel and no automatic retry.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-evidence-eval-v23'
C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(C/'plan.json');release=read(C/'release.json');execution=read(C/'generation-execution-plan.json')
assert execution['status']=='RELEASED_V23_FIXED_GENERATION_EXECUTION'
assert execution['source_files'][str(Path(__file__).relative_to(R))]==sha(__file__)
assert execution['pilot_release_sha256']==sha(C/'release.json')
D=C/'generation-pipeline';D.mkdir(exist_ok=False)
end=datetime.fromisoformat(execution['deadline_utc'])
while not (C/'terminal-pipeline-completion.json').exists():
 if datetime.now(timezone.utc)>=end:raise TimeoutError('Terminal bindings unavailable;no automaticretry')
 for child in (C/'terminal-pipeline').glob('*-exit.json'):
  if read(child)['exit_code']:raise RuntimeError('Terminal pipeline failed;no held generation')
 if any((C/j['key']/'failure.json').exists() for j in plan['jobs']):raise RuntimeError('Training failed;no held generation')
 time.sleep(10)
assert read(C/'terminal-pipeline-completion.json')['status']=='BOTH_PILOT_TERMINALS_AUDITED_PROTECTED_BOUND_NO_HELD_READOUT'
env=os.environ.copy();env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),USE_TORCH_XLA='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
registry=[]
for job in plan['jobs']:
 key=job['key'];protocol=read(job['protocol'])
 output=R/'local/nvme-cpt/evaluation'/f'document-evidence-plan-v23-{key}-held255-v1'
 modes=release['model_modes'][job['arm']]
 command=[sys.executable,'-u','-m','train.document_evidence_plan_evaluate','prepare',
  '--output',str(output),'--data-dir',str(R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'),
  '--model-dir',protocol['model_dir'],'--base-protocol',protocol['base_protocol'],
  '--binding',str(C/'frozen-models'/(key+'.json')),'--release',str(C/'release.json'),'--modes',*modes]
 with (D/(key+'-prepare.log')).open('xb') as log:
  result=subprocess.run(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
 with (D/(key+'-prepare-exit.json')).open('x') as stream:json.dump(dict(exit_code=result.returncode,utc=datetime.now(timezone.utc).isoformat()),stream)
 if result.returncode:raise RuntimeError('Preparation failed;original evidence retained,no retry')
 registry.append(dict(model=key,arm=job['arm'],modes=modes,output=str(output),control=str(C/('GEN-'+key)),
   physical=execution['physical_by_arm'][job['arm']],protocol_sha256=sha(output/'protocol.json')))
with (C/'generation-registry.json').open('x') as stream:json.dump(dict(status='FROZEN_BOTH_MODEL_PROMPTS_BEFORE_HELD_DECODING',groups=registry),stream,indent=2)
controller=R/'local/analysis/supervise_document_evidence_v23_generation_group_20261019.py'
assert execution['source_files'][str(controller.relative_to(R))]==sha(controller)
children=[]
for group in registry:
 log=(D/(group['model']+'-generation.log')).open('xb')
 command=[sys.executable,'-u',str(controller),group['output'],str(group['physical']),group['control']]
 process=subprocess.Popen(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
 children.append((group,process,log))
while any(process.poll() is None for _,process,_ in children):
 if datetime.now(timezone.utc)>=end:raise TimeoutError('Fixed generation deadline exhausted;owned child supervisors enforce closure')
 time.sleep(10)
for group,process,log in children:
 log.close()
 with (D/(group['model']+'-generation-exit.json')).open('x') as stream:json.dump(dict(exit_code=process.returncode,utc=datetime.now(timezone.utc).isoformat()),stream)
 if process.returncode:raise RuntimeError('Fixed generation failed;no partial readout orautomaticretry')
for group in registry:
 closure=read(Path(group['control'])/'closure.json')
 assert closure['status']=='ALL_FROZEN_GENERATION_SHARDS_PHYSICALLY_CLOSED_NO_SCORES_REVIEWED'
 assert closure['actual_attempts']==255*len(group['modes']) and closure['assigned_physical_device_empty']
with (C/'ALL_GENERATION_CLOSED_NO_READOUT.json').open('x') as stream:json.dump(dict(status='ALL1530_FROZEN_NATIVE_ATTEMPTS_CLOSED_NO_READOUT',utc=datetime.now(timezone.utc).isoformat(),groups=registry),stream,indent=2)
