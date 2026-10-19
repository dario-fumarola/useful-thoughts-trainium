"""Execute every frozen likelihood panel once after its original notes close.

Four batch-four shards per model;physical14/15 while generation remains active,
then freed0..11. No capability readout or automatic retry on failure.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shlex,subprocess,time
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-qa-span-v21';C=R/'local/diagnostics/document-QA-v21-study-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
D=C/'likelihood-pipeline';D.mkdir(exist_ok=False)
release=read(C/'release.json');assert release['status']=='RELEASED_DOCUMENT_QA_V21_SCIENCE_AND_FIXED_EVALUATION'
keys=['B0',*[j['key'] for j in read(C/'plan.json')['jobs']]];assert len(keys)==7
producer=R/'local/analysis/document_QA_v21_likelihood_job_20261019.py';controller=R/'local/analysis/supervise_document_QA_v21_likelihood_group_20261019.py'
python='/home/ubuntu/ema-trainium/.venv/bin/python';deadline=datetime.fromisoformat('2026-10-20T13:00:00+00:00')
env=os.environ.copy();env.update(PYTHONPATH=str(W)+':'+str(R/'local/analysis')+':'+str(R/'local/test-deps'),USE_TORCH_XLA='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
registry=[];started={};closed=set()
def save():
 (C/'likelihood-registry.json').write_text(json.dumps(dict(status='FIXED_LIKELIHOOD_PANELS_EXECUTING_NO_READOUT',groups=registry,closed=sorted(closed),producer_sha256=sha(producer),controller_sha256=sha(controller),watcher_sha256=sha(__file__)),indent=2))
def originals():
 result={'B0':dict(output=str(R/'local/nvme-cpt/evaluation/document-QA-v21-B0-held512-v1'),control=str(C/'B0-generation'))}
 p=C/'trained-generation-registry.json'
 if p.exists():
  for g in read(p)['groups']:
   if (g['model'].startswith('qa_prior-') and g['label']=='prior') or (not g['model'].startswith('qa_prior-') and g['label']=='all'):
    assert g['model'] not in result;result[g['model']]=g
 return result
save()
while len(closed)<7:
 if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Fixed likelihood deadline;no automaticretry')
 for key,job in started.items():
  if key in closed:continue
  control=Path(job['control'])
  if (control/'STOP_REASON.json').exists():raise RuntimeError('Likelihood workerfailure: '+key)
  if (control/'closure.json').exists():
   assert read(control/'closure.json')['status']=='ALL_FROZEN_LIKELIHOOD_SHARDS_PHYSICALLY_CLOSED_NO_READOUT';closed.add(key);save()
  elif (D/(key+'-exit.json')).exists():raise RuntimeError('Likelihood controller closed without success: '+key)
 occupied={v['physical'] for k,v in started.items() if k not in closed}
 pool=[14,15]+(list(range(12)) if (C/'ALL_GENERATION_CLOSED_NO_READOUT.json').exists() else [])
 free=[v for v in pool if v not in occupied]
 source=originals()
 for key in keys:
  if key in started or key not in source or not free:continue
  original=source[key]
  if not (Path(original['control'])/'closure.json').exists():continue
  output=R/'local/nvme-cpt/evaluation'/f'document-QA-v21-{key}-likelihood-v1'
  cmd=[python,'-u',str(producer),'prepare','--original',original['output'],'--output',str(output)]
  with (D/(key+'-prepare.log')).open('xb') as log:proc=subprocess.run(cmd,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
  if proc.returncode:raise RuntimeError('Likelihood preparationfailed: '+key+';no automaticretry')
  p=read(output/'protocol.json');assert p['model']['id']==key and p['release_sha256']==sha(C/'release.json')
  physical=free.pop(0);control=C/f'LL-{key}';log=D/(key+'-controller.log');exitpath=D/(key+'-exit.json')
  shell='cd /home/ubuntu/dario; source scripts/env.sh; '+shlex.join([python,'-u',str(controller),str(output),str(physical),str(control)])+' > '+shlex.quote(str(log))+' 2>&1; code=$?; printf \'{"exit_code":%s}\\n\' "$code" > '+shlex.quote(str(exitpath))
  subprocess.run(['tmux','new-session','-d','-s','QA21-LL-'+key,shell],check=True)
  job=dict(model=key,output=str(output),control=str(control),physical=physical,original=original['output'],protocol_sha256=sha(output/'protocol.json'))
  registry.append(job);started[key]=job;save()
 time.sleep(10)
assert sum(read(Path(g['control'])/'closure.json')['actual_rows'] for g in registry)==19712
(C/'ALL_LIKELIHOOD_CLOSED_NO_READOUT.json').write_text(json.dumps(dict(status='ALL_NINETEEN_THOUSAND_SEVEN_HUNDRED_TWELVE_LIKELIHOOD_ROWS_CLOSED_NO_READOUT',utc=datetime.now(timezone.utc).isoformat(),groups=registry,registry_sha256=sha(C/'likelihood-registry.json')),indent=2))
