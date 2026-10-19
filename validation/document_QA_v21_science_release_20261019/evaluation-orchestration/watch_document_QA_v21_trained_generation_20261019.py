"""Prepare audited native references,then run all fixed trained modes once."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,time
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-qa-eval-v21';C=R/'local/diagnostics/document-QA-v21-study-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
D=C/'trained-generation-pipeline';D.mkdir(exist_ok=False);plan=read(C/'plan.json');execution=read(C/'evaluation-execution-plan.json');release=read(C/'release.json');end=datetime.fromisoformat(plan['deadline_utc'])
while not (C/'terminal-pipeline-completion.json').exists():
 if datetime.now(timezone.utc)>=end:raise TimeoutError('Training terminal bindings unavailable;no retry')
 for q in (C/'terminal-pipeline').glob('*-exit.json'):
  if read(q)['exit_code']:raise RuntimeError('Terminal pipeline failed;no generation release from partial states')
 time.sleep(10)
assert read(C/'terminal-pipeline-completion.json')['status']=='ALL_SIX_TERMINAL256_MODELS_AUDITED_PROTECTED_AND_BOUND'
env=os.environ.copy();env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),USE_TORCH_XLA='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
python='/home/ubuntu/ema-trainium/.venv/bin/python';full={};registry=[]
for job in plan['jobs']:
 key=job['key'];output=R/'local/nvme-cpt/evaluation'/f'document-QA-v21-{key}-projection-v1';protocol=read(job['protocol']);modes=[v for v in release['model_modes'][key] if v not in ('own_replay','other_replay')]
 cmd=[python,'-u','-m','train.document_qa_evaluate','prepare','--output',str(output),'--data-dir',str(R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1'),'--model-dir',protocol['model_dir'],'--base-protocol',protocol['base_protocol'],'--binding',str(C/'frozen-models'/(key+'.json')),'--release',str(C/'release.json'),'--modes',*modes]
 with (D/(key+'-prepare.log')).open('xb') as f:proc=subprocess.run(cmd,cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT)
 if proc.returncode:raise RuntimeError('Native reference preparation failed:no automaticretry '+key)
 full[key]=output
controller=R/'local/analysis/supervise_document_QA_v21_generation_group_v3_20261019.py'
for group in execution['initial_groups']:
 key=group['model'];parent=full[key];prior=read(parent/'protocol.json');output=R/'local/nvme-cpt/evaluation'/f'document-QA-v21-{key}-{group["label"]}-v1';output.mkdir(exist_ok=False)
 names=['native-reference.pt','native-contexts.json',*[f'prompts-{m}.jsonl' for m in group['modes']]]
 for name in names:
  assert sha(parent/name)==prior['files'][name];os.link(parent/name,output/name)
 p=dict(prior,modes=group['modes'],files={n:prior['files'][n] for n in names},maximum_prompt_tokens={m:prior['maximum_prompt_tokens'][m] for m in group['modes']},parent_projection_protocol_sha256=sha(parent/'protocol.json'),partition='Same frozen perquery computation;disjoint fixed condition groups for parallel execution only.')
 (output/'protocol.json').write_text(json.dumps(p,indent=2));control=C/f'GEN-{key}-{group["label"]}'
 cmd=f'cd /home/ubuntu/dario; source scripts/env.sh; {python} -u {controller} {output} {group["physical"]} {control} > {D/(key+"-"+group["label"]+".log")} 2>&1'
 subprocess.run(['tmux','new-session','-d','-s','QA21-GEN-'+key+'-'+group['label'],cmd],check=True)
 registry.append(dict(**group,output=str(output),control=str(control),protocol_sha256=sha(output/'protocol.json')))
(C/'trained-generation-registry.json').write_text(json.dumps(dict(status='ALL_FIXED_INITIAL_GROUPS_STARTED_NO_SCORES',groups=registry,controller_sha256=sha(controller)),indent=2))
replays={}
while len(replays)<2:
 for seed in (2026101921,2026101922):
  key=f'qa_prior-seed{seed}'
  if key in replays:continue
  original=next(g for g in registry if g['model']==key and g['label']=='prior')
  if not (Path(original['control'])/'closure.json').exists():continue
  output=R/'local/nvme-cpt/evaluation'/f'document-QA-v21-{key}-replays-v1'
  with (D/(key+'-prepare-replays.log')).open('xb') as f:
   proc=subprocess.run([python,'-u','-m','train.document_qa_evaluate','prepare-replays','--output',str(output),'--data-dir',original['output']],cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT)
  if proc.returncode:raise RuntimeError('Replay preparation failed,no automaticretry')
  control=C/f'GEN-{key}-replays';cmd=f'cd /home/ubuntu/dario; source scripts/env.sh; {python} -u {controller} {output} {original["physical"]} {control} > {D/(key+"-replays.log")} 2>&1'
  subprocess.run(['tmux','new-session','-d','-s','QA21-GEN-'+key+'-replays',cmd],check=True)
  replays[key]=dict(model=key,label='replays',modes=['own_replay','other_replay'],physical=original['physical'],output=str(output),control=str(control),protocol_sha256=sha(output/'protocol.json'))
 for group in registry:
  log=Path(group['control'])/'STOP_REASON.json'
  if log.exists():raise RuntimeError('A fixed generation group stopped;no auto recovery or partial readout')
 if datetime.now(timezone.utc)>=end:raise TimeoutError('No automatic retry')
 time.sleep(10)
allgroups=registry+list(replays.values());(C/'trained-generation-registry.json').write_text(json.dumps(dict(status='ALL_FIXED_INITIAL_AND_REPLAY_GROUPS_STARTED_NO_SCORES',groups=allgroups,controller_sha256=sha(controller)),indent=2))
while not all((Path(g['control'])/'closure.json').exists() for g in allgroups) or not (C/'B0-generation/closure.json').exists():
 for group in allgroups:
  if (Path(group['control'])/'STOP_REASON.json').exists():raise RuntimeError('Generation error boundary;no automaticretry')
 if datetime.now(timezone.utc)>=end:raise TimeoutError('No automatic retry')
 time.sleep(10)
(C/'ALL_GENERATION_CLOSED_NO_READOUT.json').write_text(json.dumps(dict(status='ALL_TWENTY_THOUSAND_NINE_HUNDRED_NINETY_TWO_ATTEMPTS_CLOSED_NO_READOUT',utc=datetime.now(timezone.utc).isoformat(),groups=allgroups,baseline_output=str(R/'local/nvme-cpt/evaluation/document-QA-v21-B0-held512-v1')),indent=2))
