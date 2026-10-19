"""Reparse the entire declared diagnostic only after all seven jobs close."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,gzip,shutil
import numpy as np
from scripts.audit_partial_state_panels import exact_trace,independent_score,independent_state,read,rows,sha
from scripts.audit_partial_state_study import numerical
from train.partial_state_evaluation import prompt_tokens,reader_tokens
from train.partial_state_answers import score_answer
from train.partial_state_note_driver import load_bundle
from transformers import AutoTokenizer
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/variable-depth-eval-v11'
registry=read(R/'local/diagnostics/variable-depth-writer-v11-v1/evaluation-jobs.json')['jobs']
for job in registry.values():
 control=Path(job['control'])
 assert read(control/'completion.json')['status']=='COMPLETE_ALL_THREE_DECLARED_DEPTH_PANELS'
 for index in range(3):
  stage=control/f'stage-{index}'
  assert read(stage/'completion.json')['status']=='COMPLETE_DECLARED_PANEL_ALL_ATTEMPTS'
  for shard in range(4):
   exited=read(stage/f'shard-{shard}/exit.json')
   assert exited['exit_code']==0 and exited['external_stop'] is None and not exited['original_session_remaining']
tok=AutoTokenizer.from_pretrained(R/'local/assets/qwen3-1.7b-base',local_files_only=True)
PUBLIC=R/'validation/variable_depth_writer_v11_complete_20261019';PUBLIC.mkdir(exist_ok=False)
ARCHIVE=R/'local/backups/evaluation/variable-depth-writer-v11-complete';ARCHIVE.mkdir(parents=True,exist_ok=False)
def paired(delta):
 a=np.asarray(delta,float);rng=np.random.default_rng(2026101912);means=[]
 for _ in range(20000):means.append(a[rng.integers(0,len(a),size=len(a))].mean())
 return dict(difference=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),items=len(a),repeats=20000,seed=2026101912,unit='accuracy_fraction')
def archive(label,path):
 dest=ARCHIVE/label;shutil.copytree(path,dest)
 for p in path.rglob('*'):
  if p.is_file():assert sha(p)==sha(dest/p.relative_to(path))
def output(arm,index):return Path(read(Path(registry[arm]['control'])/'plan.json')['steps'][index]['output'])
for arm,job in registry.items():archive('control-'+arm,Path(job['control']))
panels={};payloads={};total_answers=0;total_notes=0
for index,panel in enumerate(('DEV_composition','DEV_depth2','DEV_depth6')):
 n=512 if index==0 else 256;bundles={};states={};checks={};cells={};metrics={}
 for arm in ('fixed','variable'):
  path=output(arm,index);p=read(path/'protocol.json');prep=Path(p['model_preparation']);qs=rows(prep/'questions.jsonl');ids=[q['item_id'] for q in qs]
  assert len(qs)==n and p['max_actions']==128 and p['prefill_width']==64
  assert [q['panel'] for q in qs]==[panel]*n and p['source_files']=={k:sha(W/k) for k in p['source_files']}
  assert read(path/'generation-closure.json')['status']=='ALL_FOUR_GENERATION_SESSIONS_AND_DEVICE_PHYSICALLY_CLOSED'
  bundles[arm]=load_bundle(path/'notes-bundle.json',ids)
  states[arm]={c['item_id']:independent_state(c['own'],q['expression_tree']) for q,c in zip(qs,bundles[arm]['controls'])}
  assert states[arm]==bundles[arm]['state_diagnostics']
  checks[arm]=[numerical(path,s,reference=prep,notes=True) for s in range(4)]
  completed=read(Path(registry[arm]['control'])/f'stage-{index}/completion.json')
  assert completed['result_sha256']==sha(path/'notes-bundle.json')
  total_notes+=n;archive(panel+'/'+arm+'-notes',path);archive(panel+'/'+arm+'-native-preparation',prep)
  file=PUBLIC/f'{panel}-{arm}-all{n}-notes.jsonl.gz'
  with gzip.GzipFile(filename=str(file),mode='wb',mtime=0) as stream:
   for c in bundles[arm]['controls']:stream.write((json.dumps(c['own'],separators=(',',':'))+'\n').encode())
  payloads[str(file.relative_to(R))]=dict(rows=n,sha256=sha(file))
 # All controls and student conditions use the exact same frozen reader.
 reader_identity=None
 for arm in ('reader','full_cot','B0','fixed-student-answers','variable-student-answers'):
  path=output(arm,index);p=read(path/'protocol.json');questions=rows(path/'questions.jsonl');controls=rows(path/'controls.jsonl')
  ids=[q['item_id'] for q in questions];assert len(ids)==n and len(set(ids))==n and p['ordered_item_ids']==ids
  assert p['direct_and_cot_cap']==160 and p['writer_action_cap']==128 and p['reader_answer_cap']==32
  assert p['source_files']=={k:sha(W/k) for k in p['source_files']}
  for name,digest in p['files'].items():assert sha(path/name)==digest
  gold={q['item_id']:exact_trace(q['expression_tree'])[0] for q in questions}
  assert gold=={v['item_id']:int(v['answer']) for v in rows(path/'gold.jsonl')}
  if arm=='reader':reader_identity=p['reader']
  if 'student-answers' in arm:
   assert p['reader']==reader_identity
   writer_arm=arm.split('-')[0];bundle=bundles[writer_arm]
   for c,b in zip(controls,bundle['controls']):
    for mode in ('own','other'):assert c['notes'][mode]==b[mode]['content_ids'] and c['note_generation'][mode]['item_id']==b[mode]['item_id']
  wanted={q['item_id']:(q,c) for q,c in zip(questions,controls)};armcells={};raw=[]
  checks[arm]=[numerical(path,s) for s in range(4)]
  for shard in range(4):
   d=path/'answers'/f'shard-{shard}';run=read(d/'run.json');values=rows(d/'answers.jsonl')
   assert run['status']=='complete_all_attempts' and run['rows']==len(range(shard,n,4))*len(p['modes']) and sha(d/'answers.jsonl')==run['answers_sha256']
   for v in values:
    key=(v['item_id'],v['mode']);assert key not in armcells and v['mode'] in p['modes']
    q,c=wanted[v['item_id']];plain=v['mode'] in ('direct','cot')
    prompt=prompt_tokens(q,tok,cot=v['mode']=='cot') if plain else reader_tokens(q,c['notes'][v['mode']],tok)
    assert v['prompt_ids']==prompt and v['cap']==(160 if plain else 32) and v['prefill_width']==(64 if plain else 256)
    assert v['n_actions']==len(v['actions'])
    independent=independent_score(v,gold[v['item_id']]);production=score_answer(v,gold[v['item_id']],cot=v['mode']=='cot')
    assert independent=={k:production[k] for k in ('score','status')}
    armcells[key]={**v,**independent};raw.append(dict(arm=arm,**armcells[key]));total_answers+=1
  assert len(armcells)==n*len(p['modes']);summary=read(path/'summary.json')
  metrics[arm]={}
  for mode in p['modes']:
   vs=[armcells[k,mode] for k in ids]
   result=dict(items=n,correct=sum(v['score'] for v in vs),unsupported=sum(v['status']!='ok' for v in vs),status_counts=dict(Counter(v['status'] for v in vs)),generated_actions=sum(v['n_actions'] for v in vs),writer_actions=sum(v['n_writer_actions'] for v in vs),decode_wall_seconds=sum(v['wall_seconds'] for v in vs),writer_wall_seconds=sum(0 if v['source_note'] is None else v['source_note']['wall_seconds'] for v in vs))
   assert result==summary['results'][mode];metrics[arm][mode]=result
  for name,value in summary['contrasts'].items():
   b,a=name.split('_minus_');actual=paired([armcells[k,b]['score']-armcells[k,a]['score'] for k in ids])
   assert actual['difference']==value['difference'] and actual['ci95']==value['ci95']
  completed=read(Path(registry[arm]['control'])/f'stage-{index}/completion.json')
  assert completed['result_sha256']==sha(path/'summary.json') and read(path/'generation-closure.json')['status']=='ALL_FOUR_GENERATION_SESSIONS_AND_DEVICE_PHYSICALLY_CLOSED'
  cells[arm]=armcells;archive(panel+'/'+arm,path)
  file=PUBLIC/f'{panel}-{arm}-all{len(raw)}-answers.jsonl.gz'
  with gzip.GzipFile(filename=str(file),mode='wb',mtime=0) as stream:
   for v in raw:stream.write((json.dumps(v,separators=(',',':'))+'\n').encode())
  payloads[str(file.relative_to(R))]=dict(rows=len(raw),sha256=sha(file))
 delta=paired([cells['variable-student-answers'][k,'own']['score']-cells['fixed-student-answers'][k,'own']['score'] for k in ids])
 canonical=paired([int(states['variable'][k]['canonical_and_natural_stop'])-int(states['fixed'][k]['canonical_and_natural_stop']) for k in ids])
 baseline={name:paired([cells['variable-student-answers'][k,'own']['score']-cells[name][k,'cot']['score'] for k in ids]) for name in ('full_cot','B0')}
 count={arm:sum(v['canonical_and_natural_stop'] for v in states[arm].values()) for arm in states}
 panels[panel]=dict(items=n,answers=metrics,writer_canonical_natural=count,variable_minus_fixed_answer=delta,variable_minus_fixed_canonical=canonical,variable_student_minus_baselines=baseline,numerical_checks=checks,
  primary_gate=(count['variable']/n>=.7 and canonical['ci95'][0]>0 and metrics['variable-student-answers']['own']['correct']/n>=.4 and delta['difference']>=.1 and delta['ci95'][0]>0) if index==0 else None)
 print('Closedpanelindependentlyverified',panel,flush=True)
assert total_answers==9216 and total_notes==2048
report=dict(status='COMPLETE_DECLARED_VARIABLE_DEPTH_DIAGNOSTIC_ALL_ATTEMPTS_AUDITED',utc=datetime.now(timezone.utc).isoformat(),panels=panels,answer_attempts=total_answers,note_attempts=total_notes,payloads=payloads,
 limits=['One diagnostic training seed,three separate panels;neverpool.','Curricula differ in depth,examples and supervised-token totals;not a pureEOS-only causal isolation.','Two-model system must justify cost against both single-model baselines. This diagnostic does not gate real-data mainline.'])
with (ARCHIVE/'AUDIT.json').open('x') as f:json.dump(report,f,indent=2)
report['EBS_archive_audit_sha256']=sha(ARCHIVE/'AUDIT.json')
with (R/'validation/variable_depth_writer_v11_complete_20261019.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({k:v for k,v in report.items() if k not in ('panels','payloads')},indent=2),flush=True)
