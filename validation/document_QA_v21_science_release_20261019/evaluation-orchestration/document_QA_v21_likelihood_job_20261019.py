"""Frozen full-answer/EOS and ordinary-text likelihood;no accuracy analysis.

prepare uses closed prefix-only prior attempts and separate gold annotations.
run opens only frozen scoring examples. Every exception remains an explicit
failed scoring attempt; it never shortens or drops a document.
"""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import argparse,hashlib,json,math,time
import torch
from train.document_qa_contexts import context,trace_from_output
from train.document_qa_span_scores import score_spans
from train.evaluate_local import setup_model
from weekend_thinking.artifacts import atomic_json,sha256_file
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-qa-span-v21'
read=lambda p:json.loads(Path(p).read_text());rows=lambda p:[json.loads(l) for l in Path(p).read_text().splitlines()]

def dependencies():
 names=['train/document_qa_span_scores.py','train/document_qa_contexts.py','train/document_qa_packing.py','train/document_evidence_data.py','train/evaluate_local.py','train/evaluate.py','train/neuron_backend.py','train/runtime_compat.py','weekend_thinking/pretrained.py']
 return {n:sha256_file(W/n) for n in names}

def prepare(original,output):
 from transformers import AutoTokenizer
 p=read(original/'protocol.json');release=read(p['release_file']);assert sha256_file(p['release_file'])==p['release_sha256']
 assert p['purpose']=='FIXED_HELD_DOCUMENT_QA' and p['items']==512 and p['shards']==4 and 'prior' in p['modes']
 notes={};origins={}
 for shard in range(4):
  d=original/'answers'/f'shard-{shard}';run=read(d/'run.json')
  assert run['status']=='complete_all_attempts' and run['protocol_sha256']==sha256_file(original/'protocol.json') and run['attempts_sha256']==sha256_file(d/'attempts.jsonl')
  origins[str(d/'attempts.jsonl')]=run['attempts_sha256']
  for row in rows(d/'attempts.jsonl'):
   if row['mode']=='prior':assert row['item_id'] not in notes;notes[row['item_id']]=row
 D=R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1';m=read(D/'manifest.json');assert sha256_file(D/'manifest.json')==release['QA_data_manifest_sha256']
 for key in ('held_questions','held_gold'):assert sha256_file(m['files'][key]['path'])==m['files'][key]['sha256']
 qs=rows(m['files']['held_questions']['path']);gold={r['item_id']:r for r in rows(m['files']['held_gold']['path'])}
 assert [q['item_id'] for q in qs]==p['ordered_item_ids'] and set(notes)==set(gold)==set(p['ordered_item_ids'])
 tok=AutoTokenizer.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False);output.mkdir(parents=True,exist_ok=False)
 files={};inventories={}
 for mode in ('direct','empty','own_replay','other_replay','oracle'):
  examples=[]
  for i,q in enumerate(qs):
   donor=None;flag=None
   if mode in ('direct','empty'):text=context(q,mode)
   elif mode=='oracle':text=context(q,'replay',gold[q['item_id']]['trace'])
   else:
    donor=qs[i if mode=='own_replay' else (i+1)%512]['item_id'];attempt=notes[donor];trace,had_boundary=trace_from_output(attempt['text']);text=context(q,'replay',trace)
    flag=dict(status=attempt['status'],stop_type=attempt['stop_type'],had_final_boundary=had_boundary)
   past=tok.encode(text,add_special_tokens=False);joint=tok.encode(text+gold[q['item_id']]['answer'],add_special_tokens=False)
   assert joint[:len(past)]==past
   target=joint[len(past):]+[151643];assert 1<len(target)<=32 and len(past)+len(target)<=3584
   examples.append(dict(item_id=q['item_id'],panel='QA',mode=mode,context_ids=past,target_ids=target,donor_item_id=donor,donor_status=flag))
  path=output/f'QA-{mode}.jsonl';path.write_text(''.join(json.dumps(v)+'\n' for v in examples));files[path.name]=sha256_file(path);inventories[path.name]=dict(rows=512,targets=sum(len(v['target_ids']) for v in examples),max_context=max(len(v['context_ids']) for v in examples))
 ordinary=R/'local/nvme-cpt/data/document-QA-v21-ordinary256-v2/manifest.json';om=read(ordinary);assert sha256_file(ordinary)==release['ordinary_manifest_sha256'] and sha256_file(om['data_file'])==om['data_sha256']
 examples=[dict(item_id=r['item_id'],panel='ordinary',mode='raw',context_ids=r['prefix_ids'],target_ids=r['target_ids']) for r in rows(om['data_file'])]
 assert len(examples)==256 and all(len(r['context_ids'])==256 and len(r['target_ids'])==32 for r in examples)
 path=output/'ordinary-raw.jsonl';path.write_text(''.join(json.dumps(v)+'\n' for v in examples));files[path.name]=sha256_file(path);inventories[path.name]=dict(rows=256,targets=8192,max_context=256)
 # Fixed TRAIN-only,own-model CPU calibration is required before any held score.
 calibration=R/'local/diagnostics/document-QA-v21-span-qualification-v1/examples.json';cal=read(calibration);torch.set_num_threads(16)
 model=p['model'];checkpoint=model['checkpoint']
 if checkpoint:assert sha256_file(checkpoint)==model['checkpoint_sha256']
 backend,_,loaded,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=Path(checkpoint) if checkpoint else None,device='cpu'))
 assert signature==p['signature']
 if checkpoint:assert all(loaded[k]==model[k] for k in ('run_id','step','spec_hash'))
 refs={};times={}
 for label,rs in cal.items():
  width=3584 if label=='QA' else 672;start=time.monotonic();refs[label]=score_spans(backend,[r['context_ids'] for r in rs],[r['target_ids'] for r in rs],width=width);times[label]=time.monotonic()-start
 (output/'calibration-inputs.json').write_bytes(calibration.read_bytes());atomic_json(output/'calibration-CPU-reference.json',dict(signature=signature,values=refs,seconds=times))
 files.update({n:sha256_file(output/n) for n in ('calibration-inputs.json','calibration-CPU-reference.json')})
 qa=read(R/'local/diagnostics/document-QA-v21-span-qualification-v1/workload-completion.json');close=read(R/'local/diagnostics/document-QA-v21-span-qualification-v1/closure.json')
 assert qa['status']=='ALL_TRAIN_ONLY_SPAN_CPU_PARITY_CHECKS_PASS_PENDING_PHYSICAL_CLOSURE' and close['physical14_empty']
 protocol=dict(schema='document-QA-v21-full-field-likelihood',created_utc=datetime.now(timezone.utc).isoformat(),model=model,model_dir=p['model_dir'],signature=signature,shards=4,batch=4,target_slots=32,widths=dict(QA=3584,ordinary=672),source_files=dependencies(),worker_sha256=sha256_file(__file__),files=files,inventories=inventories,
  original_generation_protocol_sha256=sha256_file(original/'protocol.json'),original_prior_attempts=origins,release_file=p['release_file'],release_sha256=p['release_sha256'],
  gold_answer_span='Exact native joint-context answer surface plusEOS;content andEOS logged separately. No alias substitution or repair.',execution_policy='XLA FP32 after own-model TRAIN calibration;any calibration/runtimefailure stops,no automaticretry. Qualified CPU reference is explicit fallback after rootreview. No held scoreinspection tochoosebackend.')
 atomic_json(output/'protocol.json',protocol)
 print(json.dumps(dict(status='PREPARED_FROZEN_LIKELIHOOD_NO_RESULT_REVIEW',model=model['id'],rows=2816)),flush=True)

def run(output,shard,device):
 p=read(output/'protocol.json');assert p['schema']=='document-QA-v21-full-field-likelihood' and p['source_files']==dependencies() and p['worker_sha256']==sha256_file(__file__)
 assert 0<=shard<4 and p['release_sha256']==sha256_file(p['release_file'])
 for n,v in p['files'].items():assert sha256_file(output/n)==v
 model=p['model'];path=model['checkpoint']
 if path:assert sha256_file(path)==model['checkpoint_sha256']
 backend,tok,loaded,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=Path(path) if path else None,device=device));assert signature==p['signature']
 if path:assert all(loaded[k]==model[k] for k in ('run_id','step','spec_hash'))
 d=output/'scores'/f'shard-{shard}';d.mkdir(parents=True,exist_ok=False);cal=read(output/'calibration-inputs.json');refs=read(output/'calibration-CPU-reference.json')['values'];checks={}
 for label,rs in cal.items():
  actual=score_spans(backend,[r['context_ids'] for r in rs],[r['target_ids'] for r in rs],width=p['widths'][label])
  assert len(actual)==len(refs[label])==4 and all(a['tokens']==b['tokens'] for a,b in zip(actual,refs[label]))
  error=max(abs(x-y) for a,b in zip(actual,refs[label]) for x,y in zip(a['token_logps'],b['token_logps']));mean=abs(sum(a['full_logp']-b['full_logp'] for a,b in zip(actual,refs[label]))/4)
  checks[label]=dict(max_token_error=error,absolute_mean_span_error=mean,passed=error<=.001 and mean<=.001);assert checks[label]['passed']
 atomic_json(d/'own-model-TRAIN-calibration.json',checks);count=0;errors=0;started=time.monotonic()
 with (d/'scores.jsonl').open('x') as f:
  for name in p['inventories']:
   panel='QA' if name.startswith('QA-') else 'ordinary';selected=rows(output/name)[shard::4];assert len(selected)%4==0
   for index in range(0,len(selected),4):
    batch=selected[index:index+4];before=time.monotonic()
    try:
     values=score_spans(backend,[r['context_ids'] for r in batch],[r['target_ids'] for r in batch],width=p['widths'][panel]);exception=None
    except Exception as error:values=[None]*4;exception=f'{type(error).__name__}: {error}'
    for row,value in zip(batch,values):
     result={k:row.get(k) for k in ('item_id','panel','mode','donor_item_id','donor_status')};result.update(status='error' if exception else 'ok',context_tokens=len(row['context_ids']),target_ids=row['target_ids'],batch_seconds=time.monotonic()-before)
     if value:
      result.update(value)
      if panel=='QA':result.update(answer_content_logp=sum(value['token_logps'][:-1]),EOS_logp=value['token_logps'][-1])
     else:result['error']=exception;errors+=1
     f.write(json.dumps(result,allow_nan=False)+'\n');count+=1
    f.flush();print(json.dumps(dict(shard=shard,panel=panel,rows=count,execution_errors=errors)),flush=True)
 atomic_json(d/'completion.json',dict(status='COMPLETE_ALL_LIKELIHOOD_ATTEMPTS_NOT_READOUT',rows=count,execution_errors=errors,wall_seconds=time.monotonic()-started,device=device,source_protocol_sha256=sha256_file(output/'protocol.json'),scores_sha256=sha256_file(d/'scores.jsonl')))
 assert count==704
 if errors:raise RuntimeError('Scoring executions failed;preserve every error,no automaticretry orcomplete-estimate claim')

if __name__=='__main__':
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=['prepare','run']);a.add_argument('--output',type=Path,required=True);a.add_argument('--original',type=Path);a.add_argument('--shard',type=int);a.add_argument('--device',choices=['cpu','xla'],default='xla');args=a.parse_args()
 if args.command=='prepare':
  if args.original is None:a.error('--original complete generation directory required')
  prepare(args.original.resolve(),args.output.resolve())
 else:
  if args.shard is None:a.error('--shard required')
  run(args.output.resolve(),args.shard,args.device)
