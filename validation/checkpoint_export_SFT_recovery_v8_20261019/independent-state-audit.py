"""Read-only whole-question audit, after all actual SFT/recovery sessions close."""
from pathlib import Path
from datetime import datetime,timezone
import gc,gzip,hashlib,json,math,shutil
import torch
from train.partial_state_driver import batch_at,checkpoint_identity,order_for
from train.useful_note_precision import compare_distributions
from train.useful_note_recovery import optimizer_digest
from weekend_thinking.artifacts import canonical_hash
from weekend_thinking.pretrained import load_base
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/checkpoint-export-SFT-recovery-v8-v2'
O=R/'local/nvme-study/diagnostics/checkpoint-export-SFT-recovery-v8-v2';W=R/'local/worktrees/checkpoint-memory-v8'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
 if type(a)!=type(b):return False
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

assert read(C/'completion.json')['status']=='ALL_THREE_STAGES_COMPLETE_PENDING_INDEPENDENT_MEMORY_STATE_AND_RECOVERY_AUDIT'
for stage in ('control14','candidate-fresh12','candidate-resume14'):
 d=C/stage;assert read(d/'exit.json')['exit_code']==0 and read(d/'exit.json')['external_stop'] is None
 close=read(d/'closure.json');assert close['all_original_session_processes_absent'] and close['assigned_physical8_empty']
 assert len(read(d/'first-live.json')['actual4_on_physical8'])==4
 out=O/('control' if stage=='control14' else 'candidate');step=12 if stage=='candidate-fresh12' else 14
 assert close['checkpoint_COMPLETE_sha256']==sha(out/f'step-{step:07d}/COMPLETE.json')
 completion=(out/'recoveries/actual-restart-001/previous-completion.json' if step==12 else out/'completion.json')
 assert close['completion_sha256']==sha(completion)
protocols={k:read(C/f'protocol-{k}.json') for k in ('control','candidate')}
assert {k:v for k,v in protocols['control'].items() if k!='checkpoint_export_cleanup'}=={k:v for k,v in protocols['candidate'].items() if k!='checkpoint_export_cleanup'}
assert protocols['control']['checkpoint_export_cleanup'] is False and protocols['candidate']['checkpoint_export_cleanup'] is True
p=protocols['control'];assert all(sha(W/k)==v for k,v in p['source_files'].items())
model,tok=load_base(p['model_dir']);params=list(model.named_parameters());assert len(params)==310
data=rows(p['data_file'])[:128];order=order_for(p);base=read(p['base_protocol']);ref=torch.load(Path(base['native_reference_output'])/'next-logps.pt',weights_only=True,map_location='cpu')
numerical=[]
for directory in (O/'control',O/'candidate',O/'candidate/recoveries/actual-restart-001/initialization'):
 for rank in range(4):
  q=read(directory/f'qualification-rank-{rank:05d}.json');actual=torch.load(directory/f'forward-rank-{rank:05d}.pt',weights_only=True,map_location='cpu');check=compare_distributions(ref[q['reference_index']:q['reference_index']+1],actual)[0]
  assert check['passed'] and q['status']=='PASS'
  witness=torch.load(directory/f'initial-master-rank-{rank:05d}.pt',weights_only=True,map_location='cpu');assert equal(witness['expected_cpu'],witness['actual_fp32_masters'])
  numerical.append(check)
print('All12 actual native checks and initial master witnesses pass.',flush=True)
metrics={};checkpoints={};memory={}
for condition,p in protocols.items():
 out=O/condition;run=read(out/'run.json');protocol_file=C/f'protocol-{condition}.json';ms=rows(out/'metrics.jsonl');assert [m['step'] for m in ms]==list(range(1,15));total=0
 assert run['order_hash']==canonical_hash(order) and run['protocol_sha256']==sha(protocol_file)
 for i,m in enumerate(ms):
  assert m['ordered_item_ids']==[data[k]['item_id'] for k in order[i*32:(i+1)*32]]
  labels=0
  for rank in range(4):
   batch=batch_at(data,tok,p,order,'writer',i,rank);labels+=int((batch['note_close_mask']|batch['future_mask']).sum())
  total+=labels;assert m['counters']==dict(expressions=(i+1)*32,supervised_tokens=total)
  assert math.isfinite(m['loss']) and math.isfinite(m['gradient_norm']) and m['gradient_norm']>0
 metrics[condition]=ms;checkpoints[condition]={};memory[condition]={}
 for step in range(2,15,2):
  d=out/f'step-{step:07d}';manifest=read(d/'COMPLETE.json');assert manifest['metadata']['counters']==ms[step-1]['counters']
  for name,entry in manifest['files'].items():assert (d/name).stat().st_size==entry['bytes'] and sha(d/name)==entry['sha256']
  weights=torch.load(d/'model.pt',weights_only=True,map_location='cpu',mmap=True);active=weights['model']
  assert set(active)==set(model.state_dict()) and all(v.dtype==torch.bfloat16 and torch.isfinite(v).all() for v in active.values())
  assert active['lm_head.weight'].untyped_storage().data_ptr()==active['model.embed_tokens.weight'].untyped_storage().data_ptr()
  post=[]
  for rank in range(4):
   saved=torch.load(d/f'rank-{rank:05d}.pt',weights_only=True,map_location='cpu',mmap=True)
   assert checkpoint_identity(manifest,weights,saved,p,'writer',out.name,rank,canonical_hash(order),sha(protocol_file))==step
   opt=saved['optimizer'];assert set(opt['sharded_master_weights'])==set(opt['base_state'])==set(range(310))
   for i,(name,param) in enumerate(params):
    chunk=(param.shape[0]+3)//4;shape=(chunk,*param.shape[1:]);master=opt['sharded_master_weights'][i];state=opt['base_state'][i]
    assert master.dtype==torch.float32 and tuple(master.shape)==shape and torch.isfinite(master).all()
    assert set(state)=={'exp_avg','exp_avg_sq','step','max_exp_avg_sq'}
    for key in ('exp_avg','exp_avg_sq'):
     t=state[key];assert t.dtype==torch.float32 and tuple(t.shape)==shape and torch.isfinite(t).all()
    assert (state['exp_avg_sq']>=0).all() and float(state['step'])==step and state['step'].dtype==torch.float32
    assert state['max_exp_avg_sq'].dtype==torch.float32 and state['max_exp_avg_sq'].numel()==0
    start=rank*chunk;end=min(start+chunk,param.shape[0]);used=max(0,end-start)
    if used:assert equal(active[name][start:end],master[:used].bfloat16())
   obs=read(d/f'cpu-export-release-rank-{rank:05d}.json');assert obs['cleanup_enabled']==(condition=='candidate');post.append(obs['PSS_after_bytes'])
   del saved,opt,state,master,t
  memory[condition][str(step)]=dict(four_worker_post_export_PSS_bytes=post,total_post_export_PSS_bytes=sum(post))
  checkpoints[condition][str(step)]=dict(COMPLETE_sha256=sha(d/'COMPLETE.json'),all_files=manifest['files'],all_model_master_moment_shapes_finite_counters_and_cast_bytes=True)
  del weights,active;gc.collect()
 print(condition,'all7full checkpoints independently verified',flush=True)
loss_equal=all(a['loss']==b['loss'] and a['gradient_norm']==b['gradient_norm'] and a['counters']==b['counters'] and a['ordered_item_ids']==b['ordered_item_ids'] for a,b in zip(metrics['control'],metrics['candidate']))
ta=torch.load(O/'control/step-0000014/model.pt',weights_only=True,map_location='cpu',mmap=True);tb=torch.load(O/'candidate/step-0000014/model.pt',weights_only=True,map_location='cpu',mmap=True);model_equal=equal(ta['model'],tb['model']);del ta,tb
optimizer_equal=[];rng_equal=[];restored=[]
for rank in range(4):
 a=torch.load(O/f'control/step-0000014/rank-{rank:05d}.pt',weights_only=True,map_location='cpu',mmap=True);b=torch.load(O/f'candidate/step-0000014/rank-{rank:05d}.pt',weights_only=True,map_location='cpu',mmap=True)
 optimizer_equal.append(equal(a['optimizer'],b['optimizer']));rng_equal.append(equal(a['rng'],b['rng']));assert a['counters']==b['counters']
 report=read(O/f'candidate/recoveries/actual-restart-001/restored-rank-{rank:05d}.json');s=torch.load(O/f'candidate/step-0000012/rank-{rank:05d}.pt',weights_only=True,map_location='cpu',mmap=True)
 assert report['full_active_model_bytes_equal'] and report['full_optimizer_bytes_equal'] and report['tied_parameter_objects_preserved'] and report['optimizer_digest']==optimizer_digest(s['optimizer']);restored.append(dict(rank=rank,full_saved12_digest_recomputed=True,all_restored_bytes_and_tied_objects=True));del a,b,s;gc.collect()
growth={k:(v['12']['total_post_export_PSS_bytes']-v['4']['total_post_export_PSS_bytes'])/4/2**30 for k,v in memory.items()}
gates=dict(control_growth_reproduced=growth['control']>4,candidate_growth_below_limit=growth['candidate']<2,all14_step_losses_and_norms_exact=loss_equal,terminal_model_bytes_exact=model_equal,all_four_terminal_optimizer_states_exact=all(optimizer_equal),all_four_saved_terminal_CPU_device_RNG_exact=all(rng_equal),all_four_actual_saved12_restored_digests_exact=len(restored)==4)
archive=R/'local/backups/engineering/checkpoint-export-SFT-recovery-v8-v2';need=24*2**30+4*2**30
assert shutil.disk_usage(R).free>35*2**30+need;archive.mkdir(parents=True,exist_ok=False);archived={}
for label,source in [('control',C),('output',O)]:
 for f in source.rglob('*'):
  if not f.is_file() or '__pycache__' in f.parts:continue
  if f.suffix=='.pt' and f.parent.name.startswith('step-') and (f.parent.name!='step-0000014' or f.parent.parent.name=='control' and f.name!='model.pt'):continue
  dest=archive/label/f.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest);digest=sha(f);assert sha(dest)==digest;archived[str(dest.relative_to(archive))]=dict(bytes=f.stat().st_size,sha256=digest)
write(archive/'ARCHIVE.json',dict(status='CANDIDATE_FULL14_CONTROL_MODEL14_AND_ALL_SMALL_EVIDENCE_EBS_READBACK',files=archived))
public=R/'validation/checkpoint_export_SFT_recovery_v8_20261019';public.mkdir(exist_ok=False)
for name,ms in metrics.items():
 with gzip.GzipFile(filename=str(public/f'{name}-all14-update-metrics.jsonl.gz'),mode='wb',mtime=0) as f:
  for row in ms:f.write((json.dumps(row,separators=(',',':'))+'\n').encode())
report=dict(schema='checkpoint-export-SFT-recovery-v8-v1',status='PASS_ACTUAL_SFT_MEMORY_STATE_AND_RESTART' if all(gates.values()) else 'FAIL_ACTUAL_SFT_QUALIFICATION_GATE',finished_utc=datetime.now(timezone.utc).isoformat(),gates=gates,grouped_late_PSS_GiB_per_full_save=growth,post_export_memory=memory,checkpoints=checkpoints,actual_native_checks=numerical,restored_saved12=restored,source_commit='fb72065',source_files=p['source_files'],plan_sha256=sha(C/'plan.json'),audit_script_sha256=sha(__file__),EBS_archive_receipt_sha256=sha(archive/'ARCHIVE.json'),limits=['128TRAIN mechanics,14updates,world4;no capability result or originalworld32OOM/indefinite-memory qualification.','Control14uninterrupted versus candidate12/fresh-process restore/two further updates; exact trajectory/RNG parity claimed only for gates actually passing.','All original full states retained on local SSD;candidate full14/control model14/all small evidence EBS readback. EBS remains same-instance.','Candidate remains isolated from immutable57scientific study.'])
write(C/'independent-audit.json',report);write(R/'validation/checkpoint_export_SFT_recovery_v8_20261019.json',report)
print(json.dumps(dict(status=report['status'],gates=gates,grouped_late_PSS_GiB_per_full_save=growth),indent=2),flush=True)
