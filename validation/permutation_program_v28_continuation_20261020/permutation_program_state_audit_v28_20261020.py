"""Independent actual V28 metrics, full tensor state and native witness audit."""
from pathlib import Path
import gc,hashlib,json,math,random
import torch
from weekend_thinking.artifacts import canonical_hash
from train.partial_state_driver import checkpoint_identity
from train.useful_note_precision import compare_distributions


def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


def metrics(job,step):
 p=read(job['protocol']);data=[json.loads(x) for x in Path(p['data_file']).read_text().splitlines()]
 assert len(data)==32768 and sha(p['data_file'])==p['data_sha256']
 order=[]
 for epoch in range(4):
  indices=list(range(32768));random.Random(p['seed']+epoch).shuffle(indices);order.extend(indices)
 observed=[json.loads(x) for x in (Path(job['output'])/'metrics.jsonl').read_text().splitlines()]
 assert [v['step'] for v in observed]==list(range(1,step+1))
 labels=0
 for i,v in enumerate(observed):
  indices=order[i*32:(i+1)*32];assert v['ordered_item_ids']==[data[k]['item_id'] for k in indices]
  answer=160;work=0
  for index in indices:
   row=data[index];q=row['question'];node=q['start']
   for op in q['program']:node=q[op][node]
   assert row['answer']=='ABCDEFGHIJKLMNOP'[node]
   if (i*32//32768+index)%2 and job['arm']=='trace':work+=6*len(q['program'])
  c=v['loss_components'];assert c['answer_labels']==answer and c['work_labels']==work
  assert c['answer_weight']==c['work_weight']==1.
  assert abs(v['loss']-c['answer_mean_loss']-c['work_mean_loss'])<1e-5
  if job['arm']!='trace':assert c['work_mean_loss']==0.
  labels+=answer+work;assert v['counters']==dict(expressions=(i+1)*32,supervised_tokens=labels)
  assert all(math.isfinite(v[k]) for k in ('loss','gradient_norm','elapsed_seconds')) and v['gradient_norm']>0
 run=read(Path(job['output'])/'run.json')
 assert run['order_hash']==canonical_hash(order) and run['protocol_sha256']==sha(job['protocol']) and run['arm']==job['arm']
 return p,order,observed


def full_state(job,p,order,observed,base):
 out=Path(job['output']);step=observed[-1]['step'];d=out/f'step-{step:07d}';m=read(d/'COMPLETE.json')
 assert set(m['files'])=={'model.pt',*[f'rank-{i:05d}.pt' for i in range(4)]} and m['world_size']==4
 for name,v in m['files'].items():assert (d/name).stat().st_size==v['bytes'] and sha(d/name)==v['sha256']
 weights=torch.load(d/'model.pt',weights_only=True,map_location='cpu',mmap=True);active=weights['model'];params=list(base.named_parameters())
 assert len(params)==310 and set(active)==set(base.state_dict())
 assert all(v.dtype==torch.bfloat16 and torch.isfinite(v).all() for v in active.values())
 assert active['lm_head.weight'].untyped_storage().data_ptr()==active['model.embed_tokens.weight'].untyped_storage().data_ptr()
 assert any(not equal(active[name],param.detach().bfloat16()) for name,param in params)
 for rank in range(4):
  saved=torch.load(d/f'rank-{rank:05d}.pt',weights_only=True,map_location='cpu',mmap=True)
  assert checkpoint_identity(m,weights,saved,p,job['arm'],out.name,rank,canonical_hash(order),sha(job['protocol']))==step
  assert saved['counters']==observed[-1]['counters']
  opt=saved['optimizer'];assert set(opt['sharded_master_weights'])==set(opt['base_state'])==set(range(310))
  for index,(name,param) in enumerate(params):
   chunk=(param.shape[0]+3)//4;shape=(chunk,*param.shape[1:]);master=opt['sharded_master_weights'][index];state=opt['base_state'][index]
   assert master.dtype==torch.float32 and tuple(master.shape)==shape and torch.isfinite(master).all()
   assert set(state)=={'exp_avg','exp_avg_sq','step','max_exp_avg_sq'}
   for key in ('exp_avg','exp_avg_sq'):
    value=state[key];assert value.dtype==torch.float32 and tuple(value.shape)==shape and torch.isfinite(value).all()
   assert (state['exp_avg_sq']>=0).all() and float(state['step'])==step and state['step'].dtype==torch.float32
   assert state['max_exp_avg_sq'].dtype==torch.float32 and state['max_exp_avg_sq'].numel()==0
   start=rank*chunk;used=max(0,min(start+chunk,param.shape[0])-start)
   if used:assert equal(active[name][start:start+used],master[:used].bfloat16())
  del saved,opt,master,state,value
 result=dict(step=step,COMPLETE_sha256=sha(d/'COMPLETE.json'),model_sha256=m['files']['model.pt']['sha256'],all_model_master_moments_shapes_counters_and_cast_bytes_verified=True)
 del weights,active;gc.collect();return result


def native(job,p):
 base=read(p['base_protocol']);reference=torch.load(Path(base['native_reference_output'])/'next-logps.pt',weights_only=True,map_location='cpu');out=Path(job['output']);checks=[]
 for rank in range(4):
  receipt=read(out/f'qualification-rank-{rank:05d}.json');actual=torch.load(out/f'forward-rank-{rank:05d}.pt',weights_only=True,map_location='cpu')
  v=compare_distributions(reference[rank:rank+1],actual)[0];assert v['passed'] and receipt['status']=='PASS';checks.append(v)
  witness=torch.load(out/f'initial-master-rank-{rank:05d}.pt',weights_only=True,map_location='cpu');assert equal(witness['expected_cpu'],witness['actual_fp32_masters'])
 return checks
