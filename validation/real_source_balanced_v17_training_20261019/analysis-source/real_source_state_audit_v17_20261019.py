"""Read-only full-state validation shared by protection and final study audit."""
from pathlib import Path
import json,hashlib,gc,math,random
import torch
from train.partial_state_driver import checkpoint_identity
from weekend_thinking.artifacts import canonical_hash

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


def expected_metrics(p,arm,data):
 """Independent two-pass permutation and native label-count formula."""
 order=[]
 for epoch in range((p['steps']*32+p['items']-1)//p['items']):
  indices=list(range(p['items']));random.Random(p['seed']+epoch).shuffle(indices);order+=indices
 order=order[:p['steps']*32];labels=0;expected=[]
 for step in range(1,p['steps']+1):
  batch=order[(step-1)*32:step*32]
  for pos,index in enumerate(batch,start=(step-1)*32):
   row=data[index];assert arm in ('source_direct','source_cot')
   n=16 if arm=='source_direct' else len(row['full_calculation_ids'])+2+16
   labels+=n+len(row['prefix_ids'])-1
  expected.append(dict(step=step,ordered_item_ids=[data[i]['item_id'] for i in batch],
    counters=dict(expressions=step*32,supervised_tokens=labels)))
 return order,expected


def audit_checkpoint(directory,p,arm,run_id,order,protocol_sha,expected_counter,base):
 """Hash every payload and inspect every active/master/moment tensor."""
 directory=Path(directory);m=read(directory/'COMPLETE.json');step=m['step']
 assert m['world_size']==8 and set(m['files'])=={'model.pt',*[f'rank-{i:05d}.pt' for i in range(8)]}
 assert m['metadata']['counters']==expected_counter
 for name,v in m['files'].items():assert (directory/name).stat().st_size==v['bytes'] and sha(directory/name)==v['sha256']
 weights=torch.load(directory/'model.pt',weights_only=True,map_location='cpu',mmap=True);active=weights['model']
 assert set(active)==set(base.state_dict()) and all(v.dtype==torch.bfloat16 and torch.isfinite(v).all() for v in active.values())
 assert active['lm_head.weight'].untyped_storage().data_ptr()==active['model.embed_tokens.weight'].untyped_storage().data_ptr()
 params=list(base.named_parameters());assert len(params)==310
 for rank in range(8):
  saved=torch.load(directory/f'rank-{rank:05d}.pt',weights_only=True,map_location='cpu',mmap=True)
  assert checkpoint_identity(m,weights,saved,p,arm,run_id,rank,canonical_hash(order),protocol_sha)==step
  opt=saved['optimizer'];assert set(opt['sharded_master_weights'])==set(opt['base_state'])==set(range(310))
  for index,(name,param) in enumerate(params):
   chunk=(param.shape[0]+7)//8;shape=(chunk,*param.shape[1:]);master=opt['sharded_master_weights'][index];state=opt['base_state'][index]
   assert master.dtype==torch.float32 and tuple(master.shape)==shape and torch.isfinite(master).all()
   assert set(state)=={'exp_avg','exp_avg_sq','step','max_exp_avg_sq'}
   for key in ('exp_avg','exp_avg_sq'):
    tensor=state[key];assert tensor.dtype==torch.float32 and tuple(tensor.shape)==shape and torch.isfinite(tensor).all()
   assert (state['exp_avg_sq']>=0).all() and float(state['step'])==step and state['step'].dtype==torch.float32
   assert state['max_exp_avg_sq'].dtype==torch.float32 and state['max_exp_avg_sq'].numel()==0
   start=rank*chunk;used=max(0,min(start+chunk,param.shape[0])-start)
   if used:assert equal(active[name][start:start+used],master[:used].bfloat16())
  del saved,opt,master,state,tensor
 result=dict(step=step,COMPLETE_sha256=sha(directory/'COMPLETE.json'),model_sha256=m['files']['model.pt']['sha256'],
   run_id=weights['run_id'],spec_hash=weights['spec_hash'],all_full_payload_hashes_finite_master_moment_shapes_counters_and_cast_bytes=True)
 del weights,active;gc.collect();return result
