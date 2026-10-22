"""TRAIN-only first-batch source/auxiliary gradient geometry; no parameter update."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,os,sys,time
import torch
from torch.nn import functional as F
from weekend_thinking.pretrained import load_base
from train.scaffold_learning import training_pack
R=Path('/home/ubuntu/dario');C=Path(__file__).parent;S=R/'local/diagnostics/scaffold-learning-overnight-v1';deadline=datetime(2026,10,22,11,42,tzinfo=timezone.utc);start=time.monotonic();torch.set_num_threads(24);torch.set_num_interop_threads(1)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def guard():
 if datetime.now(timezone.utc)>=deadline or time.monotonic()-start>1800:raise TimeoutError('Bounded optional derivative diagnosis')
 available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
 if available<544*2**30:raise RuntimeError('Protect existing host-memory floor and buffers')
p=json.loads((S/'scaffold_raw-seed11-protocol.json').read_text());data=Path(p['data_file']);source={r['item_id']:r for r in map(json.loads,data.open())};first={}
for arm in ['raw','format','trace']:
 h=Path('/dev/shm/scaffold-learning-overnight-v1/science')/('scaffold_'+arm+'-seed11')/'metrics.jsonl';first[arm]=json.loads(next(h.open()))
ids=first['raw']['ordered_item_ids']
if len(ids)!=272 or any(first[a]['ordered_item_ids']!=ids for a in first):raise ValueError('The exact same first native batch is required')
rows=[source[i] for i in ids];calculations=[r for r in rows if r['kind']=='calculation'];plan={'status':'FROZEN_BEFORE_DERIVATIVE_RESULTS','utc':datetime.now(timezone.utc).isoformat(),'source_file_sha256':sha(data),'native_protocol_sha256':sha(S/'scaffold_raw-seed11-protocol.json'),'native_order':11,'batch_items':ids,'source_windows':len(rows),'calculation_windows':len(calculations),'base_model_id':'Qwen/Qwen3-1.7B-Base','CPU_precision':'FP32 model/gradients, FP64 streamed geometry','source_loss_coefficient':'1/(272*527) per original target','auxiliary_loss_coefficient':'1/272 times each calculation-row mean; zero for ordinary rows','optimizer_direction_constants':{'epsilon':1e-8,'clip_norm':1.0,'source':'qualified native partial_state_driver.py optimizer constructor'},'worker_source_sha256':sha(Path(__file__)),'native_initial_preclip_norms':{k:first[k]['gradient_norm'] for k in first},'model_updates':0,'new_accelerator_calls':0,'scope':'One inspected TRAIN batch at B0. This cannot identify the unique cause of the 128-update held-out result. Initial-gradient geometry and zero-moment AdamW directional calculations are not a training ablation or proof of future improvement.'};(C/'prospective-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
model,tokenizer=load_base(R/'local/assets/qwen3-1.7b-base');model.float();model.eval();model.requires_grad_(True);parameters=dict(model.named_parameters())
def fingerprint():
 h=hashlib.sha256()
 for name,v in parameters.items():h.update(name.encode());h.update(v.detach().contiguous().numpy().tobytes())
 return h.hexdigest()
before=fingerprint();loss_source=0.;calls=0;source_started=time.monotonic()
for i,row in enumerate(rows):
 guard();x=torch.tensor([row['ids']],dtype=torch.long);hidden=model.model(input_ids=x[:,:-1],use_cache=False).last_hidden_state;logits=model.lm_head(hidden)[0].float();loss=F.cross_entropy(logits,x[0,1:])/272
 if not torch.isfinite(loss):raise ValueError('Finite source derivative required')
 loss.backward();loss_source+=float(loss.detach());calls+=1;del x,hidden,logits,loss
 if (i+1)%8==0:print(json.dumps({'phase':'source','documents':i+1,'of':272,'wall_seconds':time.monotonic()-source_started}),flush=True)
 if i==7:
  per=(time.monotonic()-source_started)/8;remaining=per*(len(rows)-8+2*len(calculations))*1.2+120;left=min(1800-(time.monotonic()-start),(deadline-datetime.now(timezone.utc)).total_seconds());(C/'measured-runtime.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'seconds_per_source_backward':per,'projected_remaining_seconds':remaining,'seconds_available':left},indent=2)+'\n')
  if remaining>left:raise TimeoutError('The complete fixed derivative batch cannot fit; do not shrink by its outcomes')
gs={name:(v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v)) for name,v in parameters.items()};source_loss=loss_source;results={};eps=1e-8;clip=1.0;
def geometry(aux):
 ss=aa=sa=0.
 for name,a in gs.items():
  b=aux[name];a=a.reshape(-1);b=b.reshape(-1)
  for k in range(0,a.numel(),2**22):
   x=a[k:k+2**22].double();y=b[k:k+2**22].double();ss+=float((x*x).sum());aa+=float((y*y).sum());sa+=float((x*y).sum())
 combined=math.sqrt(max(0.,ss+aa+2*sa));s_norm=math.sqrt(ss);a_norm=math.sqrt(aa);c_s=min(1.,clip/(s_norm+1e-12));c_t=min(1.,clip/(combined+1e-12));raw=total=0.
 for name,a in gs.items():
  b=aux[name];a=a.reshape(-1);b=b.reshape(-1)
  for k in range(0,a.numel(),2**22):
   x=a[k:k+2**22].double();y=b[k:k+2**22].double();r=c_s*x;t=c_t*(x+y);raw+=float((x*r/(r.abs()+eps)).sum());total+=float((x*t/(t.abs()+eps)).sum())
 values={'source_gradient_norm':s_norm,'auxiliary_gradient_norm':a_norm,'source_auxiliary_dot':sa,'source_auxiliary_cosine':sa/(s_norm*a_norm) if a_norm else None,'combined_gradient_norm':combined,'source_clip_multiplier':c_s,'combined_clip_multiplier':c_t,'source_dot_zero_moment_AdamW_direction_raw':raw,'source_dot_zero_moment_AdamW_direction_combined':total,'zero_moment_AdamW_source_direction_ratio':total/raw}
 if any(not math.isfinite(v) for v in values.values() if v is not None):raise ValueError('Finite complete geometry required')
 return values
for arm in ['scaffold_format','scaffold_trace']:
 model.zero_grad(set_to_none=True);value=0.;labels=0
 for row in calculations:
  guard();pack=training_pack(row,tokenizer,arm);mask=pack['work_mask'][0];pos=pack['target_positions'][0][mask]-1;y=pack['labels'][0][mask];x=pack['input_ids'];hidden=model.model(input_ids=x[:,:-1],use_cache=False).last_hidden_state[0];logits=model.lm_head(hidden[pos]).float();loss=F.cross_entropy(logits,y)/272
  if not torch.isfinite(loss):raise ValueError('Finite work derivative required')
  loss.backward();value+=float(loss.detach());labels+=len(y);calls+=1;del x,hidden,logits,loss,pack
 aux={name:(v.grad if v.grad is not None else torch.zeros_like(v)) for name,v in parameters.items()};results[arm]={'work_loss_contribution':value,'work_labels':labels,**geometry(aux)};print('COMPLETE_AUXILIARY_GEOMETRY',arm,json.dumps(results[arm]),flush=True);del aux
model.zero_grad(set_to_none=True);after=fingerprint()
if before!=after:raise ValueError('A derivative diagnosis must not change parameters')
result={'status':'COMPLETE_TRAIN_ONLY_INITIAL_GRADIENT_GEOMETRY','utc':datetime.now(timezone.utc).isoformat(),'plan':plan,'source_mean_CE':source_loss,'results':results,'parameter_fingerprint_before':before,'parameter_fingerprint_after':after,'parameters_unchanged':True,'actual_forward_backward_calls':calls,'optimizer_steps':0,'new_accelerator_calls':0,'actual_wall_seconds':time.monotonic()-start,'counterfactual_direction':'Fresh zero-moment AdamW direction with common LR omitted and common weight decay cancelling in differences; no update is applied. FP32 CPU gradients differ from native BF16 gradients. A local first-order calculation does not establish later optimization or held-out behavior.'};(C/'complete-readout.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
