"""Standard FP32 Hugging Face forwards on fixed original targets; frozen models."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gc,json,time
import numpy as np,torch
from weekend_thinking.artifacts import sha256_file
from weekend_thinking.pretrained import load_base
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--slot',type=int,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());C=a.plan.parent;D=Path(plan['output_directory']);torch.set_num_threads(24);torch.set_num_interop_threads(1);began=time.monotonic();deadline=datetime.fromisoformat(plan['CPU_stop_UTC'])
if sha256_file(Path(__file__))!=plan['worker_sha256'] or sha256_file(plan['source_file'])!=plan['source_sha256']:raise ValueError('Frozen CPU implementation and original source required')
readout=json.loads(Path(plan['native_readout']).read_text());inventory=readout['models'];records={v['item_id']:v for v in map(json.loads,Path(plan['source_file']).open())};selected=plan['item_ids'];model,_=load_base(plan['base_directory']);model.eval();model.requires_grad_(False);names=plan['worker_models'][a.slot];forwards=0
with torch.inference_mode():
 for name in names:
  if name!='B0':
   entry=inventory[name]
   if sha256_file(entry['checkpoint'])!=entry['checkpoint_sha256']:raise ValueError('Preserve original fixed native model')
   state=torch.load(entry['checkpoint'],map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(state['model'],strict=True);del state;gc.collect()
  scores=[]
  for index,item in enumerate(selected):
   if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Bounded CPU verification deadline')
   available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
   if available<512*2**30:raise RuntimeError('Preserve original host-memory floor')
   ids=torch.tensor([records[item]['ids']],dtype=torch.long)
   if ids.shape!=(1,528):raise ValueError('Full unchanged original window required')
   hidden=model.model(input_ids=ids[:,:-1],use_cache=False).last_hidden_state;parts=[]
   for first in range(0,527,64):
    logits=model.lm_head(hidden[:,first:first+64])[0].double();targets=ids[0,first+1:first+1+len(logits)];parts.append((logits.gather(1,targets[:,None]).squeeze(1)-torch.logsumexp(logits,-1)).cpu().numpy());del logits,targets
   scores.append(np.concatenate(parts));forwards+=1;del ids,hidden,parts
   if (index+1)%64==0:
    progress={'utc':datetime.now(timezone.utc).isoformat(),'slot':a.slot,'model':name,'documents':index+1,'of':len(selected),'actual_model_forwards':forwards,'wall_seconds':time.monotonic()-began};(D/f'slot-{a.slot}-progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
  array=np.asarray(scores,dtype=np.float64)
  if array.shape!=(len(selected),527) or not np.isfinite(array).all():raise ValueError('Every fixed document and target required')
  with (D/(name+'.npy')).open('xb') as output:np.save(output,array,allow_pickle=False)
  print('CPU_COMPLETE_FIXED_MODEL',name,flush=True)
(C/f'slot-{a.slot}-complete.json').write_text(json.dumps({'slot':a.slot,'models':names,'model_forwards':forwards,'wall_seconds':time.monotonic()-began},indent=2)+'\n')
