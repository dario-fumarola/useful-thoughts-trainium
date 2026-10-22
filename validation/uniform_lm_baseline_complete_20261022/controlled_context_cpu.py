from pathlib import Path
import gc,json,time
import numpy as np,torch
from weekend_thinking.pretrained import load_base
from weekend_thinking.stats import cluster_bootstrap
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/uniform-lm-native-v1';plan=json.loads((C/'controlled-context-plan.json').read_text());torch.set_num_threads(plan['CPU_threads']);torch.set_num_interop_threads(1);started=time.monotonic();rows={v['item_id']:v for v in map(json.loads,(R/'local/data/uniform-lm-baseline-v1/CONFIRM.jsonl').open())};selected=json.loads((C/'complete-readout.json').read_text())['selected_checkpoint'];model,_=load_base(R/'local/assets/qwen3-1.7b-base');model.eval();model.requires_grad_(False);records=[]
with torch.inference_mode():
 for name in plan['models']:
  if name!='B0':
   state=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(state['model'],strict=True);del state;gc.collect()
  for item in plan['item_ids']:
   original=rows[item]['ids']
   for context in plan['context_lengths']:
    if time.monotonic()-started>plan['maximum_wall_seconds']:raise TimeoutError('Controlled-context CPU budget')
    available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
    if available<512*2**30:raise RuntimeError('Preserve host-memory floor')
    seq=torch.tensor([original[512-context:528]],dtype=torch.long);hidden=model.model(input_ids=seq[:,:-1],use_cache=False).last_hidden_state;logits=model.lm_head(hidden[:,-16:])[0].double();target=seq[0,-16:];lp=logits.gather(1,target[:,None]).squeeze(1)-torch.logsumexp(logits,-1);record={'model':name,'item_id':item,'document_id':rows[item]['document_id'],'context_tokens':context,'same_target16_log_probability':float(lp.sum())};records.append(record)
    with (C/'controlled-context-forwards.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    del seq,hidden,logits,target,lp
   print(name,len([r for r in records if r['model']==name])//len(plan['context_lengths']),'of',len(plan['item_ids']),flush=True)
lookup={(v['model'],v['item_id'],v['context_tokens']):v['same_target16_log_probability'] for v in records};values=[]
for item in plan['item_ids']:
 gains=[lookup['warmup_cosine-lr1e-05',item,c]-lookup['B0',item,c] for c in plan['context_lengths']];values.append(gains+[v-gains[-1] for v in gains[:-1]])
b=cluster_bootstrap(np.asarray(values),[rows[i]['document_id'] for i in plan['item_ids']],unit='unique source document',repeats=5000,seed=17);result={'status':'COMPLETE_CONTROLLED_CONTEXT_CPU_DIAGNOSTIC','documents':32,'CPU_model_forwards':len(records),'new_accelerator_calls':0,'context_conditions':[{'available_prefix_tokens':c,'gain_nats_per_same16_token_target':float(b['mean'][i]),'CI95':b['CI95'][i].tolist()} for i,c in enumerate(plan['context_lengths'])],'interactions_shorter_gain_minus_512_gain':[{'available_prefix_tokens':c,'difference_nats_per_same16_targets':float(b['mean'][5+i]),'CI95':b['CI95'][5+i].tolist()} for i,c in enumerate(plan['context_lengths'][:-1])],'statistics':b['metadata'],'wall_seconds':time.monotonic()-started,'scope':plan['scope']};(C/'controlled-context-readout.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
