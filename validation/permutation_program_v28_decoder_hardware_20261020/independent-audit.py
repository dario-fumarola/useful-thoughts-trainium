"""Independent frozen TRAIN decoder/native/cache/raw-token and CPU greedy audit."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
from collections import Counter
import json,hashlib,os,shutil
import torch
from train.permutation_program_answer_driver import ROOT
from train.evaluate_local import setup_model
from train.useful_note_precision import compare_distributions
from weekend_thinking.core import policy_log_probs
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28-decode';assert ROOT==W
C=R/'local/diagnostics/permutation-program-v28-TRAIN4-decode-qualification-v1';O=R/'local/nvme-cpt/evaluation/permutation-program-v28-B0-TRAIN4-decode-qualification-v1'
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
 with Path(p).open('rb') as f:os.fsync(f.fileno())
p=read(O/'protocol.json');assert read(C/'exit.json')['exit_code']==0 and read(C/'closure.json')['all_original_session_processes_absent']
assert all(sha(W/k)==v for k,v in p['source_files'].items()) and all(sha(O/k)==v for k,v in p['files'].items())
record=read(O/'answers/run.json');data=rows(O/'answers/attempts.jsonl')
assert record['status']=='complete_all_attempts' and record['rows']==len(data)==28 and sha(O/'answers/attempts.jsonl')==record['attempts_sha256']
reference=torch.load(O/'native-reference.pt',map_location='cpu',weights_only=True)
actual=torch.load(O/'answers/native-logps.pt',map_location='cpu',weights_only=True)
native=compare_distributions(reference[:1],actual)[0];assert native['passed']
caches={}
for mode in p['modes']:
 vectors=torch.load(O/f'answers/cache-prefill-{mode}.pt',map_location='cpu',weights_only=True)
 value=compare_distributions(vectors['ordinary'],vectors['cached'])[0];assert value['passed'] and value['max_logp_error']<=1e-3
 caches[mode]=value
print('Actual native distribution and all7 cached/ordinary-prefill vectors independently PASS.',flush=True)
torch.set_num_threads(16)
backend,tok,_,signature,_=setup_model(SimpleNamespace(model_dir=Path(p['model_dir']),checkpoint=None,device='cpu'));assert signature==p['signature']
lookup={(r['mode'],r['item_id']):r for r in data};assert len(lookup)==28
stops=Counter();prompts={}
for mode in p['modes']:
 current=rows(O/f'prompts-{mode}.jsonl');assert [q['item_id'] for q in current]==p['ordered_item_ids']
 for q in current:
  key=(mode,q['item_id']);prompts[key]=q;v=lookup[key]
  assert v['status']=='ok' and v['stop_type']!='error' and v['n_actions']==len(v['actions']) and v['prompt_tokens']==len(q['prompt_ids'])
  if v['stop_type'].startswith('token:'):
   assert int(v['stop_type'][6:])==v['actions'][-1] and v['actions'][:-1]==v['content_ids']
  else:assert v['actions']==v['content_ids']
  if v['stop_type']=='length_cap':assert len(v['actions'])==128
  assert tok.decode(v['content_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)==v['raw_text']
  assert v['text']==v['raw_text']
  assert set(v['content_ids']).isdisjoint(backend.banned_ids)
  stops[v['stop_type']]+=1
# Fixed modes and longest prompt, chosen without generation correctness.
checks=[]
with torch.no_grad():
 for mode in ('direct','trace','pause','partial'):
  choices=[q for (m,_),q in prompts.items() if m==mode];q=max(choices,key=lambda v:(len(v['prompt_ids']),v['item_id']))
  v=lookup[(mode,q['item_id'])];sequence=list(q['prompt_ids']);count=0
  for expected in v['actions'][:8]:
   x=torch.tensor([sequence],dtype=torch.long);h=backend.hidden(x);z=backend.model.lm_head(h[:,-1]).float()
   lp=policy_log_probs(z,1.,backend.banned_ids);got=int(lp.argmax(-1)[0]);count+=1
   if got!=expected:
    top=torch.topk(lp[0],3);write(C/'CPU-greedy-mismatch.json',dict(mode=mode,item_id=q['item_id'],position=count,expected_native=expected,CPU_argmax=got,CPU_top_ids=top.indices.tolist(),CPU_top_logps=top.values.tolist()))
    raise RuntimeError('Independent ordinary CPU greedy differs; no automatic acceptance or retry')
   if got in (151643,151645):break
   sequence.append(got)
  checks.append(dict(mode=mode,item_id=q['item_id'],CPU_full_prefix_actions_equal=count))
  print('Fixed CPU noncached greedy PASS:',mode,count,flush=True)
A=R/'local/backups/engineering'/C.name/'ACTUAL';A.mkdir(parents=True,exist_ok=False);archive={}
for label,folder in [('actual',O),('control',C)]:
 for src in folder.rglob('*'):
  if not src.is_file():continue
  dst=A/label/src.relative_to(folder);dst.parent.mkdir(parents=True,exist_ok=True)
  assert shutil.disk_usage(R).free>35*2**30+src.stat().st_size
  shutil.copyfile(src,dst)
  with dst.open('rb') as f:os.fsync(f.fileno())
  assert sha(src)==sha(dst);archive[str(dst.relative_to(A))]=dict(bytes=dst.stat().st_size,sha256=sha(dst))
write(A/'ARCHIVE.json',dict(status='ALL_SMALL_ACTUAL_DECODER_EVIDENCE_READBACK_FSYNC',files=archive))
report=dict(status='PASS_ACTUAL_TRAIN_PROGRAM_DECODER_NATIVE_CACHE_RAW_TOKENS_AND_CPU_GREEDY',utc=datetime.now(timezone.utc).isoformat(),attempts=28,errors=0,stop_counts=dict(stops),native=native,cache_checks=caches,CPU_noncached_greedy=checks,source_files=p['source_files'],protocol_sha256=sha(O/'protocol.json'),audit_source_sha256=sha(__file__),archive_sha256=sha(A/'ARCHIVE.json'),limits=['UnchangedB0,4fixed TRAIN questions,not trained capability or a held result.','The native check uses one canonical context; four CPU greedy checks cover at most8actions each.','All caps and formatting failures are retained; no accuracy/cap gate for the untrained format.','Partial/donor controls are supplied offline interventions derived from available inputs.'])
write(C/'independent-audit.json',report);print(json.dumps({k:report[k] for k in ['status','attempts','errors','stop_counts']}),flush=True)
