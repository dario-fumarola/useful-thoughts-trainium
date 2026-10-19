"""Independent native decode,designated fields,metrics and bootstrap verification."""
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,math,re,string
import numpy as np
import torch
from tokenizers import Tokenizer
from train.useful_note_precision import compare_distributions
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-QA-v21-study-v1';F=R/'validation/document_QA_v21_complete_20261019.json';P=F.with_suffix('')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(F);assert r['status']=='COMPLETE_SEVEN_MODEL_TWO_SEED_DOCUMENT_QA_REFERENCE'
def payload(name):
 p=P/name;assert sha(p)==r['payloads'][str(p.relative_to(R))]['sha256'];vs=[json.loads(l) for l in gzip.decompress(p.read_bytes()).splitlines()];assert len(vs)==r['payloads'][str(p.relative_to(R))]['rows'];return vs

def norm(v):
 v=''.join(x for x in v.lower() if x not in string.punctuation)
 return ' '.join(re.sub(r'\b(a|an|the)\b',' ',v).split())

def f1(a,b):
 a,b=norm(a).split(),norm(b).split()
 if not a or not b:return float(a==b)
 return 2*sum((Counter(a)&Counter(b)).values())/(len(a)+len(b))

def answer(v):
 if v['status']!='ok':return 'execution_error',''
 text=v['text'];terminal=v['stop_type'].startswith('token:') or v['stop_type']=='length_cap'
 if v['parser_mode']=='prior':
  fields=list(re.finditer(r'(?im)^[ \t]*Final answer:[ \t]*',text))
  if len(fields)!=1:return ('unsupported' if terminal else 'pending'),''
  text=text[fields[0].end():]
 text=text.lstrip();lines=text.split('\n',1)
 if len(lines)==2 and lines[0].strip():return 'complete',lines[0].strip()
 if v['stop_type'].startswith('token:') and text.strip():return 'complete',text.strip()
 return ('unsupported' if terminal else 'pending'),''

gold={v['item_id']:v for v in payload('fixed-held-gold-metadata.jsonl.gz')};assert len(gold)==512
attempts={};LL={};ordinary={};n=0;ln=0
tok=Tokenizer.from_file(str(R/'local/assets/qwen3-1.7b-base/tokenizer.json'))
for model,tables in r['tables'].items():
 vs=payload(model+'-all-generation-attempts.jsonl.gz');assert len(vs)==512*len(tables);attempts[model]={m:{} for m in tables}
 for v in vs:
  assert v['item_id'] not in attempts[model][v['mode']];status,a=answer(v)
  assert v['parsed']['status']==status
  if status=='complete':assert v['parsed']['answer']==a
  canonical=gold[v['item_id']]['answer'];aliases=gold[v['item_id']]['answer_aliases'];complete=status=='complete'
  expected=dict(canonical_em=int(complete and norm(a)==norm(canonical)),canonical_f1=f1(a,canonical) if complete else 0.,attributed_alias_em=int(complete and any(norm(a)==norm(b) for b in [canonical,*aliases])),attributed_alias_f1=max(f1(a,b) for b in [canonical,*aliases]) if complete else 0.)
  assert all(v[k]==val for k,val in expected.items());assert tok.decode(v['content_ids'],skip_special_tokens=False)==v['raw_text']
  assert v['n_actions']==len(v['actions'])<=256
  if v['status']=='ok':
   terminal=v['stop_type'].startswith('token:');assert v['content_ids']==(v['actions'][:-1] if terminal else v['actions'])
   if terminal:assert v['stop_type']==f"token:{v['actions'][-1]}" and v['actions'][-1] in (151643,151645)
   if v['stop_type']=='length_cap':assert len(v['actions'])==256
  attempts[model][v['mode']][v['item_id']]=v
 for mode,table in tables.items():
  selected=list(attempts[model][mode].values());assert len(selected)==512 and {v['item_id'] for v in selected}==set(gold)
  assert table['correct']==sum(v['canonical_em'] for v in selected)
  for metric in ('canonical_em','canonical_f1','attributed_alias_em','attributed_alias_f1'):assert abs(table[metric]-sum(v[metric] for v in selected)/512)<1e-12
  assert table['parsed_statuses']==dict(Counter(v['parsed']['status'] for v in selected)) and table['stop_types']==dict(Counter(v['stop_type'] for v in selected))
  assert table['capped']==sum(v['stop_type']=='length_cap' for v in selected) and table['execution_errors']==sum(v['status']!='ok' for v in selected)
  assert table['generated_actions']==sum(v['n_actions'] for v in selected)
 n+=len(vs)
 ls=payload(model+'-all-likelihood-rows.jsonl.gz');assert len(ls)==2816;LL[model]={m:{} for m in ('direct','empty','own_replay','other_replay','oracle')};ordinary[model]={}
 for v in ls:
  assert v['status']=='ok' and len(v['target_ids'])==len(v['token_logps'])==v['tokens'] and all(math.isfinite(x) for x in v['token_logps'])
  assert abs(v['full_logp']-math.fsum(v['token_logps']))<1e-10
  if v['panel']=='QA':
   assert v['target_ids'][-1]==151643 and v['EOS_logp']==v['token_logps'][-1] and abs(v['answer_content_logp']-math.fsum(v['token_logps'][:-1]))<1e-10;LL[model][v['mode']][v['item_id']]=v
  else:assert v['tokens']==32;ordinary[model][v['item_id']]=-v['full_logp']/32
 for mode,values in LL[model].items():
  assert len(values)==512
  for field in ('full_logp','answer_content_logp','EOS_logp'):assert abs(r['absolute_likelihood'][model][mode][field]-sum(v[field] for v in values.values())/512)<1e-10
 assert len(ordinary[model])==256 and abs(r['absolute_likelihood'][model]['ordinary_raw']['mean_token_NLL']-sum(ordinary[model].values())/256)<1e-10
 ln+=len(ls)
indices={n:np.random.default_rng(2026101921).integers(0,n,size=(20000,n)) for n in (256,512)}
def verify(a,b,expected):
 assert a.keys()==b.keys();delta=np.array([b[k]-a[k] for k in sorted(a)],dtype=np.float64);ci=np.quantile(delta[indices[len(delta)]].mean(1),[.025,.975])
 assert expected['items']==len(delta) and expected['bootstrap_seed']==2026101921 and expected['bootstrap_draws']==20000
 assert abs(delta.mean()-expected['difference_b_minus_a'])<1e-12 and np.allclose(ci,expected['ci95'],atol=1e-12,rtol=0)
def ag(a,am,b,bm,expected,metric):verify({k:v[metric] for k,v in attempts[a][am].items()},{k:v[metric] for k,v in attempts[b][bm].items()},expected)
for seed,s in r['seeds'].items():
 d=f'qa_direct-seed{seed}';p=f'qa_prior-seed{seed}';o=f'qa_post-seed{seed}'
 pairs=dict(post_direct_minus_direct_direct=(d,'direct',o,'direct'),prior_prior_minus_prior_empty=(p,'empty',p,'prior'),prior_prior_minus_direct_direct=(d,'direct',p,'prior'),prior_prior_minus_direct_prior=(d,'prior',p,'prior'),prior_prior_minus_direct_reference_cot=(d,'reference_cot',p,'prior'),prior_prior_minus_B0_reference_cot=('B0','reference_cot',p,'prior'),prior_direct_minus_direct_direct=(d,'direct',p,'direct'),prior_prior_minus_prior_direct=(p,'direct',p,'prior'),prior_prior_minus_question_only=(p,'question_only_prior',p,'prior'),prior_prior_minus_bridge_removed=(p,'bridge_removed_prior',p,'prior'),prior_own_replay_minus_other=(p,'other_replay',p,'own_replay'),post_reference_minus_direct_reference=(d,'reference_cot',o,'reference_cot'),direct_reference_minus_B0_reference=('B0','reference_cot',d,'reference_cot'))
 for name,pair in pairs.items():
  ag(*pair,s['primary_canonical_EM_contrasts'][name],'canonical_em');ag(*pair,s['secondary_canonical_F1_contrasts'][name],'canonical_f1')
 lps=dict(post_direct_minus_direct_direct=(d,'direct',o,'direct'),prior_own_minus_prior_empty=(p,'empty',p,'own_replay'),prior_own_minus_other=(p,'other_replay',p,'own_replay'),prior_oracle_minus_direct=(p,'direct',p,'oracle'),prior_own_minus_direct_direct=(d,'direct',p,'own_replay'))
 for field,cs in s['secondary_likelihood_contrasts'].items():
  for name,(a,am,b,bm) in lps.items():verify({k:v[field] for k,v in LL[a][am].items()},{k:v[field] for k,v in LL[b][bm].items()},cs[name])
 for name,a,b in [('prior_minus_direct',d,p),('post_minus_direct',d,o),('prior_minus_B0','B0',p),('post_minus_B0','B0',o),('direct_minus_B0','B0',d)]:verify(ordinary[a],ordinary[b],s['ordinary_NLL_cost'][name])
 cs=s['primary_canonical_EM_contrasts'];passed=lambda name:cs[name]['difference_b_minus_a']>=.03 and cs[name]['ci95'][0]>0
 assert s['training_benefit_gate_pass']==passed('post_direct_minus_direct_direct')
 inference=passed('prior_prior_minus_prior_empty') and passed('prior_prior_minus_direct_direct');assert s['inference_benefit_gate_pass']==inference
 practical=inference and all(passed(n) for n in ('prior_prior_minus_direct_prior','prior_prior_minus_direct_reference_cot','prior_prior_minus_B0_reference_cot'));assert s['practical_reasoning_supervision_gate_pass']==practical
for decision,field in [('training_benefit_both_seeds','training_benefit_gate_pass'),('inference_benefit_both_seeds','inference_benefit_gate_pass'),('practical_reasoning_supervision_both_seeds','practical_reasoning_supervision_gate_pass')]:assert r['decisions'][decision]==all(s[field] for s in r['seeds'].values())
# Recompute actual per-worker native/calibrated-cache distribution checks.
for g in r['generation_inventory']:
 O=Path(g['output']);p=read(O/'protocol.json');ref=torch.load(O/'native-reference.pt',weights_only=True,map_location='cpu');refs=read(O/'native-contexts.json')['refs']
 for shard in range(4):
  D=O/'answers'/f'shard-{shard}';actual=torch.load(D/'native-logps.pt',weights_only=True,map_location='cpu');a=compare_distributions(ref[shard%len(refs):shard%len(refs)+1],actual)[0];stored=read(D/'native.json');assert a==stored and a['passed']
  vectors=torch.load(D/'cache-prefill-logps.pt',weights_only=True,map_location='cpu');a=compare_distributions(vectors['ordinary'],vectors['cached'])[0];assert a==read(D/'cache-prefill.json') and a['max_logp_error']<=.001
assert n==r['attempts']==20992 and ln==r['likelihood_rows']==19712
out=dict(status='PASS_NATIVE_ACTIONS_FIELDS_CANONICAL_ALIAS_METRICS_VECTORS_BOOTSTRAPS_AND_GATES',utc=datetime.now(timezone.utc).isoformat(),generation_rows=n,likelihood_rows=ln,report_sha256=sha(F),checker_sha256=sha(__file__),limits='Independent parsing,normalization/F1,native decode and distribution recomputation/vectorizedresampling;no fresh model executions. No selection or error exclusions.')
with (P/'independent-readout-check.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2),flush=True)
