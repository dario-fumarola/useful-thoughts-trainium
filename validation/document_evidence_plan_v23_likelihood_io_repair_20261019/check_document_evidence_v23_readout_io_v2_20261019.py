"""Independent designated-field,metric,pointer,bootstrap and decision check.

No fresh model executions. Derived FP64 summaries use absolute1e-10,declared
before outcomes;science thresholds and native device bounds are unchanged.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import gzip,hashlib,json,math,re,string
import numpy as np
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
F=R/'validation/document_evidence_plan_v23_pilot_complete_20261019.json';P=F.with_suffix('')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def norm(value):
 return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in value.lower() if c not in string.punctuation)).split())

def f1(a,b):
 a,b=norm(a).split(),norm(b).split()
 return float(a==b) if not a or not b else 2*sum((Counter(a)&Counter(b)).values())/(len(a)+len(b))

def parse(value):
 if value['status']!='ok':return 'execution_error',''
 text=value['text'];terminal=value['stop_type'].startswith('token:') or value['stop_type']=='length_cap'
 fields=list(re.finditer(r'(?im)^[ \t]*Final answer:[ \t]*',text))
 if len(fields)!=1:return ('unsupported' if terminal else 'pending'),''
 tail=text[fields[0].end():].lstrip();first=tail.split('\n',1)
 if len(first)==2 and first[0].strip():return 'complete',first[0].strip()
 if value['stop_type'].startswith('token:') and tail.strip():return 'complete',tail.strip()
 return ('unsupported' if terminal else 'pending'),''

def valid_plan(value,paragraphs):
 if value['status']!='ok':return False
 text=value['raw_text'];fields=list(re.finditer(r'(?im)^[ \t]*Final answer:[ \t]*',text))
 if len(fields)!=1:return False
 prefix=text[:fields[0].start()]
 nonempty=[line.strip() for line in prefix.splitlines() if line.strip()]
 if len(nonempty)!=1 or not prefix.rstrip().endswith(nonempty[0]):return False
 match=re.fullmatch(r'Evidence:[ \t]*((?:\[[0-9]+\][ \t]*)*)',nonempty[0],re.IGNORECASE)
 if not match:return False
 ids=list(map(int,re.findall(r'\[([0-9]+)\]',match[1])))
 return bool(ids) and len(ids)==len(set(ids)) and all(1<=v<=paragraphs for v in ids)

def main():
 r=read(F);assert r['status']=='COMPLETE_ONE_SEED_PASSAGE_SELECTION_PILOT_NOT_REPLICATION'
 def payload(name):
  path=P/name;entry=r['payloads'][str(path.relative_to(R))];assert sha(path)==entry['sha256']
  values=[json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
  assert len(values)==entry['rows'];return values
 gold={v['item_id']:v for v in payload('fixed-held-gold-annotations.jsonl.gz')};assert len(gold)==255
 manifest=read(R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2/manifest.json')
 questions={v['item_id']:v for v in map(json.loads,Path(manifest['files']['held_questions']['path']).read_text().splitlines())}
 attempts={};n=0
 for job in read(C/'plan.json')['jobs']:
  arm=job['arm'];attempts[arm]={}
  for mode,table in r['tables'][arm].items():
   values=payload(f'{job["key"]}-{mode}-native-and-scored.jsonl.gz');assert len(values)==255
   observations={}
   for value in values:
    native,stored=value['native'],value['score'];key=native['item_id'];assert key not in observations
    status,answer=parse(native);assert stored['parsed']['status']==status
    if status=='complete':assert stored['parsed']['answer']==answer
    canonical=gold[key]['answer'];aliases=gold[key]['answer_aliases'];complete=status=='complete'
    metrics=dict(canonical_em=int(complete and norm(answer)==norm(canonical)),canonical_f1=f1(answer,canonical) if complete else 0.,
     attributed_alias_em=int(complete and any(norm(answer)==norm(x) for x in [canonical,*aliases])),
     attributed_alias_f1=max(f1(answer,x) for x in [canonical,*aliases]) if complete else 0.)
    assert all(stored[k]==v for k,v in metrics.items())
    if mode=='selection':assert stored['valid_nonempty_selection']==valid_plan(native,len(questions[key]['contexts']))
    observations[key]=stored
   assert observations.keys()==gold.keys();attempts[arm][mode]=observations;n+=len(values)
   assert table['items']==255 and table['correct']==sum(v['canonical_em'] for v in observations.values())
   for metric in metrics:assert abs(table[metric]-sum(v[metric] for v in observations.values())/255)<1e-10
   assert table['parsed_statuses']==dict(Counter(v['parsed']['status'] for v in observations.values()))
   assert table['execution_errors']==sum(v['status']!='ok' for v in observations.values())
   assert table['capped']==sum(v['stop_type']=='length_cap' for v in observations.values())
   assert table['generated_actions']==sum(v['generated_actions'] for v in observations.values())
   if mode=='selection':assert table['valid_nonempty_selections']==sum(v['valid_nonempty_selection'] for v in observations.values())
 indices={n:np.random.default_rng(2026101923).integers(0,n,size=(20000,n)) for n in [255,256]}
 def verify(a,b,stored):
  assert a.keys()==b.keys();delta=np.asarray([b[k]-a[k] for k in sorted(a)],dtype=np.float64)
  interval=np.quantile(delta[indices[len(delta)]].mean(1),[.025,.975])
  assert stored['items']==len(delta) and stored['bootstrap_seed']==2026101923 and stored['bootstrap_draws']==20000
  assert abs(stored['difference_treatment_minus_control']-float(delta.mean()))<1e-12
  assert np.allclose(interval,stored['ci95'],atol=1e-12,rtol=0)
 pairs=dict(selected_minus_matched_direct=('evidence_direct','direct','evidence_select','selection'),selected_minus_same_direct=('evidence_select','direct','evidence_select','selection'),selected_minus_same_empty=('evidence_select','empty','evidence_select','selection'),selected_minus_control_selection=('evidence_direct','selection','evidence_select','selection'),trained_direct_minus_matched_direct=('evidence_direct','direct','evidence_select','direct'),annotated_oracle_minus_selected=('evidence_select','selection','evidence_select','oracle'),control_selection_minus_control_direct=('evidence_direct','direct','evidence_direct','selection'))
 for name,(a,am,b,bm) in pairs.items():
  for metric,field in [('canonical_em','canonical_EM_contrasts'),('canonical_f1','canonical_F1_secondary_contrasts')]:verify({k:v[metric] for k,v in attempts[a][am].items()},{k:v[metric] for k,v in attempts[b][bm].items()},r[field][name])
 primary=r['canonical_EM_contrasts']['selected_minus_matched_direct'];same=r['canonical_EM_contrasts'];recipe=primary['difference_treatment_minus_control']>=.03 and primary['ci95'][0]>0
 integrity=r['tables']['evidence_select']['selection']['valid_nonempty_selections']/255>=.95
 direct=same['selected_minus_same_direct']['ci95'][0]>0;empty=same['selected_minus_same_empty']['ci95'][0]>0
 assert r['decisions']==dict(net_recipe_primary_pass=recipe,selection_integrity_pass=integrity,same_model_direct_lower_bound_positive=direct,same_model_empty_lower_bound_positive=empty,inference_assistance_pass=bool(recipe and integrity and direct and empty),eligible_for_same_protocol_replication=bool(recipe and integrity),two_seed_success_established=False)
 LL={};ordinary={};ln=0
 for job in read(C/'plan.json')['jobs']:
  arm=job['arm'];values=payload(job['key']+'-all-likelihood-rows.jsonl.gz');assert len(values)==1531
  LL[arm]={mode:{} for mode in ['direct','empty','selected_id_replay','other_id_replay','oracle']};ordinary[arm]={}
  for value in values:
   assert value['status']=='ok' and not value['synthetic_padding_observation'] and len(value['token_logps'])==value['tokens']==len(value['target_ids'])
   assert all(math.isfinite(x) for x in value['token_logps']) and abs(value['full_logp']-math.fsum(value['token_logps']))<1e-10
   key=value['item_id']
   if value['panel']=='QA':
    assert value['target_ids'][-1]==151643 and abs(value['answer_content_logp']-math.fsum(value['token_logps'][:-1]))<1e-10 and value['EOS_logp']==value['token_logps'][-1]
    assert key not in LL[arm][value['mode']];LL[arm][value['mode']][key]=value
   else:assert value['tokens']==32 and key not in ordinary[arm];ordinary[arm][key]=-value['full_logp']/32
  for mode,records in LL[arm].items():
   assert records.keys()==gold.keys()
   for part in ['full_logp','answer_content_logp','EOS_logp']:assert abs(r['absolute_likelihood'][arm][mode][part]-math.fsum(v[part] for v in records.values())/255)<1e-10
  assert len(ordinary[arm])==256;ln+=len(values)
  for part,contrasts in r['secondary_pointer_replay_likelihood_contrasts'][arm].items():
   for name,a,b in [('selected_minus_empty','empty','selected_id_replay'),('selected_minus_other','other_id_replay','selected_id_replay'),('oracle_minus_direct','direct','oracle')]:verify({k:v[part] for k,v in LL[arm][a].items()},{k:v[part] for k,v in LL[arm][b].items()},contrasts[name])
 verify(ordinary['evidence_direct'],ordinary['evidence_select'],r['ordinary_NLL_selection_minus_direct'])
 old=Path(read(C/'readout-execution-plan-v2.json')['B0_ordinary_reference']['output']);base={}
 for shard in range(4):
  for v in map(json.loads,(old/'scores'/f'shard-{shard}/scores.jsonl').read_text().splitlines()):
   if v['panel']=='ordinary':base[v['item_id']]=-v['full_logp']/32
 for arm,values in ordinary.items():verify(base,values,r['ordinary_NLL_minus_reused_B0'][arm])
 assert n==r['native_attempts']==1530 and ln==r['likelihood_original_rows']==3062
 result=dict(status='PASS_INDEPENDENT_FIELDS_METRICS_POINTERS_BOOTSTRAPS_AND_DECISIONS',utc=datetime.now(timezone.utc).isoformat(),native_attempts=n,original_likelihood_rows=ln,report_sha256=sha(F),checker_sha256=sha(__file__),derived_FP64_absolute_tolerance=1e-10,limits='Independent offline parsing/normalization/pointervalidation/vectorizedresampling,no new model runs. Allnative errors retained,all255 originalquestions required. Scientific gates unchanged.')
 with (P/'independent-readout-check.json').open('x') as stream:json.dump(result,stream,indent=2)
 print(result['status'])

if __name__=='__main__':main()
