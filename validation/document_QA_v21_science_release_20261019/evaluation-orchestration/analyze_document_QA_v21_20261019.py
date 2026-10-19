"""One terminal-only readout after every frozen generation and LL panel closes.

All attempts/errors retained. Full designated answers,canonical and separately
attributed aliases. Checkpoint-conditional question bootstrap,never pooled seeds.
"""
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import csv,gzip,hashlib,json,math
import numpy as np
from tokenizers import Tokenizer
from train.document_qa_answers import ParsedAnswer,parse_answer,score
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-QA-v21-study-v1'
read=lambda p:json.loads(Path(p).read_text());rows=lambda p:[json.loads(l) for l in Path(p).read_text().splitlines()]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SEED=2026101921;DRAWS=20000

def contrast(a,b):
 assert a.keys()==b.keys() and len(a) in (256,512)
 ids=sorted(a);delta=np.array([b[k]-a[k] for k in ids],dtype=np.float64)
 assert np.isfinite(delta).all();rng=np.random.default_rng(SEED);resampled=[]
 for start in range(0,DRAWS,500):resampled.extend(delta[rng.integers(0,len(ids),size=(min(500,DRAWS-start),len(ids)))].mean(1).tolist())
 return dict(items=len(ids),difference_b_minus_a=float(delta.mean()),ci95=np.quantile(resampled,[.025,.975]).tolist(),bootstrap_draws=DRAWS,bootstrap_seed=SEED,unit='paired questions/documents;conditional on checkpoint,unadjusted95%')

def gate(v):return v['difference_b_minus_a']>=.03 and v['ci95'][0]>0

def main():
 # Do not open any outcomes before all global closures and identities check.
 assert read(C/'ALL_GENERATION_CLOSED_NO_READOUT.json')['status']=='ALL_TWENTY_THOUSAND_NINE_HUNDRED_NINETY_TWO_ATTEMPTS_CLOSED_NO_READOUT'
 assert read(C/'ALL_LIKELIHOOD_CLOSED_NO_READOUT.json')['status']=='ALL_NINETEEN_THOUSAND_SEVEN_HUNDRED_TWELVE_LIKELIHOOD_ROWS_CLOSED_NO_READOUT'
 assert read(C/'terminal-pipeline-completion.json')['status']=='ALL_SIX_TERMINAL256_MODELS_AUDITED_PROTECTED_AND_BOUND'
 release=read(C/'release.json');design=read(C/'prospective-design.json');assert sha(C/'prospective-design.json')==release['prospective_design_sha256']
 groups=[dict(model='B0',output=str(R/'local/nvme-cpt/evaluation/document-QA-v21-B0-held512-v1'),control=str(C/'B0-generation')),*read(C/'trained-generation-registry.json')['groups']]
 llgroups=read(C/'likelihood-registry.json')['groups'];assert len(llgroups)==7 and len({g['model'] for g in llgroups})==7
 for g in groups:assert read(Path(g['control'])/'closure.json')['status']=='ALL_FROZEN_GENERATION_SHARDS_PHYSICALLY_CLOSED_NO_SCORES_REVIEWED'
 for g in llgroups:assert read(Path(g['control'])/'closure.json')['status']=='ALL_FROZEN_LIKELIHOOD_SHARDS_PHYSICALLY_CLOSED_NO_READOUT'
 Q=R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1';manifest=read(Q/'manifest.json');assert sha(Q/'manifest.json')==release['QA_data_manifest_sha256']
 gold={r['item_id']:r for r in rows(manifest['files']['held_gold']['path'])};ids=list(gold);assert len(ids)==512
 ordinary_manifest=read(R/'local/nvme-cpt/data/document-QA-v21-ordinary256-v2/manifest.json');ordinary_ids={v['item_id'] for v in rows(ordinary_manifest['data_file'])};assert len(ordinary_ids)==256
 P=R/'validation/document_QA_v21_complete_20261019';P.mkdir(exist_ok=False)
 tok=Tokenizer.from_file(str(R/'local/assets/qwen3-1.7b-base/tokenizer.json'));support=set(tok.get_vocab().values())
 payloads={};attempts={m:{} for m in release['model_modes']};inventory=[];total=0
 def payload(name,values):
  p=P/name;p.write_bytes(gzip.compress((''.join(json.dumps(v,allow_nan=False)+'\n' for v in values)).encode(),mtime=0));payloads[str(p.relative_to(R))]=dict(rows=len(values),bytes=p.stat().st_size,sha256=sha(p))
 for g in groups:
  O=Path(g['output']);p=read(O/'protocol.json');model=g['model'];assert p['model']['id']==model and p['release_sha256']==sha(C/'release.json') and p['items']==512 and p['ordered_item_ids']==ids
  assert all(sha(O/n)==v for n,v in p['files'].items())
  inventory.append(dict(model=model,output=str(O),protocol_sha256=sha(O/'protocol.json'),closure_sha256=sha(Path(g['control'])/'closure.json')))
  for mode in p['modes']:assert mode not in attempts[model];attempts[model][mode]={}
  prompts={m:{v['item_id']:v for v in rows(O/f'prompts-{m}.jsonl')} for m in p['modes']}
  for shard in range(4):
   d=O/'answers'/f'shard-{shard}';report=read(d/'run.json');native=read(d/'native.json');cache=read(d/'cache-prefill.json')
   assert report['status']=='complete_all_attempts' and report['protocol_sha256']==sha(O/'protocol.json') and report['attempts_sha256']==sha(d/'attempts.jsonl') and native['passed'] and cache['max_logp_error']<=.001
   observed=rows(d/'attempts.jsonl');assert len(observed)==128*len(p['modes'])==report['rows']
   for value in observed:
    item,mode=value['item_id'],value['mode'];assert item in set(ids[shard::4]) and item not in attempts[model][mode]
    q=prompts[mode][item];a=value['actions'];content=value['content_ids'];assert value['parser_mode']==q['parser_mode'] and value['prompt_tokens']==len(q['prompt_ids']) and value['n_actions']==len(a)<=256
    assert all(type(x) is int and x in support for x in a)
    stop=value['stop_type'];terminated=stop.startswith('token:');capped=stop=='length_cap'
    if value['status']=='ok':
     assert stop in ('qa_answer_line','length_cap','token:151643','token:151645')
     if terminated:assert a and a[-1] in (151643,151645) and stop==f'token:{a[-1]}'
     assert content==(a[:-1] if terminated else a) and not any(x in (151643,151645) for x in content)
     assert not capped or len(a)==256
    raw=tok.decode(content,skip_special_tokens=False);assert raw==value['raw_text']
    if stop=='qa_answer_line':
     parsed_raw=parse_answer(raw,q['parser_mode']);assert parsed_raw.status=='complete' and value['text']==raw[:parsed_raw.end]
    else:assert value['text']==raw
    assert math.isfinite(value['wall_seconds']) and value['wall_seconds']>=0
    parsed=parse_answer(value['text'],q['parser_mode'],terminated=terminated,capped=capped)
    if value['status']!='ok':parsed=ParsedAnswer('execution_error',reason='execution_error')
    metrics=score(parsed,gold[item]['answer'],gold[item]['answer_aliases'])
    attempts[model][mode][item]=dict(**value,parsed=asdict(parsed),**metrics)
    total+=1
 assert total==20992
 tables={}
 for model,modes in attempts.items():
  assert set(modes)==set(release['model_modes'][model]);tables[model]={}
  for mode,values in modes.items():
   assert set(values)==set(ids);vs=[values[k] for k in ids]
   tables[model][mode]=dict(items=512,**{metric:sum(v[metric] for v in vs)/512 for metric in ('canonical_em','canonical_f1','attributed_alias_em','attributed_alias_f1')},correct=sum(v['canonical_em'] for v in vs),parsed_statuses=dict(Counter(v['parsed']['status'] for v in vs)),stop_types=dict(Counter(v['stop_type'] for v in vs)),execution_errors=sum(v['status']!='ok' for v in vs),capped=sum(v['stop_type']=='length_cap' for v in vs),generated_actions=sum(v['n_actions'] for v in vs),prompt_tokens=sum(v['prompt_tokens'] for v in vs),generation_wall_seconds_sum=sum(v['wall_seconds'] for v in vs))
  payload(model+'-all-generation-attempts.jsonl.gz',[modes[m][k] for m in release['model_modes'][model] for k in ids])
 likelihood={};absolute={};ordinary={};LLtotal=0
 for g in llgroups:
  O=Path(g['output']);p=read(O/'protocol.json');model=g['model'];assert p['model']['id']==model and p['release_sha256']==sha(C/'release.json') and all(sha(O/n)==v for n,v in p['files'].items())
  expected={}
  for name in p['inventories']:
   for v in rows(O/name):expected[(v['panel'],v['mode'],v['item_id'])]=v
  actual={};ls=[]
  for shard in range(4):
   d=O/'scores'/f'shard-{shard}';complete=read(d/'completion.json');cal=read(d/'own-model-TRAIN-calibration.json')
   assert complete['rows']==704 and complete['execution_errors']==0 and complete['scores_sha256']==sha(d/'scores.jsonl') and complete['source_protocol_sha256']==sha(O/'protocol.json') and all(v['passed'] for v in cal.values())
   for v in rows(d/'scores.jsonl'):
    key=(v['panel'],v['mode'],v['item_id']);assert key in expected and key not in actual and v['status']=='ok';e=expected[key]
    assert v['context_tokens']==len(e['context_ids']) and v['target_ids']==e['target_ids'] and v['tokens']==len(e['target_ids'])==len(v['token_logps'])
    assert all(math.isfinite(x) and x<=1e-5 for x in v['token_logps']) and abs(sum(v['token_logps'])-v['full_logp'])<1e-10 and v['K1_logp']==v['token_logps'][0]
    if v['panel']=='QA':assert v['target_ids'][-1]==151643 and v['EOS_logp']==v['token_logps'][-1] and abs(v['answer_content_logp']-sum(v['token_logps'][:-1]))<1e-10
    actual[key]=v;ls.append(v)
  assert set(actual)==set(expected) and len(actual)==2816;LLtotal+=len(actual)
  likelihood[model]={};absolute[model]={}
  for mode in ('direct','empty','own_replay','other_replay','oracle'):
   values={k[2]:v for k,v in actual.items() if k[:2]==('QA',mode)};assert set(values)==set(ids);likelihood[model][mode]=values
   absolute[model][mode]=dict(items=512,**{field:sum(v[field] for v in values.values())/512 for field in ('full_logp','answer_content_logp','EOS_logp')},total_target_tokens=sum(v['tokens'] for v in values.values()))
  ordinary[model]={k[2]:-v['full_logp']/32 for k,v in actual.items() if k[:2]==('ordinary','raw')};assert set(ordinary[model])==ordinary_ids
  absolute[model]['ordinary_raw']=dict(items=256,targets=8192,mean_token_NLL=sum(ordinary[model].values())/256)
  payload(model+'-all-likelihood-rows.jsonl.gz',ls)
 assert LLtotal==19712
 def comp(a,am,b,bm,metric='canonical_em'):return contrast({k:v[metric] for k,v in attempts[a][am].items()},{k:v[metric] for k,v in attempts[b][bm].items()})
 def llcomp(a,am,b,bm,part):return contrast({k:v[part] for k,v in likelihood[a][am].items()},{k:v[part] for k,v in likelihood[b][bm].items()})
 seeds={}
 for seed in (2026101921,2026101922):
  direct=f'qa_direct-seed{seed}';prior=f'qa_prior-seed{seed}';post=f'qa_post-seed{seed}'
  pairs=dict(post_direct_minus_direct_direct=(direct,'direct',post,'direct'),prior_prior_minus_prior_empty=(prior,'empty',prior,'prior'),prior_prior_minus_direct_direct=(direct,'direct',prior,'prior'),prior_prior_minus_direct_prior=(direct,'prior',prior,'prior'),prior_prior_minus_direct_reference_cot=(direct,'reference_cot',prior,'prior'),prior_prior_minus_B0_reference_cot=('B0','reference_cot',prior,'prior'),prior_direct_minus_direct_direct=(direct,'direct',prior,'direct'),prior_prior_minus_prior_direct=(prior,'direct',prior,'prior'),prior_prior_minus_question_only=(prior,'question_only_prior',prior,'prior'),prior_prior_minus_bridge_removed=(prior,'bridge_removed_prior',prior,'prior'),prior_own_replay_minus_other=(prior,'other_replay',prior,'own_replay'),post_reference_minus_direct_reference=(direct,'reference_cot',post,'reference_cot'),direct_reference_minus_B0_reference=('B0','reference_cot',direct,'reference_cot'))
  cs={name:comp(*pair) for name,pair in pairs.items()}
  training=gate(cs['post_direct_minus_direct_direct']);inference=all(gate(cs[n]) for n in ('prior_prior_minus_prior_empty','prior_prior_minus_direct_direct'))
  practical=all(gate(cs[n]) for n in ('prior_prior_minus_direct_prior','prior_prior_minus_direct_reference_cot','prior_prior_minus_B0_reference_cot'))
  llpairs=dict(post_direct_minus_direct_direct=(direct,'direct',post,'direct'),prior_own_minus_prior_empty=(prior,'empty',prior,'own_replay'),prior_own_minus_other=(prior,'other_replay',prior,'own_replay'),prior_oracle_minus_direct=(prior,'direct',prior,'oracle'),prior_own_minus_direct_direct=(direct,'direct',prior,'own_replay'))
  seeds[str(seed)]=dict(primary_canonical_EM_contrasts=cs,secondary_canonical_F1_contrasts={n:comp(*pair,metric='canonical_f1') for n,pair in pairs.items()},training_benefit_gate_pass=training,inference_benefit_gate_pass=inference,practical_reasoning_supervision_gate_pass=bool(inference and practical),secondary_likelihood_contrasts={part:{n:llcomp(*pair,part) for n,pair in llpairs.items()} for part in ('full_logp','answer_content_logp','EOS_logp')},ordinary_NLL_cost={name:contrast(ordinary[a],ordinary[b]) for name,a,b in [('prior_minus_direct',direct,prior),('post_minus_direct',direct,post),('prior_minus_B0','B0',prior),('post_minus_B0','B0',post),('direct_minus_B0','B0',direct)]})
 report=dict(status='COMPLETE_SEVEN_MODEL_TWO_SEED_DOCUMENT_QA_REFERENCE',utc=datetime.now(timezone.utc).isoformat(),release_sha256=sha(C/'release.json'),design_sha256=sha(C/'prospective-design.json'),analyzer_sha256=sha(__file__),attempts=total,likelihood_rows=LLtotal,tables=tables,absolute_likelihood=absolute,seeds=seeds,decisions=dict(training_benefit_both_seeds=all(v['training_benefit_gate_pass'] for v in seeds.values()),inference_benefit_both_seeds=all(v['inference_benefit_gate_pass'] for v in seeds.values()),practical_reasoning_supervision_both_seeds=all(v['practical_reasoning_supervision_gate_pass'] for v in seeds.values())),generation_inventory=inventory,payloads=payloads,limits=design['limits']+['All questions come from authorTRAIN;these are research held panels,not officialTEST.','Prior supervision is conventional full-trace CoT;root answer in trace. Oracleprivileged,not deployment.','Canonicalprimary;mirroraliases independently attributed,not independently authenticated.','ConcurrentFP32 wall sums are observed executioncosts,not controlled serving latency.','OrdinaryNLL measuredtradeoff;no importedFineMaththreshold,oldV17failure unchanged.'])
 with (P/'absolute-answers.csv').open('x') as f:
  writer=csv.DictWriter(f,fieldnames=['model','mode','correct','canonical_em','canonical_f1','attributed_alias_em','capped','execution_errors','generated_actions']);writer.writeheader()
  for model,ms in tables.items():
   for mode,v in ms.items():writer.writerow(dict(model=model,mode=mode,**{k:v[k] for k in writer.fieldnames if k not in ('model','mode')}))
 payload('fixed-held-gold-metadata.jsonl.gz',[gold[k] for k in ids])
 out=R/'validation/document_QA_v21_complete_20261019.json';assert not out.exists();out.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(dict(status=report['status'],decisions=report['decisions']),indent=2))

if __name__=='__main__':main()
