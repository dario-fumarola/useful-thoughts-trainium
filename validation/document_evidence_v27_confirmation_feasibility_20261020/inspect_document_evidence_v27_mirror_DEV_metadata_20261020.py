"""Two pinned DEV mirrors; complete cached TRAIN provenance; metadata only."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,os,urllib.request
from transformers import AutoTokenizer
from train.document_evidence_data import render
from train.document_qa_data import evidence_identity,fingerprint,normalized
from train.document_evidence_plan_data import fresh_candidates
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-v27-confirmation-feasibility-v1';D=R/'local/nvme-study/data/musique-DEV-mirror-metadata-v27';D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());rows=lambda p:[json.loads(v) for v in Path(p).read_text().splitlines()]
amend=read(C/'mirror-metadata-amendment.json');assert amend['status']=='CPU_ONLY_MIRROR_PROVENANCE_CHECK_BEFORE_DEV_METADATA_NO_MODELS'
# Existing same-pin TRAIN is already independently compared with every canonical author field.
M=R/'local/nvme-study/data/musique-TRAIN-mirror-v19';reference=read(M/'reconciliation.json')
assert reference['repo']=='bdsaglam/musique' and reference['revision']==amend['mirrors'][1]['revision']
assert read(M/'canonical-format-crosscheck.json')['status']=='ALL_CANONICAL_TRAIN_FIELDS_EXCEPT_ALIAS_ANNOTATIONS_MATCH_AUTHOR_CONVERTER'
assert sha(M/'musique_ans_v1.0_train.jsonl')==reference['TRAIN_sha256']=='83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a'
(D/'plan.json').write_text(json.dumps(dict(status='PINNED_MIRROR_DEV_METADATA_ONLY',utc=datetime.now(timezone.utc).isoformat(),amendment_sha256=sha(C/'mirror-metadata-amendment.json'),
 local_TRAIN_reuse='Reuse already downloaded bdsaglam TRAIN at the identical declared pin instead of downloading first-mirror TRAIN again. Recheck every primary field before any DEV parsing.',
 prior_TRAIN_receipt_sha256=sha(M/'canonical-format-crosscheck.json'),TEST_requested=False),indent=2))
S=R/'local/nvme-study/data/musique-TRAIN-source-v12';trainraw=rows(S/'musique_ans_train.raw.jsonl');shape=reference['author_ID_translation']['mapping']
def official_id(k):
 p,t=k.split('__');return shape[p]+'__'+t
base={official_id(v['id']):v for v in trainraw};seen=set()
for row in rows(M/'musique_ans_v1.0_train.jsonl'):
 v=base[row['id']];seen.add(row['id']);assert row['question']==v['composed_question_text'] and row['answer']==v['answer_text'] and row['answerable']==v['answerable']==True
 assert row['paragraphs']==[dict(idx=i,title=p['wikipedia_title'],paragraph_text=p['paragraph_text'],is_supporting=p['is_supporting']) for i,p in enumerate(v['contexts'])]
 assert [str(p['id']) for p in row['question_decomposition']]==[str(p['id']) for p in v['decomposed_instances']]
assert len(seen)==len(base)==19938
print('All19938 cached TRAIN question/answer/paragraph/support/component-ID records rechecked.',flush=True)
receipts=[];total=0
for i,mirror in enumerate(amend['mirrors']):
 url='https://huggingface.co/datasets/'+mirror['repo']+'/resolve/'+mirror['revision']+'/musique_ans_v1.0_dev.jsonl'
 p=D/f'mirror{i}-musique_ans_v1.0_dev.jsonl'
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'useful-thoughts-trainium-public-research'}),timeout=45) as stream,p.open('xb') as out:
  n=0
  while True:
   b=stream.read(2**20)
   if not b:break
   n+=len(b);total+=len(b);assert n<100*2**20 and total<amend['download_bound_MiB']*2**20;out.write(b)
  out.flush();os.fsync(out.fileno())
 assert p.open('rb').read(1)==b'{'
 receipts.append(dict(repo=mirror['repo'],revision=mirror['revision'],file=p.name,sha256=sha(p),bytes=n,url=url))
 print('Downloaded pinned DEV mirror',i,'bytes',n,flush=True)
assert receipts[0]['sha256']==receipts[1]['sha256'],'Pinned DEV mirrors disagree; stop before metadata.'
(D/'acquisition.json').write_text(json.dumps(dict(status='PINNED_MIRRORS_BYTE_IDENTICAL_DEV_ONLY',utc=datetime.now(timezone.utc).isoformat(),receipts=receipts,total_bytes=total,TRAIN_rechecked=19938,TEST_requested=False),indent=2))
raw=rows(D/receipts[0]['file']);assert len({v['id'] for v in raw})==len(raw)
T=R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2';V=R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1'
oldrows=rows(T/'TRAIN2048.jsonl')+rows(V/'held512-questions-only.jsonl')+rows(T/'held255-questions-only.jsonl');oldrawids={v['item_id'] for v in oldrows};oldids={official_id(k) for k in oldrawids};meta={v['id']:evidence_identity(v) for v in trainraw}
blocked={f:set().union(*(meta[k][f] for k in oldrawids)) for f in ('visible_text','visible_titles')}
blocked['seed_ids']={str(h['id']) for v in trainraw if v['id'] in oldrawids for h in v['decomposed_instances']};oldqs={meta[k]['question'] for k in oldrawids}
tok=AutoTokenizer.from_pretrained(R/'local/assets/qwen3-1.7b-base',local_files_only=True,trust_remote_code=False);ms=[];lengths={};length_hist=Counter()
with (D/'input-metadata.jsonl').open('x') as out:
 for i in range(0,len(raw),128):
  batch=raw[i:i+128];qs=[dict(item_id=v['id'],question=v['question'],contexts=[dict(title=p['title'],text=p['paragraph_text']) for p in v['paragraphs']]) for v in batch]
  texts=[render(q) for q in qs];tokens=tok(texts,add_special_tokens=False,truncation=False,return_attention_mask=False)['input_ids']
  for v,q,text,ids in zip(batch,qs,texts,tokens):
   assert v['answerable'] and [p['idx'] for p in v['paragraphs']]==list(range(len(v['paragraphs'])))
   support=[p for p in v['paragraphs'] if p['is_supporting']];assert support and len(q['contexts'])==20
   m=dict(item_id=v['id'],question=normalized(q['question']),support_text={fingerprint(p['paragraph_text']) for p in support},support_titles={normalized(p['title']) for p in support},seed_ids={str(h['id']) for h in v['question_decomposition']},hops=len(v['question_decomposition']))
   assert len(m['seed_ids'])==m['hops']>=2
   n=len(ids);lengths[m['item_id']]=n;length_hist[n//512*512]+=1;ms.append(m)
   out.write(json.dumps(dict(item_id=m['item_id'],native_input_tokens=n,input_sha256=hashlib.sha256(text.encode()).hexdigest(),hops=m['hops']))+'\n')
  print('Inventoried DEV metadata',min(i+128,len(raw)),flush=True)
causes=Counter();eligible=[];overlaps=Counter()
for v in ms:
 for f,t in [('support_text','visible_text'),('support_titles','visible_titles'),('seed_ids','seed_ids')]:overlaps[f]+=bool(v[f]&blocked[t])
 if lengths[v['item_id']]>3072:causes['too_long']+=1;continue
 if v['item_id'] in oldids or v['question'] in oldqs:causes['old_question']+=1;continue
 if v['seed_ids']&blocked['seed_ids']:causes['old_component_question_bare_ID_conservative']+=1;continue
 if v['support_text']&blocked['visible_text']:causes['old_visible_paragraph']+=1;continue
 if v['support_titles']&blocked['visible_titles']:causes['old_visible_page']+=1;continue
 eligible.append(v)
parent={v['item_id']:v['item_id'] for v in eligible}
def root(k):
 while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
 return k
owners={}
for v in eligible:
 for f in ('support_text','support_titles','seed_ids'):
  for val in v[f]:
   t=(f,val)
   if t in owners:
    a,b=root(v['item_id']),root(owners[t])
    if a!=b:parent[b]=a
   else:owners[t]=v['item_id']
sizes=sorted(Counter(root(k) for k in parent).values(),reverse=True)
occupied={f:set() for f in ('support_text','support_titles','seed_ids')};chosen=[];seenqs=set()
for v in sorted(eligible,key=lambda v:(hashlib.sha256(('document-evidence-v27-feasibility-only-v1:'+v['item_id']).encode()).hexdigest(),v['item_id'])):
 if v['question'] in seenqs or any(v[f]&occupied[f] for f in occupied):continue
 chosen.append(v);seenqs.add(v['question'])
 for f in occupied:occupied[f].update(v[f])
# Valid bound: each independent question consumes >=2 distinct original IDs.
union_seeds=set().union(*(v['seed_ids'] for v in eligible));independent_upper=min(len(eligible),len(union_seeds)//2)
# Original TRAIN-only zero result through the pre-existing independent implementation.
oldlengths={v['item_id']:v['native_input_tokens'] for v in rows(S/'input-inventory-v2.jsonl')};trainids={v['item_id'] for v in rows(T/'TRAIN2048.jsonl')}
assert fresh_candidates(trainraw,oldlengths,trainids,oldrawids-trainids,max_input_tokens=3072)==[]
report=dict(status='PINNED_DEV_MIRROR_METADATA_COMPLETE_NO_COHORT_OR_MODEL_RELEASE',utc=datetime.now(timezone.utc).isoformat(),source_rows=len(raw),acquisition_sha256=sha(D/'acquisition.json'),DEV_sha256=receipts[0]['sha256'],TRAIN_canonical_records_rechecked=19938,
 native_length_512_bins=dict(length_hist),raw_hops=dict(Counter(v['hops'] for v in ms)),overlaps_any_length=dict(overlaps),first_exclusion_counts=dict(causes),strict_clean_eligible_rows=len(eligible),eligible_hops=dict(Counter(v['hops'] for v in eligible)),
 support_overlap_components=len(sizes),largest_component=max(sizes,default=0),component_size_histogram=dict(Counter(sizes)),fixed_hash_greedy_independent_rows=len(chosen),greedy_hops=dict(Counter(v['hops'] for v in chosen)),
 distinct_original_IDs=len(union_seeds),independent_count_upper_bound=independent_upper,desired_independent_items=1024,confirmed_insufficient_upper_bound=independent_upper<1024,TRAIN_zero_filter_reproduced=True,
 source_sha256=sha(__file__),metadata_inventory_sha256=sha(D/'input-metadata.jsonl'),amendment_sha256=sha(C/'mirror-metadata-amendment.json'),
 limits=['Metadata only; no model forward, scores, manual new examples or selected cohort. TEST not requested.',
 'Bare original-ID exclusion is conservative because official format omits dataset namespace; it can remove additional candidates.',
 'Greedy is a lower bound; distinct original IDs/2 is an upper bound under required within-cohort original-ID disjointness.',
 'Exact paired mirrors plus full TRAIN crosscheck support provenance, not semantic entailment or novelty relative to pretraining.',
 'No smaller underpowered fallback, changed old score or automatic training release.'])
(C/'DEV-mirror-metadata-result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
