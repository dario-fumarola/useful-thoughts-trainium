"""Bounded metadata-only author DEV acquisition; no predictions or example printing."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,os,shutil,urllib.request
from transformers import AutoTokenizer
from train.document_evidence_data import question_projection,render
from train.document_qa_data import evidence_identity
from train.document_evidence_plan_data import fresh_candidates
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-v27-confirmation-feasibility-v1'
D=R/'local/nvme-study/data/musique-DEV-metadata-v27';D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]
read=lambda p:json.loads(Path(p).read_text())
plan=read(C/'DEV-metadata-plan.json');assert plan['status']=='RELEASED_CPU_ONLY_OFFICIAL_DEV_METADATA_FEASIBILITY_NOT_MODEL_EVALUATION'
assert sha(C/'DEV-metadata-plan.json')==sha(R/'validation/document_evidence_v27_confirmation_feasibility_plan_20261020.json')
assert min(shutil.disk_usage(p).free for p in (R,D))>35*2**30+100*2**20
S=R/'local/nvme-study/data/musique-TRAIN-source-v12';old=read(S/'author-data-mapping.json');meta_url=plan['allowed_downloads'][0]
request=urllib.request.Request(meta_url,headers={'User-Agent':'useful-thoughts-trainium-public-research'})
with urllib.request.urlopen(request,timeout=30) as f:metadata=f.read(2*2**20)
(D/'author-data-mapping.json').write_bytes(metadata);mapping=json.loads(metadata)
key='raw_data/musique_ans_dev.jsonl';assert mapping[key]==old[key]=='1TRXU68wveSehVbQrRRtWsUsFkKUF43QS'
url='https://drive.usercontent.google.com/download?id='+mapping[key]+'&export=download&confirm=t'
(D/'acquisition-plan.json').write_text(json.dumps(dict(status='ONLY_AUTHOR_RAW_DEV_METADATA_DOWNLOAD',utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(C/'DEV-metadata-plan.json'),source_key=key,url=url,source_limit_bytes=100*2**20,TEST_requested=False),indent=2))
path=D/'musique_ans_dev.raw.jsonl';size=0
with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'useful-thoughts-trainium-public-research'}),timeout=45) as f,path.open('xb') as out:
 content_type=f.headers.get('Content-Type');declared=f.headers.get('Content-Length')
 if declared:assert int(declared)<=100*2**20
 while True:
  block=f.read(2**20)
  if not block:break
  size+=len(block);assert size<=100*2**20;out.write(block)
 out.flush();os.fsync(out.fileno())
assert path.open('rb').read(1)==b'{'
raw=rows(path);assert len({v['id'] for v in raw})==len(raw) and len(raw)>0
(D/'acquisition.json').write_text(json.dumps(dict(status='AUTHOR_DEV_BYTES_ACQUIRED_FOR_METADATA_ONLY',utc=datetime.now(timezone.utc).isoformat(),url=url,source_key=key,bytes=size,sha256=sha(path),content_type=content_type,metadata_sha256=sha(D/'author-data-mapping.json'),TEST_requested=False,model_predictions=False),indent=2))
print('Acquired author DEV for aggregate metadata; no examples/predictions printed.',flush=True)
train_raw=rows(S/'musique_ans_train.raw.jsonl');source_meta={v['id']:evidence_identity(v) for v in train_raw}
T=R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2';V=R/'local/nvme-cpt/data/document-QA-v21-TRAIN2048-held512-v1'
oldrows=rows(T/'TRAIN2048.jsonl')+rows(V/'held512-questions-only.jsonl')+rows(T/'held255-questions-only.jsonl');oldids={v['item_id'] for v in oldrows}
blocked={f:set().union(*(source_meta[k][f] for k in oldids)) for f in ('visible_text','visible_titles','seed_ids')};oldquestions={source_meta[k]['question'] for k in oldids}
tok=AutoTokenizer.from_pretrained(R/'local/assets/qwen3-1.7b-base',local_files_only=True,trust_remote_code=False)
lengths={};metadata_rows=[];native_hist=Counter();context_counts=Counter()
with (D/'input-metadata.jsonl').open('x') as out:
 for i in range(0,len(raw),128):
  batch=raw[i:i+128];texts=[render(question_projection(v)) for v in batch]
  tokens=tok(texts,add_special_tokens=False,return_attention_mask=False,truncation=False)['input_ids']
  for v,text,ids in zip(batch,texts,tokens):
   m=dict(evidence_identity(v),hops=len(v['decomposed_instances']));n=len(ids);lengths[m['item_id']]=n
   native_hist[n//512*512]+=1;context_counts[len(v['contexts'])]+=1
   for f in ('visible_text','visible_titles','seed_ids'):assert isinstance(m[f],set)
   metadata_rows.append(m)
   record=dict(item_id=m['item_id'],input_sha256=hashlib.sha256(text.encode()).hexdigest(),native_input_tokens=n,hops=m['hops'],visible_paragraphs=len(v['contexts']))
   out.write(json.dumps(record)+'\n')
  print('Metadata inputs tokenized:',min(i+128,len(raw)),flush=True)
# Match the frozen previous source rule and report every first exclusion cause.
causes=Counter();eligible=[];overlaps=Counter()
for v in metadata_rows:
 for field,target in [('support_text','visible_text'),('support_titles','visible_titles'),('seed_ids','seed_ids')]:
  overlaps[field]+=bool(v[field]&blocked[target])
 if lengths[v['item_id']]>3072:causes['too_long']+=1;continue
 if v['item_id'] in oldids or v['question'] in oldquestions:causes['old_question']+=1;continue
 if v['seed_ids']&blocked['seed_ids']:causes['old_component_question']+=1;continue
 if v['support_text']&blocked['visible_text']:causes['old_visible_paragraph']+=1;continue
 if v['support_titles']&blocked['visible_titles']:causes['old_visible_page']+=1;continue
 eligible.append(v)
# Components of overlapping support/source identities among eligible questions.
parent={v['item_id']:v['item_id'] for v in eligible}
def root(k):
 while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
 return k
def union(a,b):
 a,b=root(a),root(b)
 if a!=b:parent[b]=a
owners={}
for v in eligible:
 k=v['item_id']
 for f in ('support_text','support_titles','seed_ids'):
  for val in v[f]:
   feature=(f,val)
   if feature in owners:union(k,owners[feature])
   else:owners[feature]=k
components=Counter(root(k) for k in parent);sizes=sorted(components.values(),reverse=True)
ranked=sorted(eligible,key=lambda v:(hashlib.sha256(('document-evidence-v27-feasibility-only-v1:'+v['item_id']).encode()).hexdigest(),v['item_id']))
chosen=[];occupied={f:set() for f in ('support_text','support_titles','seed_ids')};qs=set()
for v in ranked:
 if v['question'] in qs or any(v[f]&occupied[f] for f in occupied):continue
 chosen.append(v);qs.add(v['question'])
 for f in occupied:occupied[f].update(v[f])
# Reproduce the earlier TRAIN result through the existing, separately implemented filter.
trainids={v['item_id'] for v in rows(T/'TRAIN2048.jsonl')};heldids=oldids-trainids
oldlengths={v['item_id']:v['native_input_tokens'] for v in rows(S/'input-inventory-v2.jsonl')}
assert fresh_candidates(train_raw,oldlengths,trainids,heldids,max_input_tokens=3072)==[]
report=dict(status='AUTHOR_DEV_METADATA_FEASIBILITY_COMPLETE_NO_COHORT_OR_MODELS_RELEASED',utc=datetime.now(timezone.utc).isoformat(),
 source_rows=len(raw),source_sha256=sha(path),source_key=key,license='CC BY4.0; Stony Brook NLP MuSiQue authors',
 native_length_512_bins=dict(native_hist),context_count_histogram=dict(context_counts),raw_hops=dict(Counter(v['hops'] for v in metadata_rows)),
 overlap_counts_any_length=dict(overlaps),first_exclusion_counts=dict(causes),strict_clean_eligible_rows=len(eligible),eligible_hops=dict(Counter(v['hops'] for v in eligible)),
 support_overlap_components=len(sizes),largest_component=sizes[0] if sizes else 0,component_size_histogram=dict(Counter(sizes)),
 fixed_hash_greedy_independent_rows=len(chosen),greedy_hops=dict(Counter(v['hops'] for v in chosen)),
 eligible_questions_upper_bound=len(eligible),desired_independent_items=1024,sufficient_independent_greedy=len(chosen)>=1024,
 TRAIN_zero_independent_filter_reproduction=True,metadata_inventory_sha256=sha(D/'input-metadata.jsonl'),plan_sha256=sha(C/'DEV-metadata-plan.json'),source_sha256_code=sha(__file__),
 limits=['No model forward/training, new scores, human-read question/answer examples or selected evaluation cohort.',
 'Metadata support flags are privileged split information, not entailment verification or pretraining-novelty evidence.',
 'Greedy count is a lower bound, not a maximum. Eligible count bounds the maximum; connected components show dependence.',
 'No smaller underpowered fallback or changed old outcomes. Official TEST remains unrequested.'])
(C/'DEV-metadata-result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
