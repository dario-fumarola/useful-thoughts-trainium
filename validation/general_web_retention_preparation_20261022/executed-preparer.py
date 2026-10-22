from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,heapq,json,os,shutil,time,urllib.request
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from scripts.prepare_real_cpt_scaffold import page
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/general-web-retention-20261022';P=C/'prospective-plan.json';plan=json.loads(P.read_text());D=Path(plan['data_directory']);D.mkdir(exist_ok=False);start=time.monotonic();os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);os.environ['TOKENIZERS_PARALLELISM']='false'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check():
 if time.monotonic()-start>plan['maximum_CPU_wall_seconds']:raise TimeoutError('Source preparation wall budget')
 if any(shutil.disk_usage(p).free<35*2**30 for p in [R,D]):raise RuntimeError('Preserve filesystem floor')
 available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 if available<512*2**30:raise RuntimeError('Preserve host-memory floor')
 if sum(p.stat().st_size for p in D.iterdir() if p.is_file())>plan['maximum_storage_bytes']:raise RuntimeError('Source storage cap')
check();source=D/'source.parquet';url='https://huggingface.co/datasets/'+plan['source_dataset']+'/resolve/'+plan['source_revision']+'/'+plan['source_file']+'?download=true'
with urllib.request.urlopen(url,timeout=60) as incoming,source.open('xb') as out:
 while block:=incoming.read(4*2**20):out.write(block);check()
if source.stat().st_size!=plan['source_bytes'] or sha(source)!=plan['source_sha256']:raise ValueError('Pinned source-file byte identity differs')
print('SOURCE_DOWNLOAD_AND_SHA_PASS',flush=True)
blocked_pages=set();blocked_content=set()
for file,digest in plan['exclusion_membership_files'].items():
 path=Path(file)
 if sha(path)!=digest:raise ValueError('Original exclusion membership differs')
 for line in path.open():
  row=json.loads(line);blocked_pages.add(page(row['url']));blocked_content.update(row[k] for k in ['document_id','content_sha256'] if row.get(k))
print('BLOCKED',len(blocked_pages),len(blocked_content),flush=True)
tok=AutoTokenizer.from_pretrained(plan['tokenizer'],local_files_only=True,trust_remote_code=False);heap=[];counts=Counter();seen_pages=set();seen_content=set();position=0
with (D/'eligible-index.jsonl').open('x') as index:
 for batch in pq.ParquetFile(source).iter_batches(batch_size=256,columns=['text','url','id']):
  for raw in batch.to_pylist():
   if position>=plan['maximum_source_rows']:break
   n=position;position+=1;counts['rows_examined']+=1
   if position%256==0:check()
   text=raw['text'];url=raw['url'];canonical=page(url);doc=hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
   if canonical in blocked_pages or doc in blocked_content:counts['excluded_prior_math_source']+=1;continue
   if canonical in seen_pages or doc in seen_content:counts['duplicate_page_or_content']+=1;continue
   seen_pages.add(canonical);seen_content.add(doc)
   ids=tok(text[:65536],add_special_tokens=False,truncation=True,max_length=8208,return_attention_mask=False)['input_ids'];positions=max(0,min(8192,len(ids)-16)-512+1)
   if not positions:counts['short_source']+=1;continue
   choice=hashlib.sha256((plan['salt']+':select:'+doc).encode()).hexdigest();cut=512+int(hashlib.sha256((plan['salt']+':cut:'+doc).encode()).hexdigest(),16)%positions
   meta={'document_id':doc,'canonical_page':canonical,'parquet_row':n,'positions':positions,'selection_hash':choice,'cut':cut};index.write(json.dumps(meta)+'\n')
   row={**meta,'item_id':doc+':general-web:'+str(cut),'cohort':'ENGLISH_RETENTION','ids':ids[cut-512:cut+16],'source_record_id':raw['id'],'source_text_sha256':hashlib.sha256(text.encode()).hexdigest()}
   if len(row['ids'])!=528:raise ValueError('Complete original window required')
   counts['eligible_unique_documents']+=1;entry=(-int(choice,16),doc,row)
   if len(heap)<plan['selected_documents']:heapq.heappush(heap,entry)
   elif entry>heap[0]:heapq.heapreplace(heap,entry)
  if position>=plan['maximum_source_rows']:break
if len(heap)!=plan['selected_documents']:raise ValueError('Fixed source scan supplies insufficient documents')
selected=sorted((entry[2] for entry in heap),key=lambda row:row['selection_hash']);path=D/'ENGLISH_RETENTION.jsonl';path.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in selected))
result={'status':'COMPLETE_GENERAL_WEB_SOURCE_PREPARATION','utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(P),'source_sha256':sha(source),'counts':dict(counts),'blocked_pages':len(blocked_pages),'blocked_content':len(blocked_content),'file':str(path),'data_sha256':sha(path),'documents':len(selected),'source_tokens_per_document':528,'predicted_tokens_per_document':527,'wall_seconds':time.monotonic()-start,'model_calls':0,'source_dataset':plan['source_dataset'],'source_revision':plan['source_revision'],'scope':plan['interpretation']};(D/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
