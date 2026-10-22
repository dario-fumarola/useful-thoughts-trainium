from pathlib import Path
from collections import defaultdict
import gzip,hashlib,heapq,json,os,time
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from scripts.prepare_real_cpt_scaffold import page
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/uniform-mixture-20261022';plan=json.loads((C/'prospective-plan.json').read_text());D=Path(plan['data_directory']);start=time.monotonic();os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);os.environ['TOKENIZERS_PARALLELISM']='false'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(spec):
 if sha(spec['path'])!=spec['sha256']:raise ValueError('Pinned source/proof changed')
 return Path(spec['path'])
p=json.loads(bound(plan['parent_uniform_source_plan']).read_text());web=json.loads(bound(plan['English_source_plan']).read_text());manifest=json.loads((D/'complete.json').read_text());cohorts={}
for cohort,spec in manifest['files'].items():
 if sha(spec['path'])!=spec['sha256']:raise ValueError('Complete cohort data changed')
 rows=[json.loads(x) for x in Path(spec['path']).open()]
 if len(rows)!=spec['rows'] or any(len(v['ids'])!=528 for v in rows):raise ValueError('Complete original windows required')
 cohorts[cohort]=rows
allids=[v['document_id'] for rows in cohorts.values() for v in rows]
if len(allids)!=len(set(allids)) or len(allids)!=35840:raise ValueError('Training and confirmation sources overlap or are missing')
parent_rows=[json.loads(x) for x in bound(plan['parent_uniform_TRAIN']).open()];bound(plan['parent_independent_source_receipt'])
expected_parent=sorted(parent_rows,key=lambda v:hashlib.sha256((plan['salt']+':math-train:'+v['document_id']).encode()).hexdigest())[:16384];math_train=[v for v in cohorts['TRAIN'] if v['domain']=='math'];English_train=[v for v in cohorts['TRAIN'] if v['domain']=='English']
if len(math_train)!=len(English_train)!=16384 or [{k:v for k,v in row.items() if k!='domain'} for row in math_train]!=expected_parent:raise ValueError('Original independently verified mathematical fitting subset changed')
blocked=set()
for spec in plan['fresh_math_exclusions']+p['additional_used_sources']:blocked.update(json.loads(x)['document_id'] for x in bound(spec).open())
with gzip.open(bound(p['population']),'rt') as f:
 selection=heapq.nsmallest(2048,({'document_id':v['document_id'],'positions':v['positions']} for line in f if (v:=json.loads(line))['positions']>0 and not v['used'] and v['document_id'] not in blocked),key=lambda v:hashlib.sha256((plan['salt']+':math:split:'+v['document_id']).encode()).hexdigest())
if [v['document_id'] for v in selection]!=[v['document_id'] for v in cohorts['MATH_CONFIRM']]:raise ValueError('Fresh mathematical identity selection differs')
for choice,row in zip(selection,cohorts['MATH_CONFIRM']):
 expected=512+int(hashlib.sha256((plan['salt']+':math:cut:'+row['document_id']).encode()).hexdigest(),16)%choice['positions']
 if row['provenance']['native_cut']!=expected:raise ValueError('Fresh mathematical position differs')
blocked_pages=set();blocked_content=set()
for filename,digest in web['exclusion_membership_files'].items():
 if sha(filename)!=digest:raise ValueError('Known math source exclusion changed')
 for line in Path(filename).open():
  row=json.loads(line);blocked_pages.add(page(row['url']));blocked_content.update(row[k] for k in ['document_id','content_sha256'] if row.get(k))
old=[json.loads(x) for x in bound(plan['English_exclude_original_diagnostic']).open()];blocked_pages.update(v['canonical_page'] for v in old);blocked_content.update(v['document_id'] for v in old)
tok=AutoTokenizer.from_pretrained(web['tokenizer'],local_files_only=True,trust_remote_code=False);source=Path(plan['English_source_file'])
if sha(source)!=plan['English_source_sha256']:raise ValueError('Original English Parquet changed')
English=English_train+cohorts['ENGLISH_CONFIRM'];saved={v['document_id']:v for v in English};found=set();eligible=[];seen_pages=set();seen_content=set();offset=0
for batch in pq.ParquetFile(source).iter_batches(batch_size=256,columns=['text','url','id']):
 for raw in batch.to_pylist():
  if offset>=plan['English_maximum_source_rows']:break
  number=offset;offset+=1;canonical=page(raw['url']);doc=hashlib.sha256(' '.join(raw['text'].split()).encode()).hexdigest()
  if canonical in blocked_pages or doc in blocked_content or canonical in seen_pages or doc in seen_content:continue
  seen_pages.add(canonical);seen_content.add(doc);ids=tok(raw['text'][:65536],add_special_tokens=False,truncation=True,max_length=8208,return_attention_mask=False)['input_ids'];positions=max(0,min(8192,len(ids)-16)-512+1)
  if not positions:continue
  priority=hashlib.sha256((plan['salt']+':English-select:'+doc).encode()).hexdigest();eligible.append((priority,doc))
  if doc in saved:
   row=saved[doc];cut=512+int(hashlib.sha256((plan['salt']+':English-cut:'+doc).encode()).hexdigest(),16)%positions
   if row['ids']!=ids[cut-512:cut+16] or row['canonical_page']!=canonical or row['provenance']['parquet_row']!=number or row['provenance']['native_cut']!=cut or row['provenance']['source_record_id']!=raw['id']:raise ValueError('Original English window changed')
   found.add(doc)
 if offset>=plan['English_maximum_source_rows']:break
if found!=set(saved) or [doc for _,doc in sorted(eligible)[:17408]]!=[v['document_id'] for v in English]:raise ValueError('Every selected English window and source hash allocation must reproduce')
print('ENGLISH_ORIGINAL_RECORDS_VERIFIED',len(found),flush=True)
downloads={v['filename']:v for v in json.loads(bound(p['parquet_downloads']).read_text())};by_file=defaultdict(list)
for row in cohorts['MATH_CONFIRM']:by_file[row['provenance']['filename']].append(row)
checked_math=0
for filename,rows in by_file.items():
 spec=downloads[filename]
 if sha(spec['path'])!=spec['sha256']:raise ValueError('Original mathematical Parquet changed')
 parquet=pq.ParquetFile(spec['path']);ends=[];total=0
 for g in range(parquet.num_row_groups):total+=parquet.metadata.row_group(g).num_rows;ends.append(total)
 groups=defaultdict(list)
 for row in rows:
  pos=row['provenance']['parquet_row'];g=next(i for i,end in enumerate(ends) if pos<end);groups[g].append((pos-(ends[g-1] if g else 0),row))
 for g,wanted in groups.items():
  original=parquet.read_row_group(g,columns=['text','url'])
  for index,row in wanted:
   raw=original.slice(index,1).to_pylist()[0];doc=hashlib.sha256(' '.join(raw['text'].split()).encode()).hexdigest();ids=tok(raw['text'][:65536],add_special_tokens=False,truncation=True,max_length=8208,return_attention_mask=False)['input_ids'];cut=row['provenance']['native_cut']
   if doc!=row['document_id'] or row['ids']!=ids[cut-512:cut+16] or row['canonical_page']!=page(raw['url']):raise ValueError('Original fresh mathematical record/window differs')
   checked_math+=1
 print('MATH_PARQUET_VERIFIED',filename,checked_math,flush=True)
if checked_math!=2048:raise ValueError('Every mathematical confirmation window required')
result={'status':'PASS_INDEPENDENT_UNIFORM_MIXTURE_SOURCES','utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'files':{k:{'rows':len(v),'sha256':sha(D/(k+'.jsonl'))} for k,v in cohorts.items()},'original_English_windows_reconstructed':17408,'original_fresh_math_windows_reconstructed':2048,'math_fitting_windows_exact_subset_of_previously_verified_sources':16384,'all_sources_separated':True,'identity_and_cut_selection_recomputed':True,'model_calls':0,'wall_seconds':time.monotonic()-start,'scope':'Independent original-record/window reconstruction for all new sources, exact inherited mathematical fitting subset, shared tokenizer and canonical-page definition.'};(D/'independent-source-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
