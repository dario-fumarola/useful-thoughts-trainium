from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,os,shutil,time
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from scripts.prepare_uniform_lm import select_sources
from scripts.prepare_real_cpt_scaffold import page
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/uniform-mixture-20261022';P=C/'prospective-plan.json';p=json.loads(P.read_text());D=Path(p['data_directory']);D.mkdir(parents=True,exist_ok=False);start=time.monotonic();os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);os.environ['TOKENIZERS_PARALLELISM']='false'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(spec):
 path=Path(spec['path'])
 if sha(path)!=spec['sha256']:raise ValueError('Frozen source input differs: '+str(path))
 return path
def check():
 if time.monotonic()-start>p['source_preparation_wall_seconds']:raise TimeoutError('Finite mixture source preparation')
 if min(shutil.disk_usage(v).free for v in [R,D])<35*2**30:raise RuntimeError('Preserve filesystem floor')
 if sum(f.stat().st_size for f in D.rglob('*') if f.is_file())>p['additional_storage_cap_GiB']*2**30:raise RuntimeError('Finite source/output allocation')
parent=json.loads(bound(p['parent_uniform_source_plan']).read_text());web=json.loads(bound(p['English_source_plan']).read_text());train_parent=[json.loads(x) for x in bound(p['parent_uniform_TRAIN']).open()];bound(p['parent_independent_source_receipt']);blocked=set()
for spec in p['fresh_math_exclusions']+parent['additional_used_sources']:blocked.update(json.loads(x)['document_id'] for x in bound(spec).open())
with gzip.open(bound(parent['population']),'rt') as f:math_selected=select_sources((json.loads(x) for x in f),{'MATH_CONFIRM':2048},blocked,p['salt']+':math')
(D/'math-selection.json').write_text(json.dumps(math_selected,indent=2)+'\n');math_by_id={v['document_id']:v for v in math_selected};tok=AutoTokenizer.from_pretrained(web['tokenizer'],local_files_only=True,trust_remote_code=False);math_rows={}
for pool in parent['source_pools']:
 metadata=[v for line in bound(pool['membership']).open() if (v:=json.loads(line))['document_id'] in math_by_id]
 with bound(pool['raw']).open('rb') as f:
  for m in metadata:
   f.seek(m['raw_offset']);line=f.read(m['raw_bytes']);raw=json.loads(line);doc=hashlib.sha256(' '.join(raw['text'].split()).encode()).hexdigest();choice=math_by_id[doc];ids=tok(raw['text'][:65536],add_special_tokens=False,truncation=True,max_length=8208,return_attention_mask=False)['input_ids'];cut=choice['cut']
   if doc!=m['document_id'] or raw['url']!=m['url'] or max(0,min(8192,len(ids)-16)-512+1)!=choice['positions']:raise ValueError('Fixed math source/window population changed')
   math_rows[doc]={'schema':'uniform-next-token-v1','document_id':doc,'item_id':doc+':mixture-math:'+str(cut),'cohort':'MATH_CONFIRM','split':'confirm','domain':'math','canonical_page':page(raw['url']),'ids':ids[cut-512:cut+16],'provenance':{'pool':pool['name'],'filename':m['parquet_filename'],'parquet_row':m['parquet_row'],'native_cut':cut,'raw_line_sha256':hashlib.sha256(line).hexdigest()}}
 if len(math_rows)==2048:break
if set(math_rows)!=set(math_by_id):raise ValueError('Every fresh math source must reconstruct')
print('FRESH_MATH2048_RECONSTRUCTED',flush=True)
blocked_pages=set();blocked_content=set()
for file,digest in web['exclusion_membership_files'].items():
 if sha(file)!=digest:raise ValueError('Original mathematical source exclusions changed')
 for line in Path(file).open():
  row=json.loads(line);blocked_pages.add(page(row['url']));blocked_content.update(row[k] for k in ['document_id','content_sha256'] if row.get(k))
old_English=[json.loads(x) for x in bound(p['English_exclude_original_diagnostic']).open()];blocked_pages.update(v['canonical_page'] for v in old_English);blocked_content.update(v['document_id'] for v in old_English)
source=Path(p['English_source_file'])
if sha(source)!=p['English_source_sha256']:raise ValueError('Pinned English source changed')
seen_pages=set();seen_content=set();eligible=[];offset=0
for batch in pq.ParquetFile(source).iter_batches(batch_size=256,columns=['text','url','id']):
 for raw in batch.to_pylist():
  if offset>=p['English_maximum_source_rows']:break
  number=offset;offset+=1;canonical=page(raw['url']);doc=hashlib.sha256(' '.join(raw['text'].split()).encode()).hexdigest()
  if canonical in blocked_pages or doc in blocked_content or canonical in seen_pages or doc in seen_content:continue
  seen_pages.add(canonical);seen_content.add(doc);ids=tok(raw['text'][:65536],add_special_tokens=False,truncation=True,max_length=8208,return_attention_mask=False)['input_ids'];positions=max(0,min(8192,len(ids)-16)-512+1)
  if not positions:continue
  cut=512+int(hashlib.sha256((p['salt']+':English-cut:'+doc).encode()).hexdigest(),16)%positions;selection=hashlib.sha256((p['salt']+':English-select:'+doc).encode()).hexdigest()
  eligible.append({'schema':'uniform-next-token-v1','document_id':doc,'item_id':doc+':mixture-English:'+str(cut),'domain':'English','canonical_page':canonical,'ids':ids[cut-512:cut+16],'selection_hash':selection,'provenance':{'dataset':web['source_dataset'],'revision':web['source_revision'],'parquet_row':number,'native_cut':cut,'source_record_id':raw['id'],'source_text_sha256':hashlib.sha256(raw['text'].encode()).hexdigest()}})
 if offset>=p['English_maximum_source_rows']:break
 check()
eligible.sort(key=lambda v:v['selection_hash'])
if len(eligible)<17408:raise ValueError('Fixed additional English frame does not supply the declared allocation')
English_train=[dict(v,cohort='TRAIN',split='train') for v in eligible[:16384]];English_confirm=[dict(v,cohort='ENGLISH_CONFIRM',split='confirm') for v in eligible[16384:17408]]
math_train=[dict(v,domain='math') for v in sorted(train_parent,key=lambda v:hashlib.sha256((p['salt']+':math-train:'+v['document_id']).encode()).hexdigest())[:16384]]
cohorts={'TRAIN':math_train+English_train,'MATH_CONFIRM':[math_rows[v['document_id']] for v in math_selected],'ENGLISH_CONFIRM':English_confirm};allids=[v['document_id'] for rows in cohorts.values() for v in rows]
if len(allids)!=len(set(allids)):raise ValueError('Every training and confirmation source must be disjoint')
files={}
for cohort,rows in cohorts.items():
 if any(len(v['ids'])!=528 for v in rows):raise ValueError('Keep every original window intact')
 file=D/(cohort+'.jsonl');file.write_text(''.join(json.dumps(v,separators=(',',':'))+'\n' for v in rows));files[cohort]={'path':str(file),'rows':len(rows),'sha256':sha(file),'role':'training' if cohort=='TRAIN' else 'confirmation'}
result={'status':'COMPLETE_UNIFORM_LM_SOURCE_PREPARATION','identity':p['identity'],'utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(P),'files':files,'TRAIN_domain_counts':{'math':16384,'English':16384},'English_frame_rows_scanned':offset,'eligible_unused_English_documents':len(eligible),'source_models_queried':0,'parent_TRAIN_source_proof':p['parent_independent_source_receipt'],'wall_seconds':time.monotonic()-start,'target_weight':'Every original shifted token equal; equal domain counts and window lengths make the fitting objective50/50'};(D/'complete.json').write_text(json.dumps(result,indent=2)+'\n');check();print(json.dumps(result,indent=2),flush=True)
