"""Format-neutral real partial-state data. No model, interface or training release."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
from fractions import Fraction
import hashlib,json,shutil
from transformers import AutoTokenizer
from train.calculation_boundaries import maximal_lhs,parse_numeric
from train.symbolic_note_labels import postorder_steps
from weekend_thinking.artifacts import canonical_hash
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/real-prefix-partial-labels-v9-v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def derive_from_prefix(prefix_ids,tokenizer):
 # No source continuation, gold number, stored note or future-dependent field.
 past=tokenizer.decode(prefix_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
 expression=maximal_lhs(past);value,steps=postorder_steps(expression)
 if len(steps)<2:raise ValueError('Nonroot supervision needs a compound expression')
 partial='\n'.join(steps[:-1])+'\n';full='\n'.join(steps)+'\n'
 return dict(expression=parse_numeric(expression).expression,value=value,
             partial_text=partial,partial_ids=tokenizer.encode(partial,add_special_tokens=False),
             full_text=full,full_ids=tokenizer.encode(full,add_special_tokens=False),steps=steps,past=past)
def writer_question(row):
 # The future and all labels stay outside the actual writer-input projection.
 return dict(item_id=row['item_id'],document_id=row['document_id'],prefix_ids=list(row['prefix_ids']))
p=read(C/'plan.json');assert p['script_sha256']==sha(__file__)
assert all(sha(Path(k))==v for k,v in p['label_source_files'].items())
source_completion=read(R/'local/diagnostics/real-raw-compound-prefix-v9-v1/completion.json');assert source_completion['status']=='COMPLETE_RAW_COMPOUND_CPU_CURATION_NOT_TRAINING_RELEASE'
manifest_file=Path(p['source_manifest']);assert source_completion['manifest_sha256']==sha(manifest_file)
m=read(manifest_file);assert m['minimum_binary_ops']==2 and sha(m['accepted_file'])==m['accepted_sha256']
O=Path(p['output']);assert not O.exists();O.mkdir(parents=True,exist_ok=False)
tok=AutoTokenizer.from_pretrained(R/'local/assets/qwen3-1.7b-base',local_files_only=True,trust_remote_code=False)
assert isinstance(tok.eos_token_id,int);cohorts={'train':[],'dev':[]};documents=set();excluded=[];all_inventory={k:[] for k in cohorts}
with Path(m['accepted_file']).open() as source:
 for lineno,line in enumerate(source,1):
  row=json.loads(line);assert row['split'] in cohorts and row['document_id'] not in documents;documents.add(row['document_id'])
  assert all(row.get(k) is not None for k in ['raw_line_sha256','raw_text_sha256','prefix_source_char_end','raw_rhs_validation'])
  d=derive_from_prefix(row['prefix_ids'],tok);assert d['past']==row['prefix_text'] and d['value']==Fraction(row['numeric_value'])
  assert len(row['natural_future16_ids'])==16 and row['boundary']['prefix_sha256']==canonical_hash(row['prefix_ids'])
  inventory=dict(item_id=row['item_id'],document_id=row['document_id'],split=row['split'],expression=d['expression'],nonroot_steps=len(d['steps'])-1,partial_actions=len(d['partial_ids'])+1,full_trace_actions=len(d['full_ids'])+1,incidental_intermediate_equals_root=any(Fraction(s.rsplit(' = ',1)[1])==d['value'] for s in d['steps'][:-1]))
  all_inventory[row['split']].append(inventory)
  if inventory['partial_actions']>64:
   excluded.append(dict(**inventory,reason='declared64action_partial_budget'));continue
  out=dict(item_id=row['item_id'],document_id=row['document_id'],split=row['split'],prefix_ids=row['prefix_ids'],prefix_text=d['past'],continuation16_ids=row['natural_future16_ids'],numeric_surface_ids=row['numeric_surface_ids'],numeric_surface_text=row['numeric_surface_text'],closing_boundary_ids=row['closing_boundary_ids'],numeric_value=row['numeric_value'],normalized_expression=d['expression'],partial_note_text=d['partial_text'],partial_note_ids=d['partial_ids'],full_calculation_text=d['full_text'],full_calculation_ids=d['full_ids'],stop_id=tok.eos_token_id,source_accepted_line=lineno,source_accepted_line_sha256=hashlib.sha256(line.encode()).hexdigest(),source_provenance={k:row[k] for k in ['raw_line_sha256','raw_text_sha256','raw_source_line','source_url','prefix_source_char_end','source_chunk_char_start','source_chunk_char_end','absolute_source_cut','raw_rhs_validation']})
  assert set(writer_question(out))=={'item_id','document_id','prefix_ids'}
  cohorts[row['split']].append(out)
print('All raw accepted rows rederived from causal prefixes',len(documents),flush=True)
for split,rows in cohorts.items():rows.sort(key=lambda r:canonical_hash(dict(salt=p['selection_salt'],split=split,document_id=r['document_id'])))
available={k:len(v) for k,v in cohorts.items()};enough=available['train']>=5120 and available['dev']>=256
selected={k:v[:(5120 if k=='train' else 256)] for k,v in cohorts.items()} if enough else {}
files={}
for split,rs in selected.items():
 path=O/f'{split.upper()}.jsonl'
 with path.open('x') as f:
  for row in rs:f.write(json.dumps(row,separators=(',',':'))+'\n')
 files[path.name]=dict(rows=len(rs),sha256=sha(path),bytes=path.stat().st_size)
with (O/'all-candidate-inventory.jsonl').open('x') as f:
 for vs in all_inventory.values():
  for x in vs:f.write(json.dumps(x,separators=(',',':'))+'\n')
write(O/'exclusions.json',excluded)
assert all(sha(Path(k))==v for k,v in p['label_source_files'].items())
train_expr={r['normalized_expression'] for r in selected.get('train',[])}
result=dict(schema='real-prefix-partial-labels-v9-v1',status='COMPLETE_FORMAT_NEUTRAL_DATA_NOT_TRAINING_RELEASE' if enough else 'INSUFFICIENT_DECLARED_DATA_POOL_NO_COHORT',utc=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=sha(manifest_file),source_accepted_sha256=m['accepted_sha256'],source_curator_commit=p['source_curator_commit'],projection_script_sha256=sha(__file__),selection_salt=p['selection_salt'],available_fitting64=available,raw_documents_by_split={k:len(v) for k,v in all_inventory.items()},selected_files=files,excluded_by_reason=dict(Counter(x['reason'] for x in excluded)),selected_document_overlap=len({x['document_id'] for x in selected.get('train',[])}&{x['document_id'] for x in selected.get('dev',[])}),selected_DEV_expression_seen_in_TRAIN=sum(x['normalized_expression'] in train_expr for x in selected.get('dev',[])),selected_DEV_novel_expression_item_ids=[x['item_id'] for x in selected.get('dev',[]) if x['normalized_expression'] not in train_expr],writer_projection_fields=['item_id','document_id','prefix_ids'],teacher_label_inputs='Native decoded causal prefix only. Exact interpreter nonroot postorder; final root operation omitted. Full trace supplied separately for competitive baselines.',limitations=['No model/interface/learning release or capability claim.','Real FineMath document-disjoint TRAIN/DEV; developmental, not official TEST.','Source equation acceptance uses observed RHS truth; label derivation itself receives no future.','Incidental equality of an intermediate and final value is logged and not filtered.','Existing full-answer oracle note fields are not copied into writer targets.'])
write(O/'manifest.json',result)
A=R/'local/backups/data/real-prefix-partial-labels-v9-v1';need=sum(f.stat().st_size for f in O.iterdir() if f.is_file());assert shutil.disk_usage(R).free>35*2**30+need;A.mkdir(parents=True,exist_ok=False);archived={}
for label,source in [('output',O),('control',C)]:
 for q in source.rglob('*'):
  if not q.is_file():continue
  dest=A/label/q.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dest);digest=sha(q);assert sha(dest)==digest;archived[str(dest.relative_to(A))]=dict(bytes=q.stat().st_size,sha256=digest)
write(A/'ARCHIVE.json',dict(status='ALL_FORMAT_NEUTRAL_DATA_READBACK_VERIFIED',files=archived))
write(C/'completion.json',dict(status=result['status'],manifest_sha256=sha(O/'manifest.json'),EBS_archive_receipt_sha256=sha(A/'ARCHIVE.json')))
print(json.dumps(dict(status=result['status'],available=available,selected_files=files,selected_DEV_expression_seen_in_TRAIN=result['selected_DEV_expression_seen_in_TRAIN']),indent=2),flush=True)
