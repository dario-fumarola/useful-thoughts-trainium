"""Descriptive format/answer-alias/hop inspection;never changes frozen scores.

Prepared without viewing outcomes. These are interpretation aids,not additional
primary endpoints,causal effects or new study gates.
"""
from pathlib import Path
from datetime import datetime,timezone
import csv,gzip,hashlib,json
from collections import Counter
from tokenizers import Tokenizer
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-QA-v21-study-v1';F=R/'validation/document_QA_v21_complete_20261019.json';P=F.with_suffix('')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(F);assert read(C/'final-readout-completion.json')['status']=='ALL_SEVEN_MODELS_COMPLETE_ANALYZED_INDEPENDENTLY_CHECKED_RENDERED_AND_PROTECTED';assert read(P/'independent-readout-check.json')['report_sha256']==sha(F)
def values(name):return [json.loads(l) for l in gzip.decompress((P/name).read_bytes()).splitlines()]
gold={v['item_id']:v for v in values('fixed-held-gold-metadata.jsonl.gz')};data={};strata=[];diagnostics=[]
tok=Tokenizer.from_file(str(R/'local/assets/qwen3-1.7b-base/tokenizer.json'));native_targets={};checked_targets=0
for model in r['tables']:
 for v in values(model+'-all-likelihood-rows.jsonl.gz'):
  if v['panel']!='QA':continue
  assert v['target_ids'][-1]==151643 and tok.decode(v['target_ids'][:-1],skip_special_tokens=False)==gold[v['item_id']]['answer']
  if v['item_id'] not in native_targets:native_targets[v['item_id']]=v['target_ids']
  assert native_targets[v['item_id']]==v['target_ids'];checked_targets+=1
assert checked_targets==17920 and len(native_targets)==512
for model,modes in r['tables'].items():
 data[model]={m:{} for m in modes}
 for v in values(model+'-all-generation-attempts.jsonl.gz'):
  data[model][v['mode']][v['item_id']]=v
  if v['parsed']['status']!='complete' or (v['attributed_alias_em'] and not v['canonical_em']):
   diagnostics.append(dict(model=model,mode=v['mode'],item_id=v['item_id'],parsed_status=v['parsed']['status'],execution_status=v['status'],stop_type=v['stop_type'],canonical_em=v['canonical_em'],canonical_f1=v['canonical_f1'],attributed_alias_em=v['attributed_alias_em'],prediction=v['parsed']['answer'],canonical=gold[v['item_id']]['answer'],answer_aliases=gold[v['item_id']]['answer_aliases'],text=v['text']))
 for mode,byid in data[model].items():
  for hops in sorted({v['hops'] for v in gold.values()}):
   selected=[v for k,v in byid.items() if gold[k]['hops']==hops];n=len(selected)
   strata.append(dict(model=model,mode=mode,hops=hops,questions=n,canonical_correct=sum(v['canonical_em'] for v in selected),canonical_f1=sum(v['canonical_f1'] for v in selected)/n,attributed_alias_correct=sum(v['attributed_alias_em'] for v in selected),unsupported=sum(v['parsed']['status']!='complete' for v in selected),capped=sum(v['stop_type']=='length_cap' for v in selected)))
comparisons={}
for seed in r['seeds']:
 d=f'qa_direct-seed{seed}';p=f'qa_prior-seed{seed}';post=f'qa_post-seed{seed}'
 pairs=dict(post_direct_minus_direct_direct=(d,'direct',post,'direct'),prior_prior_minus_prior_empty=(p,'empty',p,'prior'),prior_prior_minus_direct_reference=(d,'reference_cot',p,'prior'),prior_prior_minus_B0_reference=('B0','reference_cot',p,'prior'))
 comparisons[seed]={}
 for name,(a,am,b,bm) in pairs.items():
  left,right=data[a][am],data[b][bm];cells={}
  for key in ('both_complete','reference_only_complete','treatment_only_complete','neither_complete'):
   selected=[]
   for item in gold:
    first=left[item]['parsed']['status']=='complete';second=right[item]['parsed']['status']=='complete'
    group='both_complete' if first and second else 'reference_only_complete' if first else 'treatment_only_complete' if second else 'neither_complete'
    if group==key:selected.append(item)
   cells[key]=dict(questions=len(selected),reference_correct=sum(left[k]['canonical_em'] for k in selected),treatment_correct=sum(right[k]['canonical_em'] for k in selected),net_extra_canonical_correct=sum(right[k]['canonical_em']-left[k]['canonical_em'] for k in selected))
  assert sum(v['questions'] for v in cells.values())==512 and sum(v['net_extra_canonical_correct'] for v in cells.values())==sum(right[k]['canonical_em']-left[k]['canonical_em'] for k in gold)
  comparisons[seed][name]=cells
out=dict(status='DESCRIPTIVE_FORMAT_ALIAS_AND_HOP_INSPECTION_NO_SCORE_OR_GATE_CHANGES',utc=datetime.now(timezone.utc).isoformat(),report_sha256=sha(F),source_sha256=sha(__file__),hop_counts=dict(Counter(v['hops'] for v in gold.values())),full_native_canonical_answer_target_vectors_verified=checked_targets,format_accounting=comparisons,hop_cells=strata,diagnostic_rows=len(diagnostics),limits=['Netcorrect accounting retains all512questiondenominators;bothcomplete subset is descriptive,not a causal estimate.','Hopstrata are unequal/small,especially4hop;not unseenstructure or pretrainingnovelty tests.','Alias-only matches use attributedmirrorannotations;not hand-repaired canonicalscores or independentlyverifiedvalidparaphrases.','No newthresholds,checkpointselection,questions,modelcalls orprimaryclaims.'])
(P/'format-alias-hop-inspection.json').write_text(json.dumps(out,indent=2));(P/'all-unsupported-or-alias-only-diagnostics.jsonl.gz').write_bytes(gzip.compress((''.join(json.dumps(v)+'\n' for v in diagnostics)).encode(),mtime=0))
with (P/'hop-strata-descriptive.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(strata[0]));w.writeheader();w.writerows(strata)
print(out['status'],flush=True)
