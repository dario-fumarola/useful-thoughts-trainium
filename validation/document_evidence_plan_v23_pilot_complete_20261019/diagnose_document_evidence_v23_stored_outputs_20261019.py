"""Posthoc stored-output diagnosis;no model runs,rescoring or gate changes."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import gzip,hashlib,json,re
from train.document_qa_answers import normalize_answer
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
F=R/'validation/document_evidence_plan_v23_pilot_complete_20261019.json';P=F.with_suffix('')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();original=sha(F)
read=lambda p:json.loads(Path(p).read_text())
def payload(name):return [json.loads(line) for line in gzip.decompress((P/name).read_bytes()).splitlines()]
gold={v['item_id']:v for v in payload('fixed-held-gold-annotations.jsonl.gz')}
allrows={};prefixes={}
for job in read(C/'plan.json')['jobs']:
 arm=job['arm'];allrows[arm]={};prefixes[arm]={}
 for mode in read(F)['tables'][arm]:
  values=payload(f'{job["key"]}-{mode}-native-and-scored.jsonl.gz');allrows[arm][mode]={v['native']['item_id']:v for v in values}
  rows=[]
  for value in values:
   text=value['native']['raw_text'];match=re.search(r'(?im)^[ \t]*Final answer:[ \t]*',text);prefix=text[:match.start()] if match else text
   ids=list(map(int,re.findall(r'\[([0-9]+)\]',prefix)))
   rows.append(dict(item_id=value['native']['item_id'],prefix=prefix,ids=ids))
  prefixes[arm][mode]=rows
summaries={arm:{mode:dict(items=len(values),nonempty_generated_prefixes=sum(bool(v['prefix'].strip()) for v in values),generated_bracket_ID_lists=sum(bool(v['ids']) for v in values),ID_count_distribution=dict(Counter(len(v['ids']) for v in values))) for mode,values in modes.items()} for arm,modes in prefixes.items()}
B=allrows['evidence_select'];own=B['selection'];direct=B['direct'];oracle=B['oracle']
exact=[key for key,v in own.items() if v['score']['annotated_support_overlap']['exact_match']]
notexact=sorted(set(own)-set(exact))
def group(keys):
 return dict(items=len(keys),own_correct=sum(own[k]['score']['canonical_em'] for k in keys),same_direct_correct=sum(direct[k]['score']['canonical_em'] for k in keys),oracle_correct=sum(oracle[k]['score']['canonical_em'] for k in keys),own_oracle_correctness_disagreements=sum(own[k]['score']['canonical_em']!=oracle[k]['score']['canonical_em'] for k in keys),own_oracle_normalized_answer_disagreements=sum(normalize_answer(own[k]['score']['parsed']['answer'])!=normalize_answer(oracle[k]['score']['parsed']['answer']) for k in keys))
def transitions(control,treatment):
 return dict(improved=sum(treatment[k]['score']['canonical_em']>control[k]['score']['canonical_em'] for k in control),regressed=sum(treatment[k]['score']['canonical_em']<control[k]['score']['canonical_em'] for k in control),both_correct=sum(treatment[k]['score']['canonical_em']==control[k]['score']['canonical_em']==1 for k in control),both_incorrect=sum(treatment[k]['score']['canonical_em']==control[k]['score']['canonical_em']==0 for k in control))
bydepth={}
for hop in sorted({v['hops'] for v in gold.values()}):
 keys=[k for k,v in gold.items() if v['hops']==hop]
 bydepth[str(hop)]=dict(items=len(keys),exact_annotated_support_sets=sum(own[k]['score']['annotated_support_overlap']['exact_match'] for k in keys),generated_ID_counts=dict(Counter(len(own[k]['score']['selection']['indices']) for k in keys)),teacher_ID_counts=dict(Counter(len(gold[k]['support_indices']) for k in keys)))
result=dict(status='POSTHOC_STORED_SELECTION_AND_EMPTY_PREFIX_DIAGNOSIS_NO_SCORE_CHANGE',utc=datetime.now(timezone.utc).isoformat(),primary_report_sha256=original,source_sha256=sha(__file__),
 generated_prefix_summary=summaries,exact_annotation_set_subgroup=group(exact),nonexact_annotation_set_subgroup=group(notexact),
 transitions_own_minus_same_direct=transitions(direct,own),transitions_own_minus_matched_direct=transitions(allrows['evidence_direct']['direct'],own),
 support_selection_by_depth_descriptive=bydepth,
 interpretation=['SuppliedemptyEvidencefield isnot guaranteed noteomission when modelgenerates anIDline afterward. Reportthis controlas operationally contaminated,retainitsoriginal score. Same-modeldirect andmatched-direct contrasts areunchanged.',
  'Annotationagreement andsubgroups aredescriptive,not semanticgold orcausalestimates. Exactsetconditioning isposthoc andcorrelated withquestiondifficulty.',
  'Alloriginalgates/data/predictions unchanged. No new model calls. Four-hop subgroupn4 cannotestablishdepthgeneralization.'])
(P/'stored-selection-and-empty-prefix-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
assert sha(F)==original
print(json.dumps(result,indent=2))
